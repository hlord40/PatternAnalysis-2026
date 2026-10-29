import argparse
import warnings
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')                      # save plots to files (no screen needed)
import matplotlib.pyplot as plt
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')   # out-of-gamut notices


def parse_args():
    p = argparse.ArgumentParser(description='Fine-tune MambaIRv2 for Lab colourisation')
    p.add_argument('--data', required=True)
    p.add_argument('--out', default=os.path.expanduser('~/runs/main'))
    p.add_argument('--weights', default=None)
    p.add_argument('--iters', type=int, default=10000)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--crop', type=int, default=128)
    p.add_argument('--lr_new', type=float, default=2e-4)
    p.add_argument('--lr_body', type=float, default=2e-5)
    p.add_argument('--val_every', type=int, default=500)
    p.add_argument('--log_every', type=int, default=50)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=67)
    p.add_argument('--time_limit', type=float, default=None)
    p.add_argument('--no_checkpoint', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--test_only', action='store_true')
    return p.parse_args()


#defining colourfulness
def colourfulness(rgb):
    """Hasler & Suesstrunk (2003) colourfulness of an RGB uint8 image (higher = more colourful)."""
    r, g, b = (rgb[..., i].astype(np.float64) for i in range(3))
    rg, yb = r - g, 0.5 * (r + g) - b
    return np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)

#class for all metrics (PSNR, SSIM, LPIPS and colourfulness)
class Metrics:
    def __init__(self, device):
        self.device = device
        self.lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(device).eval()

    def _to_tensor(self, rgb):
        # uint8 (H, W, 3) into float (1, 3, H, W) within [-1, 1] for LPIPS
        return torch.from_numpy(rgb).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(self.device)

    @torch.no_grad() #we need this here to stop gradient tracking for LPIPS
    def __call__(self, pred, gt):
        return {'psnr': peak_signal_noise_ratio(gt, pred, data_range=255),
                'ssim': structural_similarity(gt, pred, channel_axis=-1, data_range=255),
                'lpips': self.lpips_fn(self._to_tensor(pred), self._to_tensor(gt)).item(),
                'colour_pred': colourfulness(pred), #predicted colorfulness
                'colour_gt': colourfulness(gt)} #ground truth colorfulness


def summarise(rows):
#for each metric, take the mean across all images in the validation or test set (e.g. 100 images in the held-out DIV2K test set)
    m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    m['colour_ratio'] = m['colour_pred'] / m['colour_gt'] # colourfulness ratio (prediction / truth)
    return m



#Small Helper functions

@torch.no_grad()#no gradient tracking for this function, as we are just predicting ab channels from L channel
def predict_ab(model, L):
#takes the input L (B, 1, H, W) on the GPU, passes through model for the predicted ab (B, 2, H, W), and then clamped to the valid range
    with torch.autocast('cuda', dtype=torch.bfloat16):
        ab = model(L)
    return ab.float().clamp(-1, 1)


def save_panel(path, images, titles): #used in validation and testing
    fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5))
    for ax, im, title in zip(axes, images, titles):
        ax.imshow(im)
        ax.set_title(title)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_history(history, path):
#shows the training loss in top left, validation loss top middle, the other 4 in the 2 x 3 grid are the other metrics
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    if history['train_loss']: #for training loss, top left
        its, loss = zip(*history['train_loss'])
        axes[0, 0].plot(its, loss, alpha=0.25, label='per step')
        if len(loss) >= 50: #makes a smoother line for the training loss by taking a moving average over 50 steps
            smooth = np.convolve(loss, np.ones(50) / 50, mode='valid')
            axes[0, 0].plot(its[49:], smooth, label='moving average (50 steps)')
        axes[0, 0].set_title('Training loss (L1 on ab)')
        axes[0, 0].set_xlabel('iteration')
        axes[0, 0].legend()

    val = history['val']
    val_its = [v['iter'] for v in val]
    panels = [('l1_ab', 'Validation loss (L1 on ab)'),
              ('psnr', 'Validation PSNR (dB)'),
              ('ssim', 'Validation SSIM'),
              ('lpips', 'Validation LPIPS (lower is better)'),
              ('colour_ratio', 'Colourfulness ratio (prediction / truth)')]
    for ax, (key, title) in zip(axes.flat[1:], panels):
        ax.plot(val_its, [v[key] for v in val], marker='o')
        ax.set_title(title)
        ax.set_xlabel('iteration')
    axes[1, 2].axhspan(0.85, 1.15, color='green', alpha=0.15, label='AC1 target (within 15%)')
    axes[1, 2].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)