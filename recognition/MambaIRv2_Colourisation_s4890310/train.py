import argparse
import warnings
import os
import json
import numpy as np
import torch
import time
import random
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')                      # save plots to files (no screen needed)
import matplotlib.pyplot as plt
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from modules import build_pretrained_MambaDenoiser, LabColouriser
from dataset import get_data_loaders, lab_tensors_to_rgb

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


#Validation

@torch.no_grad() #no gradient tracking since we are just evaluating the model on the validation set
def evaluate(model, loader, metrics, device, save_dir=None, n_save=3):
    """Colourise every image in the loader and return the averaged metrics."""
    model.eval() #no checkpointing
    rows, l1 = [], []
    for k, (L, ab) in enumerate(loader):
        L, ab = L.to(device), ab.to(device)
        ab_pred = predict_ab(model, L)
        l1.append(F.l1_loss(ab_pred, ab).item())

        pred = lab_tensors_to_rgb(L[0], ab_pred[0])
        gt = lab_tensors_to_rgb(L[0], ab[0])
        rows.append(metrics(pred, gt))

        if save_dir is not None and k < n_save: #save 3 images
            grey = lab_tensors_to_rgb(L[0], torch.zeros_like(ab[0]))
            save_panel(os.path.join(save_dir, f'val_{k}.png'), [grey, pred, gt],
                       ['Greyscale input', 'Prediction', 'Ground truth'])
    model.train() #back into training mode
    result = summarise(rows)
    result['l1_ab'] = float(np.mean(l1))
    return result


def save_checkpoint(path, model, opt, sched, it, history, best): #everything able to resume training is saved
    torch.save({'params': model.state_dict(), 'optimizer': opt.state_dict(),
                'scheduler': sched.state_dict(), 'iter': it, 'history': history, 'best': best}, path)


def validate_and_save(model, val_loader, metrics, device, args, it, history, opt, sched, best):
#run validation, save the best checkpoint, and save the training history and plots
    sample_dir = os.path.join(args.out, 'samples', f'iter{it:06d}')
    os.makedirs(sample_dir, exist_ok=True)
    v = evaluate(model, val_loader, metrics, device, save_dir=sample_dir)
    v['iter'] = it
    history['val'].append(v)
    print(f'[val] iter {it}: L1 {v["l1_ab"]:.4f}  PSNR {v["psnr"]:.2f}  SSIM {v["ssim"]:.4f}  '
          f'LPIPS {v["lpips"]:.4f}  colourfulness ratio {v["colour_ratio"]:.3f}', flush=True)

    if v['lpips'] < best:
        best = v['lpips']
        torch.save({'params': model.state_dict(), 'iter': it, 'val': v},
                   os.path.join(args.out, 'best.pth'))
        print(f'  new best checkpoint (LPIPS {best:.4f})', flush=True)

    save_checkpoint(os.path.join(args.out, 'last.pth'), model, opt, sched, it, history, best)
    plot_history(history, os.path.join(args.out, 'training_curves.png'))
    with open(os.path.join(args.out, 'history.json'), 'w') as f:
        json.dump(history, f)
    return best


#Testing

@torch.no_grad() #no gradient tracking since we are just evaluating the model on the test set
def test(model, test_loader, metrics, device, args):
#test the best checkpoint saved and compare to the untuned pretrained MambaIRv2 model
    ckpt = torch.load(os.path.join(args.out, 'best.pth'), map_location='cpu')
    model.load_state_dict(ckpt['params'])
    model.eval() #evluation mode (no checkpointing)
    print(f'testing best checkpoint from iteration {ckpt["iter"]}', flush=True) #says the iteration number of the best chekpoint

    untuned = build_pretrained_MambaDenoiser(args.weights).to(device).eval()   # baseline: RGB in, RGB out

    names = ['Greyscale input', 'Untuned MambaIRv2', 'Fine-tuned']
    rows = {n: [] for n in names}
    times, peak = [], 0.0
    bleed_dir = os.path.join(args.out, 'test', 'bleed_check')       # for AC4, inspect these by eye
    os.makedirs(bleed_dir, exist_ok=True)
    bleed_ids = set(random.Random(args.seed).sample(range(len(test_loader)), min(20, len(test_loader))))

    for k, (L, ab) in enumerate(test_loader):
        L, ab = L.to(device), ab.to(device)

        # for finetuned model: time and peak memory for AC5
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t = time.time()
        ab_pred = predict_ab(model, L)
        torch.cuda.synchronize()
        times.append(time.time() - t)
        peak = max(peak, torch.cuda.max_memory_allocated() / 1e9)

        gt = lab_tensors_to_rgb(L[0], ab[0])
        grey = lab_tensors_to_rgb(L[0], torch.zeros_like(ab[0]))
        tuned = lab_tensors_to_rgb(L[0], ab_pred[0])

        # untuned pretrained model, given the greyscale image as RGB
        grey_t = torch.from_numpy(grey).permute(2, 0, 1)[None].float().div(255).to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            base = untuned(grey_t).float().clamp(0, 1)
        base = (base[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)

        for name, img in zip(names, [grey, base, tuned]):
            rows[name].append(metrics(img, gt))
        if k in bleed_ids:
            save_panel(os.path.join(bleed_dir, f'test_{k:03d}.png'), [grey, base, tuned, gt],
                       ['Greyscale input', 'Untuned MambaIRv2', 'Fine-tuned (ours)', 'Ground truth'])

    results = {n: summarise(rows[n]) for n in names}
    mean_time = float(np.mean(times[1:] if len(times) > 1 else times))   # skip the warm-up image
    results['inference'] = {'mean_seconds_per_image': mean_time, 'peak_vram_gb': peak}

    tuned, untuned_r = results['Fine-tuned'], results['Untuned MambaIRv2']
    checks = {
        'AC1 colourfulness within 15% of ground truth': abs(tuned['colour_ratio'] - 1) <= 0.15,
        'AC2 LPIPS lower than untuned MambaIRv2': tuned['lpips'] < untuned_r['lpips'],
        'AC3 SSIM no lower than untuned MambaIRv2': tuned['ssim'] >= untuned_r['ssim'],
        'AC5 peak VRAM < 15 GB and < 1 s per image': peak < 15 and mean_time < 1,
    }
    results['acceptance_criteria'] = checks

    print('\nTEST RESULTS (100 held-out DIV2K images, longest side 512 px)')
    print(f'{"method":<20} {"PSNR":>7} {"SSIM":>7} {"LPIPS":>7} {"colour ratio":>13}')
    for n in names:
        r = results[n]
        print(f'{n:<20} {r["psnr"]:7.2f} {r["ssim"]:7.4f} {r["lpips"]:7.4f} {r["colour_ratio"]:13.3f}')
    print(f'inference: {mean_time:.3f} s/image, peak VRAM {peak:.2f} GB')
    for name, ok in checks.items():
        print(f'  {"PASS" if ok else "FAIL"}  {name}')
    print(f'  CHECK BY EYE  AC4 colour bleeding: see {bleed_dir}')

    with open(os.path.join(args.out, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

#Main

def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('A CUDA GPU is required (MambaIRv2 uses mamba_ssm CUDA kernels).')
    device = torch.device('cuda')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    train_loader, val_loader, test_loader = get_data_loaders(args.data, crop=args.crop, batch_size=args.batch, seed=args.seed, num_workers=args.workers)
    metrics = Metrics(device)
    model = LabColouriser(build_pretrained_MambaDenoiser(args.weights),
                          use_checkpoint=not args.no_checkpoint).to(device)

    if not args.test_only:
        # two learning rates: new layers learn fast, the pretrained body adapts slowly
        new_params = list(model.conv_in.parameters()) + list(model.conv_out.parameters())
        new_ids = {id(p) for p in new_params}
        body_params = [p for p in model.parameters() if p.requires_grad and id(p) not in new_ids]
        opt = torch.optim.Adam([{'params': new_params, 'lr': args.lr_new},
                                {'params': body_params, 'lr': args.lr_body}])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters)

        it, best = 0, float('inf')
        history = {'train_loss': [], 'val': [], 'args': vars(args)}
        last_path = os.path.join(args.out, 'last.pth')
        if args.resume and os.path.exists(last_path):
            ckpt = torch.load(last_path, map_location='cpu')
            model.load_state_dict(ckpt['params'])
            opt.load_state_dict(ckpt['optimizer'])
            sched.load_state_dict(ckpt['scheduler'])
            it, history, best = ckpt['iter'], ckpt['history'], ckpt['best']
            print(f'resumed from iteration {it}', flush=True)

        if it == 0:   # "before fine-tuning" numbers
            best = validate_and_save(model, val_loader, metrics, device, args, it, history, opt, sched, best)

        print(f'training for {args.iters} iterations: batch {args.batch}, crop {args.crop}', flush=True)
        model.train()
        start = time.time()
        stop = False
        while it < args.iters and not stop:
            for L, ab in train_loader:
                L, ab = L.to(device, non_blocking=True), ab.to(device, non_blocking=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):    # faster, less memory
                    pred = model(L)
                loss = F.l1_loss(pred.float(), ab)                     # L1 on the ab channels

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                it += 1
                history['train_loss'].append((it, loss.item()))

                hours = (time.time() - start) / 3600
                stop = args.time_limit is not None and hours >= args.time_limit
                if it % args.log_every == 0:
                    print(f'iter {it}/{args.iters}  loss {loss.item():.4f}  '
                          f'lr {sched.get_last_lr()[1]:.2e}  '
                          f'peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.1f} GB  '
                          f'{hours:.2f} h', flush=True)
                if it % args.val_every == 0 or it == args.iters or stop:
                    best = validate_and_save(model, val_loader, metrics, device, args,
                                             it, history, opt, sched, best)
                if it >= args.iters or stop:
                    if stop:
                        print(f'time limit of {args.time_limit} h reached at iteration {it}', flush=True)
                    break

    test(model, test_loader, metrics, device, args)


if __name__ == '__main__':
    main()