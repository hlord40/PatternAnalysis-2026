import os
import re
from modules import build_pretrained_MambaDenoiser, LabColouriser
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import glob
import torch.nn.functional as F

from dataset import lab_tensors_to_rgb, rgb_to_lab_tensors

fine_tuned_weights = os.path.expanduser('~/runs/main/best.pth')
pretrained_model = os.path.expanduser('~/comp3170/MambaIR/experiments/mambairv2_ColorDN_15.pth')
input_test = os.path.expanduser('~/comp3170/greyscale-colorization/test-dataset')
output_test = os.path.expanduser('~/runs/predictions')

max_side = 1024

def sort_test_folder(path): #sorts the test folder so that images are in numerical order
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', os.path.basename(path))]

def load_model(device):
    model = LabColouriser(build_pretrained_MambaDenoiser(pretrained_model), use_checkpoint=False)
    model.load_state_dict(torch.load(fine_tuned_weights, map_location='cpu')['params'])
    return model.to(device).eval()

@torch.no_grad()
def predict(model, rgb, device):
    L, ab_input = rgb_to_lab_tensors(rgb)

    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        small = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        L_model, _ = rgb_to_lab_tensors(small)
    else:
        L_model = L
    with torch.autocast('cuda', dtype=torch.bfloat16): #autocast for faster inference
        ab = model(L_model[None].to(device))
    ab = ab.float().clamp(-1, 1) #need values between -1 and 1 for lab2rgb to work

    if scale <1.0:
        ab = F.interpolate(ab, size=(h, w), mode='bilinear', align_corners=False)
    return lab_tensors_to_rgb(L, ab[0].cpu())

def save_overview(path, results, mean_time, pairs_per_row=2): #shows the greyscale input and colourised output side by side in a grid, with inference times
    rows = int(np.ceil(len(results) / pairs_per_row))
    fig, axes = plt.subplots(rows, 2 * pairs_per_row, figsize=(6 * pairs_per_row, 2.6 * rows)) #makes the grid
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis('off')
    for i, (name, grey, colour, seconds) in enumerate(results):
        r, c = divmod(i, pairs_per_row)
        axes[r, 2 * c].imshow(grey)
        axes[r, 2 * c].set_title(f'{name}: greyscale input', fontsize=9)
        axes[r, 2 * c + 1].imshow(colour)
        axes[r, 2 * c + 1].set_title(f'{name}: colourised in {seconds:.2f} s', fontsize=9)
    fig.suptitle(f'{len(results)} greyscale test images and their colourisations '
                 f'(average inference time {mean_time:.2f} s per image)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=80)
    plt.close(fig)

def main():
    if not torch.cuda.is_available():
        raise SystemExit('A CUDA GPU is required (MambaIRv2 uses mamba_ssm CUDA kernels).')
    device = torch.device('cuda')
 

    paths = sorted(glob.glob(os.path.join(input_test, '*.jpg')), key=sort_test_folder) #finds test images
    if not paths:
        raise SystemExit(f'No .jpg images found in {input_test}')
    os.makedirs(output_test, exist_ok=True)
 
    model = load_model(device)
    print(f'loaded {fine_tuned_weights}; colourising {len(paths)} images on {torch.cuda.get_device_name(0)}')
 
    results, times = [], []
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        # cv2.imread loads a greyscale JPEG as 3 identical channels; convert BGR -> RGB
        rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
 
        torch.cuda.synchronize()
        t = time.time()
        colour = predict(model, rgb, device)
        torch.cuda.synchronize()
        times.append(time.time() - t)
 
        results.append((name, rgb, colour, times[-1]))
        h, w = rgb.shape[:2]
        print(f'{name:>5}  {w} x {h}  {times[-1]:.2f} s', flush=True)
 
    mean_time = np.mean(times[1:] if len(times) > 1 else times)   # skip the warm-up image since its time is signifacntly longer
    save_overview(os.path.join(output_test, 'overview.png'), results, mean_time)
    print(f'\ncolourised {len(paths)} images, mean {mean_time:.2f} s per image '
          f'(first image excluded as warm-up), peak GPU memory '
          f'{torch.cuda.max_memory_allocated() / 1e9:.2f} GB') #how many images were colourised, mean time per image, and peak GPU memory usage
    print(f'overview figure saved to {os.path.join(output_test, "overview.png")}')
 
 
if __name__ == '__main__':
    main()