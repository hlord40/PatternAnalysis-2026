import numpy as np
import torch
import cv2
import random
import glob
import os

from torch.utils.data import Dataset, Dataloader
from skimage.color import rgb2lab, lab2rgb



#converting from rgb to lab tensors (for input)

def rgb_to_lab_tensors(rgb_uint8):
    lab = rgb2lab(rgb_uint8)
    L = lab[..., :1] / 100.0
    ab = lab[..., 1:] / 110.0
    to_tensor = lambda a: torch.from_numpy(a.astype(np.float32)).permute(2, 0, 1)
    return to_tensor(L), to_tensor(ab)

#converting from lab tensors to rgb (for output)

def lab_tensors_to_rgb(L, ab):
    lab = torch.cat([L * 100.0, ab * 110.0], dim=0).permute(1, 2, 0).cpu().numpy()
    rgb = lab2rgb(lab.astype(np.float64))
    return (rgb * 255).round().clip(0,255).astype(np.uint8)

#dataset loading, augmentation, and preprocessing

class DIV2K(Dataset):
    def __init__(self, folder, crop, crop_amount=50):
        self.paths = sorted(glob.glob(os.path.join(folder, '*.png')))
        self.crop = crop
        self.crop_amount = crop_amount

    def __len__(self):
        return len(self.paths) * self.crop_amount

    def __getitem__(self, i):
        img = cv2.cvtColor(cv2.imread(self.paths[i % len(self.paths)]), cv2.COLOR_BRG2RGB)
        h, w = img.shape[:2]
        y, x = random.randint(0, h - self.crop), random.randint(0, w - self.crop)
        patch = img[y:y + self.crop, x:x + self.crop] #crop
        if random.random() < 0.5: patch = patch[:, ::-1] #horisontal flip
        if random.random() < 0.5: patch = patch[::-1] #vertical flip
        return rgb_to_lab_tensors(np.ascontiguousarray(patch))