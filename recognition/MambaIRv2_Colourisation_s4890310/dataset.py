import numpy as np
import torch
import cv2
import random
import glob
import os

from torch.utils.data import Dataset, DataLoader
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
    lab = torch.cat([L * 100.0, ab * 110.0], dim=0).permute(1, 2, 0).detach().cpu().numpy()
    rgb = lab2rgb(lab.astype(np.float64))
    return (rgb * 255).round().clip(0,255).astype(np.uint8)


def list_images(folder):
    return sorted(glob.glob(os.path.join(folder, '*.png')))

def split_div2k(root, n_val=100, seed=67):
    train_pool = list_images(os.path.join(root, 'DIV2K_train_HR'))
    test_paths = list_images(os.path.join(root, 'DIV2K_valid_HR'))

    order = torch.randperm(len((train_pool)), generator=torch.Generator().manual_seed(seed))
    val_idx = set(order[:n_val].tolist())

    train_paths = [p for i, p in enumerate(train_pool) if i not in val_idx]
    val_paths = [p for i, p in enumerate(train_pool) if i in val_idx]
    return train_paths, val_paths, test_paths

#dataset loading, augmentation, and preprocessing

class DIV2K(Dataset):
    def __init__(self, paths, crop=128, crop_amount=50, train=True, max_side=512):
        self.paths = list(paths)
        self.crop = crop
        self.train = train
        self.crop_amount = crop_amount if train else 1
        self.max_side = max_side

    def __len__(self):
        return len(self.paths) * self.crop_amount

    def __getitem__(self, i):
        img = cv2.cvtColor(cv2.imread(self.paths[i % len(self.paths)]), cv2.COLOR_BGR2RGB)

        if self.train:
            h, w = img.shape[:2]
            y, x = random.randint(0, h - self.crop), random.randint(0, w - self.crop)
            patch = img[y:y + self.crop, x:x + self.crop] #crop
            if random.random() < 0.5: patch = patch[:, ::-1] #horisontal flip
            if random.random() < 0.5: patch = patch[::-1] #vertical flip

        else: #test and validation images are resized to a maximum side length of 512 pixels, and then cropped to a multiple of 16 pixels
            h, w = img.shape[:2]
            s = min(1.0, self.max_side / max(h, w))
            patch = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            ph, pw = patch.shape[:2]
            patch = patch[:ph // 16 * 16, :pw // 16 * 16] #crop to multiple of 16

        return rgb_to_lab_tensors(np.ascontiguousarray(patch))

#dataloaders for training.py simplicity
def get_data_loaders(root, crop=128, batch_size=8, seed=67, num_workers=4, crop_amount=50):
    train_paths, val_paths, test_paths = split_div2k(root, seed=seed)
    train_loader = DataLoader(DIV2K(train_paths, crop, crop_amount, train=True), batch_size=batch_size, shuffle=True, num_workers=num_workers, 
                              pin_memory=True, drop_last=True)

    val_loader = DataLoader(DIV2K(val_paths, train=False), batch_size=1, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(DIV2K(test_paths, train=False), batch_size=1, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    # quick check: python dataset.py /path/to/folder/containing/DIV2K_train_HR
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else 'data'
    train_loader, val_loader, test_loader = get_data_loaders(root, num_workers=0)
    print('train samples:', len(train_loader.dataset),
          '| val images:', len(val_loader.dataset),
          '| test images:', len(test_loader.dataset))
    L, ab = next(iter(train_loader))
    print('train batch:', tuple(L.shape), tuple(ab.shape))
