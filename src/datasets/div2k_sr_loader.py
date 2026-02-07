import os
from pathlib import Path
from typing import Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    # RGB PIL -> float32 tensor in [0,1], shape (C,H,W)
    x = torch.from_numpy(__import__("numpy").array(img)).permute(2, 0, 1).float() / 255.0
    return x

class Div2KSRPaired(Dataset):
    """
    Paired SR dataset:
      input  = LR image (H/2, W/2)
      target = HR image (H, W)
    """
    def __init__(self, root: str, split: str, scale: int = 2, hr_crop: int = 256, augment: bool = True):
        self.root = Path(root)
        self.split = split
        self.scale = scale
        self.hr_crop = hr_crop
        self.lr_crop = hr_crop // scale
        self.augment = augment

        self.input_dir = self.root / split / "input"
        self.target_dir = self.root / split / "target"

        if not self.input_dir.exists() or not self.target_dir.exists():
            raise FileNotFoundError(f"Missing folders: {self.input_dir} or {self.target_dir}")

        self.files = sorted([p.name for p in self.input_dir.iterdir()
                             if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])

        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {self.input_dir}")

        # Ensure pairs exist
        for name in self.files[:10]:
            if not (self.target_dir / name).exists():
                raise RuntimeError(f"Pair missing for {name} in target/")

    def __len__(self) -> int:
        return len(self.files)

    def _random_crop_pair(self, lr: Image.Image, hr: Image.Image) -> Tuple[Image.Image, Image.Image]:
        lr_w, lr_h = lr.size
        hr_w, hr_h = hr.size

        # Safety: ensure sizes match scale
        if hr_w != lr_w * self.scale or hr_h != lr_h * self.scale:
            # If mismatch, center-crop to match
            new_hr_w = (hr_w // self.scale) * self.scale
            new_hr_h = (hr_h // self.scale) * self.scale
            hr = hr.crop((0, 0, new_hr_w, new_hr_h))
            lr = lr.resize((new_hr_w // self.scale, new_hr_h // self.scale), resample=Image.BICUBIC)
            lr_w, lr_h = lr.size
            hr_w, hr_h = hr.size

        # If crop bigger than image, fallback to center crop with min size
        if self.lr_crop > lr_w or self.lr_crop > lr_h:
            # Resize up small images (rare in DIV2K), keep ratio
            lr = lr.resize((max(lr_w, self.lr_crop), max(lr_h, self.lr_crop)), Image.BICUBIC)
            hr = hr.resize((lr.size[0] * self.scale, lr.size[1] * self.scale), Image.BICUBIC)
            lr_w, lr_h = lr.size

        x_lr = random.randint(0, lr_w - self.lr_crop)
        y_lr = random.randint(0, lr_h - self.lr_crop)

        x_hr = x_lr * self.scale
        y_hr = y_lr * self.scale

        lr_patch = lr.crop((x_lr, y_lr, x_lr + self.lr_crop, y_lr + self.lr_crop))
        hr_patch = hr.crop((x_hr, y_hr, x_hr + self.hr_crop, y_hr + self.hr_crop))
        return lr_patch, hr_patch

    def _augment_pair(self, lr: Image.Image, hr: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if not self.augment:
            return lr, hr
        # Random horizontal flip
        if random.random() < 0.5:
            lr = lr.transpose(Image.FLIP_LEFT_RIGHT)
            hr = hr.transpose(Image.FLIP_LEFT_RIGHT)
        # Random vertical flip
        if random.random() < 0.5:
            lr = lr.transpose(Image.FLIP_TOP_BOTTOM)
            hr = hr.transpose(Image.FLIP_TOP_BOTTOM)
        return lr, hr

    def __getitem__(self, idx: int):
        name = self.files[idx]
        lr = Image.open(self.input_dir / name).convert("RGB")
        hr = Image.open(self.target_dir / name).convert("RGB")

        if self.split == "train":
            lr, hr = self._random_crop_pair(lr, hr)
            lr, hr = self._augment_pair(lr, hr)
        else:
            # val/test: center-crop to fixed size for consistent metrics
            # (simple and deterministic)
            lr_w, lr_h = lr.size
            x = max(0, (lr_w - self.lr_crop) // 2)
            y = max(0, (lr_h - self.lr_crop) // 2)
            lr = lr.crop((x, y, x + self.lr_crop, y + self.lr_crop))
            hr = hr.crop((x * self.scale, y * self.scale, x * self.scale + self.hr_crop, y * self.scale + self.hr_crop))

        lr_t = pil_to_tensor(lr)
        hr_t = pil_to_tensor(hr)
        return lr_t, hr_t, name

def make_loaders(
    dataset_root: str,
    train_bs: int = 8,
    val_bs: int = 8,
    num_workers: int = 0,
    hr_crop: int = 256,
    scale: int = 2
):
    train_ds = Div2KSRPaired(dataset_root, "train", scale=scale, hr_crop=hr_crop, augment=True)
    val_ds   = Div2KSRPaired(dataset_root, "val",   scale=scale, hr_crop=hr_crop, augment=False)

    train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=val_bs,   shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
