"""
pytorch_dataset.py

Clean, corrected, stable implementation for synthetic art damage simulation.
"""

import os
import io
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import random
from pathlib import Path
from sklearn.model_selection import train_test_split


# ============================================================
#   DAMAGE EFFECTS
# ============================================================

def apply_varnish(img):
    """
    Simulates the yellow/brown oxidation of old varnish layers.
    More subtle & realistic than sepia.
    """
    r_tint = random.randint(225, 255)
    g_tint = random.randint(200, 240)
    b_tint = random.randint(150, 190)   # low blue → yellow tone

    varnish = Image.new("RGB", img.size, (r_tint, g_tint, b_tint))

    alpha = random.uniform(0.15, 0.35)  # Subtle (FIXED)
    return Image.blend(img, varnish, alpha)



# ============================================================
#   MAIN DATASET
# ============================================================

class SyntheticArtDataset(Dataset):
    """
    Loads CLEAN images and synthetically DEGRADES them on-the-fly.
    Returns (damaged, clean).
    """

    def __init__(self, clean_paths, transform=False, resize_transform=None):
        self.clean_paths = clean_paths
        self.transform = transform        # geometric augmentations
        self.resize_transform = resize_transform

        # Prebuilt transforms (faster)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )

    def __len__(self):
        return len(self.clean_paths)

    def __getitem__(self, idx):
        clean_path = self.clean_paths[idx]
        clean_img = Image.open(clean_path).convert("RGB")

        # --------------------------------------------------------------
        #   1. Resize
        # --------------------------------------------------------------
        if self.resize_transform:
            clean_img = self.resize_transform(clean_img)

        # --------------------------------------------------------------
        #   2. Geometric augmentations (TRAIN ONLY)
        # --------------------------------------------------------------
        if self.transform:
            if random.random() > 0.5:
                clean_img = TF.hflip(clean_img)
            if random.random() > 0.5:
                clean_img = TF.vflip(clean_img)

            angle = transforms.RandomRotation.get_params([-10, 10])
            clean_img = TF.rotate(clean_img, angle)

        # --------------------------------------------------------------
        #   3. DEGRADATION PIPELINE
        # --------------------------------------------------------------
        damaged_img = clean_img.copy()

        aging_type = random.random()

        # SCENARIO A — Yellow Varnish
        if aging_type < 0.33:
            damaged_img = apply_varnish(damaged_img)
            damaged_img = TF.adjust_contrast(damaged_img, random.uniform(1.0, 1.15))

        # SCENARIO B — Sun Bleached
        elif aging_type < 0.66:
            # Strong desaturation
            damaged_img = TF.adjust_saturation(damaged_img, random.uniform(0.3, 0.65))
            # Overexposure
            damaged_img = TF.adjust_brightness(damaged_img, random.uniform(1.1, 1.25))

        # SCENARIO C — Dirt / Grime
        else:
            damaged_img = TF.adjust_brightness(damaged_img, random.uniform(0.6, 0.85))
            damaged_img = TF.adjust_contrast(damaged_img, random.uniform(0.7, 0.95))
            damaged_img = TF.adjust_saturation(damaged_img, random.uniform(0.6, 0.9))

        # --------------------------------------------------------------
        #   4. Optional blur + JPEG compression
        # --------------------------------------------------------------
        if random.random() > 0.5:
            damaged_img = TF.gaussian_blur(
                damaged_img,
                kernel_size=5,
                sigma=random.uniform(0.4, 1.1)
            )

        # JPEG compression artifact simulation
        quality = random.randint(60, 90)
        buffer = io.BytesIO()
        damaged_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        damaged_img = Image.open(buffer).convert("RGB")

        # --------------------------------------------------------------
        #   5. Convert both to tensor (0–1)
        # --------------------------------------------------------------
        clean_tensor = self.to_tensor(clean_img)
        damaged_tensor = self.to_tensor(damaged_img)

        # --------------------------------------------------------------
        #   6. PHYSICAL DAMAGE (SCRATCHES)
        # --------------------------------------------------------------
        num_cracks = random.randint(8, 20)
        for _ in range(num_cracks):
            damaged_tensor = transforms.RandomErasing(
                p=1.0,
                scale=(0.0002, 0.002),
                ratio=(0.05, 20.0),
                value='random'
            )(damaged_tensor)

        if random.random() > 0.7:
            damaged_tensor = transforms.RandomErasing(
                p=1.0,
                scale=(0.01, 0.04),
                ratio=(0.5, 2.0),
                value='random'
            )(damaged_tensor)

        # --------------------------------------------------------------
        #   7. Normalize to [-1,1]  (important for tanh)
        # --------------------------------------------------------------
        clean_tensor = self.normalize(clean_tensor)
        damaged_tensor = self.normalize(damaged_tensor)

        return damaged_tensor, clean_tensor



# ============================================================
#   DATALOADER FACTORY
# ============================================================

def get_dataloaders(base_path, img_size=(256, 256), batch_size=8,
                    split_ratio=0.2, num_workers=2):

    base_path = Path(base_path)
    undamaged_dir = base_path / "paired_dataset_art" / "undamaged"

    if not undamaged_dir.exists():
        raise FileNotFoundError(f"Undamaged directory missing: {undamaged_dir}")

    # Collect clean image paths
    image_files = [f for f in os.listdir(undamaged_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    clean_paths = [undamaged_dir / f for f in image_files]

    # Train-val split
    train_paths, val_paths = train_test_split(
        clean_paths,
        test_size=split_ratio,
        random_state=42
    )

    resize_transform = transforms.Resize(img_size)

    train_dataset = SyntheticArtDataset(
        train_paths,
        transform=True,
        resize_transform=resize_transform
    )

    val_dataset = SyntheticArtDataset(
        val_paths,
        transform=False,          # no flips/rotations
        resize_transform=resize_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
