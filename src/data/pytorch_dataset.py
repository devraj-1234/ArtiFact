"""
pytorch_dataset.py

PyTorch-specific Dataset and DataLoader for the art restoration project.
This module provides a PyTorch-compatible dataset class that includes
data augmentation capabilities using torchvision.
"""
import os
import io  # Moved to top
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

def apply_sepia(img):
    """Applies a sepia tone filter to a PIL Image."""
    # Sepia matrix transformation
    sepia_matrix = [
        0.393, 0.769, 0.189, 0,
        0.349, 0.686, 0.168, 0,
        0.272, 0.534, 0.131, 0
    ]
    # Convert to RGB first just in case, then apply matrix
    img = img.convert("RGB")
    sepia_img = img.convert("RGB", sepia_matrix)
    return sepia_img

class SyntheticArtDataset(Dataset):
    """
    Dataset that loads clean art images and synthetically degrades them
    on-the-fly to create (damaged, clean) pairs for training.
    """
    def __init__(self, clean_paths, transform=None, resize_transform=None):
        """
        Args:
            clean_paths (list): List of file paths to clean (undamaged) images.
            transform (bool, optional): Whether to apply geometric augmentations (flips, rotations).
            resize_transform (callable, optional): A transform to resize the image.
        """
        self.clean_paths = clean_paths
        self.transform = transform
        self.resize_transform = resize_transform

    def __len__(self):
        return len(self.clean_paths)

    def __getitem__(self, idx):
        clean_path = self.clean_paths[idx]
        clean_img = Image.open(clean_path).convert("RGB")

        # 1. Resize (Ground Truth Preprocessing)
        if self.resize_transform:
            clean_img = self.resize_transform(clean_img)

        # 2. Geometric Transforms (Data Augmentation on Ground Truth)
        if self.transform:
            if random.random() > 0.5: clean_img = TF.hflip(clean_img)
            if random.random() > 0.5: clean_img = TF.vflip(clean_img)
            angle = transforms.RandomRotation.get_params([-10, 10])
            clean_img = TF.rotate(clean_img, angle)

        # --- START SYNTHETIC DEGRADATION PIPELINE ---
        damaged_img = clean_img.copy()

        # A. Extreme Color Aging (Sepia, B/W, or Fading)
        color_choice = random.random()
        if color_choice < 0.25:
            # Simulate old B/W photo
            damaged_img = TF.to_grayscale(damaged_img, num_output_channels=3)
            # Add heavy "light pollution" / uneven exposure
            brightness = random.uniform(0.5, 1.5)
            contrast = random.uniform(0.5, 1.5)
            damaged_img = TF.adjust_brightness(damaged_img, brightness)
            damaged_img = TF.adjust_contrast(damaged_img, contrast)
        elif color_choice < 0.50:
            # Simulate Sepia Tone
            damaged_img = apply_sepia(damaged_img)
             # Sepia photos are often darker/faded
            damaged_img = TF.adjust_brightness(damaged_img, random.uniform(0.7, 1.1))
            damaged_img = TF.adjust_contrast(damaged_img, random.uniform(0.8, 1.2))
        else:
            # General Fading and Color Shifts
            # High chance of desaturation to look "old"
            saturation = random.uniform(0.2, 0.8) 
            brightness = random.uniform(0.7, 1.3)
            contrast = random.uniform(0.7, 1.3)
            hue = random.uniform(-0.05, 0.05)
            damaged_img = TF.adjust_saturation(damaged_img, saturation)
            damaged_img = TF.adjust_brightness(damaged_img, brightness)
            damaged_img = TF.adjust_contrast(damaged_img, contrast)
            damaged_img = TF.adjust_hue(damaged_img, hue)

        # B. Blur & JPEG (Simulate bad camera/storage)
        if random.random() > 0.3:
            sigma = random.uniform(1.0, 3.0)
            damaged_img = TF.gaussian_blur(damaged_img, kernel_size=9, sigma=sigma)

        # Always apply some JPEG compression artifacts
        quality = random.randint(30, 70)
        buffer = io.BytesIO()
        damaged_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        damaged_img = Image.open(buffer).convert("RGB")

        # --- Convert to Tensor for Physical Damage ---
        to_tensor = transforms.ToTensor()
        clean_tensor = to_tensor(clean_img)
        damaged_tensor = to_tensor(damaged_img)

        # C. Heavy Surface Texture/Creases (Simulated by Noise)
        # We use heavy noise to simulate the rough, damaged surface texture.
        if random.random() > 0.2:
            noise_level = random.uniform(0.05, 0.15)
            noise = torch.randn_like(damaged_tensor) * noise_level
            damaged_tensor = damaged_tensor + noise
            damaged_tensor = torch.clamp(damaged_tensor, 0, 1)

        # D. Aggressive Holes and Scratches (Random Erasing)
        # Pass 1: Big Holes (like tears in the canvas)
        if random.random() > 0.3:
            re_holes = transforms.RandomErasing(p=1.0, scale=(0.05, 0.2), ratio=(0.3, 3.3))
            damaged_tensor = re_holes(damaged_tensor)

        # Pass 2: Long Scratches (thin lines)
        # We force extreme aspect ratios to look like scratches
        if random.random() > 0.3:
            # Vertical-ish scratch
            re_scratch_v = transforms.RandomErasing(p=1.0, scale=(0.01, 0.03), ratio=(0.1, 0.3))
            damaged_tensor = re_scratch_v(damaged_tensor)
        if random.random() > 0.3:
             # Horizontal-ish scratch
            re_scratch_h = transforms.RandomErasing(p=1.0, scale=(0.01, 0.03), ratio=(3.0, 10.0))
            damaged_tensor = re_scratch_h(damaged_tensor)

        # Normalize tensors to [-1, 1] range for standard U-Net training
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        clean_tensor = normalize(clean_tensor)
        damaged_tensor = normalize(damaged_tensor)

        return damaged_tensor, clean_tensor
    
    
def get_dataloaders(base_path, img_size=(256, 256), batch_size=16, split_ratio=0.2, num_workers=0):
    """
    Creates and returns training and validation DataLoaders using Synthetic Degradation.

    Args:
        base_path (str): Path to the dataset folder.
        img_size (tuple): Size to resize images to.
        batch_size (int): How many samples per batch to load.
        split_ratio (float): Proportion of the dataset to include in the validation split.
        num_workers (int): How many subprocesses to use for data loading.

    Returns:
        tuple: (train_loader, val_loader)
    """
    base_path = Path(base_path)
    # We only look for the 'undamaged' folder now, as we generate damage synthetically
    undamaged_dir = base_path / "paired_dataset_art" / "undamaged"

    if not undamaged_dir.exists():
        raise FileNotFoundError(f"Undamaged directory not found at {undamaged_dir}")

    image_files = [f for f in os.listdir(undamaged_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    clean_paths = [undamaged_dir / f for f in image_files]

    # Split files into training and validation sets
    train_paths, val_paths = train_test_split(clean_paths, test_size=split_ratio, random_state=42)

    # Define transformations
    resize_transform = transforms.Resize(img_size)

    # Create datasets
    # transform=True enables geometric augmentations (flips/rotations) for training
    train_dataset = SyntheticArtDataset(train_paths, transform=True, resize_transform=resize_transform)
    # transform=False disables geometric augmentations for validation, but degradations still apply
    val_dataset = SyntheticArtDataset(val_paths, transform=False, resize_transform=resize_transform)

    # Create dataloaders
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