"""
pytorch_dataset.py

PyTorch-specific Dataset and DataLoader for the art restoration project.
This module provides a PyTorch-compatible dataset class that includes
data augmentation capabilities using torchvision.
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

class PairedArtDataset(Dataset):
    """
    PyTorch Dataset for paired damaged and undamaged art images.
    Handles data augmentation for training.
    """
    def __init__(self, damaged_paths, undamaged_paths, transform=None, resize_transform=None):
        """
        Args:
            damaged_paths (list): List of file paths to damaged images.
            undamaged_paths (list): List of file paths to undamaged (ground truth) images.
            transform (callable, optional): A function/transform to be applied on a sample.
            resize_transform (callable, optional): A transform to resize the image.
        """
        self.damaged_paths = damaged_paths
        self.undamaged_paths = undamaged_paths
        self.transform = transform
        self.resize_transform = resize_transform

    def __len__(self):
        return len(self.damaged_paths)

    def __getitem__(self, idx):
        damaged_img_path = self.damaged_paths[idx]
        undamaged_img_path = self.undamaged_paths[idx]

        damaged_img = Image.open(damaged_img_path).convert("RGB")
        undamaged_img = Image.open(undamaged_img_path).convert("RGB")

        if self.resize_transform:
            damaged_img = self.resize_transform(damaged_img)
            undamaged_img = self.resize_transform(undamaged_img)

        if self.transform:
            # Apply geometric transformations consistently to both images
            if random.random() > 0.5:
                damaged_img = TF.hflip(damaged_img)
                undamaged_img = TF.hflip(undamaged_img)
            
            if random.random() > 0.5:
                damaged_img = TF.vflip(damaged_img)
                undamaged_img = TF.vflip(undamaged_img)

            angle = transforms.RandomRotation.get_params([-15, 15])
            damaged_img = TF.rotate(damaged_img, angle)
            undamaged_img = TF.rotate(undamaged_img, angle)
            
            # Apply color jitter only to the damaged image to improve model robustness
            color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            damaged_img = color_jitter(damaged_img)

        # Convert to tensor and normalize
        to_tensor = transforms.ToTensor()
        damaged_tensor = to_tensor(damaged_img)
        undamaged_tensor = to_tensor(undamaged_img)

        return damaged_tensor, undamaged_tensor

def get_dataloaders(base_path, img_size=(256, 256), batch_size=16, split_ratio=0.2, num_workers=0):
    """
    Creates and returns training and validation DataLoaders.

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
    damaged_dir = base_path / "paired_dataset_art" / "damaged"
    undamaged_dir = base_path / "paired_dataset_art" / "undamaged"

    image_files = [f for f in os.listdir(damaged_dir) if os.path.exists(undamaged_dir / f)]
    
    damaged_paths = [damaged_dir / f for f in image_files]
    undamaged_paths = [undamaged_dir / f for f in image_files]

    # Split files into training and validation sets
    train_damaged, val_damaged, train_undamaged, val_undamaged = train_test_split(
        damaged_paths, undamaged_paths, test_size=split_ratio, random_state=42
    )

    # Define transformations
    # Resize is applied first, then augmentations in the dataset class
    resize_transform = transforms.Resize(img_size)

    # Create datasets
    # We apply augmentation by passing `transform=True` for the training set
    train_dataset = PairedArtDataset(train_damaged, train_undamaged, transform=True, resize_transform=resize_transform)
    val_dataset = PairedArtDataset(val_damaged, val_undamaged, transform=False, resize_transform=resize_transform) # No augmentation for validation

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
