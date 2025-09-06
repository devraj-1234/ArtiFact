"""
dataset.py
Dataset loading and preprocessing utilities for the ArtifactVision project.
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path


class ArtDataset:
    """Dataset class for art restoration and fake detection tasks."""
    
    def __init__(self, base_path, img_size=(256, 256), split_ratio=0.2):
        """
        Initialize the dataset.
        
        Args:
            base_path (str): Path to the dataset folder
            img_size (tuple): Size to resize images to
            split_ratio (float): Validation split ratio
        """
        self.base_path = Path(base_path)
        self.img_size = img_size
        self.split_ratio = split_ratio
        self.damaged_dir = self.base_path / "paired_dataset_art" / "damaged"
        self.undamaged_dir = self.base_path / "paired_dataset_art" / "undamaged"
        
    def load_image_pairs(self):
        """
        Load paired damaged and undamaged images.
        
        Returns:
            tuple: Lists of damaged and undamaged images
        """
        damaged_images = []
        undamaged_images = []
        
        image_files = os.listdir(self.damaged_dir)
        
        for img_file in image_files:
            damaged_path = os.path.join(self.damaged_dir, img_file)
            undamaged_path = os.path.join(self.undamaged_dir, img_file)
            
            # Check if both damaged and undamaged versions exist
            if not os.path.exists(undamaged_path):
                continue
                
            # Load images
            damaged_img = cv2.imread(damaged_path)
            undamaged_img = cv2.imread(undamaged_path)
            
            if damaged_img is None or undamaged_img is None:
                continue
                
            # Resize images
            damaged_img = cv2.resize(damaged_img, self.img_size)
            undamaged_img = cv2.resize(undamaged_img, self.img_size)
            
            damaged_images.append(damaged_img)
            undamaged_images.append(undamaged_img)
            
        return np.array(damaged_images), np.array(undamaged_images)
    
    def create_train_val_split(self):
        """
        Create train/validation split of the dataset.
        
        Returns:
            tuple: (x_train, x_val, y_train, y_val) arrays
        """
        damaged_imgs, undamaged_imgs = self.load_image_pairs()
        
        # Split into train and validation sets
        x_train, x_val, y_train, y_val = train_test_split(
            damaged_imgs, undamaged_imgs, 
            test_size=self.split_ratio, 
            random_state=42
        )
        
        return x_train, x_val, y_train, y_val
    
    def create_fake_detection_dataset(self):
        """
        Create a dataset for fake detection (binary classification).
        
        Returns:
            tuple: (X, y) where X is image data and y is binary labels (0=real, 1=fake)
        """
        damaged_imgs, undamaged_imgs = self.load_image_pairs()
        
        # Create labels: 0 for real (undamaged) and 1 for fake/damaged
        real_labels = np.zeros(len(undamaged_imgs))
        fake_labels = np.ones(len(damaged_imgs))
        
        # Combine images and labels
        X = np.concatenate((undamaged_imgs, damaged_imgs), axis=0)
        y = np.concatenate((real_labels, fake_labels), axis=0)
        
        # Shuffle the data
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        
        # Split into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.split_ratio, random_state=42
        )
        
        return X_train, X_val, y_train, y_val


def extract_fft_features(image, return_magnitude=True):
    """
    Extract FFT features from an image.
    
    Args:
        image (numpy.ndarray): Input image
        return_magnitude (bool): Whether to return magnitude spectrum
        
    Returns:
        numpy.ndarray: FFT features
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    
    if return_magnitude:
        # Calculate magnitude spectrum
        magnitude = 20 * np.log(np.abs(fshift) + 1)
        return magnitude
    else:
        return fshift
