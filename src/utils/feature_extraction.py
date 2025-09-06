"""
feature_extraction.py
Feature extraction utilities using FFT, wavelets, and other image processing techniques.
"""

import cv2
import numpy as np
from pathlib import Path


def extract_fft_features(image, normalize=True):
    """
    Extract FFT features from an image.
    
    Args:
        image (numpy.ndarray): Input image
        normalize (bool): Whether to normalize the features
        
    Returns:
        numpy.ndarray: FFT magnitude spectrum
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    
    # Calculate magnitude spectrum
    magnitude = 20 * np.log(np.abs(fshift) + 1)
    
    if normalize:
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return magnitude


def extract_fft_bands(image, bands=3):
    """
    Extract features from different frequency bands in FFT.
    
    Args:
        image (numpy.ndarray): Input image
        bands (int): Number of frequency bands to extract
        
    Returns:
        list: Features from different frequency bands
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    
    features = []
    
    # Create bands of different sizes
    for i in range(bands):
        radius = (i + 1) * min(rows, cols) // (2 * bands)
        mask = np.zeros((rows, cols), np.uint8)
        
        if i == 0:
            # Low frequency band (center)
            cv2.circle(mask, (ccol, crow), radius, 1, -1)
        else:
            # Band between previous radius and current radius
            prev_radius = i * min(rows, cols) // (2 * bands)
            cv2.circle(mask, (ccol, crow), radius, 1, -1)
            cv2.circle(mask, (ccol, crow), prev_radius, 0, -1)
        
        # Apply mask and compute inverse FFT
        filtered = fshift * mask
        img_back = np.fft.ifft2(np.fft.ifftshift(filtered))
        img_back = np.abs(img_back)
        
        # Calculate statistics from this band
        mean = np.mean(img_back)
        std = np.std(img_back)
        features.extend([mean, std])
    
    return np.array(features)


def extract_texture_features(image):
    """
    Extract texture features using GLCM (Gray-Level Co-occurrence Matrix).
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Texture features
    """
    from skimage.feature import greycomatrix, greycoprops
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Normalize to fewer gray levels to reduce computation
    gray = cv2.normalize(gray, None, 0, 7, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Calculate GLCM
    distances = [1, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = greycomatrix(gray, distances, angles, levels=8, symmetric=True, normed=True)
    
    # Calculate GLCM properties
    contrast = greycoprops(glcm, 'contrast')
    dissimilarity = greycoprops(glcm, 'dissimilarity')
    homogeneity = greycoprops(glcm, 'homogeneity')
    energy = greycoprops(glcm, 'energy')
    correlation = greycoprops(glcm, 'correlation')
    
    # Average over directions and distances
    features = np.hstack([
        contrast.mean(), dissimilarity.mean(), 
        homogeneity.mean(), energy.mean(), 
        correlation.mean()
    ])
    
    return features


def apply_filters(image, cutoff=30):
    """
    Apply low-pass and high-pass filters using FFT.
    
    Args:
        image (numpy.ndarray): Input image
        cutoff (int): Frequency cutoff for filters
        
    Returns:
        tuple: (low_pass_filtered, high_pass_filtered) images
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    
    # Apply FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    
    # Low-pass filter
    mask_lp = np.zeros((rows, cols), np.uint8)
    mask_lp[crow-cutoff:crow+cutoff, ccol-cutoff:ccol+cutoff] = 1
    img_lp = np.fft.ifft2(np.fft.ifftshift(fshift * mask_lp))
    img_lp = np.abs(img_lp)
    
    # High-pass filter
    mask_hp = np.ones((rows, cols), np.uint8)
    mask_hp[crow-cutoff:crow+cutoff, ccol-cutoff:ccol+cutoff] = 0
    img_hp = np.fft.ifft2(np.fft.ifftshift(fshift * mask_hp))
    img_hp = np.abs(img_hp)
    
    return img_lp, img_hp
