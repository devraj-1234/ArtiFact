"""
Feature Extractor for Artwork Analysis
Extracts features from images for ML classification.

Features extracted:
1-4: Statistical (mean, std, skewness, kurtosis)
5-7: Frequency bands (low, high, ratio)
8-12: Radial profile (center, 25%, 50%, 75%, edge)
13: Color balance needed (0-1 score)
14: Sharpening needed (0-1 score)

Total: 14 features
"""

import cv2
import numpy as np
from scipy import stats


def load_image(path, grayscale=True):
    """Load an image in grayscale."""
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if img is None:
        raise FileNotFoundError(f"Could not load image at {path}")
    
    return img


def compute_fft(img):
    """Compute the 2D FFT of an image and return the shifted spectrum."""
    # Apply FFT
    f = np.fft.fft2(img)
    # Shift zero frequency to center
    fshift = np.fft.fftshift(f)
    return fshift


def compute_magnitude_spectrum(fshift):
    """Compute magnitude spectrum from shifted FFT."""
    # Calculate magnitude spectrum (log scale for better visualization)
    magnitude = 20 * np.log(np.abs(fshift) + 1)
    return magnitude


def extract_fft_statistics(magnitude):
    """
    Extract statistical features from FFT magnitude spectrum.
    
    Returns dict with 9 statistical features.
    """
    # Basic statistics
    mean = np.mean(magnitude)
    std = np.std(magnitude)
    max_val = np.max(magnitude)
    min_val = np.min(magnitude)
    
    # Calculate skewness (measures asymmetry)
    skewness = np.mean(((magnitude - mean) / std) ** 3) if std > 0 else 0
    
    # Calculate kurtosis (measures "tailedness")
    kurtosis = np.mean(((magnitude - mean) / std) ** 4) - 3 if std > 0 else 0
    
    # Calculate energy in different frequency bands
    rows, cols = magnitude.shape
    crow, ccol = rows // 2, cols // 2
    
    # Low frequency energy (center region - smooth content)
    low_freq_region = magnitude[crow-30:crow+30, ccol-30:ccol+30]
    low_freq_energy = np.sum(low_freq_region) / low_freq_region.size if low_freq_region.size > 0 else 0
    
    # High frequency energy (outer regions - details/noise)
    high_freq_mask = np.ones_like(magnitude)
    high_freq_mask[crow-30:crow+30, ccol-30:ccol+30] = 0
    high_freq_region = magnitude * high_freq_mask
    high_freq_energy = np.sum(high_freq_region) / np.count_nonzero(high_freq_mask)
    
    # Energy ratio (high/low) - key indicator of damage
    energy_ratio = high_freq_energy / low_freq_energy if low_freq_energy > 0 else float('inf')
    
    return {
        'mean': mean,
        'std_dev': std,
        'max': max_val,
        'min': min_val,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'low_freq_energy': low_freq_energy,
        'high_freq_energy': high_freq_energy,
        'energy_ratio': energy_ratio
    }


def compute_radial_profile(magnitude):
    """
    Compute the radial profile of an FFT magnitude spectrum.
    Shows how frequency energy changes from center to edge.
    """
    rows, cols = magnitude.shape
    center_row, center_col = rows // 2, cols // 2
    
    # Create coordinate grid
    y, x = np.ogrid[:rows, :cols]
    
    # Calculate distance from center for each pixel
    distance_from_center = np.sqrt((x - center_col)**2 + (y - center_row)**2)
    distance_from_center = distance_from_center.astype(int)
    
    # Maximum radius
    max_radius = min(center_row, center_col)
    
    # Initialize array for radial profile
    radial_profile = np.zeros(max_radius)
    
    # Compute average magnitude at each radius
    for r in range(max_radius):
        mask = (distance_from_center == r)
        if np.any(mask):
            radial_profile[r] = np.mean(magnitude[mask])
    
    return np.arange(max_radius), radial_profile


def detect_color_balance_need(img_bgr):
    """
    Detect if an image needs color balance correction.
    
    Analyzes:
    - Channel mean balance (should be similar across RGB)
    - Color cast detection (dominant color shifts)
    - Gray world assumption deviation
    
    Args:
        img_bgr: Color image in BGR format
        
    Returns:
        score: 0-1 score (0=no correction needed, 1=correction needed)
    """
    if len(img_bgr.shape) == 2:
        # Grayscale image - no color balance needed
        return 0.0
    
    # Convert to RGB for analysis
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Calculate channel means
    r_mean = np.mean(img_rgb[:, :, 0])
    g_mean = np.mean(img_rgb[:, :, 1])
    b_mean = np.mean(img_rgb[:, :, 2])
    
    # Calculate overall mean
    overall_mean = (r_mean + g_mean + b_mean) / 3.0
    
    # Calculate deviation from balanced (gray world assumption)
    # In a balanced image, all channels should have similar means
    if overall_mean > 0:
        r_deviation = abs(r_mean - overall_mean) / overall_mean
        g_deviation = abs(g_mean - overall_mean) / overall_mean
        b_deviation = abs(b_mean - overall_mean) / overall_mean
        
        # Average deviation
        balance_deviation = (r_deviation + g_deviation + b_deviation) / 3.0
    else:
        balance_deviation = 0.0
    
    # Calculate color cast score
    # Check if one channel dominates significantly
    channel_means = np.array([r_mean, g_mean, b_mean])
    max_mean = np.max(channel_means)
    min_mean = np.min(channel_means)
    
    if max_mean > 0:
        color_cast_score = (max_mean - min_mean) / max_mean
    else:
        color_cast_score = 0.0
    
    # Calculate standard deviation of channel std devs
    # Damaged/aged images often have uneven color distribution
    r_std = np.std(img_rgb[:, :, 0])
    g_std = np.std(img_rgb[:, :, 1])
    b_std = np.std(img_rgb[:, :, 2])
    
    std_variation = np.std([r_std, g_std, b_std]) / (np.mean([r_std, g_std, b_std]) + 1e-6)
    
    # Combine metrics (weighted average)
    # Higher values indicate more need for color correction
    color_balance_score = (
        0.4 * balance_deviation +    # Channel mean imbalance
        0.4 * color_cast_score +      # Dominant color cast
        0.2 * min(std_variation, 1.0) # Channel variation
    )
    
    # Normalize to 0-1 range and clip
    color_balance_score = min(color_balance_score, 1.0)
    
    return color_balance_score


def detect_sharpening_need(img):
    """
    Detect if an image needs sharpening enhancement.
    
    Analyzes:
    - Edge strength (weak edges = needs sharpening)
    - High-frequency content (low = blurry/needs sharpening)
    - Laplacian variance (low = blurry)
    
    Args:
        img: Grayscale or color image
        
    Returns:
        score: 0-1 score (0=sharp already, 1=needs sharpening)
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Method 1: Laplacian variance (standard blur detection)
    # Sharp images have high variance, blurry images have low variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_var = laplacian.var()
    
    # Normalize (typical range: 0-500 for 8-bit images)
    # Lower values = more blur = more sharpening needed
    laplacian_score = max(0, 1.0 - (laplacian_var / 500.0))
    
    # Method 2: High-frequency energy (from FFT)
    # Calculate FFT to measure high-frequency content
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    
    rows, cols = magnitude.shape
    crow, ccol = rows // 2, cols // 2
    
    # Calculate high-frequency energy (edges and details)
    # Exclude center region (low frequencies)
    mask = np.ones_like(magnitude)
    radius = 30
    mask[crow-radius:crow+radius, ccol-radius:ccol+radius] = 0
    
    high_freq_energy = np.sum(magnitude * mask)
    total_energy = np.sum(magnitude)
    
    if total_energy > 0:
        high_freq_ratio = high_freq_energy / total_energy
    else:
        high_freq_ratio = 0
    
    # Lower ratio = more sharpening needed
    # Typical sharp images: 0.4-0.6, blurry images: 0.2-0.3
    freq_score = max(0, 1.0 - (high_freq_ratio / 0.5))
    
    # Method 3: Gradient magnitude (edge strength)
    # Calculate gradients using Sobel
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Calculate mean gradient strength
    mean_gradient = np.mean(gradient_magnitude)
    
    # Normalize (typical range: 0-30 for 8-bit images)
    # Lower values = weaker edges = more sharpening needed
    gradient_score = max(0, 1.0 - (mean_gradient / 30.0))
    
    # Combine metrics (weighted average)
    sharpening_score = (
        0.4 * laplacian_score +  # Laplacian variance
        0.3 * freq_score +        # High-frequency content
        0.3 * gradient_score      # Edge strength
    )
    
    # Clip to 0-1 range
    sharpening_score = np.clip(sharpening_score, 0.0, 1.0)
    
    return sharpening_score


def extract_ml_features(image_path):
    """
    Extract 14-dimensional feature vector from an image for ML.
    
    Features:
    1-4: FFT statistics (mean, std, skewness, kurtosis)
    5-7: Frequency bands (low_freq, high_freq, energy_ratio)
    8-12: Radial profile (center, 25%, 50%, 75%, edge)
    13: Color balance need score (0-1)
    14: Sharpening need score (0-1)
    
    Args:
        image_path: Path to image file
        
    Returns:
        features: numpy array of 14 features
        feature_names: list of feature names
    """
    # Load image in grayscale for FFT analysis
    img_gray = load_image(image_path, grayscale=True)
    
    # Load image in color for color balance detection
    img_color = load_image(image_path, grayscale=False)
    
    # Compute FFT features
    fshift = compute_fft(img_gray)
    magnitude = compute_magnitude_spectrum(fshift)
    
    # Get statistical features
    stats = extract_fft_statistics(magnitude)
    
    # Get radial profile
    radii, profile = compute_radial_profile(magnitude)
    
    # Extract 5 representative points from radial profile
    if len(profile) >= 5:
        profile_features = np.array([
            profile[0] if len(profile) > 0 else 0,              # Center (DC component)
            profile[len(profile)//4] if len(profile) > 4 else 0,  # 25% position
            profile[len(profile)//2] if len(profile) > 2 else 0,  # 50% position
            profile[3*len(profile)//4] if len(profile) > 4 else 0,  # 75% position
            profile[-1] if len(profile) > 0 else 0               # Edge (highest freq)
        ])
    else:
        profile_features = np.zeros(5)
    
    # Detect color balance and sharpening needs
    color_balance_score = detect_color_balance_need(img_color)
    sharpening_score = detect_sharpening_need(img_color)
    
    # Create feature vector (14 features total)
    features = np.array([
        stats['mean'],
        stats['std_dev'],
        stats['skewness'],
        stats['kurtosis'],
        stats['low_freq_energy'],
        stats['high_freq_energy'],
        stats['energy_ratio']
    ])
    
    # Add radial profile features
    features = np.concatenate([features, profile_features])
    
    # Add color balance and sharpening detection features
    features = np.concatenate([features, [color_balance_score, sharpening_score]])
    
    # Feature names for reference
    feature_names = [
        'mean', 'std_dev', 'skewness', 'kurtosis',
        'low_freq_energy', 'high_freq_energy', 'energy_ratio',
        'radial_center', 'radial_25', 'radial_50', 'radial_75', 'radial_edge',
        'color_balance_need', 'sharpening_need'
    ]
    
    return features, feature_names


def extract_features_batch(image_paths, verbose=True):
    """
    Extract features from multiple images.
    
    Args:
        image_paths: List of image file paths
        verbose: Print progress
        
    Returns:
        features_matrix: numpy array of shape (n_images, 14)
        feature_names: list of feature names
    """
    features_list = []
    
    for i, path in enumerate(image_paths):
        try:
            features, feature_names = extract_ml_features(path)
            features_list.append(features)
            
            if verbose and (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(image_paths)} images")
                
        except Exception as e:
            print(f"Error processing {path}: {e}")
            # Add zeros for failed images
            features_list.append(np.zeros(14))
    
    return np.array(features_list), feature_names


if __name__ == "__main__":
    # Test the feature extractor
    import sys
    import os
    
    # Try to load a sample image
    test_image_dir = "data/raw/AI_for_Art_Restoration_2/paired_dataset_art/damaged"
    
    if os.path.exists(test_image_dir):
        files = os.listdir(test_image_dir)
        if files:
            test_path = os.path.join(test_image_dir, files[0])
            print(f"Testing Enhanced Feature Extraction on: {files[0]}")
            print("="*70)
            
            features, names = extract_ml_features(test_path)
            
            print("\nExtracted Features (14 total):")
            print("-"*70)
            for name, value in zip(names, features):
                print(f"{name:<25}: {value:.4f}")
            
            print("\n" + "="*70)
            print("NEW DETECTION FEATURES:")
            print(f"  Color Balance Need  : {features[-2]:.4f} (0=balanced, 1=needs correction)")
            print(f"  Sharpening Need     : {features[-1]:.4f} (0=sharp, 1=needs sharpening)")
            print("="*70)
    else:
        print(f"Test directory not found: {test_image_dir}")
        print("Run from project root directory.")
