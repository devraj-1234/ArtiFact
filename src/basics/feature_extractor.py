"""
feature_extractor.py
Extract features from images for machine learning.
This module provides functions to extract various features from images that can be
used for forgery detection.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from basics.basic_fft import convert_to_grayscale, compute_fft


def extract_basic_features(img):
    """
    Extract basic statistical features from an image.
    
    Args:
        img: Input image
        
    Returns:
        dict: Dictionary of basic features
    """
    # Convert to grayscale
    gray = convert_to_grayscale(img)
    
    # Basic statistics
    mean = np.mean(gray)
    std = np.std(gray)
    min_val = np.min(gray)
    max_val = np.max(gray)
    
    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()  # Normalize
    
    # Calculate skewness and kurtosis from histogram
    indices = np.arange(256)
    skewness = np.sum((indices - mean)**3 * hist) / (std**3) if std > 0 else 0
    kurtosis = np.sum((indices - mean)**4 * hist) / (std**4) - 3 if std > 0 else 0
    
    return {
        'mean': mean,
        'std_dev': std,
        'min': min_val,
        'max': max_val,
        'skewness': skewness,
        'kurtosis': kurtosis
    }


def extract_fft_features(img):
    """
    Extract features from the FFT spectrum.
    
    Args:
        img: Input image
        
    Returns:
        dict: Dictionary of FFT features
    """
    # Convert to grayscale
    gray = convert_to_grayscale(img)
    
    # Compute FFT
    fshift, magnitude = compute_fft(gray)
    
    # Calculate basic statistics of magnitude spectrum
    mean = np.mean(magnitude)
    std = np.std(magnitude)
    max_val = np.max(magnitude)
    
    # Calculate energy in different frequency regions
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    
    # Define regions (center, mid, edge)
    center_radius = min(rows, cols) // 8
    mid_radius = min(rows, cols) // 4
    
    # Create masks for different regions
    y, x = np.ogrid[:rows, :cols]
    distances = np.sqrt((x - ccol)**2 + (y - crow)**2)
    
    center_mask = distances <= center_radius
    mid_mask = (distances > center_radius) & (distances <= mid_radius)
    edge_mask = distances > mid_radius
    
    # Calculate energy in each region
    center_energy = np.sum(magnitude[center_mask]) / np.sum(center_mask)
    mid_energy = np.sum(magnitude[mid_mask]) / np.sum(mid_mask)
    edge_energy = np.sum(magnitude[edge_mask]) / np.sum(edge_mask)
    
    # Calculate ratios
    center_to_edge_ratio = center_energy / edge_energy if edge_energy > 0 else float('inf')
    center_to_mid_ratio = center_energy / mid_energy if mid_energy > 0 else float('inf')
    mid_to_edge_ratio = mid_energy / edge_energy if edge_energy > 0 else float('inf')
    
    return {
        'fft_mean': mean,
        'fft_std': std,
        'fft_max': max_val,
        'center_energy': center_energy,
        'mid_energy': mid_energy,
        'edge_energy': edge_energy,
        'center_to_edge_ratio': center_to_edge_ratio,
        'center_to_mid_ratio': center_to_mid_ratio,
        'mid_to_edge_ratio': mid_to_edge_ratio
    }


def extract_texture_features(img):
    """
    Extract texture features using GLCM (Gray-Level Co-occurrence Matrix).
    
    Args:
        img: Input image
        
    Returns:
        dict: Dictionary of texture features
    """
    try:
        from skimage.feature import graycomatrix, graycoprops
    except ImportError:
        print("scikit-image is required for texture features. Install with: pip install scikit-image")
        return {}
    
    # Convert to grayscale
    gray = convert_to_grayscale(img)
    
    # Quantize to fewer gray levels to reduce computation
    bins = 8
    gray_scaled = (gray / (256 / bins)).astype(np.uint8)
    
    # Compute GLCM
    distances = [1, 3, 5]  # Pixel distances
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Angles
    glcm = graycomatrix(gray_scaled, distances, angles, levels=bins, symmetric=True, normed=True)
    
    # Calculate GLCM properties
    contrast = graycoprops(glcm, 'contrast').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    
    return {
        'contrast': contrast,
        'dissimilarity': dissimilarity,
        'homogeneity': homogeneity,
        'energy': energy,
        'correlation': correlation
    }


def extract_edge_features(img):
    """
    Extract edge features using Canny edge detector.
    
    Args:
        img: Input image
        
    Returns:
        dict: Dictionary of edge features
    """
    # Convert to grayscale
    gray = convert_to_grayscale(img)
    
    # Apply Canny edge detection with different thresholds
    edges_low = cv2.Canny(gray, 50, 150)
    edges_high = cv2.Canny(gray, 100, 200)
    
    # Calculate edge density
    edge_density_low = np.sum(edges_low > 0) / edges_low.size
    edge_density_high = np.sum(edges_high > 0) / edges_high.size
    
    return {
        'edge_density_low': edge_density_low,
        'edge_density_high': edge_density_high
    }


def extract_all_features(img):
    """
    Extract all features from an image.
    
    Args:
        img: Input image
        
    Returns:
        dict: Dictionary of all features
    """
    # Extract all feature types
    basic = extract_basic_features(img)
    fft = extract_fft_features(img)
    texture = extract_texture_features(img)
    edge = extract_edge_features(img)
    
    # Combine all features
    features = {}
    features.update(basic)
    features.update(fft)
    features.update(texture)
    features.update(edge)
    
    return features


def visualize_features(features):
    """
    Visualize extracted features.
    
    Args:
        features: Dictionary of features
    """
    # Group features by type
    basic_features = {k: v for k, v in features.items() if k in extract_basic_features(np.zeros((2, 2)))}
    fft_features = {k: v for k, v in features.items() if k in extract_fft_features(np.zeros((2, 2)))}
    texture_features = {k: v for k, v in features.items() if k.lower() in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']}
    edge_features = {k: v for k, v in features.items() if 'edge' in k.lower()}
    
    # Create bar charts
    plt.figure(figsize=(15, 10))
    
    # Basic features
    plt.subplot(2, 2, 1)
    plt.bar(basic_features.keys(), basic_features.values())
    plt.title('Basic Image Features')
    plt.xticks(rotation=45)
    
    # FFT features
    plt.subplot(2, 2, 2)
    plt.bar(fft_features.keys(), fft_features.values())
    plt.title('FFT Features')
    plt.xticks(rotation=45)
    
    # Texture features
    plt.subplot(2, 2, 3)
    plt.bar(texture_features.keys(), texture_features.values())
    plt.title('Texture Features')
    plt.xticks(rotation=45)
    
    # Edge features
    plt.subplot(2, 2, 4)
    plt.bar(edge_features.keys(), edge_features.values())
    plt.title('Edge Features')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()


def compare_features(features1, features2, title1="Image 1", title2="Image 2"):
    """
    Compare features from two images.
    
    Args:
        features1: Features from first image
        features2: Features from second image
        title1: Title for first image
        title2: Title for second image
    """
    # Get common features
    common_features = set(features1.keys()) & set(features2.keys())
    
    # Calculate differences
    diff = {k: features1[k] - features2[k] for k in common_features}
    
    # Sort by absolute difference
    sorted_diff = sorted(diff.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Display top differences
    plt.figure(figsize=(12, 6))
    
    # Plot the top 10 differences or all if less than 10
    top_n = min(10, len(sorted_diff))
    top_features = [item[0] for item in sorted_diff[:top_n]]
    top_diffs = [item[1] for item in sorted_diff[:top_n]]
    
    plt.bar(top_features, top_diffs)
    plt.title(f'Top Feature Differences ({title1} vs {title2})')
    plt.xticks(rotation=45)
    plt.ylabel('Difference')
    plt.tight_layout()
    plt.show()
    
    # Print table of all differences
    print(f"Feature Comparison: {title1} vs {title2}")
    print("-" * 50)
    print(f"{'Feature':<20} {title1:>12} {title2:>12} {'Difference':>12}")
    print("-" * 50)
    
    for feature, value in sorted_diff:
        print(f"{feature:<20} {features1[feature]:>12.4f} {features2[feature]:>12.4f} {value:>12.4f}")


def main():
    """Main function to demonstrate feature extraction."""
    # Get directory of this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set paths to dataset
    data_path = os.path.join(current_dir, "../../data/raw/AI_for_Art_Restoration_2/paired_dataset_art")
    damaged_dir = os.path.join(data_path, "damaged")
    undamaged_dir = os.path.join(data_path, "undamaged")
    
    try:
        # Find a pair of images
        damaged_files = os.listdir(damaged_dir)
        undamaged_files = os.listdir(undamaged_dir)
        
        common_files = [f for f in damaged_files if f in undamaged_files]
        
        if common_files:
            sample_file = common_files[0]
            print(f"Selected sample file: {sample_file}")
            
            # Load images
            damaged_path = os.path.join(damaged_dir, sample_file)
            undamaged_path = os.path.join(undamaged_dir, sample_file)
            
            damaged_img = cv2.imread(damaged_path)
            undamaged_img = cv2.imread(undamaged_path)
            
            if damaged_img is not None and undamaged_img is not None:
                # Extract features
                print("Extracting features from damaged image...")
                damaged_features = extract_all_features(damaged_img)
                
                print("Extracting features from undamaged image...")
                undamaged_features = extract_all_features(undamaged_img)
                
                # Visualize features
                print("Visualizing features for damaged image...")
                visualize_features(damaged_features)
                
                print("Visualizing features for undamaged image...")
                visualize_features(undamaged_features)
                
                # Compare features
                print("Comparing features...")
                compare_features(damaged_features, undamaged_features, "Damaged", "Undamaged")
            else:
                print("Failed to load images.")
        else:
            print("No matching files found.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
