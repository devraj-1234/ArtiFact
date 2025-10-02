"""
basic_restoration.py
Basic image restoration techniques using OpenCV and FFT.
This module provides simple methods for restoring degraded images.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Use relative import since basic_fft is in the same package
from .basic_fft import convert_to_grayscale, compute_fft, inverse_fft


def display_images(images, titles, cmaps=None, figsize=(15, 5)):
    """
    Display multiple images side by side for comparison.
    
    Args:
        images: List of images to display
        titles: List of titles for each image
        cmaps: List of colormaps for each image (or None for default)
        figsize: Figure size as (width, height)
    """
    n_images = len(images)
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    
    # If only one image, axes is not a list, so we convert it to a list
    if n_images == 1:
        axes = [axes]
    
    for i, (img, title) in enumerate(zip(images, titles)):
        cmap = cmaps[i] if cmaps else ('gray' if len(img.shape) == 2 else None)
        axes[i].imshow(img, cmap=cmap)
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def denoise_gaussian(image, kernel_size=5):
    """
    Apply Gaussian blur to remove noise.
    
    Args:
        image: Input image
        kernel_size: Size of the Gaussian kernel
        
    Returns:
        Denoised image
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def denoise_median(image, kernel_size=5):
    """
    Apply median filter to remove salt-and-pepper noise.
    
    Args:
        image: Input image
        kernel_size: Size of the median filter kernel
        
    Returns:
        Denoised image
    """
    return cv2.medianBlur(image, kernel_size)


def denoise_bilateral(image, d=9, sigma_color=75, sigma_space=75):
    """
    Apply bilateral filter to remove noise while preserving edges.
    
    Args:
        image: Input image
        d: Diameter of each pixel neighborhood
        sigma_color: Filter sigma in the color space
        sigma_space: Filter sigma in the coordinate space
        
    Returns:
        Denoised image
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def denoise_nlm(image, h=10, template_window_size=7, search_window_size=21):
    """
    Apply Non-Local Means denoising.
    
    Args:
        image: Input image
        h: Filter strength
        template_window_size: Size of template patch
        search_window_size: Size of window for searching similar patches
        
    Returns:
        Denoised image
    """
    if len(image.shape) == 3:  # Color image
        return cv2.fastNlMeansDenoisingColored(
            image, None, h, h, template_window_size, search_window_size
        )
    else:  # Grayscale image
        return cv2.fastNlMeansDenoising(
            image, None, h, template_window_size, search_window_size
        )


def enhance_contrast(image):
    """
    Enhance image contrast using histogram equalization.
    
    Args:
        image: Input image
        
    Returns:
        Contrast-enhanced image
    """
    if len(image.shape) == 3:  # Color image
        # Convert to YUV and apply histogram equalization to the Y channel
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    else:  # Grayscale image
        return cv2.equalizeHist(image)


def enhance_clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    """
    Enhance image using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        image: Input image
        clip_limit: Threshold for contrast limiting
        grid_size: Size of grid for histogram equalization
        
    Returns:
        Enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    
    if len(image.shape) == 3:  # Color image
        # Convert to LAB color space and apply CLAHE to L channel
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:  # Grayscale image
        return clahe.apply(image)


def sharpen_image(image, kernel_size=5, sigma=1.0, amount=1.5, threshold=0):
    """
    Sharpen image using unsharp masking.
    
    Args:
        image: Input image
        kernel_size: Size of Gaussian kernel
        sigma: Standard deviation of Gaussian kernel
        amount: Strength of sharpening effect
        threshold: Minimum difference to apply sharpening
        
    Returns:
        Sharpened image
    """
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    
    return sharpened


def remove_scratches(image, kernel_size=5):
    """
    Remove scratches using morphological operations.
    
    Args:
        image: Input image
        kernel_size: Size of kernel for morphological operations
        
    Returns:
        Image with reduced scratches
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Create kernel for morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Apply closing operation (dilation followed by erosion)
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
    # If original was color, merge result back
    if len(image.shape) == 3:
        # Get difference between original and processed
        diff = cv2.subtract(closed, gray)
        
        # Apply difference to all channels
        result = image.copy()
        for i in range(3):
            result[:,:,i] = cv2.add(image[:,:,i], diff)
        return result
    
    return closed


def fft_denoise(image, threshold_percentage=0.1):
    """
    Remove noise using FFT-based frequency filtering.
    
    Args:
        image: Input image
        threshold_percentage: Percentage of frequencies to remove
        
    Returns:
        Denoised image
    """
    # Convert to grayscale
    gray = convert_to_grayscale(image)
    
    # Apply FFT
    fshift, magnitude = compute_fft(gray)
    
    # Create a high-pass filter
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    
    # Calculate threshold based on magnitude
    threshold = threshold_percentage * np.max(magnitude)
    
    # Apply threshold to filter out low magnitudes (likely noise)
    mask = magnitude > threshold
    
    # Apply mask to FFT result
    fshift_filtered = fshift * mask
    
    # Inverse FFT
    restored = inverse_fft(fshift_filtered)
    
    return restored


def inpaint_image(image, mask=None, method=cv2.INPAINT_NS):
    """
    Inpaint damaged regions using Navier-Stokes or Fast Marching Method.
    
    Args:
        image: Input image
        mask: Binary mask of damaged regions (255 for damaged, 0 for undamaged)
        method: Inpainting method (cv2.INPAINT_NS or cv2.INPAINT_TELEA)
        
    Returns:
        Inpainted image
    """
    if mask is None:
        # If no mask provided, try to create one by thresholding
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Use simple thresholding to detect potential damaged areas
        # This is a naive approach and may not work well for all images
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    
    # Inpaint
    return cv2.inpaint(image, mask.astype(np.uint8), 3, method)


def restore_image(image, techniques=None):
    """
    Apply multiple restoration techniques to an image.
    
    Args:
        image: Input image
        techniques: List of techniques to apply, e.g., ['denoise', 'sharpen', 'enhance']
                  If None, applies a default restoration pipeline
                  
    Returns:
        Restored image
    """
    if techniques is None:
        techniques = ['denoise', 'enhance', 'sharpen']
    
    result = image.copy()
    
    for technique in techniques:
        if technique == 'denoise':
            # Apply bilateral filter to preserve edges while removing noise
            result = denoise_bilateral(result)
        elif technique == 'enhance':
            # Enhance contrast with CLAHE
            result = enhance_clahe(result)
        elif technique == 'sharpen':
            # Apply sharpening
            result = sharpen_image(result)
        elif technique == 'remove_scratches':
            # Remove scratches
            result = remove_scratches(result)
        elif technique == 'inpaint':
            # Apply inpainting (will try to auto-detect damaged regions)
            result = inpaint_image(result)
            
    return result


def main():
    """Main function to demonstrate basic restoration techniques."""
    # Get directory of this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set paths to dataset
    data_path = os.path.join(current_dir, "../../data/raw/AI_for_Art_Restoration_2/paired_dataset_art")
    damaged_dir = os.path.join(data_path, "damaged")
    undamaged_dir = os.path.join(data_path, "undamaged")
    
    try:
        # Find a pair of images
        damaged_files = os.listdir(damaged_dir)
        
        if damaged_files:
            # Select first image
            sample_file = damaged_files[0]
            print(f"Selected sample file: {sample_file}")
            
            # Load damaged image
            damaged_path = os.path.join(damaged_dir, sample_file)
            damaged_img = cv2.imread(damaged_path)
            
            if damaged_img is not None:
                # Check if there's a ground truth (undamaged) version
                undamaged_img = None
                undamaged_path = os.path.join(undamaged_dir, sample_file)
                if os.path.exists(undamaged_path):
                    undamaged_img = cv2.imread(undamaged_path)
                    print("Found matching undamaged image for comparison.")
                
                # Convert from BGR to RGB for display
                damaged_rgb = cv2.cvtColor(damaged_img, cv2.COLOR_BGR2RGB)
                
                # Apply different restoration techniques
                print("Applying restoration techniques...")
                
                # 1. Denoise with different methods
                denoised_gaussian = cv2.cvtColor(denoise_gaussian(damaged_img), cv2.COLOR_BGR2RGB)
                denoised_bilateral = cv2.cvtColor(denoise_bilateral(damaged_img), cv2.COLOR_BGR2RGB)
                denoised_nlm = cv2.cvtColor(denoise_nlm(damaged_img), cv2.COLOR_BGR2RGB)
                
                # Display denoising results
                display_images(
                    [damaged_rgb, denoised_gaussian, denoised_bilateral, denoised_nlm],
                    ["Original Damaged", "Gaussian Denoise", "Bilateral Denoise", "NLM Denoise"]
                )
                
                # 2. Enhancement techniques
                enhanced_contrast = cv2.cvtColor(enhance_contrast(damaged_img), cv2.COLOR_BGR2RGB)
                enhanced_clahe = cv2.cvtColor(enhance_clahe(damaged_img), cv2.COLOR_BGR2RGB)
                
                # Display enhancement results
                display_images(
                    [damaged_rgb, enhanced_contrast, enhanced_clahe],
                    ["Original Damaged", "Histogram Equalization", "CLAHE Enhancement"]
                )
                
                # 3. Scratch removal and sharpening
                no_scratches = cv2.cvtColor(remove_scratches(damaged_img), cv2.COLOR_BGR2RGB)
                sharpened = cv2.cvtColor(sharpen_image(damaged_img), cv2.COLOR_BGR2RGB)
                
                # Display scratch removal and sharpening results
                display_images(
                    [damaged_rgb, no_scratches, sharpened],
                    ["Original Damaged", "Scratch Removal", "Sharpened"]
                )
                
                # 4. Full restoration pipeline
                restored = cv2.cvtColor(restore_image(damaged_img), cv2.COLOR_BGR2RGB)
                
                # Display final restoration with ground truth if available
                if undamaged_img is not None:
                    undamaged_rgb = cv2.cvtColor(undamaged_img, cv2.COLOR_BGR2RGB)
                    display_images(
                        [damaged_rgb, restored, undamaged_rgb],
                        ["Original Damaged", "Restored", "Ground Truth"]
                    )
                else:
                    display_images(
                        [damaged_rgb, restored],
                        ["Original Damaged", "Restored"]
                    )
                
                print("Restoration demonstration complete.")
            else:
                print("Failed to load image.")
        else:
            print("No damaged images found.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
