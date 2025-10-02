"""
basic_fft.py
Basic Fast Fourier Transform functions for image processing.
This module provides simple functions for working with FFT in image analysis.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def convert_to_grayscale(img):
    """
    Convert an image to grayscale.
    
    Args:
        img: Input image (numpy array)
        
    Returns:
        numpy.ndarray: Grayscale image
    """
    if len(img.shape) == 3:  # Check if image has multiple channels
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img  # Image is already grayscale
    return gray


def compute_fft(img):
    """
    Compute the 2D FFT of an image.
    
    Args:
        img: Input image (numpy array)
        
    Returns:
        tuple: (fshift, magnitude_spectrum) where:
            - fshift is the shifted Fourier transform
            - magnitude_spectrum is the log-scaled magnitude for visualization
    """
    # Ensure image is grayscale
    gray = convert_to_grayscale(img)
    
    # Compute 2D FFT
    f = np.fft.fft2(gray)
    
    # Shift zero frequency component to center of the spectrum
    fshift = np.fft.fftshift(f)
    
    # Calculate magnitude spectrum (log scale for better visualization)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    return fshift, magnitude_spectrum


def apply_filter(fshift, filter_mask):
    """
    Apply a filter in the frequency domain and compute inverse FFT.
    
    Args:
        fshift: Shifted Fourier transform
        filter_mask: Filter mask to apply
        
    Returns:
        numpy.ndarray: Filtered image in spatial domain
    """
    # Apply filter
    filtered_fshift = fshift * filter_mask
    
    # Inverse shift
    f_ishift = np.fft.ifftshift(filtered_fshift)
    
    # Inverse FFT
    img_back = np.fft.ifft2(f_ishift)
    
    # Get the real part (magnitude) for the image
    img_back = np.abs(img_back)
    
    # Normalize to 0-255 range
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return img_back


def create_lowpass_filter(shape, cutoff):
    """
    Create a circular low-pass filter mask.
    
    Args:
        shape: Shape of the image (rows, cols)
        cutoff: Cutoff radius for the filter
        
    Returns:
        numpy.ndarray: Low-pass filter mask
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2  # Center coordinates
    
    # Create a grid of coordinates
    y, x = np.ogrid[:rows, :cols]
    
    # Calculate distance from center for each pixel
    distance = np.sqrt((x - ccol)**2 + (y - crow)**2)
    
    # Create circular mask
    mask = np.zeros((rows, cols), dtype=np.float32)
    mask[distance <= cutoff] = 1
    
    return mask


def create_highpass_filter(shape, cutoff):
    """
    Create a circular high-pass filter mask.
    
    Args:
        shape: Shape of the image (rows, cols)
        cutoff: Cutoff radius for the filter
        
    Returns:
        numpy.ndarray: High-pass filter mask
    """
    # Create the inverse of a lowpass filter
    lowpass = create_lowpass_filter(shape, cutoff)
    highpass = 1 - lowpass
    
    return highpass


def create_bandpass_filter(shape, inner_cutoff, outer_cutoff):
    """
    Create a band-pass filter mask.
    
    Args:
        shape: Shape of the image (rows, cols)
        inner_cutoff: Inner radius (high-pass component)
        outer_cutoff: Outer radius (low-pass component)
        
    Returns:
        numpy.ndarray: Band-pass filter mask
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2  # Center coordinates
    
    # Create a grid of coordinates
    y, x = np.ogrid[:rows, :cols]
    
    # Calculate distance from center for each pixel
    distance = np.sqrt((x - ccol)**2 + (y - crow)**2)
    
    # Create band-pass mask
    mask = np.zeros((rows, cols), dtype=np.float32)
    mask[(distance >= inner_cutoff) & (distance <= outer_cutoff)] = 1
    
    return mask


def visualize_spectrum(img, spectrum, title="FFT Analysis"):
    """
    Visualize an image and its frequency spectrum side by side.
    
    Args:
        img: Input image
        spectrum: Magnitude spectrum
        title: Plot title
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    ax1.imshow(convert_to_grayscale(img), cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Magnitude spectrum
    ax2.imshow(spectrum, cmap='viridis')
    ax2.set_title('FFT Magnitude Spectrum')
    ax2.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig


def apply_fft_filter(img, filter_type='lowpass', cutoff=30):
    """
    Apply an FFT filter to an image.
    
    Args:
        img: Input image
        filter_type: Type of filter ('lowpass', 'highpass', or 'bandpass')
        cutoff: Cutoff radius for the filter
        
    Returns:
        tuple: (filtered_image, magnitude_spectrum, filter_mask)
    """
    # Convert to grayscale
    gray = convert_to_grayscale(img)
    
    # Compute FFT
    fshift, magnitude = compute_fft(gray)
    
    # Get dimensions
    rows, cols = gray.shape
    
    # Create filter mask
    if filter_type == 'lowpass':
        mask = create_lowpass_filter((rows, cols), cutoff)
    elif filter_type == 'highpass':
        mask = create_highpass_filter((rows, cols), cutoff)
    elif filter_type == 'bandpass':
        mask = create_bandpass_filter((rows, cols), cutoff // 2, cutoff)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    
    # Apply filter
    filtered = apply_filter(fshift, mask)
    
    return filtered, magnitude, mask


def inverse_fft(fshift):
    """
    Compute the inverse FFT to reconstruct the image.
    
    Args:
        fshift: Shifted frequency spectrum
        
    Returns:
        Reconstructed image (grayscale)
    """
    # Inverse shift to move the zero frequency component back to the top-left
    f_ishift = np.fft.ifftshift(fshift)
    
    # Compute the inverse FFT
    img_back = np.fft.ifft2(f_ishift)
    
    # Take the absolute value to get the reconstructed image
    img_back = np.abs(img_back)
    
    # Normalize to 0-255 and convert to uint8
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return img_back


if __name__ == "__main__":
    # Example usage
    img_path = "../../data/raw/AI_for_Art_Restoration_2/paired_dataset_art/damaged/sample_image.jpg"
    try:
        img = cv2.imread(img_path)
        if img is not None:
            # Apply different filters
            lowpass, magnitude, lp_mask = apply_fft_filter(img, 'lowpass', 30)
            highpass, _, hp_mask = apply_fft_filter(img, 'highpass', 30)
            bandpass, _, bp_mask = apply_fft_filter(img, 'bandpass', 50)
            
            # Visualize results
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Original and spectrum
            axes[0, 0].imshow(convert_to_grayscale(img), cmap='gray')
            axes[0, 0].set_title('Original')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(magnitude, cmap='viridis')
            axes[0, 1].set_title('FFT Magnitude')
            axes[0, 1].axis('off')
            
            # Filters
            axes[0, 2].imshow(lp_mask, cmap='gray')
            axes[0, 2].set_title('Low-pass Filter')
            axes[0, 2].axis('off')
            
            # Results
            axes[1, 0].imshow(lowpass, cmap='gray')
            axes[1, 0].set_title('Low-pass Result')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(highpass, cmap='gray')
            axes[1, 1].set_title('High-pass Result')
            axes[1, 1].axis('off')
            
            axes[1, 2].imshow(bandpass, cmap='gray')
            axes[1, 2].set_title('Band-pass Result')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            plt.show()
        else:
            print(f"Could not load image at {img_path}")
    except Exception as e:
        print(f"Error: {e}")
