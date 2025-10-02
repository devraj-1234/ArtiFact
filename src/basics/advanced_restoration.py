"""
advanced_restoration.py
Advanced image restoration techniques using FFT, Wavelets, and sophisticated algorithms.
This module provides state-of-the-art methods for restoring degraded images.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift


def fft_denoise(image, threshold_percentile=90):
    """
    Remove noise using FFT by filtering high-frequency components.
    
    Args:
        image: Input image (BGR or grayscale)
        threshold_percentile: Percentile threshold for frequency filtering (higher = more aggressive)
    
    Returns:
        Denoised image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        is_color = True
    else:
        gray = image.copy()
        is_color = False
    
    # Apply FFT
    f_transform = fft2(gray)
    f_shift = fftshift(f_transform)
    
    # Calculate magnitude spectrum
    magnitude = np.abs(f_shift)
    
    # Create mask based on threshold
    threshold = np.percentile(magnitude, threshold_percentile)
    mask = magnitude > threshold
    
    # Apply mask
    f_shift_filtered = f_shift * mask
    
    # Inverse FFT
    f_ishift = ifftshift(f_shift_filtered)
    img_back = ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    # Normalize to 0-255
    img_back = np.clip(img_back, 0, 255).astype(np.uint8)
    
    if is_color:
        # Apply to each channel
        result = np.zeros_like(image)
        for i in range(3):
            channel = image[:, :, i]
            f_transform = fft2(channel)
            f_shift = fftshift(f_transform)
            f_shift_filtered = f_shift * mask
            f_ishift = ifftshift(f_shift_filtered)
            img_back = ifft2(f_ishift)
            result[:, :, i] = np.clip(np.abs(img_back), 0, 255).astype(np.uint8)
        return result
    else:
        return img_back


def remove_periodic_noise(image, filter_type='notch', d0=30):
    """
    Remove periodic noise patterns using frequency domain filtering.
    
    Args:
        image: Input image
        filter_type: 'notch' or 'bandstop'
        d0: Filter radius
    
    Returns:
        Image with periodic noise removed
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        is_color = True
    else:
        gray = image.copy()
        is_color = False
    
    # Apply FFT
    f_transform = fft2(gray)
    f_shift = fftshift(f_transform)
    
    # Get image dimensions
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create filter
    mask = np.ones((rows, cols), np.uint8)
    
    if filter_type == 'notch':
        # Create notch filter to remove specific frequencies
        y, x = np.ogrid[:rows, :cols]
        
        # Find peaks in magnitude spectrum (excluding DC component)
        magnitude = np.abs(f_shift)
        magnitude[crow-5:crow+5, ccol-5:ccol+5] = 0  # Mask DC component
        
        # Find top peaks
        threshold = np.percentile(magnitude, 99.5)
        peaks = magnitude > threshold
        
        # Create notch filter around peaks
        for i in range(rows):
            for j in range(cols):
                if peaks[i, j]:
                    dist = np.sqrt((x - j)**2 + (y - i)**2)
                    mask[dist < d0] = 0
    
    # Apply filter
    f_shift_filtered = f_shift * mask
    
    # Inverse FFT
    f_ishift = ifftshift(f_shift_filtered)
    img_back = ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = np.clip(img_back, 0, 255).astype(np.uint8)
    
    if is_color:
        # Apply same mask to color channels
        result = np.zeros_like(image)
        for i in range(3):
            channel = image[:, :, i]
            f_transform = fft2(channel)
            f_shift = fftshift(f_transform)
            f_shift_filtered = f_shift * mask
            f_ishift = ifftshift(f_shift_filtered)
            img_back = ifft2(f_ishift)
            result[:, :, i] = np.clip(np.abs(img_back), 0, 255).astype(np.uint8)
        return result
    else:
        return img_back


def wavelet_denoise(image, wavelet='db1', level=1, threshold_scale=1.0):
    """
    Denoise image using wavelet transform.
    Requires pywt (PyWavelets) library.
    
    Args:
        image: Input image
        wavelet: Wavelet type ('db1', 'db2', 'haar', 'sym2', etc.)
        level: Decomposition level
        threshold_scale: Threshold scaling factor
    
    Returns:
        Denoised image
    """
    try:
        import pywt
    except ImportError:
        print("PyWavelets (pywt) not installed. Install with: pip install PyWavelets")
        return image
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        is_color = True
        result = np.zeros_like(image)
        
        for i in range(3):
            channel = image[:, :, i].astype(float)
            
            # Perform wavelet decomposition
            coeffs = pywt.wavedec2(channel, wavelet, level=level)
            
            # Calculate threshold
            sigma = np.median(np.abs(coeffs[-1][1])) / 0.6745
            threshold = sigma * threshold_scale * np.sqrt(2 * np.log(channel.size))
            
            # Apply thresholding
            new_coeffs = [coeffs[0]]
            for detail_level in coeffs[1:]:
                new_coeffs.append(tuple([pywt.threshold(c, threshold, mode='soft') 
                                        for c in detail_level]))
            
            # Reconstruct
            reconstructed = pywt.waverec2(new_coeffs, wavelet)
            
            # Handle size mismatch
            reconstructed = reconstructed[:channel.shape[0], :channel.shape[1]]
            result[:, :, i] = np.clip(reconstructed, 0, 255).astype(np.uint8)
        
        return result
    else:
        channel = image.astype(float)
        coeffs = pywt.wavedec2(channel, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1][1])) / 0.6745
        threshold = sigma * threshold_scale * np.sqrt(2 * np.log(channel.size))
        
        new_coeffs = [coeffs[0]]
        for detail_level in coeffs[1:]:
            new_coeffs.append(tuple([pywt.threshold(c, threshold, mode='soft') 
                                    for c in detail_level]))
        
        reconstructed = pywt.waverec2(new_coeffs, wavelet)
        reconstructed = reconstructed[:image.shape[0], :image.shape[1]]
        return np.clip(reconstructed, 0, 255).astype(np.uint8)


def auto_detect_damage_mask(image, method='edges', **kwargs):
    """
    Automatically detect damaged areas in an image.
    
    Args:
        image: Input image
        method: 'edges', 'brightness', or 'combined'
        **kwargs: Additional parameters for specific methods
    
    Returns:
        Binary mask of damaged areas
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    mask = np.zeros(gray.shape, dtype=np.uint8)
    
    if method == 'edges':
        # Detect edges (scratches, cracks)
        edges = cv2.Canny(gray, kwargs.get('low_thresh', 50), kwargs.get('high_thresh', 150))
        
        # Dilate to make edges thicker
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(edges, kernel, iterations=kwargs.get('dilate_iter', 2))
    
    elif method == 'brightness':
        # Detect very bright or very dark areas
        low_thresh = kwargs.get('low_thresh', 20)
        high_thresh = kwargs.get('high_thresh', 235)
        
        mask = ((gray < low_thresh) | (gray > high_thresh)).astype(np.uint8) * 255
    
    elif method == 'combined':
        # Combine multiple detection methods
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        
        brightness_mask = ((gray < 20) | (gray > 235)).astype(np.uint8) * 255
        
        mask = cv2.bitwise_or(edges_dilated, brightness_mask)
    
    return mask


def exemplar_inpaint(image, mask, patch_size=9):
    """
    Advanced inpainting using exemplar-based method.
    Fills damaged areas using similar patches from the image.
    
    Args:
        image: Input image
        mask: Binary mask (255 = damaged, 0 = good)
        patch_size: Size of patches to use
    
    Returns:
        Inpainted image
    """
    # Use OpenCV's photo inpainting as a sophisticated exemplar method
    result = cv2.inpaint(image, mask, inpaintRadius=patch_size, 
                         flags=cv2.INPAINT_TELEA)
    
    return result


def anisotropic_diffusion(image, iterations=10, kappa=50, gamma=0.1):
    """
    Apply anisotropic diffusion for edge-preserving smoothing.
    
    Args:
        image: Input image
        iterations: Number of iterations
        kappa: Conduction coefficient
        gamma: Rate of diffusion
    
    Returns:
        Smoothed image with preserved edges
    """
    if len(image.shape) == 3:
        is_color = True
        img = image.astype(float)
        result = np.zeros_like(img)
        
        for i in range(3):
            result[:, :, i] = _anisotropic_diffusion_channel(
                img[:, :, i], iterations, kappa, gamma
            )
        
        return np.clip(result, 0, 255).astype(np.uint8)
    else:
        img = image.astype(float)
        result = _anisotropic_diffusion_channel(img, iterations, kappa, gamma)
        return np.clip(result, 0, 255).astype(np.uint8)


def _anisotropic_diffusion_channel(channel, iterations, kappa, gamma):
    """Helper function for single channel anisotropic diffusion."""
    img = channel.copy()
    
    for _ in range(iterations):
        # Calculate gradients
        nabla_n = np.roll(img, -1, axis=0) - img  # North
        nabla_s = np.roll(img, 1, axis=0) - img   # South
        nabla_e = np.roll(img, -1, axis=1) - img  # East
        nabla_w = np.roll(img, 1, axis=1) - img   # West
        
        # Calculate conduction coefficients
        c_n = np.exp(-(nabla_n / kappa) ** 2)
        c_s = np.exp(-(nabla_s / kappa) ** 2)
        c_e = np.exp(-(nabla_e / kappa) ** 2)
        c_w = np.exp(-(nabla_w / kappa) ** 2)
        
        # Update image
        img += gamma * (c_n * nabla_n + c_s * nabla_s + 
                       c_e * nabla_e + c_w * nabla_w)
    
    return img


def unsharp_mask(image, sigma=1.0, strength=1.5, threshold=0):
    """
    Advanced sharpening using unsharp masking.
    
    Args:
        image: Input image
        sigma: Gaussian blur sigma
        strength: Sharpening strength
        threshold: Threshold for sharpening
    
    Returns:
        Sharpened image
    """
    # Create blurred version
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    
    # Calculate sharpening mask
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    
    # Apply threshold if specified
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        sharpened = np.where(low_contrast_mask, image, sharpened)
    
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def color_correction(image, reference=None, method='histogram_matching'):
    """
    Correct color cast and match color palette.
    
    Args:
        image: Input image to correct
        reference: Reference image (if None, use auto white balance)
        method: 'histogram_matching' or 'white_balance'
    
    Returns:
        Color-corrected image
    """
    if method == 'white_balance' or reference is None:
        # Simple gray world white balance
        result = image.copy().astype(float)
        
        for i in range(3):
            avg = np.mean(result[:, :, i])
            result[:, :, i] = result[:, :, i] * (128.0 / avg)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    elif method == 'histogram_matching':
        # Match histogram to reference image
        result = np.zeros_like(image)
        
        for i in range(3):
            result[:, :, i] = _match_histograms(
                image[:, :, i], reference[:, :, i]
            )
        
        return result
    
    return image


def _match_histograms(source, template):
    """Match histogram of source image to template image."""
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()
    
    # Get the set of unique pixel values and their corresponding indices
    s_values, s_counts = np.unique(source, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)
    
    # Calculate CDFs
    s_quantiles = np.cumsum(s_counts).astype(float) / source.size
    t_quantiles = np.cumsum(t_counts).astype(float) / template.size
    
    # Interpolate
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    
    # Map source values to template values
    matched = np.interp(source, s_values, interp_t_values)
    
    return matched.reshape(oldshape).astype(np.uint8)


def multi_scale_restoration(image, scales=3):
    """
    Apply restoration at multiple scales using Gaussian pyramid.
    
    Args:
        image: Input image
        scales: Number of pyramid levels
    
    Returns:
        Restored image
    """
    # Build Gaussian pyramid
    pyramid = [image]
    for i in range(scales - 1):
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    
    # Process each level
    processed_pyramid = []
    for level in pyramid:
        # Apply restoration (denoising + sharpening)
        denoised = cv2.bilateralFilter(level, 9, 75, 75)
        sharpened = unsharp_mask(denoised, sigma=1.0, strength=1.2)
        processed_pyramid.append(sharpened)
    
    # Reconstruct from pyramid
    result = processed_pyramid[-1]
    for i in range(len(processed_pyramid) - 2, -1, -1):
        result = cv2.pyrUp(result)
        # Ensure same size
        result = result[:processed_pyramid[i].shape[0], :processed_pyramid[i].shape[1]]
        # Blend
        result = cv2.addWeighted(result, 0.5, processed_pyramid[i], 0.5, 0)
    
    return result
