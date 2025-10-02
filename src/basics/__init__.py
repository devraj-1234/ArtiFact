"""
Basic image processing module with fundamental FFT and restoration operations.

This module provides access to:
- FFT-related functions from basic_fft
- Restoration techniques from basic_restoration
"""

from src.basics.basic_fft import (
    convert_to_grayscale,
    compute_fft,
    inverse_fft,
    visualize_spectrum
)

from src.basics.basic_restoration import (
    denoise_gaussian,
    denoise_bilateral,
    denoise_nlm,
    enhance_contrast,
    enhance_clahe,
    sharpen_image,
    remove_scratches,
    restore_image
)
