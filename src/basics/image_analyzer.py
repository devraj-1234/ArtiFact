"""
image_analyzer.py
A simple image analysis tool for examining artwork.
This tool can load images, perform FFT analysis, and compare images.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

# Add parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from basics.basic_fft import convert_to_grayscale, compute_fft, apply_fft_filter


class ImageAnalyzer:
    """A class for analyzing and comparing images using FFT."""
    
    def __init__(self):
        """Initialize the analyzer."""
        pass
    
    def load_image(self, path):
        """
        Load an image from a file.
        
        Args:
            path: Path to the image file
            
        Returns:
            numpy.ndarray: Loaded image or None if loading failed
        """
        img = cv2.imread(path)
        if img is None:
            print(f"Error: Could not load image at {path}")
            return None
        return img
    
    def analyze_image(self, img, title="Image Analysis"):
        """
        Perform basic analysis of an image.
        
        Args:
            img: Input image
            title: Title for the analysis
            
        Returns:
            dict: Dictionary containing analysis results
        """
        # Convert to grayscale
        gray = convert_to_grayscale(img)
        
        # Get basic image statistics
        mean = np.mean(gray)
        std = np.std(gray)
        min_val = np.min(gray)
        max_val = np.max(gray)
        
        # Compute FFT
        fshift, magnitude = compute_fft(gray)
        
        # Compute histograms
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Display results
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=16)
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Grayscale image
        axes[0, 1].imshow(gray, cmap='gray')
        axes[0, 1].set_title('Grayscale')
        axes[0, 1].axis('off')
        
        # FFT magnitude spectrum
        axes[1, 0].imshow(magnitude, cmap='viridis')
        axes[1, 0].set_title('FFT Magnitude Spectrum')
        axes[1, 0].axis('off')
        
        # Histogram
        axes[1, 1].plot(hist)
        axes[1, 1].set_title('Grayscale Histogram')
        axes[1, 1].set_xlabel('Pixel Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Return analysis results
        return {
            'mean': mean,
            'std_dev': std,
            'min': min_val,
            'max': max_val,
            'fft_magnitude': magnitude,
            'histogram': hist
        }
    
    def compare_images(self, img1, img2, title1="Image 1", title2="Image 2"):
        """
        Compare two images and their frequency characteristics.
        
        Args:
            img1: First image
            img2: Second image
            title1: Title for the first image
            title2: Title for the second image
        """
        # Convert to grayscale
        gray1 = convert_to_grayscale(img1)
        gray2 = convert_to_grayscale(img2)
        
        # Resize the second image to match the first if needed
        if gray1.shape != gray2.shape:
            gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
        
        # Compute FFT
        fshift1, magnitude1 = compute_fft(gray1)
        fshift2, magnitude2 = compute_fft(gray2)
        
        # Calculate difference
        diff_img = cv2.absdiff(gray1, gray2)
        diff_magnitude = np.abs(magnitude1 - magnitude2)
        
        # Display results
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Original images
        axes[0, 0].imshow(gray1, cmap='gray')
        axes[0, 0].set_title(title1)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(gray2, cmap='gray')
        axes[0, 1].set_title(title2)
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(diff_img, cmap='hot')
        axes[0, 2].set_title('Image Difference')
        axes[0, 2].axis('off')
        
        # FFT spectrums
        axes[1, 0].imshow(magnitude1, cmap='viridis')
        axes[1, 0].set_title(f'FFT of {title1}')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(magnitude2, cmap='viridis')
        axes[1, 1].set_title(f'FFT of {title2}')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(diff_magnitude, cmap='hot')
        axes[1, 2].set_title('FFT Difference')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics about the differences
        mean_diff = np.mean(diff_img)
        std_diff = np.std(diff_img)
        mean_fft_diff = np.mean(diff_magnitude)
        std_fft_diff = np.std(diff_magnitude)
        
        print("Comparison Statistics:")
        print(f"Mean pixel difference: {mean_diff:.2f}")
        print(f"Std dev of pixel difference: {std_diff:.2f}")
        print(f"Mean FFT difference: {mean_fft_diff:.2f}")
        print(f"Std dev of FFT difference: {std_fft_diff:.2f}")
    
    def restore_image(self, img, method='basic', cutoff=30):
        """
        Attempt to restore a damaged image using FFT filtering.
        
        Args:
            img: Input image
            method: Restoration method ('basic', 'advanced')
            cutoff: Cutoff radius for filters
            
        Returns:
            numpy.ndarray: Restored image
        """
        # Convert to grayscale
        gray = convert_to_grayscale(img)
        
        if method == 'basic':
            # Use simple lowpass filter
            restored, _, _ = apply_fft_filter(gray, 'lowpass', cutoff)
            
        elif method == 'advanced':
            # Apply a more sophisticated approach
            # 1. Apply bandpass filter to remove certain frequency components
            _, magnitude, _ = apply_fft_filter(gray, 'bandpass', cutoff)
            
            # 2. Get FFT
            fshift, _ = compute_fft(gray)
            
            # 3. Create custom filter based on magnitude
            rows, cols = gray.shape
            crow, ccol = rows // 2, cols // 2
            
            # Create mask that preserves strong frequencies
            mask = np.ones((rows, cols), dtype=np.float32)
            
            # Remove mid-range frequencies that often contain damage patterns
            mask[crow-cutoff//2:crow+cutoff//2, ccol-cutoff//2:ccol+cutoff//2] = 0.5
            
            # Apply custom filter
            filtered_fshift = fshift * mask
            f_ishift = np.fft.ifftshift(filtered_fshift)
            restored = np.fft.ifft2(f_ishift)
            restored = np.abs(restored)
            restored = cv2.normalize(restored, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
        else:
            raise ValueError(f"Unknown restoration method: {method}")
        
        # Display results
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(gray, cmap='gray')
        axes[0].set_title('Original Damaged')
        axes[0].axis('off')
        
        axes[1].imshow(restored, cmap='gray')
        axes[1].set_title(f'Restored ({method})')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return restored


def main():
    """Main function to demonstrate the ImageAnalyzer."""
    analyzer = ImageAnalyzer()
    
    # Get directory of this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set paths to dataset
    data_path = os.path.join(current_dir, "../../data/raw/AI_for_Art_Restoration_2/paired_dataset_art")
    damaged_dir = os.path.join(data_path, "damaged")
    undamaged_dir = os.path.join(data_path, "undamaged")
    
    # List files in both directories
    try:
        damaged_files = os.listdir(damaged_dir)
        undamaged_files = os.listdir(undamaged_dir)
        
        # Find common files
        common_files = [f for f in damaged_files if f in undamaged_files]
        
        if common_files:
            # Select a sample file
            sample_file = common_files[0]
            print(f"Selected sample file: {sample_file}")
            
            # Load images
            damaged_img = analyzer.load_image(os.path.join(damaged_dir, sample_file))
            undamaged_img = analyzer.load_image(os.path.join(undamaged_dir, sample_file))
            
            if damaged_img is not None and undamaged_img is not None:
                # Analyze damaged image
                print("Analyzing damaged image...")
                analyzer.analyze_image(damaged_img, "Analysis of Damaged Artwork")
                
                # Compare damaged and undamaged images
                print("Comparing images...")
                analyzer.compare_images(damaged_img, undamaged_img, "Damaged", "Undamaged")
                
                # Try to restore the damaged image
                print("Attempting restoration...")
                analyzer.restore_image(damaged_img, 'basic')
                analyzer.restore_image(damaged_img, 'advanced')
            else:
                print("Failed to load images.")
        else:
            print("No matching files found in both damaged and undamaged directories.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
