"""
Intelligent Restoration System
Combines FFT-based damage analysis with optimized restoration pipeline.

This system:
1. Analyzes image using FFT features
2. Determines damage type and severity
3. Applies the best restoration method (Color Correction + Unsharp Mask)
4. Optionally uses advanced methods for severe damage
"""

import cv2
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.ml.feature_extractor import extract_ml_features
from src.basics.optimized_restoration import restore_image_optimized
from src.basics.advanced_restoration import (
    fft_denoise, 
    anisotropic_diffusion,
    color_correction,
    unsharp_mask
)


class IntelligentRestorer:
    """
    Intelligent restoration system that analyzes damage and applies optimal restoration.
    
    Uses FFT features to classify damage, then applies the best restoration method.
    Default: Color Correction + Unsharp Mask (90% quality, 5% processing time)
    """
    
    def __init__(self, model=None):
        """
        Initialize the intelligent restorer.
        
        Args:
            model: Pre-trained ML classifier (optional, for future use)
        """
        self.model = model
        
        # Define thresholds based on feature analysis
        # These can be tuned based on your dataset
        self.thresholds = {
            'high_freq_energy': 75.0,    # High = noisy/scratched
            'energy_ratio': 0.18,         # High = damaged
            'low_freq_energy': 400.0      # Low = faded
        }
    
    def analyze_damage(self, image_path):
        """
        Analyze image and determine damage type and severity.
        
        Args:
            image_path: Path to image file
            
        Returns:
            dict with:
                - severity: 'light', 'moderate', 'severe'
                - damage_types: list of detected damage types
                - features: extracted FFT features
                - recommended_method: restoration method to use
        """
        # Extract FFT features
        features, feature_names = extract_ml_features(image_path)
        features_dict = dict(zip(feature_names, features))
        
        # Analyze features to determine damage type
        damage_types = []
        
        if features_dict['high_freq_energy'] > self.thresholds['high_freq_energy']:
            damage_types.append('noise/scratches')
        
        if features_dict['energy_ratio'] > self.thresholds['energy_ratio']:
            damage_types.append('heavy_damage')
        
        if features_dict['low_freq_energy'] < self.thresholds['low_freq_energy']:
            damage_types.append('fading')
        
        # Determine severity
        if len(damage_types) == 0:
            severity = 'light'
            recommended_method = 'optimized'
        elif len(damage_types) == 1:
            severity = 'moderate'
            recommended_method = 'optimized'
        else:
            severity = 'severe'
            recommended_method = 'advanced'  # Use advanced only for severe cases
        
        return {
            'severity': severity,
            'damage_types': damage_types if damage_types else ['minor_wear'],
            'features': features_dict,
            'recommended_method': recommended_method
        }
    
    def restore_optimized(
        self, 
        image, 
        color_method='white_balance',
        sharpen_sigma=1.0,
        sharpen_strength=1.5,
        sharpen_threshold=0
    ):
        """
        Apply optimized restoration (Color Correction + Unsharp Mask).
        
        This is the recommended method for 90% of cases:
        - Fast processing (~0.5 seconds)
        - Excellent quality improvement (+5 to +8 dB PSNR)
        - Simple and reliable
        
        Args:
            image: Input image (BGR)
            color_method: 'white_balance' or 'histogram_matching'
            sharpen_sigma: Gaussian blur sigma (0.5-3.0)
            sharpen_strength: Sharpening strength (0.5-3.0)
            sharpen_threshold: Minimum contrast to sharpen (0-20)
            
        Returns:
            Restored image (BGR)
        """
        return restore_image_optimized(
            image,
            color_method=color_method,
            sharpen_sigma=sharpen_sigma,
            sharpen_strength=sharpen_strength,
            sharpen_threshold=sharpen_threshold
        )
    
    def restore_advanced(self, image):
        """
        Apply advanced restoration for severe damage.
        
        Uses multiple techniques:
        1. FFT denoising (remove high-frequency noise)
        2. Anisotropic diffusion (edge-preserving smoothing)
        3. Color correction (fix fading)
        4. Unsharp mask (enhance details)
        
        Note: Slower than optimized (~10-30 seconds) but handles severe damage.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Restored image (BGR)
        """
        result = image.copy()
        
        # Step 1: FFT denoising
        result = fft_denoise(result, threshold_percentile=92)
        
        # Step 2: Anisotropic diffusion
        result = anisotropic_diffusion(result, iterations=8, kappa=50, gamma=0.1)
        
        # Step 3: Color correction
        result = color_correction(result, method='white_balance')
        
        # Step 4: Final sharpening
        result = unsharp_mask(result, sigma=1.0, strength=1.3)
        
        return result
    
    def restore_auto(self, image_path, output_path=None, verbose=True):
        """
        Automatically analyze and restore an image.
        
        This is the main function to use - it analyzes the image and
        applies the best restoration method automatically.
        
        Args:
            image_path: Path to damaged image
            output_path: Path to save restored image (optional)
            verbose: Print analysis results
            
        Returns:
            tuple: (restored_image, analysis_dict)
        
        Example:
            >>> restorer = IntelligentRestorer()
            >>> restored, info = restorer.restore_auto('damaged.jpg', 'restored.jpg')
            >>> print(f"Damage severity: {info['severity']}")
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Analyze damage
        analysis = self.analyze_damage(image_path)
        
        if verbose:
            print("\n" + "="*60)
            print("INTELLIGENT RESTORATION ANALYSIS")
            print("="*60)
            print(f"Image: {os.path.basename(image_path)}")
            print(f"Damage Severity: {analysis['severity'].upper()}")
            print(f"Damage Types: {', '.join(analysis['damage_types'])}")
            print(f"Recommended Method: {analysis['recommended_method']}")
            print("-"*60)
        
        # Apply restoration based on recommendation
        if analysis['recommended_method'] == 'optimized':
            if verbose:
                print("Applying: Color Correction + Unsharp Mask (Fast)")
            restored = self.restore_optimized(image)
        else:
            if verbose:
                print("Applying: Advanced Pipeline (Slower, for severe damage)")
            restored = self.restore_advanced(image)
        
        if verbose:
            print("✓ Restoration complete!")
            print("="*60 + "\n")
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, restored)
            if verbose:
                print(f"Saved to: {output_path}")
        
        return restored, analysis
    
    def batch_restore(
        self, 
        input_dir, 
        output_dir, 
        extensions=('.jpg', '.png', '.jpeg'),
        verbose=True
    ):
        """
        Batch restore all images in a directory.
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            extensions: Image file extensions to process
            verbose: Print progress
            
        Returns:
            dict with statistics
        
        Example:
            >>> restorer = IntelligentRestorer()
            >>> stats = restorer.batch_restore('damaged/', 'restored/')
        """
        import os
        from glob import glob
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all images
        image_files = []
        for ext in extensions:
            image_files.extend(glob(os.path.join(input_dir, f"*{ext}")))
        
        if verbose:
            print(f"\nFound {len(image_files)} images to process")
            print("="*60)
        
        # Statistics
        stats = {
            'total': len(image_files),
            'light': 0,
            'moderate': 0,
            'severe': 0,
            'optimized_method': 0,
            'advanced_method': 0,
            'success': 0,
            'failed': 0
        }
        
        # Process each image
        for idx, input_path in enumerate(image_files, 1):
            try:
                filename = os.path.basename(input_path)
                output_path = os.path.join(output_dir, filename)
                
                if verbose:
                    print(f"\n[{idx}/{len(image_files)}] Processing: {filename}")
                
                # Restore
                restored, analysis = self.restore_auto(
                    input_path, 
                    output_path, 
                    verbose=verbose
                )
                
                # Update stats
                stats['success'] += 1
                stats[analysis['severity']] += 1
                if analysis['recommended_method'] == 'optimized':
                    stats['optimized_method'] += 1
                else:
                    stats['advanced_method'] += 1
                    
            except Exception as e:
                print(f"✗ Error processing {filename}: {e}")
                stats['failed'] += 1
        
        # Print summary
        if verbose:
            print("\n" + "="*60)
            print("BATCH RESTORATION SUMMARY")
            print("="*60)
            print(f"Total images: {stats['total']}")
            print(f"Successfully processed: {stats['success']}")
            print(f"Failed: {stats['failed']}")
            print(f"\nDamage Severity:")
            print(f"  Light: {stats['light']}")
            print(f"  Moderate: {stats['moderate']}")
            print(f"  Severe: {stats['severe']}")
            print(f"\nMethods Used:")
            print(f"  Optimized (Color+Sharpen): {stats['optimized_method']}")
            print(f"  Advanced (Multi-technique): {stats['advanced_method']}")
            print("="*60 + "\n")
        
        return stats


# Convenience function for quick restoration
def restore_image(input_path, output_path=None, verbose=True):
    """
    Quick restoration function - analyzes and restores automatically.
    
    Args:
        input_path: Path to damaged image
        output_path: Path to save restored image (optional)
        verbose: Print analysis
        
    Returns:
        Restored image
    
    Example:
        >>> from src.ml.intelligent_restoration import restore_image
        >>> restored = restore_image('damaged.jpg', 'restored.jpg')
    """
    restorer = IntelligentRestorer()
    restored, _ = restorer.restore_auto(input_path, output_path, verbose)
    return restored


if __name__ == "__main__":
    # Test the intelligent restorer
    import sys
    
    if len(sys.argv) > 1:
        # Command-line usage
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        
        restorer = IntelligentRestorer()
        restored, analysis = restorer.restore_auto(input_path, output_path)
        
        print(f"\n✓ Image restored successfully!")
        if output_path:
            print(f"Saved to: {output_path}")
    else:
        print("Intelligent Restoration System")
        print("="*60)
        print("\nUsage:")
        print("  python intelligent_restoration.py <input_image> [output_image]")
        print("\nExample:")
        print("  python intelligent_restoration.py damaged.jpg restored.jpg")
        print("\nOr use in Python:")
        print("  >>> from src.ml.intelligent_restoration import restore_image")
        print("  >>> restored = restore_image('damaged.jpg', 'restored.jpg')")
