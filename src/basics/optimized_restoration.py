"""
optimized_restoration.py
Optimized restoration using only Color Correction + Unsharp Mask.

Based on experiments, these two techniques provide the best quality-to-speed ratio
for aged artwork restoration.
"""

import cv2
import numpy as np
from .advanced_restoration import color_correction, unsharp_mask


def restore_image_optimized(
    image,
    color_method='white_balance',
    reference_image=None,
    sharpen_sigma=1.0,
    sharpen_strength=1.5,
    sharpen_threshold=0
):
    """
    Optimized restoration pipeline using only Color Correction + Unsharp Mask.
    
    This simplified pipeline provides excellent results with fast processing time,
    making it ideal for batch processing and production use.
    
    Args:
        image: Input image (BGR format)
        color_method: 'white_balance' or 'histogram_matching'
        reference_image: Reference image for histogram matching (optional)
        sharpen_sigma: Gaussian blur sigma for unsharp mask (0.5-3.0)
        sharpen_strength: Sharpening strength (0.5-3.0)
        sharpen_threshold: Minimum contrast to sharpen (0-20)
    
    Returns:
        Restored image (BGR format)
    
    Example:
        >>> import cv2
        >>> from src.basics.optimized_restoration import restore_image_optimized
        >>> 
        >>> img = cv2.imread('damaged.jpg')
        >>> restored = restore_image_optimized(img)
        >>> cv2.imwrite('restored.jpg', restored)
    """
    result = image.copy()
    
    # Step 1: Color Correction
    if color_method == 'histogram_matching' and reference_image is not None:
        result = color_correction(
            result, 
            reference=reference_image, 
            method='histogram_matching'
        )
    else:
        result = color_correction(result, method='white_balance')
    
    # Step 2: Unsharp Masking
    result = unsharp_mask(
        result, 
        sigma=sharpen_sigma, 
        strength=sharpen_strength, 
        threshold=sharpen_threshold
    )
    
    return result


def batch_restore(
    input_files,
    output_dir,
    color_method='white_balance',
    sharpen_sigma=1.0,
    sharpen_strength=1.5,
    sharpen_threshold=0,
    verbose=True
):
    """
    Batch restore multiple images using the optimized pipeline.
    
    Args:
        input_files: List of input file paths
        output_dir: Output directory path
        color_method: 'white_balance' or 'histogram_matching'
        sharpen_sigma: Gaussian blur sigma (0.5-3.0)
        sharpen_strength: Sharpening strength (0.5-3.0)
        sharpen_threshold: Minimum contrast to sharpen (0-20)
        verbose: Print progress messages
    
    Returns:
        List of successfully processed file paths
    
    Example:
        >>> import glob
        >>> from src.basics.optimized_restoration import batch_restore
        >>> 
        >>> files = glob.glob('damaged_images/*.jpg')
        >>> batch_restore(files, 'restored_images/')
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    successful = []
    failed = []
    
    for i, input_path in enumerate(input_files, 1):
        try:
            if verbose:
                print(f"Processing [{i}/{len(input_files)}]: {os.path.basename(input_path)}")
            
            # Read image
            img = cv2.imread(input_path)
            if img is None:
                raise ValueError(f"Could not read image: {input_path}")
            
            # Restore
            restored = restore_image_optimized(
                img,
                color_method=color_method,
                sharpen_sigma=sharpen_sigma,
                sharpen_strength=sharpen_strength,
                sharpen_threshold=sharpen_threshold
            )
            
            # Save
            filename = os.path.basename(input_path)
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, restored)
            
            successful.append(output_path)
            
        except Exception as e:
            if verbose:
                print(f"  ✗ Failed: {e}")
            failed.append((input_path, str(e)))
    
    # Summary
    if verbose:
        print("\n" + "="*60)
        print(f"Batch Processing Complete")
        print("="*60)
        print(f"Successful: {len(successful)}/{len(input_files)}")
        if failed:
            print(f"Failed: {len(failed)}")
            for path, error in failed:
                print(f"  - {os.path.basename(path)}: {error}")
        print("="*60)
    
    return successful


def calculate_quality_metrics(original, restored, ground_truth=None):
    """
    Calculate quality metrics for restoration.
    
    Args:
        original: Original damaged image
        restored: Restored image
        ground_truth: Ground truth image (optional)
    
    Returns:
        Dictionary with quality metrics
    """
    metrics = {}
    
    # Helper function to calculate PSNR
    def psnr(img1, img2):
        # Crop to minimum common size
        min_h = min(img1.shape[0], img2.shape[0])
        min_w = min(img1.shape[1], img2.shape[1])
        img1_crop = img1[:min_h, :min_w]
        img2_crop = img2[:min_h, :min_w]
        
        mse = np.mean((img1_crop.astype(np.float64) - img2_crop.astype(np.float64)) ** 2)
        if mse == 0:
            return float('inf')
        return 10 * np.log10((255.0 ** 2) / mse)
    
    # Calculate metrics
    if ground_truth is not None:
        metrics['psnr_original'] = psnr(original, ground_truth)
        metrics['psnr_restored'] = psnr(restored, ground_truth)
        metrics['psnr_improvement'] = metrics['psnr_restored'] - metrics['psnr_original']
        
        # Try to calculate SSIM if scikit-image is available
        try:
            from skimage.metrics import structural_similarity as ssim
            
            # Crop to common size
            min_h = min(original.shape[0], ground_truth.shape[0], restored.shape[0])
            min_w = min(original.shape[1], ground_truth.shape[1], restored.shape[1])
            
            orig_crop = original[:min_h, :min_w]
            rest_crop = restored[:min_h, :min_w]
            gt_crop = ground_truth[:min_h, :min_w]
            
            # Calculate win_size
            min_side = min(orig_crop.shape[0], orig_crop.shape[1])
            win_size = min(7, min_side)
            if win_size % 2 == 0:
                win_size -= 1
            
            metrics['ssim_original'] = ssim(orig_crop, gt_crop, channel_axis=2, 
                                           data_range=255, win_size=win_size)
            metrics['ssim_restored'] = ssim(rest_crop, gt_crop, channel_axis=2, 
                                           data_range=255, win_size=win_size)
            metrics['ssim_improvement'] = metrics['ssim_restored'] - metrics['ssim_original']
        except ImportError:
            pass
    
    return metrics


# CLI interface
if __name__ == "__main__":
    import sys
    import glob
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single image: python optimized_restoration.py input.jpg [output.jpg]")
        print("  Batch mode:   python optimized_restoration.py 'input/*.jpg' output_dir/")
        sys.exit(1)
    
    input_pattern = sys.argv[1]
    
    # Check if it's a pattern or single file
    files = glob.glob(input_pattern)
    
    if not files:
        print(f"No files found matching: {input_pattern}")
        sys.exit(1)
    
    if len(files) == 1:
        # Single file mode
        input_path = files[0]
        output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.replace('.', '_restored.')
        
        print(f"Restoring: {input_path}")
        img = cv2.imread(input_path)
        restored = restore_image_optimized(img)
        cv2.imwrite(output_path, restored)
        print(f"Saved: {output_path}")
        
    else:
        # Batch mode
        output_dir = sys.argv[2] if len(sys.argv) > 2 else 'restored/'
        print(f"Batch processing {len(files)} files...")
        batch_restore(files, output_dir)
