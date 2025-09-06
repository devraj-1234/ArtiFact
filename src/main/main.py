"""
main.py
Main module for the ArtifactVision project.
This module provides a command-line interface for both artifact restoration and forgery detection.
"""

import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.restoration_model import ArtifactRestorer
from src.models.detection_model import ArtForgeryDetector
from src.utils.feature_extraction import extract_fft_features
from src.utils.visualization import plot_restoration_results, plot_image_comparison, save_visualization


def restore_image(args):
    """
    Restore a damaged artwork image.
    
    Args:
        args: Command line arguments
    """
    # Load the model
    input_shape = (256, 256, 3)  # Default shape
    model = ArtifactRestorer(input_shape=input_shape, model_path=args.model_path)
    
    # Load the damaged image
    damaged_img = cv2.imread(args.input_image)
    if damaged_img is None:
        print(f"Error: Could not load image {args.input_image}")
        return
    
    # Restore the image
    print(f"Restoring image {args.input_image}...")
    restored_img = model.restore(damaged_img)
    
    # Save the result
    output_file = args.output_image or os.path.join(
        os.path.dirname(args.input_image),
        "restored_" + os.path.basename(args.input_image)
    )
    cv2.imwrite(output_file, restored_img)
    print(f"Saved restored image to {output_file}")
    
    # Create visualization
    if args.visualize:
        # Convert BGR to RGB for visualization
        damaged_rgb = cv2.cvtColor(damaged_img, cv2.COLOR_BGR2RGB)
        restored_rgb = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
        
        fig = plot_image_comparison(damaged_rgb, restored_rgb, 
                                  titles=['Damaged', 'Restored'])
        
        viz_file = os.path.splitext(output_file)[0] + "_comparison.png"
        save_visualization(fig, os.path.basename(viz_file), os.path.dirname(viz_file))
        print(f"Saved visualization to {viz_file}")


def detect_forgery(args):
    """
    Detect if an artwork is genuine or a forgery.
    
    Args:
        args: Command line arguments
    """
    # Load the model
    detector = ArtForgeryDetector(model_type=args.model_type, model_path=args.model_path)
    
    # Load the image
    img = cv2.imread(args.input_image)
    if img is None:
        print(f"Error: Could not load image {args.input_image}")
        return
    
    # Predict forgery
    print(f"Analyzing image {args.input_image}...")
    prediction, probability = detector.predict(img)
    
    # Output results
    status = "FAKE/DAMAGED" if prediction == 1 else "GENUINE"
    print(f"Prediction: {status}")
    print(f"Confidence: {probability:.2%}")
    
    # Create visualization if requested
    if args.visualize:
        # Get FFT magnitude
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        fft_magnitude = extract_fft_features(gray)
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Prediction: {status} ({probability:.2%} confidence)")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(fft_magnitude, cmap='viridis')
        plt.title("FFT Magnitude Spectrum")
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        output_dir = os.path.dirname(args.input_image)
        viz_file = os.path.join(output_dir, "analysis_" + os.path.basename(args.input_image))
        plt.savefig(viz_file)
        plt.close()
        print(f"Saved analysis visualization to {viz_file}")


def main():
    # Create main parser
    parser = argparse.ArgumentParser(
        description="ArtifactVision - Art Restoration and Forgery Detection"
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Restoration parser
    restore_parser = subparsers.add_parser('restore', help='Restore a damaged artwork')
    restore_parser.add_argument('--input_image', '-i', type=str, required=True,
                             help='Path to the damaged image')
    restore_parser.add_argument('--output_image', '-o', type=str,
                             help='Path to save the restored image')
    restore_parser.add_argument('--model_path', '-m', type=str, 
                             default='outputs/models/restoration_model.h5',
                             help='Path to the restoration model')
    restore_parser.add_argument('--visualize', '-v', action='store_true',
                             help='Generate visualization of before and after')
    
    # Forgery detection parser
    detect_parser = subparsers.add_parser('detect', help='Detect if an artwork is genuine or fake')
    detect_parser.add_argument('--input_image', '-i', type=str, required=True,
                             help='Path to the image to analyze')
    detect_parser.add_argument('--model_path', '-m', type=str,
                             default='outputs/models/detection_model_rf.joblib',
                             help='Path to the detection model')
    detect_parser.add_argument('--model_type', '-t', type=str, choices=['rf', 'svm'],
                             default='rf', help='Type of model (Random Forest or SVM)')
    detect_parser.add_argument('--visualize', '-v', action='store_true',
                             help='Generate visualization of the analysis')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run appropriate function
    if args.command == 'restore':
        restore_image(args)
    elif args.command == 'detect':
        detect_forgery(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
