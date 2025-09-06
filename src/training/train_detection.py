"""
train_detection.py
Train the forgery detection model.
"""

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset import ArtDataset
from models.detection_model import ArtForgeryDetector
from utils.feature_extraction import extract_fft_features, extract_fft_bands, extract_texture_features
from utils.visualization import plot_confusion_matrix, plot_feature_importance, save_visualization


def extract_features_batch(images):
    """
    Extract features from a batch of images.
    
    Args:
        images (numpy.ndarray): Batch of images
        
    Returns:
        numpy.ndarray: Feature matrix
    """
    features_list = []
    
    for img in tqdm(images, desc="Extracting features"):
        # Extract FFT features
        fft_magnitude = extract_fft_features(img, normalize=True)
        
        # Calculate statistical features from FFT magnitude
        fft_mean = np.mean(fft_magnitude)
        fft_std = np.std(fft_magnitude)
        fft_skew = np.mean(((fft_magnitude - fft_mean) / fft_std) ** 3) if fft_std > 0 else 0
        fft_kurtosis = np.mean(((fft_magnitude - fft_mean) / fft_std) ** 4) - 3 if fft_std > 0 else 0
        
        # Extract frequency band features
        band_features = extract_fft_bands(img, bands=3)
        
        # Extract texture features
        texture_features = extract_texture_features(img)
        
        # Combine all features
        features = np.hstack([
            fft_mean, fft_std, fft_skew, fft_kurtosis,
            band_features,
            texture_features
        ])
        
        features_list.append(features)
    
    return np.array(features_list)


def train_detection_model(args):
    """
    Train the forgery detection model.
    
    Args:
        args: Command line arguments
    """
    # Create dataset
    print(f"Loading dataset from {args.data_path}...")
    dataset = ArtDataset(args.data_path, img_size=(args.img_size, args.img_size), split_ratio=args.val_split)
    
    # Load and split data for forgery detection
    print("Creating forgery detection dataset...")
    X_train, X_val, y_train, y_val = dataset.create_fake_detection_dataset()
    
    print(f"Training images: {X_train.shape}")
    print(f"Validation images: {X_val.shape}")
    
    # Extract features from images
    print("Extracting features from training images...")
    X_train_features = extract_features_batch(X_train)
    
    print("Extracting features from validation images...")
    X_val_features = extract_features_batch(X_val)
    
    print(f"Feature matrix shape: {X_train_features.shape}")
    
    # Create and train the model
    print(f"Creating {args.model_type} detection model...")
    detector = ArtForgeryDetector(model_type=args.model_type)
    
    print("Training model...")
    train_accuracy = detector.train(X_train_features, y_train)
    print(f"Training accuracy: {train_accuracy:.4f}")
    
    # Evaluate model
    print("Evaluating model...")
    eval_results = detector.evaluate(X_val_features, y_val)
    
    print(f"Validation accuracy: {eval_results['accuracy']:.4f}")
    print(f"Validation precision: {eval_results['precision']:.4f}")
    print(f"Validation recall: {eval_results['recall']:.4f}")
    print(f"Validation F1 score: {eval_results['f1']:.4f}")
    
    # Save the model
    model_path = os.path.join(args.output_dir, 'models', f'detection_model_{args.model_type}.joblib')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    detector.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Plot confusion matrix
    cm = eval_results['confusion_matrix']
    fig = plot_confusion_matrix(cm, class_names=['Real', 'Fake'], title='Confusion Matrix')
    save_visualization(fig, 'detection_confusion_matrix.png', os.path.join(args.output_dir, 'figures'))
    
    # Plot feature importance (for Random Forest only)
    if args.model_type == 'rf':
        try:
            importance_dict = detector.get_feature_importance()
            feature_names = list(importance_dict.keys())
            importances = np.array(list(importance_dict.values()))
            
            fig = plot_feature_importance(feature_names, importances)
            save_visualization(fig, 'detection_feature_importance.png', os.path.join(args.output_dir, 'figures'))
        except Exception as e:
            print(f"Could not plot feature importance: {e}")


def main():
    parser = argparse.ArgumentParser(description='Train forgery detection model')
    parser.add_argument('--data_path', type=str, default='../../data/raw/AI_for_Art_Restoration_2',
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='../../outputs',
                        help='Output directory for models and visualizations')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size for feature extraction (square)')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--model_type', type=str, default='rf', choices=['rf', 'svm'],
                        help='Type of model to train (rf: Random Forest, svm: Support Vector Machine)')
    
    args = parser.parse_args()
    
    train_detection_model(args)


if __name__ == "__main__":
    main()
