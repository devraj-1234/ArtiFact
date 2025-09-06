"""
train_restoration.py
Train the artifact restoration model.
"""

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset import ArtDataset
from models.restoration_model import ArtifactRestorer
from utils.visualization import plot_restoration_results, save_visualization


def train_restoration_model(args):
    """
    Train the artifact restoration model.
    
    Args:
        args: Command line arguments
    """
    # Create dataset
    print(f"Loading dataset from {args.data_path}...")
    dataset = ArtDataset(args.data_path, img_size=(args.img_size, args.img_size), split_ratio=args.val_split)
    
    # Load and split data
    print("Creating train/validation split...")
    x_train, x_val, y_train, y_val = dataset.create_train_val_split()
    
    # Normalize images
    x_train = x_train.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0
    y_train = y_train.astype('float32') / 255.0
    y_val = y_val.astype('float32') / 255.0
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Validation data shape: {x_val.shape}")
    
    # Create model
    print("Creating restoration model...")
    input_shape = (args.img_size, args.img_size, 3)
    model = ArtifactRestorer(input_shape=input_shape)
    
    # Train model
    print("Training model...")
    model_path = os.path.join(args.output_dir, 'models', 'restoration_model.h5')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    history = model.train(
        x_train, y_train, 
        x_val, y_val, 
        batch_size=args.batch_size, 
        epochs=args.epochs, 
        save_path=model_path
    )
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['mae'])
    plt.plot(history['val_mae'])
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    history_plot_path = os.path.join(args.output_dir, 'figures', 'restoration_training_history.png')
    os.makedirs(os.path.dirname(history_plot_path), exist_ok=True)
    plt.savefig(history_plot_path)
    print(f"Saved training history to {history_plot_path}")
    
    # Generate example restorations
    print("Generating example restorations...")
    for i in range(min(5, len(x_val))):
        # Get original, damaged, and restored images
        original = y_val[i]
        damaged = x_val[i]
        
        # Restore damaged image
        restored = model.model.predict(np.expand_dims(damaged, axis=0))[0]
        
        # Convert back to uint8 range for visualization
        original_img = (original * 255).astype(np.uint8)
        damaged_img = (damaged * 255).astype(np.uint8)
        restored_img = (restored * 255).astype(np.uint8)
        
        # Visualize results
        fig = plot_restoration_results(
            original_img, damaged_img, restored_img,
            titles=['Original', 'Damaged', 'Restored']
        )
        
        # Save visualization
        example_path = os.path.join(args.output_dir, 'figures', f'restoration_example_{i+1}.png')
        save_visualization(fig, f'restoration_example_{i+1}.png', os.path.join(args.output_dir, 'figures'))
        print(f"Saved example restoration to {example_path}")


def main():
    parser = argparse.ArgumentParser(description='Train artifact restoration model')
    parser.add_argument('--data_path', type=str, default='../../data/raw/AI_for_Art_Restoration_2',
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='../../outputs',
                        help='Output directory for models and visualizations')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size for training (square)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs for training')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    
    args = parser.parse_args()
    
    train_restoration_model(args)


if __name__ == "__main__":
    main()
