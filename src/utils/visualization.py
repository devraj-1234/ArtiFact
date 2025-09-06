"""
visualization.py
Visualization utilities for the ArtifactVision project.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os


def plot_image_comparison(original, processed, titles=None, figsize=(10, 5)):
    """
    Plot original and processed images side by side.
    
    Args:
        original (numpy.ndarray): Original image
        processed (numpy.ndarray): Processed image
        titles (list): List of titles for subplots
        figsize (tuple): Figure size
    """
    if titles is None:
        titles = ['Original', 'Processed']
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB) if len(original.shape) == 3 else original, cmap='gray')
    axes[0].set_title(titles[0])
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB) if len(processed.shape) == 3 else processed, cmap='gray')
    axes[1].set_title(titles[1])
    axes[1].axis('off')
    
    plt.tight_layout()
    return fig


def plot_fft_analysis(image, magnitude_spectrum, low_pass, high_pass, figsize=(15, 5)):
    """
    Plot FFT analysis of an image including original, magnitude spectrum, 
    low-pass filtered and high-pass filtered versions.
    
    Args:
        image (numpy.ndarray): Original image
        magnitude_spectrum (numpy.ndarray): FFT magnitude spectrum
        low_pass (numpy.ndarray): Low-pass filtered image
        high_pass (numpy.ndarray): High-pass filtered image
        figsize (tuple): Figure size
    """
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(magnitude_spectrum, cmap='gray')
    axes[1].set_title('FFT Magnitude Spectrum')
    axes[1].axis('off')
    
    axes[2].imshow(low_pass, cmap='gray')
    axes[2].set_title('Low-pass Filter')
    axes[2].axis('off')
    
    axes[3].imshow(high_pass, cmap='gray')
    axes[3].set_title('High-pass Filter')
    axes[3].axis('off')
    
    plt.tight_layout()
    return fig


def plot_feature_importance(feature_names, importances, title='Feature Importance', figsize=(10, 6)):
    """
    Plot feature importance from a machine learning model.
    
    Args:
        feature_names (list): Names of features
        importances (numpy.ndarray): Importance scores
        title (str): Plot title
        figsize (tuple): Figure size
    """
    indices = np.argsort(importances)[::-1]
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(importances)), importances[indices])
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=90)
    ax.set_title(title)
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    plt.tight_layout()
    return fig


def plot_restoration_results(original, damaged, restored, titles=None, figsize=(15, 5)):
    """
    Plot original, damaged, and restored images.
    
    Args:
        original (numpy.ndarray): Original undamaged image
        damaged (numpy.ndarray): Damaged image
        restored (numpy.ndarray): Restored image
        titles (list): List of titles for subplots
        figsize (tuple): Figure size
    """
    if titles is None:
        titles = ['Original', 'Damaged', 'Restored']
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Convert BGR to RGB if images are color
    if len(original.shape) == 3:
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        damaged = cv2.cvtColor(damaged, cv2.COLOR_BGR2RGB)
        restored = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)
    
    axes[0].imshow(original, cmap='gray' if len(original.shape) == 2 else None)
    axes[0].set_title(titles[0])
    axes[0].axis('off')
    
    axes[1].imshow(damaged, cmap='gray' if len(damaged.shape) == 2 else None)
    axes[1].set_title(titles[1])
    axes[1].axis('off')
    
    axes[2].imshow(restored, cmap='gray' if len(restored.shape) == 2 else None)
    axes[2].set_title(titles[2])
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', figsize=(8, 6)):
    """
    Plot confusion matrix.
    
    Args:
        cm (numpy.ndarray): Confusion matrix
        class_names (list): Names of classes
        title (str): Plot title
        figsize (tuple): Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create a custom colormap from white to deep blue
    cmap = LinearSegmentedColormap.from_list('blue_cmap', ['white', 'steelblue'])
    
    im = ax.imshow(cm, cmap=cmap)
    
    # Show all ticks and label them
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", 
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    ax.set_title(title)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    
    fig.tight_layout()
    return fig


def save_visualization(fig, filename, output_dir="outputs/figures"):
    """
    Save visualization to file.
    
    Args:
        fig (matplotlib.figure.Figure): Figure to save
        filename (str): Filename
        output_dir (str): Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)
