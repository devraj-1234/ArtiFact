import sys
import os
import torch
from torchvision import transforms

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.pytorch_dataset import SyntheticArtDataset, get_dataloaders

def test_dataset():
    print("Testing SyntheticArtDataset...")
    
    # Mock path - assuming the user has this path
    base_path = r"d:\R&D Project\image_processing\data\raw\AI_for_Art_Restoration_2"
    
    # Check if path exists
    if not os.path.exists(base_path):
        print(f"Path {base_path} does not exist. Skipping test.")
        return

    try:
        train_loader, val_loader = get_dataloaders(base_path, batch_size=4)
        print("DataLoaders created successfully.")
        
        # Get a batch
        damaged, clean = next(iter(train_loader))
        
        print(f"Damaged batch shape: {damaged.shape}")
        print(f"Clean batch shape: {clean.shape}")
        
        if damaged.shape == clean.shape == (4, 3, 256, 256):
            print("Shapes are correct.")
        else:
            print("Shapes are INCORRECT.")
            
        # Check value range
        print(f"Damaged range: [{damaged.min():.3f}, {damaged.max():.3f}]")
        print(f"Clean range: [{clean.min():.3f}, {clean.max():.3f}]")
        
        if damaged.max() <= 1.0 and damaged.min() >= 0.0:
            print("Values are in [0, 1] range.")
        else:
            print("Values are OUT of [0, 1] range.")

    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset()
