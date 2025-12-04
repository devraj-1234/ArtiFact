import torch
import torchvision
import matplotlib.pyplot as plt
import os
from src.data.pytorch_dataset import get_dataloaders

# Config
BASE_PATH = r"data/raw/AI_for_Art_Restoration_2" # Update if needed
OUTPUT_FILE = "preview_training_batch.png"

def show_batch():
    # Load a small batch
    train_loader, _ = get_dataloaders(BASE_PATH, batch_size=4)
    damaged, clean = next(iter(train_loader))

    # Create a grid: Top row = Damaged, Bottom row = Clean
    # We stack them to visualize pairs vertically
    combined = torch.cat((damaged, clean), dim=0)
    
    # Make a grid image
    grid_img = torchvision.utils.make_grid(combined, nrow=4, padding=2)
    
    # Convert tensor to numpy for plotting
    plt.figure(figsize=(15, 8))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis('off')
    plt.title("Top: Synthetically Damaged Input | Bottom: Clean Ground Truth")
    
    # Save it
    plt.savefig(OUTPUT_FILE)
    print(f"Saved preview to {os.path.abspath(OUTPUT_FILE)}")
    print("Open this image to verify the damage looks realistic (scratches, blur, noise).")

if __name__ == "__main__":
    show_batch()