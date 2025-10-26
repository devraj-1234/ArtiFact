"""
train_unet.py

This script trains the PyTorch U-Net model for color and light restoration.
It uses the PairedArtDataset with data augmentation and saves the best
model based on validation loss.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import logging

# Make sure the project root is in the Python path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.dl.pytorch_unet import UNet
from src.data.pytorch_dataset import get_dataloaders

# --- Configuration ---
# Construct paths relative to the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_BASE_PATH = os.path.join(PROJECT_ROOT, "data/raw/AI_for_Art_Restoration_2")
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "outputs/models/unet")
LOGS_DIR = os.path.join(PROJECT_ROOT, "outputs/logs/unet")
BATCH_SIZE = 8 # Lowered for potentially high memory usage of U-Net
EPOCHS = 100
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.2
IMG_SIZE = (256, 256)

# --- Setup Logging ---
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "training.log")),
        logging.StreamHandler()
    ]
)

def train_model():
    """Main function to train the U-Net model."""
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- DataLoaders ---
    logging.info("Loading datasets...")
    train_loader, val_loader = get_dataloaders(
        base_path=DATA_BASE_PATH,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        split_ratio=VAL_SPLIT,
        num_workers=2 # Use multiple workers if your OS and setup support it
    )
    logging.info(f"Training samples: {len(train_loader.dataset)}, Validation samples: {len(val_loader.dataset)}")

    # --- Model, Optimizer, Loss ---
    logging.info("Initializing model...")
    model = UNet(n_channels=3, n_classes=3).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.L1Loss() # L1 Loss (Mean Absolute Error) is often good for image restoration
    
    best_val_loss = float('inf')

    # --- Training Loop ---
    logging.info("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [T]", unit="batch")
        for damaged_imgs, undamaged_imgs in progress_bar:
            damaged_imgs = damaged_imgs.to(device)
            undamaged_imgs = undamaged_imgs.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(damaged_imgs)
            loss = criterion(outputs, undamaged_imgs)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1} - Average Training Loss: {avg_train_loss:.6f}")

        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [V]", unit="batch")
            for damaged_imgs, undamaged_imgs in progress_bar_val:
                damaged_imgs = damaged_imgs.to(device)
                undamaged_imgs = undamaged_imgs.to(device)
                
                outputs = model(damaged_imgs)
                loss = criterion(outputs, undamaged_imgs)
                val_loss += loss.item()
                progress_bar_val.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / len(val_loader)
        logging.info(f"Epoch {epoch+1} - Average Validation Loss: {avg_val_loss:.6f}")

        # --- Save Best Model ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(MODEL_SAVE_DIR, "best_unet_model.pth")
            torch.save(model.state_dict(), model_path)
            logging.info(f"New best model saved to {model_path} with validation loss: {avg_val_loss:.6f}")

    logging.info("Finished Training.")

if __name__ == "__main__":
    train_model()
