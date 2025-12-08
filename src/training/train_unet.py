"""
train_unet.py

This script trains the PyTorch U-Net model for color and light restoration.
Optimized for 6GB VRAM GPUs to prevent CUDA OOM errors.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import logging
import segmentation_models_pytorch as smp
import gc
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.dl.perceptual_loss import PerceptualLoss
from src.data.pytorch_dataset import get_dataloaders

# --- Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_BASE_PATH = os.path.join(PROJECT_ROOT, "data/raw/AI_for_Art_Restoration_2")
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "outputs/models/unet")
LOGS_DIR = os.path.join(PROJECT_ROOT, "outputs/logs/unet")

BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.2
IMG_SIZE = (256, 256)
L1_WEIGHT = 5
PERCEPTUAL_WEIGHT = 0.05

# --- Setup Logging ---
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "training_perceptual.log")),
        logging.StreamHandler()
    ]
)

def train_model():
    # Memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --------------------- Data ---------------------
    logging.info("Loading datasets...")
    train_loader, val_loader = get_dataloaders(
        base_path=DATA_BASE_PATH,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        split_ratio=VAL_SPLIT,
        num_workers=0
    )
    logging.info(f"Training samples: {len(train_loader.dataset)}")

    # --------------------- Model ---------------------
    logging.info("Initializing U-Net...")

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=3,
        activation='tanh'
    ).to(device)

    # --------------------- Optimizer ---------------------
    scaler = torch.cuda.amp.GradScaler()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --------------------- Loss Functions ---------------------
    l1_criterion = nn.L1Loss().to(device)

    logging.info("Setting up Perceptual Loss...")
    perceptual_criterion = PerceptualLoss(device=device).to(device)

    best_val_loss = float('inf')

    logging.info("Starting training...")

    # ===========================================================
    #                        TRAINING LOOP
    # ===========================================================
    for epoch in range(EPOCHS):

        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [T]", unit="batch")

        for damaged_imgs, undamaged_imgs in progress_bar:

            damaged_imgs = damaged_imgs.to(device, non_blocking=True).float()
            undamaged_imgs = undamaged_imgs.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)

            # Mixed precision
            with torch.cuda.amp.autocast():

                outputs = model(damaged_imgs)

                # ORDER MATTERS (better memory reuse)
                perceptual_loss = perceptual_criterion(outputs, undamaged_imgs)
                l1_loss = l1_criterion(outputs, undamaged_imgs)

                total_loss = (L1_WEIGHT * l1_loss) + (PERCEPTUAL_WEIGHT * perceptual_loss)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_loss = total_loss.item()
            running_loss += batch_loss

            # Cleanup
            del outputs, l1_loss, perceptual_loss, total_loss
            torch.cuda.empty_cache()

            progress_bar.set_postfix(
                loss=batch_loss,
                vram=f"{torch.cuda.memory_allocated()/1024**3:.2f}GB"
            )

        avg_train_loss = running_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1} - Avg Train Loss: {avg_train_loss:.4f}")

        # ===========================================================
        #                        VALIDATION LOOP
        # ===========================================================
        model.eval()
        val_loss = 0.0

        torch.cuda.empty_cache()

        with torch.no_grad():
            progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [V]", unit="batch")

            for damaged_imgs, undamaged_imgs in progress_bar_val:

                damaged_imgs = damaged_imgs.to(device, non_blocking=True).float()
                undamaged_imgs = undamaged_imgs.to(device, non_blocking=True).float()

                with torch.cuda.amp.autocast():

                    outputs = model(damaged_imgs)

                    l1 = l1_criterion(outputs, undamaged_imgs)
                    percep = perceptual_criterion(outputs, undamaged_imgs)

                    total = (L1_WEIGHT * l1) + (PERCEPTUAL_WEIGHT * percep)

                val_loss += total.item()

                del outputs, l1, percep, total
                torch.cuda.empty_cache()

        avg_val_loss = val_loss / len(val_loader)
        logging.info(f"Epoch {epoch+1} - Avg Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(MODEL_SAVE_DIR, "best_unet_resnet34_perceptual.pth")
            torch.save(model.state_dict(), model_path)
            logging.info(f"New best model saved! Loss: {avg_val_loss:.4f}")

    logging.info("Training Complete.")


if __name__ == "__main__":
    train_model()
