import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from torchvision import transforms
from PIL import Image
import os
import sys
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.realesrgan_wrapper import RealESRGANWrapper
from src.models.gfpgan_wrapper import GFPGANWrapper

# --- CONFIGURATION ---
# Path to your best trained U-Net
UNET_PATH = r"outputs/models/unet/best_unet_resnet34_perceptual.pth"
# Path to the damaged image you want to test
TEST_IMAGE_PATH = r"data/raw/AI_for_Art_Restoration_2/paired_dataset_art/damaged/4.jpg"
REAL_IMAGE_PATH = r"data/raw/AI_for_Art_Restoration_2/paired_dataset_art/undamaged/"
temp = ""
i = len(TEST_IMAGE_PATH)-1
while TEST_IMAGE_PATH[i] != '/':
    temp = TEST_IMAGE_PATH[i] + temp
    i -= 1
REAL_IMAGE_PATH += temp
 
# Where to save the results
OUTPUT_DIR = "outputs/restored_results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Processing: {TEST_IMAGE_PATH}")

    # ==========================================
    # 0. PREPARE INPUT
    # ==========================================
    original_pil = Image.open(TEST_IMAGE_PATH).convert("RGB")
    original_np = np.array(original_pil) # Keep for display (RGB)
    actual_pil = Image.open(REAL_IMAGE_PATH).convert("RGB")
    actual_np = np.array(actual_pil) # Keep for display (RGB)

    # ==========================================
    # STAGE 1: U-Net (Structural Repair)
    # ==========================================
    print("\n>>> Stage 1: Running U-Net (Structure & Color)...")
    unet = smp.Unet(
        encoder_name="resnet34", 
        encoder_weights=None, 
        in_channels=3, 
        classes=3, 
        activation='tanh' # Matches your training
    ).to(DEVICE)
    
    if not os.path.exists(UNET_PATH):
        print("Error: U-Net model not found!")
        return

    unet.load_state_dict(torch.load(UNET_PATH, map_location=DEVICE))
    unet.eval()

    # Preprocess for U-Net (Resize to 256x256)
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    input_tensor = preprocess(original_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        unet_output = unet(input_tensor)

    # Denormalize Tanh [-1, 1] -> [0, 1]
    unet_output = (unet_output + 1) / 2.0
    unet_output = torch.clamp(unet_output, 0, 1)
    
    # Convert to Numpy (RGB) for Display/Next Stage
    unet_np = unet_output.squeeze().cpu().numpy().transpose(1, 2, 0)
    unet_uint8 = (unet_np * 255).astype(np.uint8)
    
    # Convert RGB -> BGR for Stage 2 (OpenCV format)
    stage1_bgr = cv2.cvtColor(unet_uint8, cv2.COLOR_RGB2BGR)
    
    # Save Stage 1
    cv2.imwrite(f"{OUTPUT_DIR}/1_stage1_unet.png", stage1_bgr)

    # ==========================================
    # STAGE 2: Real-ESRGAN (Texture Synthesis)
    # ==========================================
    print(">>> Stage 2: Running Real-ESRGAN (Texture x4)...")
    esrgan = RealESRGANWrapper(model_str='x4') 
    
    # Upscale 4x
    stage2_bgr, _ = esrgan.restore(stage1_bgr, outscale=4)
    
    # Convert BGR -> RGB for Display
    stage2_rgb = cv2.cvtColor(stage2_bgr, cv2.COLOR_BGR2RGB)
    
    # Save Stage 2
    cv2.imwrite(f"{OUTPUT_DIR}/2_stage2_realesrgan.png", stage2_bgr)

    # ==========================================
    # STAGE 3: GFPGAN (Face Restoration)
    # ==========================================
    print(">>> Stage 3: Running GFPGAN (Face Polish)...")
    # Initialize with upscale=1 (Image is already big from Stage 2)
    gfpgan = GFPGANWrapper(model_version='v1.3', device=DEVICE, upscale=1)
    
    final_bgr = gfpgan.enhance(stage2_bgr)
    
    # Convert BGR -> RGB for Display
    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
    
    # Save Final
    # cv2.imwrite(f"{OUTPUT_DIR}/3_final_masterpiece.png", final_bgr)

    # ==========================================
    # FINAL DISPLAY (The 4-Way Plot)
    # ==========================================
    print("\n>>> Displaying Results...")
    plt.figure(figsize=(20, 10))

    # 1. Damaged Input
    plt.subplot(1, 5, 1)
    plt.imshow(original_np)
    plt.title("1. Original Damaged", fontsize=14)
    plt.axis("off")

    # 2. U-Net Only
    plt.subplot(1, 5, 2)
    plt.imshow(unet_uint8)
    plt.title("2. U-Net (Color/Structure)", fontsize=14)
    plt.axis("off")

    # 3. U-Net + Real-ESRGAN
    plt.subplot(1, 5, 3)
    plt.imshow(stage2_rgb)
    plt.title("3. + Real-ESRGAN (Texture)", fontsize=14)
    plt.axis("off")

    # 4. U-Net + Real-ESRGAN + GFPGAN
    plt.subplot(1, 5, 4)
    plt.imshow(final_rgb)
    plt.title("4. + GFPGAN (Face Polish)", fontsize=14)
    plt.axis("off")

    # 5. Actual Undamaged
    plt.subplot(1, 5, 5)
    plt.imshow(actual_np)
    plt.title("5. Actual Undamaged", fontsize=14)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print("\nDone! Images saved to 'outputs/restored_results/'")

if __name__ == "__main__":
    main()