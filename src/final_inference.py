import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from torchvision import transforms
from PIL import Image
import os
import sys
import matplotlib.pyplot as plt

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.realesrgan_wrapper import RealESRGANWrapper
from src.models.gfpgan_wrapper import GFPGANWrapper

# --- CONFIG ---
UNET_PATH = r"outputs/models/unet/best_unet_resnet34_perceptual.pth"
TEST_IMAGE_PATH = r"data/raw/AI_for_Art_Restoration_2/paired_dataset_art/damaged/18.png" 
REAL_IMAGE_PATH = r"data/raw/AI_for_Art_Restoration_2/paired_dataset_art/undamaged/18.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 1. TRADITIONAL CV TOOLS (Your Old Code)
# ==========================================
def detect_scratch_mask(img_bgr):
    """
    Creates a mask for BRIGHT white scratches (like on the blue dress).
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Threshold: Look for pixels that are unusually bright compared to neighbors
    # We use adaptive thresholding to find thin bright lines
    mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 15, -5 # Negative C looks for bright spots
    )
    
    # Filter noise: Only keep distinct lines
    kernel = np.ones((2,2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def pre_process_scratches(img_bgr):
    """
    Fills big white scratches with 'dummy' texture so U-Net doesn't get confused.
    """
    print("   [Traditional] detecting and filling scratches...")
    mask = detect_scratch_mask(img_bgr)
    
    # Inpaint the masked area using Telea algorithm (fast)
    # This creates a blurry fill, which U-Net will later fix/sharpen.
    inpainted = cv2.inpaint(img_bgr, mask, 3, cv2.INPAINT_TELEA)
    return inpainted

def post_process_color(img_bgr):
    """
    Removes the 'Old Varnish' Yellow Tint using LAB Auto-White Balance.
    """
    print("   [Traditional] Applying Auto-White Balance...")
    result = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    avg_a = np.mean(result[:, :, 1])
    avg_b = np.mean(result[:, :, 2])
    
    # Shift A (Green-Red) and B (Blue-Yellow) towards Neutral Gray (128)
    # The 1.1 multiplier makes it slightly aggressive against yellow
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

# ==========================================
# 2. MAIN PIPELINE
# ==========================================
def main():
    print(f"Processing: {TEST_IMAGE_PATH}")
    
    # Load Original
    original_bgr = cv2.imread(TEST_IMAGE_PATH)
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    real_bgr = cv2.imread(REAL_IMAGE_PATH)
    real_rgb = cv2.cvtColor(real_bgr, cv2.COLOR_BGR2RGB)

    # --- STEP 0: PRE-PROCESSING (The Filler) ---
    # We fix the big scratches first using CV
    pre_processed_bgr = pre_process_scratches(original_bgr)
    
    # Convert to PIL for U-Net
    pre_processed_rgb = cv2.cvtColor(pre_processed_bgr, cv2.COLOR_BGR2RGB)
    input_pil = Image.fromarray(pre_processed_rgb)

    # --- STEP 1: U-NET (The Mason) ---
    print("\n>>> Stage 1: U-Net (Refining Structure)...")
    unet = smp.Unet(encoder_name="resnet34", in_channels=3, classes=3, activation='tanh').to(DEVICE)
    unet.load_state_dict(torch.load(UNET_PATH, map_location=DEVICE))
    unet.eval()

    # Resize/Normalize
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    input_tensor = preprocess(input_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = unet(input_tensor)

    # Denormalize
    output = (output + 1) / 2.0
    output = torch.clamp(output, 0, 1)
    stage1_np = output.squeeze().cpu().numpy().transpose(1, 2, 0)
    stage1_uint8 = (stage1_np * 255).astype(np.uint8)
    stage1_bgr = cv2.cvtColor(stage1_uint8, cv2.COLOR_RGB2BGR)

    # --- STEP 2: REAL-ESRGAN (The Detailer) ---
    print(">>> Stage 2: Real-ESRGAN (Adding Texture)...")
    esrgan = RealESRGANWrapper(model_str='x4')
    stage2_bgr, _ = esrgan.restore(stage1_bgr, outscale=4)

    # --- STEP 3: GFPGAN (The Portrait Artist) ---
    print(">>> Stage 3: GFPGAN (Fixing Faces)...")
    gfpgan = GFPGANWrapper(model_version='v1.3', device=DEVICE, upscale=1)
    stage3_bgr = gfpgan.enhance(stage2_bgr)

    # --- STEP 4: POST-PROCESSING (The Colorist) ---
    # This is where we fix the yellow tint
    print("\n>>> Stage 4: Post-Processing Color Correction...")
    final_bgr = post_process_color(stage3_bgr)
    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)

    # ==========================================
    # DISPLAY
    # ==========================================
    plt.figure(figsize=(20, 10))
    
    plt.subplot(1, 6, 1)
    plt.imshow(original_rgb)
    plt.title("1. Original Damaged")
    plt.axis("off")

    plt.subplot(1, 6, 2)
    plt.imshow(cv2.cvtColor(stage1_bgr, cv2.COLOR_BGR2RGB))
    plt.title("2. U-Net only")
    plt.axis("off")
    
    plt.subplot(1, 6, 3)
    plt.imshow(cv2.cvtColor(stage2_bgr, cv2.COLOR_BGR2RGB))
    plt.title("3. Real-ESRGAN")
    plt.axis("off")

    plt.subplot(1, 6, 4)
    plt.imshow(cv2.cvtColor(stage3_bgr, cv2.COLOR_BGR2RGB))
    plt.title("4. GFPGAN")
    plt.axis("off")

    plt.subplot(1, 6, 5)
    plt.imshow(final_rgb)
    plt.title("5. Final Hybrid Result")
    plt.axis("off")
    
    plt.subplot(1, 6, 6)
    plt.imshow(real_rgb)
    plt.title("6. Ground Truth")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()