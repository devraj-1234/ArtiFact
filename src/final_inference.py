import os
# Optimize memory allocation to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from torchvision import transforms
from PIL import Image
import sys
import matplotlib.pyplot as plt
import gc

# NEW: State-of-the-Art Inpainting
from simple_lama_inpainting import SimpleLama

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.realesrgan_wrapper import RealESRGANWrapper
from src.models.gfpgan_wrapper import GFPGANWrapper

# --- CONFIG ---
UNET_PATH = r"outputs/models/unet/best_unet_resnet34_perceptual.pth"
TEST_IMAGE_PATH = r""
REAL_IMAGE_PATH = r""
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def clean_memory():
    """Forcibly frees GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()

def create_white_crack_mask(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([180, 60, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask

def auto_white_balance(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    avg_a = np.mean(lab[:, :, 1])
    avg_b = np.mean(lab[:, :, 2])
    lab[:, :, 1] = lab[:, :, 1] - ((avg_a - 128) * (lab[:, :, 0] / 255.0) * 1.1)
    lab[:, :, 2] = lab[:, :, 2] - ((avg_b - 128) * (lab[:, :, 0] / 255.0) * 1.1)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def run_unet(model, img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    input_tensor = preprocess(img_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)

    output = (output + 1) / 2.0
    output = torch.clamp(output, 0, 1)

    np_out = output.squeeze().cpu().numpy().transpose(1, 2, 0)
    return cv2.cvtColor((np_out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

def lama_tiled_inpaint(lama, image_rgb: np.ndarray, mask_gray: np.ndarray, tile_size=512, overlap=32):
    """
    Splits image into tiles, inpaints each, and stitches them back.
    Robust to edge cases where dimensions are not multiples of 8.
    """
    h, w, _ = image_rgb.shape
    result = image_rgb.copy()
    stride = tile_size - overlap

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y2 = min(y + tile_size, h)
            x2 = min(x + tile_size, w)
            
            tile_img = image_rgb[y:y2, x:x2]
            tile_mask = mask_gray[y:y2, x:x2]

            if np.count_nonzero(tile_mask) == 0:
                continue
            
            # CRITICAL FIX FOR BROADCAST ERROR: Remember expected size
            h_exp = y2 - y
            w_exp = x2 - x

            tile_img_pil = Image.fromarray(tile_img)
            tile_mask_pil = Image.fromarray(tile_mask)

            with torch.no_grad():
                tile_out = lama(tile_img_pil, tile_mask_pil)
            
            tile_out_np = np.array(tile_out)
            
            # CROP back to expected size if LaMa added padding
            tile_out_np = tile_out_np[:h_exp, :w_exp]

            result[y:y2, x:x2] = tile_out_np

    return result

# ==========================================
# 2. MAIN PIPELINE
# ==========================================
def main():
    print(f"Processing: {TEST_IMAGE_PATH}")

    original_bgr = cv2.imread(TEST_IMAGE_PATH)
    if original_bgr is None:
        print("Error: Could not load damaged image.")
        return
    orig_h, orig_w = original_bgr.shape[:2]

    real_bgr = cv2.imread(REAL_IMAGE_PATH)
    real_rgb = cv2.cvtColor(real_bgr, cv2.COLOR_BGR2RGB) if real_bgr is not None else np.zeros_like(original_bgr)
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    # --- STAGE 0: LAMA (Load -> Run -> Delete) ---
    print("\n>>> Stage 0: LaMa Inpainting (TILED)")
    clean_memory()
    
    lama = SimpleLama(device=DEVICE)
    lama.model.eval()
    
    crack_mask = create_white_crack_mask(original_bgr)

    inpainted_rgb = lama_tiled_inpaint(
        lama=lama,
        image_rgb=original_rgb,
        mask_gray=crack_mask,
        tile_size=512, 
        overlap=32
    )
    inpainted_bgr = cv2.cvtColor(inpainted_rgb, cv2.COLOR_RGB2BGR)
    
    del lama
    clean_memory()

    # --- STAGE 1: U-NET ---
    print(">>> Stage 1: U-Net")
    unet = smp.Unet("resnet34", in_channels=3, classes=3, activation="tanh").to(DEVICE)
    unet.load_state_dict(torch.load(UNET_PATH, map_location=DEVICE))
    unet.eval()

    unet_bgr = run_unet(unet, inpainted_bgr)
    
    del unet
    clean_memory()

    # --- STAGE 2: REAL-ESRGAN ---
    print(">>> Stage 2: Real-ESRGAN")
    esrgan = RealESRGANWrapper(model_str="x4")
    esrgan_bgr, _ = esrgan.restore(unet_bgr, outscale=4)
    
    del esrgan
    clean_memory()

    # --- STAGE 3: GFPGAN ---
    print(">>> Stage 3: GFPGAN")
    gfpgan = GFPGANWrapper(model_version="v1.3", device=DEVICE, upscale=1)
    gfpgan_bgr = gfpgan.enhance(esrgan_bgr)
    
    del gfpgan
    clean_memory()

    # --- STAGE 4: POST ---
    print(">>> Stage 4: Auto White Balance")
    final_bgr_high_res = auto_white_balance(gfpgan_bgr)
    # final_rgb is generated in the display section below
    
    print(f">>> Final Step: Resizing back to {orig_w}x{orig_h}...")
    # Use LANCZOS4 for high-quality resampling
    final_bgr = cv2.resize(final_bgr_high_res, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
    # -----------------------------------------

    # ==========================================
    # DISPLAY ALL STAGES
    # ==========================================
    print("\nPreparing Visualization...")
    
    # Convert intermediate BGR outputs to RGB for Matplotlib
    unet_rgb = cv2.cvtColor(unet_bgr, cv2.COLOR_BGR2RGB)
    esrgan_rgb = cv2.cvtColor(esrgan_bgr, cv2.COLOR_BGR2RGB)
    gfpgan_rgb = cv2.cvtColor(gfpgan_bgr, cv2.COLOR_BGR2RGB)
    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(28, 6)) # Wide figure for 7 panels

    # 1. Original
    plt.subplot(1, 7, 1)
    plt.imshow(original_rgb)
    plt.title("1. Input Damaged", fontweight='bold')
    plt.axis("off")

    # 2. LaMa
    plt.subplot(1, 7, 2)
    plt.imshow(inpainted_rgb)
    plt.title("2. Stage 0: LaMa\n(Structural Inpainting)", fontweight='bold')
    plt.axis("off")

    # 3. U-Net
    plt.subplot(1, 7, 3)
    plt.imshow(unet_rgb)
    plt.title("3. Stage 1: U-Net\n(Color Correction)", fontweight='bold')
    plt.axis("off")

    # 4. Real-ESRGAN
    plt.subplot(1, 7, 4)
    plt.imshow(esrgan_rgb)
    plt.title("4. Stage 2: Real-ESRGAN\n(Texture x4)", fontweight='bold')
    plt.axis("off")

    # 5. GFPGAN
    plt.subplot(1, 7, 5)
    plt.imshow(gfpgan_rgb)
    plt.title("5. Stage 3: GFPGAN\n(Face Restoration)", fontweight='bold')
    plt.axis("off")

    # 6. Final AWB
    plt.subplot(1, 7, 6)
    plt.imshow(final_rgb)
    plt.title("6. Stage 4: Final AWB\n(Remove Yellow Tint)", fontweight='bold')
    plt.axis("off")
    
    # 7. Ground Truth
    plt.subplot(1, 7, 7)
    plt.imshow(real_rgb)
    plt.title("7. Ground Truth", fontweight='bold')
    plt.axis("off")
    
    plt.tight_layout()
    print("Displaying plot...")
    plt.show()

if __name__ == "__main__":
    main()