import os
from pathlib import Path
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR)) 

import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import gc

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from simple_lama_inpainting import SimpleLama
from src.models.realesrgan_wrapper import RealESRGANWrapper
from src.models.gfpgan_wrapper import GFPGANWrapper

UNET_PATH = str(ROOT_DIR / "checkpoints" / "best_unet_resnet34_perceptual.pth")     # Update this path to the downloaded U-Net weights

TEST_IMAGE_PATH = str(ROOT_DIR / "assets" / "sample_damaged.png")   # Update this path to your test image

REAL_IMAGE_PATH = str(ROOT_DIR / "assets" / "")                     # If you have a ground truth image for evaluation, place it in the assets folder and update this path.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"             # Use GPU if available, otherwise fallback to CPU

def clean_memory():
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
            h_exp = y2 - y
            w_exp = x2 - x
            tile_img_pil = Image.fromarray(tile_img)
            tile_mask_pil = Image.fromarray(tile_mask)
            with torch.no_grad():
                tile_out = lama(tile_img_pil, tile_mask_pil)
            tile_out_np = np.array(tile_out)
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
    has_ground_truth = real_bgr is not None
    real_rgb = cv2.cvtColor(real_bgr, cv2.COLOR_BGR2RGB) if has_ground_truth else np.zeros_like(original_bgr)
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    # --- STAGE 0: LAMA ---
    print("\n>>> Stage 0: LaMa Inpainting (TILED)")
    clean_memory()
    lama = SimpleLama(device=DEVICE)
    lama.model.eval()
    crack_mask = create_white_crack_mask(original_bgr)
    inpainted_rgb = lama_tiled_inpaint(lama, original_rgb, crack_mask, tile_size=512, overlap=32)
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
    
    if has_ground_truth:
        target_h, target_w = real_bgr.shape[:2]
    else:
        target_h, target_w = orig_h, orig_w
    
    print(f">>> Final Step: Resizing back to {target_w}x{target_h}...")
    final_bgr = cv2.resize(final_bgr_high_res, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    # ==========================================
    # NEW: CALCULATE METRICS
    # ==========================================
    print("\n==========================================")
    print("EVALUATION METRICS")
    print("==========================================")
    
    psnr_value = 0.0
    ssim_value = 0.0
    
    if has_ground_truth:
        # Check to ensure shapes are identical to prevent ValueError
        if real_bgr.shape != final_bgr.shape:
            print(f"Warning: Forcing shape match. Real: {real_bgr.shape} | Final: {final_bgr.shape}")
            final_bgr = cv2.resize(final_bgr, (real_bgr.shape[1], real_bgr.shape[0]))

        # Calculate PSNR
        psnr_value = psnr(real_bgr, final_bgr, data_range=255)
        
        # Calculate SSIM
        ssim_value = ssim(real_bgr, final_bgr, data_range=255, channel_axis=-1)
        
        # print(f"PSNR : {psnr_value:.2f} dB (Higher is better, > 30 is excellent)")
        # print(f"SSIM : {ssim_value:.4f} (Closer to 1.0 is better)")
    else:
        print("Ground truth image not found. Metrics skipped.")
        
    print("==========================================\n")

    # ==========================================
    # DISPLAY ALL STAGES
    # ==========================================
    print("Preparing Visualization...")
    unet_rgb = cv2.cvtColor(unet_bgr, cv2.COLOR_BGR2RGB)
    esrgan_rgb = cv2.cvtColor(esrgan_bgr, cv2.COLOR_BGR2RGB)
    gfpgan_rgb = cv2.cvtColor(gfpgan_bgr, cv2.COLOR_BGR2RGB)
    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(28, 6)) 

    # Dynamic title for the plot based on metrics
    if has_ground_truth:
        plt.suptitle(f"Pipeline Evaluation | PSNR: {psnr_value:.2f} dB | SSIM: {ssim_value:.4f}", fontsize=16, fontweight='bold', y=1.05)

    # plt.subplot(1, 6, 1)
    # plt.imshow(original_rgb)
    # plt.title("1. Input Damaged", fontweight='bold')
    # plt.axis("off")

    # plt.subplot(1, 6, 2)
    # plt.imshow(inpainted_rgb)
    # plt.title("2. Stage 0: LaMa\n(Structural Inpainting)", fontweight='bold')
    # plt.axis("off")

    # plt.subplot(1, 6, 3)
    # plt.imshow(unet_rgb)
    # plt.title("3. Stage 1: U-Net\n(Color Correction)", fontweight='bold')
    # plt.axis("off")

    # plt.subplot(1, 6, 4)
    # plt.imshow(esrgan_rgb)
    # plt.title("4. Stage 2: Real-ESRGAN\n(Texture x4)", fontweight='bold')
    # plt.axis("off")

    # plt.subplot(1, 6, 5)
    # plt.imshow(gfpgan_rgb)
    # plt.title("5. Stage 3: GFPGAN\n(Face Restoration)", fontweight='bold')
    # plt.axis("off")

    # plt.subplot(1, 6, 6)
    # plt.imshow(final_rgb)
    # plt.title("6. Stage 4: Final AWB\n(Remove Yellow Tint)", fontweight='bold')
    # plt.axis("off")
    
    # plt.subplot(1, 7, 7)
    # plt.imshow(real_rgb)
    # plt.title("7. Ground Truth", fontweight='bold')
    # plt.axis("off")
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_rgb)
    plt.title("Input Damaged", fontweight='bold')
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(final_rgb)
    plt.title("Restored Image", fontweight='bold')
    plt.axis("off")
    
    plt.tight_layout()
    print("Displaying plot...")
    plt.show()

if __name__ == "__main__":
    main()