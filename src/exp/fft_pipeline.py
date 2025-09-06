"""
fft_pipeline.py
Full experimental FFT pipeline for ArtifactVision project.
- Processes all images in a folder (damaged/undamaged)
- Computes FFT, magnitude spectrum
- Applies low-pass and high-pass filtering
- Saves processed images in outputs/figures/
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------- CONFIG ----------
INPUT_FOLDERS = {
    "damaged": "../../data/raw/AI_for_Art_Restoration_2/paired_dataset_art/damaged",
    "undamaged": "../../data/raw/AI_for_Art_Restoration_2/paired_dataset_art/undamaged"
}

OUTPUT_PATH = "outputs/figures/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

CUTOFF = 30  # frequency cutoff for filters

# ---------- FUNCTIONS ----------
def apply_fft(img, cutoff=CUTOFF):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    # Low-pass filter
    mask_lp = np.zeros((rows, cols), np.uint8)
    mask_lp[crow-cutoff:crow+cutoff, ccol-cutoff:ccol+cutoff] = 1
    img_lp = np.fft.ifft2(np.fft.ifftshift(fshift * mask_lp))
    img_lp = np.abs(img_lp)

    # High-pass filter
    mask_hp = np.ones((rows, cols), np.uint8)
    mask_hp[crow-cutoff:crow+cutoff, ccol-cutoff:ccol+cutoff] = 0
    img_hp = np.fft.ifft2(np.fft.ifftshift(fshift * mask_hp))
    img_hp = np.abs(img_hp)

    return magnitude, img_lp, img_hp

def save_images(img, magnitude, img_lp, img_hp, label, fname):
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    axes[0].imshow(img, cmap='gray'); axes[0].set_title("Original"); axes[0].axis('off')
    axes[1].imshow(magnitude, cmap='gray'); axes[1].set_title("FFT Spectrum"); axes[1].axis('off')
    axes[2].imshow(img_lp, cmap='gray'); axes[2].set_title("Low-pass"); axes[2].axis('off')
    axes[3].imshow(img_hp, cmap='gray'); axes[3].set_title("High-pass"); axes[3].axis('off')
    plt.tight_layout()
    save_file = os.path.join(OUTPUT_PATH, f"{label}_{fname}_fft.png")
    plt.savefig(save_file)
    plt.close()
    print(f"✅ Saved {save_file}")

# ---------- MAIN ----------
if __name__ == "__main__":
    for label, folder in INPUT_FOLDERS.items():
        files = os.listdir(folder)
        for f in files:
            img_path = os.path.join(folder, f)
            img = cv2.imread(img_path, 0)
            if img is None:
                print(f"⚠️ Skipping {img_path}")
                continue

            mag, img_lp, img_hp = apply_fft(img, CUTOFF)
            save_images(img, mag, img_lp, img_hp, label, os.path.splitext(f)[0])
