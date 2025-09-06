"""
fft_playground.py
Experimental FFT script for ArtifactVision project.
- Loads random damaged & undamaged images
- Applies FFT (2D)
- Visualizes & saves low-pass and high-pass filtered results
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# ---------- CONFIG ----------
DAMAGED_PATH = "../../data/raw/AI_for_Art_Restoration_2/paired_dataset_art/damaged"
UNDAMAGED_PATH = "../../data/raw/AI_for_Art_Restoration_2/paired_dataset_art/undamaged"
OUTPUT_PATH = "../../outputs/figures/"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ---------- UTILS ----------
def show_and_save(images, titles, save_name, cmap="gray"):
    """Show multiple images side by side & save them"""
    n = len(images)
    plt.figure(figsize=(15, 5))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i], cmap=cmap)
        plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_PATH, save_name)
    plt.savefig(save_path)
    plt.show()
    print(f"✅ Saved results at {save_path}")


def apply_fft(img, cutoff=30):
    """Apply FFT, return magnitude spectrum, low-pass and high-pass filtered images"""
    # 1. FFT
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

    # 2. Masks
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    # Low-pass
    mask_lp = np.zeros((rows, cols), np.uint8)
    mask_lp[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 1
    fshift_lp = fshift * mask_lp
    img_lp = np.fft.ifft2(np.fft.ifftshift(fshift_lp))
    img_lp = np.abs(img_lp)

    # High-pass
    mask_hp = np.ones((rows, cols), np.uint8)
    mask_hp[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 0
    fshift_hp = fshift * mask_hp
    img_hp = np.fft.ifft2(np.fft.ifftshift(fshift_hp))
    img_hp = np.abs(img_hp)

    return magnitude_spectrum, img_lp, img_hp


def process_random_image(folder, label, cutoff=30):
    """Pick random image from folder, apply FFT, save results"""
    files = os.listdir(folder)
    img_file = random.choice(files)
    img_path = os.path.join(folder, img_file)

    # Load grayscale
    img = cv2.imread(img_path, 0)
    if img is None:
        print(f"⚠️ Could not load {img_path}")
        return

    # Apply FFT
    mag, img_lp, img_hp = apply_fft(img, cutoff)

    # Show & Save
    save_name = f"{label}_{os.path.splitext(img_file)[0]}_fft.png"
    show_and_save(
        [img, mag, img_lp, img_hp],
        ["Original", "FFT Spectrum", "Low-pass (Smooth)", "High-pass (Edges/Cracks)"],
        save_name,
    )


# ---------- MAIN ----------
if __name__ == "__main__":
    print("🔍 Processing one damaged and one undamaged image...")

    process_random_image(DAMAGED_PATH, "damaged", cutoff=30)
    process_random_image(UNDAMAGED_PATH, "undamaged", cutoff=30)
