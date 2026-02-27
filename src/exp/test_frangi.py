import cv2
import numpy as np
import matplotlib.pyplot as plt
from simple_lama_inpainting import SimpleLama

# ---------------------------------------------------------
# Initialize LaMa ONCE (important for performance)
# ---------------------------------------------------------
lama = SimpleLama(device="cuda")  # change to "cpu" if needed


# ---------------------------------------------------------
# Gabor Filter Bank
# ---------------------------------------------------------
def build_gabor_bank():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 4):
        kern = cv2.getGaborKernel(
            (ksize, ksize),
            sigma=4.0,
            theta=theta,
            lambd=10.0,
            gamma=0.5,
            psi=0,
            ktype=cv2.CV_32F
        )
        kern /= (kern.sum() + 1e-6)
        filters.append(kern)
    return filters


def apply_gabor_bank(img_gray, filters):
    accum = np.zeros_like(img_gray, dtype=np.float32)
    for kern in filters:
        resp = cv2.filter2D(img_gray, cv2.CV_32F, kern)
        accum = np.maximum(accum, resp)
    return cv2.normalize(accum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# ---------------------------------------------------------
# Crack Mask Generation
# ---------------------------------------------------------
def generate_crack_mask(image_path):
    img = cv2.imread(image_path)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, _, _ = cv2.split(lab)

    # --- BlackHat (dark cracks) ---
    bh_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, bh_kernel)
    _, bh_mask = cv2.threshold(blackhat, 15, 255, cv2.THRESH_BINARY)

    # --- Gabor (directional cracks) ---
    gabor_filters = build_gabor_bank()
    gabor_resp = apply_gabor_bank(L, gabor_filters)
    _, gabor_mask = cv2.threshold(gabor_resp, 30, 255, cv2.THRESH_BINARY)

    # --- Canny edges ---
    edges = cv2.Canny(L, 50, 150)

    # --- Smart intersection ---
    combined = cv2.bitwise_and(bh_mask, cv2.bitwise_or(gabor_mask, edges))

    # --- Cleanup ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    combined = cv2.dilate(combined, kernel, iterations=1)

    return img, combined


# ---------------------------------------------------------
# Split Mask: Craquelure vs Deep Cracks
# ---------------------------------------------------------
def split_mask_by_size(mask, threshold_area=200):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    small_mask = np.zeros_like(mask)
    large_mask = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < threshold_area:
            small_mask[labels == i] = 255
        else:
            large_mask[labels == i] = 255

    return small_mask, large_mask


# ---------------------------------------------------------
# Full Restoration Pipeline
# ---------------------------------------------------------
def inpainting_pipeline(img_path, output_prefix="restored"):
    original, full_mask = generate_crack_mask(img_path)

    small_mask, large_mask = split_mask_by_size(full_mask, threshold_area=200)

    # ---------- Stage 1: OpenCV Telea (Fine Craquelure) ----------
    small_mask = cv2.dilate(small_mask, np.ones((3, 3), np.uint8))
    stage1 = cv2.inpaint(original, small_mask, 3, cv2.INPAINT_TELEA)

    # ---------- Stage 2: LaMa (Deep Cracks Only) ----------
    stage1_rgb = cv2.cvtColor(stage1, cv2.COLOR_BGR2RGB)

    lama_mask = cv2.erode(large_mask, np.ones((3, 3), np.uint8))
    lama_result_rgb = lama(stage1_rgb, lama_mask)
    lama_result_rgb = np.array(lama_result_rgb)
    lama_result = cv2.cvtColor(lama_result_rgb, cv2.COLOR_RGB2BGR)

    # ---------- Texture Reinjection ----------
    detail = cv2.subtract(
        stage1,
        cv2.GaussianBlur(stage1, (0, 0), 2.5)
    )

    # 2. CRITICAL FIX: Resize LaMa output to match the original image exactly.
    # LaMa often adds padding (e.g., 512x512 output for a 503x503 input).
    h, w = stage1.shape[:2]  # Get original height and width
    if lama_result.shape[:2] != (h, w):
        print(f"Resizing LaMa output from {lama_result.shape[:2]} to {(h, w)}...")
        lama_result = cv2.resize(lama_result, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # 3. Now it is safe to blend
    final = cv2.addWeighted(lama_result, 1.0, detail, 0.35, 0)

    # ---------- Visualization ----------
    plt.figure(figsize=(14, 9))

    plt.subplot(2, 2, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.title("Detected Crack Mask")
    plt.imshow(full_mask, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.title("After OpenCV (Craquelure)")
    plt.imshow(cv2.cvtColor(stage1, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.title("Final Restoration (OpenCV + LaMa)")
    plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# Run
# ---------------------------------------------------------
inpainting_pipeline("D:\\R&D Project\\image_processing\\data\\raw\\AI_for_Art_Restoration_2\\paired_dataset_art\\damaged\\34.png", output_prefix="restored_34")
