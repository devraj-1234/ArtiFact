import cv2
import math
import numpy as np
import scipy.ndimage


# ===============================
# Non-Maximal Suppression
# ===============================

def non_max_suppression(data, win):
    data_max = scipy.ndimage.maximum_filter(
        data, footprint=win, mode='constant'
    )
    data[data != data_max] = 0
    return data


def oriented_nms(mag, ang):
    ang_quant = np.round(ang / (np.pi / 4)) % 4

    winE  = np.array([[0,0,0],[1,1,1],[0,0,0]])
    winSE = np.array([[1,0,0],[0,1,0],[0,0,1]])
    winS  = np.array([[0,1,0],[0,1,0],[0,1,0]])
    winSW = np.array([[0,0,1],[0,1,0],[1,0,0]])

    magE  = non_max_suppression(mag.copy(), winE)
    magSE = non_max_suppression(mag.copy(), winSE)
    magS  = non_max_suppression(mag.copy(), winS)
    magSW = non_max_suppression(mag.copy(), winSW)

    out = np.zeros_like(mag)
    out[ang_quant == 0] = magE[ang_quant == 0]
    out[ang_quant == 1] = magSE[ang_quant == 1]
    out[ang_quant == 2] = magS[ang_quant == 2]
    out[ang_quant == 3] = magSW[ang_quant == 3]

    return out


# ===============================
# Crack Detection
# ===============================

def detect_cracks(image_bgr):
    """
    Returns a binary crack mask (uint8: 0 or 255)
    """

    # --- Use VALUE channel (best for white cracks) ---
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = hsv[:, :, 2].astype(np.float32) / 255.0

    # --- High-pass (DoG) ---
    sigma = 5
    kernel = 2 * math.ceil(2 * sigma) + 1
    blur = cv2.GaussianBlur(gray, (kernel, kernel), sigma)
    highpass = cv2.subtract(gray, blur)

    # --- Sobel gradients ---
    sobelx = cv2.Sobel(highpass, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(highpass, cv2.CV_64F, 0, 1, ksize=3)

    mag = np.hypot(sobelx, sobely)
    ang = np.arctan2(sobely, sobelx)

    # --- Adaptive threshold ---
    thresh = np.percentile(mag, 85)
    mag[mag < thresh] = 0

    # --- Oriented Non-Max Suppression ---
    mag = oriented_nms(mag, ang)

    # --- Binary mask ---
    mask = np.zeros_like(mag, dtype=np.uint8)
    mask[mag > 0] = 255

    # --- Thicken & connect cracks ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask


# ===============================
# Example Usage
# ===============================

if __name__ == "__main__":
    img = cv2.imread(r"data\raw\AI_for_Art_Restoration_2\paired_dataset_art\damaged\86.png")
    crack_mask = detect_cracks(img)

    cv2.imshow("Crack Mask", crack_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
