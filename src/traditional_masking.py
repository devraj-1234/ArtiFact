import cv2
import numpy as np

def create_craquelure_mask(gray_img, kernel_size=(5,5)):
   
    # 1. Define structuring element (rectangular kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    # 2. Black Hat: Closing - Original
    # Closing fills dark holes. Subtracting Original leaves ONLY the dark holes.
    blackhat = cv2.morphologyEx(gray_img, cv2.MORPH_BLACKHAT, kernel)
    
    # 3. Threshold to binary
    # We use a low threshold because cracks are subtle
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    return mask

def create_paint_loss_mask(img_bgr):
   
    # 1. Texture Analysis (Laplacian Variance)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Calculate local texture magnitude
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_mag = np.abs(laplacian)
    
    # Blur slightly to average the texture score over a region
    texture_map = cv2.blur(texture_mag, (9,9))
    
    # Threshold: Areas with VERY low texture (flat canvas)
    # 5.0 is an empirical cutoff for "flatness"
    _, low_texture_mask = cv2.threshold(texture_map, 5.0, 255, cv2.THRESH_BINARY_INV)
    low_texture_mask = low_texture_mask.astype(np.uint8)

    # 2. Chroma Analysis (LAB Color Space)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Calculate distance from Neutral Gray (a=128, b=128)
    # distance = sqrt((a-128)^2 + (b-128)^2)
    a_dist = np.abs(a.astype(np.float32) - 128)
    b_dist = np.abs(b.astype(np.float32) - 128)
    chroma_dist = np.sqrt(a_dist**2 + b_dist**2)
    
    # Threshold: Areas with very low color saturation (Grey/White gesso)
    # 10.0 is empirical cutoff for "grey-ish"
    _, low_chroma_mask = cv2.threshold(chroma_dist, 10.0, 255, cv2.THRESH_BINARY_INV)
    low_chroma_mask = low_chroma_mask.astype(np.uint8)
    
    # 3. Combine: Damage must be BOTH Low Texture AND Low Chroma
    combined = cv2.bitwise_and(low_texture_mask, low_chroma_mask)
    
    # Clean up noise
    kernel = np.ones((3,3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=2)
    
    return combined

def create_conservative_damage_mask(img_bgr):
    """
    Orchestrator: Combines Cracks (BlackHat) and Patches (Texture/Chroma).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Get masks
    crack_mask = create_craquelure_mask(gray)
    patch_mask = create_paint_loss_mask(img_bgr)
    
    # 2. Combine (Union)
    final_mask = cv2.bitwise_or(crack_mask, patch_mask)
    
    # 3. Conservative Dilation
    # Just enough to cover the anti-aliased edges of the damage
    kernel = np.ones((3,3), np.uint8)
    final_mask = cv2.dilate(final_mask, kernel, iterations=1)
    
    return final_mask, crack_mask, patch_mask