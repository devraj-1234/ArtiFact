"""
test_model.py
Loads the trained U-Net and restores a specific image.
Updated for Tanh activation ([-1, 1] range).
"""
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt

# --- CONFIG ---
MODEL_PATH = r"outputs/models/unet/best_unet_resnet34_perceptual.pth"
# Update this path to whatever real image you want to test
TEST_IMAGE_PATH = r"data/raw/AI_for_Art_Restoration_2/paired_dataset_art/damaged/2.png" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def restore_image():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print(f"Loading model from {MODEL_PATH}...")
    
    # 1. Initialize Model
    # Architecture must match training EXACTLY
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=3,
        activation='tanh' # Correct activation for [-1, 1] range
    ).to(DEVICE)

    # 2. Load Weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 3. Load and Preprocess Image
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"Error: Image not found at {TEST_IMAGE_PATH}")
        return
        
    print(f"Processing {TEST_IMAGE_PATH}...")
    img = Image.open(TEST_IMAGE_PATH).convert("RGB")
    original_size = img.size
    
    # Preprocessing
    # We resize to 256x256 for the U-Net.
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # 4. Inference
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # 5. Post-process (Tensor -> Image)
    
    # --- TANH FIX: Denormalize [-1, 1] -> [0, 1] ---
    # Since Tanh outputs values from -1 to 1, we must shift and scale them.
    # Formula: (x + 1) / 2
    output_tensor = (output_tensor + 1) / 2.0
    
    # Clamp to ensure we stay in valid [0, 1] range (handles minor overshoot)
    output_tensor = torch.clamp(output_tensor, 0, 1)

    # Convert to Numpy (H, W, C)
    output_image = output_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    
    # Scale to [0, 255] for display/saving
    output_image = (output_image * 255).astype(np.uint8)
    
    # Resize back to original dimensions (optional, keeps aspect ratio of result)
    # Using Lanczos interpolation for better sharpness when upsizing
    output_image = cv2.resize(output_image, original_size, interpolation=cv2.INTER_LANCZOS4)
    
    # 6. Visualization
    rgb_in = np.array(img)
    # output_image is already RGB (since we didn't convert to BGR yet)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_in)
    plt.title("Input Damaged Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(output_image)
    plt.title("Restored Output (U-Net)")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

    # Optional: Save result
    # cv2.imwrite("restored_result.png", cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    restore_image()