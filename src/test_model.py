"""
test_model.py
Loads the trained U-Net and restores a specific image.
"""
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from torchvision import transforms
from PIL import Image
import os

# --- CONFIG ---
# Path to the BEST saved model
MODEL_PATH = r"outputs/models/unet/best_unet_resnet34_perceptual.pth"
# Path to a REAL damaged image you want to fix
TEST_IMAGE_PATH = r"data/raw/AI_for_Art_Restoration_2/paired_dataset_art/damaged/download.jpg" 
# (If you don't have a real one handy, pick one from your dataset or download one)
OUTPUT_PATH = "restored_result.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def restore_image():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print(f"Loading model from {MODEL_PATH}...")
    
    # 1. Initialize Model (Must match training config exactly)
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None, # No need to download weights, we load our own
        in_channels=3,
        classes=3,
        activation='sigmoid'
    ).to(DEVICE)

    # 2. Load Weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 3. Load and Preprocess Image
    print(f"Processing {TEST_IMAGE_PATH}...")
    img = Image.open(TEST_IMAGE_PATH).convert("RGB")
    original_size = img.size
    
    # Resize to training size (256x256) for the U-Net
    # Note: In a production pipeline, we would tile the image. 
    # For now, we resize to see if the restoration logic works.
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
    # The model output is in [-1, 1] (due to normalization) or [0, 1] (sigmoid).
    # Since we used Sigmoid activation, output is [0, 1].
    output_image = output_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    
    # Denormalize if necessary? 
    # Wait, our target was normalized to [-1, 1] during training, 
    # but the Sigmoid activation forces [0, 1]. 
    # Ideally, we usually train without sigmoid for reconstruction or handle ranges carefully.
    # Let's assume standard visual output [0, 1].
    
    output_image = np.clip(output_image * 255, 0, 255).astype(np.uint8)
    
    # Resize back to original dimensions (optional, might be blurry)
    output_image = cv2.resize(output_image, original_size, interpolation=cv2.INTER_CUBIC)
    
    # Save
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR) # OpenCV uses BGR
    cv2.imwrite(OUTPUT_PATH, output_image)
    print(f"Restored image saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    restore_image()