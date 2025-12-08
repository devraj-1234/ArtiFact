"""
hybrid_pipeline.py

This script defines the 2-step hybrid restoration pipeline. It combines the
PyTorch U-Net for color/light correction and the Real-ESRGAN model for
detail and texture enhancement.
"""
import torch
import numpy as np
from PIL import Image
import os
import cv2
import segmentation_models_pytorch as smp
from gfpgan import GFPGANer

# Make sure the project root is in the Python path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.realesrgan_wrapper import RealESRGANRestorer
from torchvision import transforms

class HybridPipeline:
    def __init__(self, unet_model_path, realesrgan_model_str, device=None):
        """
        Initializes the hybrid restoration pipeline.

        Args:
            unet_model_path (str): Path to the trained U-Net model state_dict (.pth file).
            realesrgan_model_str (str): Model string for RealESRGAN (e.g., 'x4').
            device (torch.device, optional): The device to run the models on. 
                                             Autodetects if None.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        print(f"HybridPipeline using device: {self.device}")

        # --- Load U-Net Model ---
        print(f"Loading U-Net model from {unet_model_path}...")
        self.unet = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None, # No need to load weights here, will be loaded from state_dict
            in_channels=3,
            classes=3,
            activation='sigmoid'
        )
        self.unet.load_state_dict(torch.load(unet_model_path, map_location=self.device))
        self.unet.to(self.device)
        self.unet.eval()
        print("U-Net model loaded successfully.")

        # --- Load Real-ESRGAN Model ---
        print(f"Loading Real-ESRGAN model '{realesrgan_model_str}'...")
        model_name_map = {
            'x4': 'RealESRGAN_x4plus',
            'x2': 'RealESRGAN_x2plus',
            'anime': 'RealESRGAN_x4plus_anime_6B'
        }
        model_name = model_name_map.get(realesrgan_model_str, 'RealESRGAN_x4plus')
        self.realesrgan = RealESRGANRestorer(model_name=model_name, device=self.device, tile = 256, half=True)
        print("Real-ESRGAN model loaded successfully.")

        # --- Load GFPGAN Model ---
        print("Loading GFPGAN for face enhancement...")
        self.gfpgan = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=4, # Corresponds to RealESRGAN's scale
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None # We use RealESRGAN for background
        )
        print("GFPGAN model loaded successfully.")

        # --- Define transformations for U-Net ---
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def restore_image(self, image: Image.Image, output_dir: str = None, img_name: str = "restored.png"):
        """
        Applies the full 3-step restoration pipeline to a single image.
        U-Net -> Real-ESRGAN -> GFPGAN
        
        Args:
            image (PIL.Image.Image): The input image.
            output_dir (str, optional): Directory to save intermediate and final results. Defaults to None.
            img_name (str, optional): The base name for saved files. Defaults to "restored.png".

        Returns:
            PIL.Image.Image: The final restored image.
        """
        print(f"Processing {img_name}...")
        
        # The input is already a PIL image, no need to open it.
        img = image.copy()

        # --- Step 1: U-Net for Color/Light Correction ---
        print("Step 1: Applying U-Net for color and light correction...")
        with torch.no_grad():
            input_tensor = self.transform(img).unsqueeze(0).to(self.device)
            unet_output_tensor = self.unet(input_tensor)
        
        # Convert U-Net output tensor to a BGR cv2 image for the next steps
        unet_output_bgr = self.tensor_to_cv2(unet_output_tensor)
        
        if output_dir:
            unet_output_path = os.path.join(output_dir, f"unet_{img_name}")
            cv2.imwrite(unet_output_path, unet_output_bgr)
            print(f"Saved U-Net output to {unet_output_path}")

        # --- Step 2: Real-ESRGAN for Upscaling and Detail ---
        print("Step 2: Applying Real-ESRGAN for detail enhancement...")
        realesrgan_output_bgr, _ = self.realesrgan.restore(unet_output_bgr)
        
        if output_dir:
            realesrgan_output_path = os.path.join(output_dir, f"realesrgan_{img_name}")
            cv2.imwrite(realesrgan_output_path, realesrgan_output_bgr)
            print(f"Saved Real-ESRGAN output to {realesrgan_output_path}")

        # --- Step 3: GFPGAN for Face Polishing ---
        print("Step 3: Applying GFPGAN for face polishing...")
        _, _, final_output_bgr = self.gfpgan.enhance(
            realesrgan_output_bgr, has_aligned=False, only_center_face=False, paste_back=True
        )

        if output_dir:
            final_output_path = os.path.join(output_dir, f"final_{img_name}")
            cv2.imwrite(final_output_path, final_output_bgr)
            print(f"Saved final output to {final_output_path}")
            
        # Convert final BGR cv2 image back to PIL RGB for return
        final_output_rgb = cv2.cvtColor(final_output_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(final_output_rgb)

    def tensor_to_cv2(self, tensor):
        """Converts a PyTorch tensor (C, H, W) to a CV2 image (H, W, C) in BGR format."""
        # Squeeze the batch dimension, move channels to last, and detach from graph
        img_np = tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        # Convert from [0, 1] float to [0, 255] uint8
        img_np = (img_np * 255).astype(np.uint8)
        # Convert RGB (from PIL/PyTorch) to BGR (for OpenCV)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        return img_bgr

def main():
    """Example usage of the HybridPipeline."""
    # --- Configuration ---
    UNET_MODEL = "outputs/models/unet/best_unet_resnet34_perceptual.pth"
    REALESRGAN_MODEL = "x4"
    INPUT_IMAGE = "data/raw/AI_for_Art_Restoration_2/paired_dataset_art/damaged/2.jpg"
    OUTPUT_DIR = "outputs/hybrid_restored_v2"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check if the U-Net model exists
    if not os.path.exists(UNET_MODEL):
        print(f"Error: U-Net model not found at {UNET_MODEL}")
        print("Please train the U-Net model first by running 'src/training/train_unet.py'")
        return

    # --- Initialize and run pipeline ---
    pipeline = HybridPipeline(UNET_MODEL, REALESRGAN_MODEL)
    
    output_filename = os.path.join(OUTPUT_DIR, os.path.basename(INPUT_IMAGE))
    pipeline.restore_image(INPUT_IMAGE, output_filename)

if __name__ == '__main__':
    main()
