"""
GFPGAN Wrapper for Face Restoration
- Auto-downloads GFPGANv1.3 weights
- specialized for face enhancement on already-upscaled images
"""

import cv2
import numpy as np
import warnings
from pathlib import Path
from typing import Optional, Tuple
import torch

# Try to import gfpgan
try:
    from gfpgan import GFPGANer
    GFPGAN_AVAILABLE = True
except ImportError:
    GFPGAN_AVAILABLE = False
    warnings.warn("GFPGAN not installed. Install with: pip install gfpgan")

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

def ensure_gfpgan_weights(model_name: str, dst_dir: str | Path) -> str:
    """
    Download GFPGAN weights if missing.
    """
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / f"{model_name}.pth"

    if dst_path.exists() and dst_path.stat().st_size > 0:
        return str(dst_path)

    print(f"Downloading {model_name} weights...")
    
    # Official releases
    urls = {
        'GFPGANv1.3': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
        'GFPGANv1.4': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth',
        'RestoreFormer': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
    }

    if model_name not in urls:
        raise ValueError(f"Unknown model {model_name}. Available: {list(urls.keys())}")

    try:
        # Use simple urllib to avoid heavy deps if possible, or torch.hub
        import requests
        response = requests.get(urls[model_name], stream=True)
        response.raise_for_status()
        
        with open(dst_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        print(f"Downloaded to {dst_path}")
        return str(dst_path)
    except Exception as e:
        raise RuntimeError(f"Failed to download GFPGAN weights: {e}")

class GFPGANWrapper:
    def __init__(self, model_version='v1.3', device='cuda', upscale=1):
        """
        Args:
            model_version (str): 'v1.3', 'v1.4', or 'RestoreFormer'
            upscale (int): Upscaling factor. 
                           Set to 1 if image is already upscaled by Real-ESRGAN.
        """
        if not GFPGAN_AVAILABLE:
            raise ImportError("GFPGAN not installed")

        self.device = device
        
        # internal name mapping
        if '1.3' in model_version: name = 'GFPGANv1.3'
        elif '1.4' in model_version: name = 'GFPGANv1.4'
        else: name = 'GFPGANv1.3'

        weights_dir = str(_repo_root() / "outputs" / "models" / "gfpgan")
        model_path = ensure_gfpgan_weights(name, weights_dir)

        self.restorer = GFPGANer(
            model_path=model_path,
            upscale=upscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None, # We use Real-ESRGAN separately
            device=device
        )
        print(f"Loaded GFPGAN: {name} | device={device}")

    def enhance(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Enhance faces in the image.
        Input: BGR numpy array (OpenCV format)
        Output: BGR numpy array
        """
        # cropped_faces, restored_faces, restored_img
        _, _, output = self.restorer.enhance(
            img_bgr, 
            has_aligned=False, 
            only_center_face=False, 
            paste_back=True,
            weight=0.5 # Blend factor (0.5 is balanced, 1.0 is strong face correction)
        )
        return output