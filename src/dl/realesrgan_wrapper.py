"""
Real-ESRGAN Wrapper for Art Restoration
Pre-trained model integration for production-ready restoration
"""

import cv2
import numpy as np
import os
from typing import Optional, Tuple
import warnings

try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    REALESRGAN_AVAILABLE = True
except ImportError:
    REALESRGAN_AVAILABLE = False
    warnings.warn("Real-ESRGAN not installed. Install with: pip install realesrgan")


class RealESRGANRestorer:
    """
    Wrapper for Real-ESRGAN pre-trained model.
    Provides super-resolution and enhancement for damaged artwork.
    """
    
    def __init__(self, 
                 model_name: str = 'RealESRGAN_x4plus',
                 tile_size: int = 0,
                 tile_pad: int = 10,
                 pre_pad: int = 0,
                 half_precision: bool = False,
                 device: str = 'cuda'):
        """
        Initialize Real-ESRGAN model.
        
        Args:
            model_name: Model variant to use
                - 'RealESRGAN_x4plus': General 4x upscaling (recommended)
                - 'RealESRGAN_x4plus_anime_6B': Anime/illustration style
                - 'RealESRGAN_x2plus': 2x upscaling (faster)
            tile_size: Tile size for processing (0 = no tiling)
            tile_pad: Padding for tiles
            pre_pad: Pre-padding
            half_precision: Use FP16 for faster inference (requires GPU)
            device: 'cuda' or 'cpu'
        """
        if not REALESRGAN_AVAILABLE:
            raise ImportError("Real-ESRGAN not installed. Run: pip install realesrgan")
        
        self.model_name = model_name
        self.device = device
        self.upscaler = None
        
        # Model configurations
        self.model_configs = {
            'RealESRGAN_x4plus': {
                'scale': 4,
                'model_path': 'weights/RealESRGAN_x4plus.pth',
                'netscale': 4,
                'num_blocks': 23
            },
            'RealESRGAN_x2plus': {
                'scale': 2,
                'model_path': 'weights/RealESRGAN_x2plus.pth',
                'netscale': 2,
                'num_blocks': 23
            },
            'RealESRGAN_x4plus_anime_6B': {
                'scale': 4,
                'model_path': 'weights/RealESRGAN_x4plus_anime_6B.pth',
                'netscale': 4,
                'num_blocks': 6
            }
        }
        
        # Initialize model
        self._load_model(tile_size, tile_pad, pre_pad, half_precision)
    
    def _load_model(self, tile_size, tile_pad, pre_pad, half_precision):
        """Load the Real-ESRGAN model."""
        if self.model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        config = self.model_configs[self.model_name]
        
        # Define model architecture
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=config['num_blocks'],
            num_grow_ch=32,
            scale=config['netscale']
        )
        
        # Initialize upsampler
        self.upscaler = RealESRGANer(
            scale=config['netscale'],
            model_path=config['model_path'],
            model=model,
            tile=tile_size,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            half=half_precision,
            device=self.device
        )
        
        print(f"Loaded Real-ESRGAN model: {self.model_name}")
        print(f"Scale: {config['scale']}x, Device: {self.device}")
    
    def restore(self, 
                image: np.ndarray,
                outscale: float = 1.0,
                face_enhance: bool = False) -> np.ndarray:
        """
        Restore/enhance image using Real-ESRGAN.
        
        Args:
            image: Input image (BGR format, 0-255)
            outscale: Output scale factor (1.0 = same size as upscaled)
            face_enhance: Enable face enhancement (requires GFPGAN)
        
        Returns:
            Enhanced image (BGR format, 0-255)
        """
        if self.upscaler is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Enhance
            output, _ = self.upscaler.enhance(
                image,
                outscale=outscale,
                face_enhance=face_enhance
            )
            
            return output
            
        except Exception as e:
            print(f"Enhancement failed: {e}")
            return image
    
    def restore_file(self,
                    input_path: str,
                    output_path: Optional[str] = None,
                    outscale: float = 1.0,
                    face_enhance: bool = False) -> Tuple[np.ndarray, str]:
        """
        Restore image from file.
        
        Args:
            input_path: Path to input image
            output_path: Path to save output (optional)
            outscale: Output scale factor
            face_enhance: Enable face enhancement
        
        Returns:
            Tuple of (enhanced_image, output_path)
        """
        # Read image
        image = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {input_path}")
        
        # Enhance
        enhanced = self.restore(image, outscale, face_enhance)
        
        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, enhanced)
            print(f"Saved enhanced image to: {output_path}")
        
        return enhanced, output_path
    
    def batch_restore(self,
                     input_dir: str,
                     output_dir: str,
                     outscale: float = 1.0,
                     face_enhance: bool = False,
                     extensions: list = ['.jpg', '.png', '.jpeg']):
        """
        Batch process multiple images.
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            outscale: Output scale factor
            face_enhance: Enable face enhancement
            extensions: File extensions to process
        """
        import glob
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all images
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(input_dir, f'*{ext}')))
        
        print(f"Found {len(image_files)} images to process")
        
        # Process each image
        for i, image_path in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
            
            output_path = os.path.join(
                output_dir,
                f"restored_{os.path.basename(image_path)}"
            )
            
            try:
                self.restore_file(image_path, output_path, outscale, face_enhance)
            except Exception as e:
                print(f"Failed to process {image_path}: {e}")
        
        print(f"Batch processing complete. Output saved to: {output_dir}")


class GFPGANRestorer:
    """
    Wrapper for GFPGAN face restoration model.
    Use for portrait artwork specifically.
    """
    
    def __init__(self, 
                 model_version: str = '1.3',
                 upscale: int = 2,
                 device: str = 'cuda'):
        """
        Initialize GFPGAN model.
        
        Args:
            model_version: GFPGAN version ('1.3' or '1.4')
            upscale: Upscale factor
            device: 'cuda' or 'cpu'
        """
        try:
            from gfpgan import GFPGANer
            self.GFPGANer = GFPGANer
        except ImportError:
            raise ImportError("GFPGAN not installed. Run: pip install gfpgan")
        
        self.model_version = model_version
        self.upscale = upscale
        self.device = device
        self.restorer = None
        
        self._load_model()
    
    def _load_model(self):
        """Load GFPGAN model."""
        model_path = f'weights/GFPGANv{self.model_version}.pth'
        
        self.restorer = self.GFPGANer(
            model_path=model_path,
            upscale=self.upscale,
            arch='clean',
            channel_multiplier=2,
            device=self.device
        )
        
        print(f"Loaded GFPGAN v{self.model_version}")
    
    def restore_faces(self, 
                     image: np.ndarray,
                     has_aligned: bool = False,
                     only_center_face: bool = False,
                     weight: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Restore faces in image.
        
        Args:
            image: Input image (BGR, 0-255)
            has_aligned: Whether input is aligned face
            only_center_face: Only process center face
            weight: Balance between enhancement and original (0-1)
        
        Returns:
            Tuple of (restored_image, cropped_faces)
        """
        _, _, restored_img = self.restorer.enhance(
            image,
            has_aligned=has_aligned,
            only_center_face=only_center_face,
            paste_back=True,
            weight=weight
        )
        
        return restored_img, None


def demo_realesrgan():
    """Demo Real-ESRGAN on sample images."""
    print("Real-ESRGAN Demo")
    print("=" * 70)
    
    # Initialize
    restorer = RealESRGANRestorer(
        model_name='RealESRGAN_x4plus',
        device='cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
    )
    
    # Test on sample image
    test_image = '../data/raw/AI_for_Art_Restoration_2/paired_dataset_art/damaged/image_1.jpg'
    
    if os.path.exists(test_image):
        enhanced, output_path = restorer.restore_file(
            test_image,
            output_path='../outputs/realesrgan_demo.jpg',
            outscale=1.0
        )
        
        print(f"Input shape: {cv2.imread(test_image).shape}")
        print(f"Output shape: {enhanced.shape}")
        print("Demo complete!")
    else:
        print(f"Test image not found: {test_image}")


if __name__ == '__main__':
    demo_realesrgan()
