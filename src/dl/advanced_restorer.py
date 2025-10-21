"""
Advanced Art Restoration System
Handles structural damage like tears, holes, and missing parts
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
from typing import Tuple, Dict, List, Optional
from .hybrid_restorer import HybridRestorer


class StructuralDamageDetector:
    """
    Detects tears, holes, cracks, and missing parts in artwork.
    """
    
    def __init__(self):
        self.min_hole_area = 100  # Minimum area for hole detection
        self.max_hole_area = 10000  # Maximum area for hole detection
        
    def detect_tears_and_cracks(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect linear damage like tears and cracks.
        
        Args:
            image: Input damaged image
            
        Returns:
            mask: Binary mask of detected tears/cracks
            cracks: List of crack information
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection to find potential cracks
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Use HoughLinesP to detect line segments (potential cracks)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                               minLineLength=50, maxLineGap=10)
        
        # Create mask for detected lines
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cracks = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Calculate line properties
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                
                # Filter lines that look like cracks (long, irregular)
                if length > 30:
                    cv2.line(mask, (x1, y1), (x2, y2), 255, 3)
                    cracks.append({
                        'start': (x1, y1),
                        'end': (x2, y2),
                        'length': length,
                        'angle': angle
                    })
        
        return mask, cracks
    
    def detect_holes_and_missing_parts(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect holes and missing parts using color/texture analysis.
        
        Args:
            image: Input damaged image
            
        Returns:
            mask: Binary mask of detected holes
            holes: List of hole information
        """
        # Convert to LAB color space for better color analysis
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Find very dark or very bright anomalous regions
        dark_mask = l_channel < 30  # Very dark regions
        bright_mask = l_channel > 220  # Very bright regions
        
        # Combine anomalous regions
        anomaly_mask = dark_mask | bright_mask
        
        # Clean up mask with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        anomaly_mask = cv2.morphologyEx(anomaly_mask.astype(np.uint8), 
                                       cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(anomaly_mask, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        holes = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_hole_area < area < self.max_hole_area:
                # Fill the contour in the mask
                cv2.fillPoly(mask, [contour], 255)
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                holes.append({
                    'contour': contour,
                    'area': area,
                    'bbox': (x, y, w, h),
                    'center': (x + w//2, y + h//2)
                })
        
        return mask, holes
    
    def detect_all_damage(self, image: np.ndarray) -> Dict:
        """
        Comprehensive damage detection.
        
        Args:
            image: Input damaged image
            
        Returns:
            Dict with all detected damage information
        """
        # Detect different types of damage
        crack_mask, cracks = self.detect_tears_and_cracks(image)
        hole_mask, holes = self.detect_holes_and_missing_parts(image)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(crack_mask, hole_mask)
        
        # Calculate damage statistics
        total_pixels = image.shape[0] * image.shape[1]
        damaged_pixels = np.sum(combined_mask > 0)
        damage_percentage = (damaged_pixels / total_pixels) * 100
        
        return {
            'crack_mask': crack_mask,
            'hole_mask': hole_mask,
            'combined_mask': combined_mask,
            'cracks': cracks,
            'holes': holes,
            'damage_percentage': damage_percentage,
            'has_structural_damage': len(cracks) > 0 or len(holes) > 0
        }


class DeepInpainter:
    """
    Deep learning-based inpainting for structural damage restoration.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.build_inpainting_model()
    
    def build_inpainting_model(self):
        """
        Build a context encoder model for inpainting.
        Based on "Context Encoders: Feature Learning by Inpainting"
        """
        # Input: damaged image + mask
        img_input = keras.layers.Input(shape=(256, 256, 3), name='image')
        mask_input = keras.layers.Input(shape=(256, 256, 1), name='mask')
        
        # Combine image and mask
        masked_img = keras.layers.Multiply()([img_input, mask_input])
        combined_input = keras.layers.Concatenate()([masked_img, mask_input])
        
        # Encoder
        x = keras.layers.Conv2D(64, 4, strides=2, padding='same', activation='relu')(combined_input)
        x = keras.layers.BatchNormalization()(x)
        
        x = keras.layers.Conv2D(128, 4, strides=2, padding='same', activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        
        x = keras.layers.Conv2D(256, 4, strides=2, padding='same', activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        
        x = keras.layers.Conv2D(512, 4, strides=2, padding='same', activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        
        # Bottleneck
        x = keras.layers.Conv2D(512, 4, strides=2, padding='same', activation='relu')(x)
        
        # Decoder
        x = keras.layers.Conv2DTranspose(512, 4, strides=2, padding='same', activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        
        x = keras.layers.Conv2DTranspose(256, 4, strides=2, padding='same', activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        
        x = keras.layers.Conv2DTranspose(128, 4, strides=2, padding='same', activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        
        x = keras.layers.Conv2DTranspose(64, 4, strides=2, padding='same', activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        
        # Output layer
        output = keras.layers.Conv2DTranspose(3, 4, strides=2, padding='same', 
                                            activation='sigmoid', name='inpainted')(x)
        
        self.model = keras.Model(inputs=[img_input, mask_input], outputs=output)
        
        # Compile with perceptual loss
        self.model.compile(
            optimizer=keras.optimizers.Adam(0.0002),
            loss=self.perceptual_loss,
            metrics=['mae']
        )
    
    def perceptual_loss(self, y_true, y_pred):
        """
        Perceptual loss using VGG16 features.
        """
        # Load VGG16 for perceptual loss
        vgg = keras.applications.VGG16(include_top=False, weights='imagenet')
        vgg.trainable = False
        
        # Extract features
        true_features = vgg(y_true)
        pred_features = vgg(y_pred)
        
        # Calculate feature loss
        feature_loss = tf.reduce_mean(tf.square(true_features - pred_features))
        
        # Add L1 loss
        l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
        
        return feature_loss + l1_loss
    
    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Inpaint damaged regions using deep learning.
        
        Args:
            image: Damaged image (0-255)
            mask: Binary mask (255 = damaged, 0 = intact)
            
        Returns:
            Inpainted image
        """
        if self.model is None:
            print("⚠️ Inpainting model not available, using traditional method")
            return self.traditional_inpaint(image, mask)
        
        # Preprocess
        img_resized = cv2.resize(image, (256, 256)) / 255.0
        mask_resized = cv2.resize(mask, (256, 256)) / 255.0
        
        # Create inverse mask (1 for intact, 0 for damaged)
        inverse_mask = 1.0 - mask_resized
        inverse_mask = np.expand_dims(inverse_mask, axis=-1)
        
        # Prepare inputs
        img_batch = np.expand_dims(img_resized, axis=0)
        mask_batch = np.expand_dims(inverse_mask, axis=0)
        
        # Predict
        inpainted = self.model.predict([img_batch, mask_batch])[0]
        
        # Resize back and convert to uint8
        inpainted = cv2.resize(inpainted, (image.shape[1], image.shape[0]))
        inpainted = (inpainted * 255).astype(np.uint8)
        
        return inpainted
    
    def traditional_inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Traditional inpainting using OpenCV methods.
        """
        # Use Navier-Stokes based inpainting
        inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)
        return inpainted
    
    def load_model(self, model_path: str):
        """Load pre-trained inpainting model."""
        try:
            self.model = keras.models.load_model(model_path, compile=False)
            self.model.compile(
                optimizer=keras.optimizers.Adam(0.0002),
                loss=self.perceptual_loss,
                metrics=['mae']
            )
            print(f"✅ Loaded inpainting model from {model_path}")
        except Exception as e:
            print(f"⚠️ Could not load model: {e}")
            self.model = None


class AdvancedRestorer(HybridRestorer):
    """
    Advanced restoration system that handles both quality degradation
    and structural damage like tears, holes, and missing parts.
    """
    
    def __init__(self, ml_model_path: str, ml_scaler_path: str, 
                 inpainting_model_path: Optional[str] = None, **kwargs):
        # Initialize parent hybrid restorer
        super().__init__(ml_model_path, ml_scaler_path, **kwargs)
        
        # Initialize structural damage components
        self.damage_detector = StructuralDamageDetector()
        self.inpainter = DeepInpainter(inpainting_model_path)
        
    def restore_advanced(self, image_path: str, output_path: Optional[str] = None, 
                        strategy: str = 'auto') -> Tuple[np.ndarray, Dict]:
        """
        Advanced restoration handling both quality and structural damage.
        
        Args:
            image_path: Path to damaged image
            output_path: Optional output path
            strategy: 'auto', 'inpaint_first', or 'quality_first'
            
        Returns:
            Tuple of (restored_image, restoration_info)
        """
        print(f"🎨 Advanced restoration of {os.path.basename(image_path)}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Step 1: Detect all types of damage
        damage_info = self.damage_detector.detect_all_damage(image)
        quality_analysis = self.analyze_damage(image_path)
        
        print(f"📊 Damage Analysis:")
        print(f"  - Structural damage: {damage_info['has_structural_damage']}")
        print(f"  - Tears/cracks: {len(damage_info['cracks'])}")
        print(f"  - Holes/missing: {len(damage_info['holes'])}")
        print(f"  - Damage area: {damage_info['damage_percentage']:.1f}%")
        print(f"  - Quality severity: {quality_analysis['severity']}")
        
        restored_image = image.copy()
        restoration_steps = []
        
        # Step 2: Handle structural damage first (if present)
        if damage_info['has_structural_damage']:
            print("🔧 Repairing structural damage with inpainting...")
            
            # Inpaint damaged regions
            inpainted = self.inpainter.inpaint(image, damage_info['combined_mask'])
            
            # Blend inpainted regions with original
            mask_3d = np.stack([damage_info['combined_mask']] * 3, axis=-1) / 255.0
            restored_image = (inpainted * mask_3d + restored_image * (1 - mask_3d)).astype(np.uint8)
            
            restoration_steps.append('structural_inpainting')
        
        # Step 3: Apply quality restoration to the structurally repaired image
        if strategy == 'quality_first' or not damage_info['has_structural_damage']:
            print("✨ Applying quality restoration...")
            
            # Use parent class method for quality restoration
            temp_path = 'temp_structural_repaired.jpg'
            cv2.imwrite(temp_path, restored_image)
            
            quality_restored, quality_info = super().restore(
                temp_path, 
                strategy='auto'
            )
            
            restored_image = quality_restored
            restoration_steps.append(f"quality_{quality_info['method']}")
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        # Step 4: Save result
        if output_path:
            cv2.imwrite(output_path, restored_image)
            print(f"💾 Saved to {output_path}")
        
        # Compile restoration info
        restoration_info = {
            'structural_damage': damage_info,
            'quality_analysis': quality_analysis,
            'restoration_steps': restoration_steps,
            'has_structural_repair': damage_info['has_structural_damage'],
            'final_method': 'advanced_hybrid'
        }
        
        return restored_image, restoration_info
    
    def compare_all_methods(self, image_path: str) -> Dict:
        """
        Compare original hybrid methods + advanced restoration.
        """
        # Get standard comparison
        results = super().compare_methods(image_path)
        
        # Add advanced restoration
        results['advanced'], info = self.restore_advanced(image_path)
        results['advanced_info'] = info
        
        return results
    
    def batch_restore_advanced(self, input_dir: str, output_dir: str, 
                              file_pattern: str = "*.jpg") -> List[Dict]:
        """
        Batch advanced restoration with detailed reporting.
        """
        import glob
        
        os.makedirs(output_dir, exist_ok=True)
        results = []
        
        files = glob.glob(os.path.join(input_dir, file_pattern))
        
        for i, file_path in enumerate(files):
            print(f"\n🎯 Processing {i+1}/{len(files)}: {os.path.basename(file_path)}")
            
            try:
                output_path = os.path.join(output_dir, f"restored_{os.path.basename(file_path)}")
                restored, info = self.restore_advanced(file_path, output_path)
                
                results.append({
                    'input_file': file_path,
                    'output_file': output_path,
                    'success': True,
                    'info': info
                })
                
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
                results.append({
                    'input_file': file_path,
                    'success': False,
                    'error': str(e)
                })
        
        return results


def demo_advanced_restoration():
    """
    Demo the advanced restoration system.
    """
    print('🚀 Advanced Art Restoration System Demo')
    print('='*70)
    print('Features:')
    print('  - Detects tears, cracks, holes, missing parts')
    print('  - Deep learning inpainting for structural repair')
    print('  - Combined with ML-guided quality restoration')
    print('  - Handles complex multi-type damage')
    
    # Initialize advanced restorer
    restorer = AdvancedRestorer(
        ml_model_path='../outputs/models/restoration_parameter_predictor.pkl',
        ml_scaler_path='../outputs/models/parameter_feature_scaler.pkl',
        use_dl=True
    )
    
    # Test on sample image
    test_image = '../data/raw/AI_for_Art_Restoration_2/paired_dataset_art/damaged/image_1.jpg'
    
    if os.path.exists(test_image):
        restored, info = restorer.restore_advanced(
            test_image,
            output_path='../outputs/advanced_restored.jpg'
        )
        
        print('\n📈 Advanced Restoration Complete!')
        print(f"Steps applied: {', '.join(info['restoration_steps'])}")
        
        if info['has_structural_repair']:
            print(f"Structural damage repaired: {info['structural_damage']['damage_percentage']:.1f}% of image")
    else:
        print(f'⚠️ Test image not found: {test_image}')


if __name__ == '__main__':
    demo_advanced_restoration()