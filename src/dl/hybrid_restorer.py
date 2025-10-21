"""
Hybrid ML + DL Restoration System

Combines traditional ML (for fast decisions) with Deep Learning (for high-quality restoration).

Strategy:
1. Use ML to analyze damage characteristics (fast, uses FFT features)
2. ML decides restoration strategy based on damage type
3. Apply appropriate restoration method:
   - Light damage → Classical methods (fast)
   - Moderate damage → ML-guided classical restoration
   - Severe damage → Deep learning U-Net (high quality)
"""

import numpy as np
import cv2
import joblib
import os

# Import our modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.ml.feature_extractor import extract_ml_features
from src.basics.optimized_restoration import restore_image_optimized
from src.basics.advanced_restoration import unsharp_mask

try:
    from src.dl.unet_model import UNetRestorer
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False
    print('Warning: Deep learning models not available (TensorFlow not installed)')

try:
    from src.dl.realesrgan_wrapper import RealESRGANRestorer, GFPGANRestorer
    REALESRGAN_AVAILABLE = True
except ImportError:
    REALESRGAN_AVAILABLE = False
    print('Warning: Real-ESRGAN not available (install with: pip install realesrgan)')


class HybridRestorer:
    """
    Intelligent hybrid restoration system.
    
    Uses ML to analyze damage and decide optimal restoration strategy.
    Combines speed of classical methods with quality of deep learning.
    """
    
    def __init__(self, 
                 ml_model_path='outputs/models/restoration_parameter_predictor.pkl',
                 ml_scaler_path='outputs/models/parameter_feature_scaler.pkl',
                 dl_model_path=None,
                 use_dl=True,
                 use_realesrgan=True,
                 realesrgan_model='RealESRGAN_x4plus'):
        """
        Initialize hybrid restoration system.
        
        Args:
            ml_model_path: Path to ML parameter prediction model
            ml_scaler_path: Path to feature scaler
            dl_model_path: Path to U-Net model weights (optional)
            use_dl: Whether to use U-Net for severe damage
            use_realesrgan: Whether to use Real-ESRGAN (recommended)
            realesrgan_model: Real-ESRGAN model variant
        """
        # Load ML model
        self.ml_model = joblib.load(ml_model_path)
        self.scaler = joblib.load(ml_scaler_path)
        print(f'Loaded ML model from {ml_model_path}')
        
        # Load Real-ESRGAN (preferred for production)
        self.realesrgan_restorer = None
        if use_realesrgan and REALESRGAN_AVAILABLE:
            try:
                self.realesrgan_restorer = RealESRGANRestorer(
                    model_name=realesrgan_model,
                    device='cuda'
                )
                print(f'Loaded Real-ESRGAN model: {realesrgan_model}')
            except Exception as e:
                print(f'Warning: Could not load Real-ESRGAN: {e}')
                self.realesrgan_restorer = None
        
        # Load U-Net DL model (fallback)
        self.use_dl = use_dl and DL_AVAILABLE
        if self.use_dl:
            if dl_model_path and os.path.exists(dl_model_path):
                self.dl_restorer = UNetRestorer(model_path=dl_model_path)
                print(f'Loaded U-Net model from {dl_model_path}')
            else:
                self.dl_restorer = None
                self.use_dl = False
                print('Warning: U-Net model not found')
        else:
            self.dl_restorer = None
        
        # Thresholds for damage severity
        self.severe_damage_threshold = 0.7  # Sharpening need > 0.7 = severe
        self.moderate_damage_threshold = 0.4  # 0.4-0.7 = moderate
    
    def analyze_damage(self, image_path):
        """
        Analyze damage characteristics using ML features.
        
        Args:
            image_path: Path to damaged image
        
        Returns:
            dict: Damage analysis with severity and characteristics
        """
        # Extract features
        features, feature_names = extract_ml_features(image_path)
        
        # Predict optimal parameters
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        predicted_params = self.ml_model.predict(features_scaled)[0]
        
        # Extract specific features
        feature_dict = dict(zip(feature_names, features))
        
        # Determine severity
        sharpening_need = feature_dict['sharpening_need']
        color_balance_need = feature_dict['color_balance_need']
        high_freq_energy = feature_dict['high_freq_energy']
        
        if sharpening_need > self.severe_damage_threshold or high_freq_energy < 0.1:
            severity = 'severe'
        elif sharpening_need > self.moderate_damage_threshold:
            severity = 'moderate'
        else:
            severity = 'light'
        
        return {
            'severity': severity,
            'sharpening_need': sharpening_need,
            'color_balance_need': color_balance_need,
            'high_freq_energy': high_freq_energy,
            'predicted_params': {
                'apply_color_correction': round(predicted_params[0]),
                'sharpen_sigma': predicted_params[1],
                'sharpen_strength': predicted_params[2]
            },
            'features': feature_dict
        }
    
    def restore(self, image_path, output_path=None, strategy='auto'):
        """
        Restore damaged image using optimal strategy.
        
        Args:
            image_path: Path to damaged image
            output_path: Path to save restored image (optional)
            strategy: Restoration strategy:
                - 'auto': Automatically choose based on damage (default)
                - 'classical': Force classical methods
                - 'ml_guided': Force ML-guided classical
                - 'dl': Force deep learning (if available)
        
        Returns:
            tuple: (restored_image, restoration_info)
        """
        # Analyze damage
        damage_analysis = self.analyze_damage(image_path)
        
        # Load original image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f'Could not load image from {image_path}')
        
        # Determine restoration method
        if strategy == 'auto':
            if damage_analysis['severity'] == 'severe' and self.use_dl and self.dl_restorer:
                method = 'deep_learning'
            elif damage_analysis['severity'] == 'moderate':
                method = 'ml_guided'
            else:
                method = 'classical'
        elif strategy == 'dl' and self.use_dl and self.dl_restorer:
            method = 'deep_learning'
        elif strategy == 'ml_guided':
            method = 'ml_guided'
        else:
            method = 'classical'
        
        # Apply restoration
        if method == 'deep_learning':
            restored = self._restore_with_dl(image)
            print(f'🔬 Applied deep learning restoration (severity: {damage_analysis["severity"]})')
        
        elif method == 'ml_guided':
            restored = self._restore_with_ml(image, damage_analysis['predicted_params'])
            print(f'🤖 Applied ML-guided restoration (severity: {damage_analysis["severity"]})')
        
        else:  # classical
            restored = self._restore_classical(image, damage_analysis)
            print(f'⚡ Applied fast classical restoration (severity: {damage_analysis["severity"]})')
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, restored)
            print(f'✅ Saved restored image to {output_path}')
        
        # Prepare restoration info
        info = {
            'method': method,
            'damage_analysis': damage_analysis,
            'parameters': damage_analysis['predicted_params']
        }
        
        return restored, info
    
    def _restore_with_dl(self, image):
        """Apply deep learning restoration - prefers Real-ESRGAN if available"""
        # Try Real-ESRGAN first (best quality)
        if self.realesrgan_restorer is not None:
            try:
                restored = self.realesrgan_restorer.restore(
                    image,
                    outscale=1.0,
                    face_enhance=False
                )
                return restored
            except Exception as e:
                print(f'Real-ESRGAN failed: {e}, falling back to U-Net')
        
        # Fallback to U-Net
        if self.dl_restorer is not None:
            return self.dl_restorer.restore(image)
        
        # Final fallback to ML-guided
        print('Warning: No DL models available, using ML-guided restoration')
        features, _ = extract_ml_features(image)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        predicted_params = self.ml_model.predict(features_scaled)[0]
        params = {
            'apply_color_correction': round(predicted_params[0]),
            'sharpen_sigma': predicted_params[1],
            'sharpen_strength': predicted_params[2]
        }
        return self._restore_with_ml(image, params)
    
    def _restore_with_ml(self, image, params):
        """Apply ML-guided classical restoration"""
        if params['apply_color_correction'] == 1:
            restored = restore_image_optimized(
                image.copy(),
                color_method='white_balance',
                sharpen_sigma=params['sharpen_sigma'],
                sharpen_strength=params['sharpen_strength']
            )
        else:
            restored = unsharp_mask(
                image.copy(),
                sigma=params['sharpen_sigma'],
                strength=params['sharpen_strength']
            )
        return restored
    
    def _restore_classical(self, image, damage_analysis):
        """Apply fast classical restoration with default parameters"""
        # Use gentle restoration for light damage
        if damage_analysis['color_balance_need'] > 0.3:
            restored = restore_image_optimized(
                image.copy(),
                color_method='white_balance',
                sharpen_sigma=1.0,
                sharpen_strength=1.0
            )
        else:
            restored = unsharp_mask(
                image.copy(),
                sigma=1.0,
                strength=1.0
            )
        return restored
    
    def batch_restore(self, input_dir, output_dir, file_pattern='*.jpg'):
        """
        Restore all images in a directory.
        
        Args:
            input_dir: Directory containing damaged images
            output_dir: Directory to save restored images
            file_pattern: File pattern to match (e.g., '*.jpg', '*.png')
        
        Returns:
            list: Results for each restored image
        """
        import glob
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all matching files
        pattern = os.path.join(input_dir, file_pattern)
        files = glob.glob(pattern)
        
        print(f'📂 Found {len(files)} images to restore')
        print('='*70)
        
        results = []
        for idx, file_path in enumerate(files, 1):
            filename = os.path.basename(file_path)
            output_path = os.path.join(output_dir, filename)
            
            print(f'\n[{idx}/{len(files)}] Processing {filename}...')
            
            try:
                restored, info = self.restore(file_path, output_path)
                results.append({
                    'filename': filename,
                    'success': True,
                    'method': info['method'],
                    'severity': info['damage_analysis']['severity'],
                    'output_path': output_path
                })
            except Exception as e:
                print(f'❌ Error processing {filename}: {e}')
                results.append({
                    'filename': filename,
                    'success': False,
                    'error': str(e)
                })
        
        # Summary
        print()
        print('='*70)
        print('📊 Batch Restoration Summary:')
        successful = sum(1 for r in results if r['success'])
        print(f'✅ Successfully restored: {successful}/{len(files)} images')
        
        if successful > 0:
            methods = {}
            for r in results:
                if r['success']:
                    method = r['method']
                    methods[method] = methods.get(method, 0) + 1
            
            print('\nMethods used:')
            for method, count in methods.items():
                print(f'  {method}: {count} images')
        
        return results
    
    def compare_methods(self, image_path):
        """
        Compare all restoration methods on a single image.
        
        Args:
            image_path: Path to damaged image
        
        Returns:
            dict: Results from each method
        """
        image = cv2.imread(image_path)
        damage_analysis = self.analyze_damage(image_path)
        
        results = {
            'original': image,
            'damage_analysis': damage_analysis
        }
        
        # Classical
        results['classical'] = self._restore_classical(image, damage_analysis)
        
        # ML-guided
        results['ml_guided'] = self._restore_with_ml(image, damage_analysis['predicted_params'])
        
        # DL (if available)
        if self.use_dl and self.dl_restorer:
            results['deep_learning'] = self._restore_with_dl(image)
        
        return results


def demo_hybrid_system():
    """
    Demo function to show hybrid system capabilities.
    """
    print('🎨 Hybrid ML + DL Restoration System Demo')
    print('='*70)
    
    # Initialize hybrid restorer
    restorer = HybridRestorer(
        ml_model_path='../outputs/models/restoration_parameter_predictor.pkl',
        ml_scaler_path='../outputs/models/parameter_feature_scaler.pkl',
        use_dl=True
    )
    
    # Example restoration
    test_image = '../data/raw/AI_for_Art_Restoration_2/paired_dataset_art/damaged/image_1.jpg'
    
    if os.path.exists(test_image):
        restored, info = restorer.restore(
            test_image,
            output_path='../outputs/hybrid_demo_restored.jpg',
            strategy='auto'
        )
        
        print('\n📊 Restoration Info:')
        print(f"Method used: {info['method']}")
        print(f"Damage severity: {info['damage_analysis']['severity']}")
        print(f"Sharpening need: {info['damage_analysis']['sharpening_need']:.3f}")
        print(f"Color balance need: {info['damage_analysis']['color_balance_need']:.3f}")
    else:
        print(f'⚠️ Test image not found: {test_image}')


if __name__ == '__main__':
    demo_hybrid_system()
