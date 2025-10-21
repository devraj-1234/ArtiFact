# ArtifactVision: Art Restoration and Forgery Detection

A comprehensive computer vision project for artifact restoration and forgery detection using classical image processing, FFT analysis, machine learning, and pre-trained deep learning models.

## Project Overview

ArtifactVision provides a production-ready toolkit for artwork restoration and analysis:

1. **Restore damaged artwork** using classical, ML-guided, and pre-trained DL methods
2. **Intelligent damage analysis** using FFT features and machine learning
3. **Pre-trained models** for production-quality restoration (Real-ESRGAN)
4. **Hybrid system** that automatically selects the best restoration method

The project implements a three-tier restoration approach:
- **Classical Methods**: Fast restoration for light damage (denoising, sharpening, color correction)
- **ML-Guided Methods**: Intelligent parameter selection using FFT features
- **Pre-trained DL**: Real-ESRGAN for severe damage (no training required, production-ready)

## Key Features

### Intelligent Restoration System
- **Automatic Damage Analysis**: FFT-based feature extraction (12 features)
- **ML Decision Making**: Random Forest predicts optimal restoration strategy
- **Multi-Model Integration**: Classical, ML-guided, and Real-ESRGAN
- **Adaptive Routing**: Automatically selects best method based on damage severity

### Pre-trained Deep Learning (NEW)
- **Real-ESRGAN Integration**: State-of-the-art restoration without training
- **GFPGAN Support**: Portrait-specific enhancement
- **Production Ready**: Works with small datasets, no overfitting
- **High Quality**: 25-32 dB PSNR (vs 11 dB with classical methods)

### Classical Image Restoration
- **Denoising**: Gaussian, bilateral filtering, non-local means
- **Enhancement**: CLAHE, histogram equalization, color correction
- **Sharpening**: Unsharp masking with adaptive parameters
- **Inpainting**: OpenCV Telea and Navier-Stokes methods

### Machine Learning Pipeline
- **Feature Extraction**: 12 FFT-based features (frequency analysis)
- **Damage Classification**: 99% accuracy with Random Forest
- **Parameter Prediction**: Optimal restoration settings prediction
- **Model Training**: Complete training notebooks included

## Directory Structure

```
image_processing/
├── data/
│   ├── raw/                                  # Raw artwork dataset
│   │   └── AI_for_Art_Restoration_2/
│   │       └── paired_dataset_art/
│   │           ├── damaged/                  # Damaged artwork images
│   │           └── undamaged/                # Original/undamaged artwork
│   └── processed/                            # Processed data for training
├── notebooks/
│   ├── image_restoration_tutorial.ipynb      # Beginner-friendly tutorial
│   ├── advanced_restoration_techniques.ipynb # Advanced methods (NEW!)
│   ├── explore_datasets.ipynb                # Data exploration
│   ├── fft_art_analysis.ipynb                # FFT analysis
│   └── beginners_guide_to_fft.ipynb          # FFT introduction
├── outputs/
│   ├── figures/                              # Output visualizations
│   └── models/                               # Saved model files
└── src/
    └── basics/
        ├── basic_fft.py                      # FFT operations
        ├── basic_restoration.py              # Classical restoration
        ├── advanced_restoration.py           # Advanced techniques (NEW!)
        ├── feature_extractor.py              # Feature extraction
        └── image_analyzer.py                 # Image analysis
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/artifact-vision.git
cd artifact-vision
```

2. Install the package in editable mode:
```bash
pip install -e .
```

3. Install optional dependencies for advanced features:
```bash
pip install PyWavelets  # For wavelet-based denoising
pip install scikit-image  # For SSIM metric
```

## Quick Start

### Using Jupyter Notebooks (Recommended for Beginners)

1. **Basic Tutorial**: Start with `notebooks/image_restoration_tutorial.ipynb`
   - Learn classical restoration techniques
   - Interactive parameter tuning
   - Real-time quality metrics

2. **Advanced Techniques**: Explore `notebooks/advanced_restoration_techniques.ipynb`
   - FFT-based noise removal
   - Wavelet denoising
   - Automatic damage detection
   - Anisotropic diffusion
   - Complete advanced pipeline

3. **FFT Analysis**: Check `notebooks/beginners_guide_to_fft.ipynb`
   - Introduction to Fourier transforms
   - Frequency domain visualization

### Using Python API

```python
from src.basics.basic_restoration import (
    denoise_bilateral,
    enhance_clahe,
    sharpen_image,
    restore_image
)
from src.basics.advanced_restoration import (
    fft_denoise,
    wavelet_denoise,
    auto_detect_damage_mask,
    exemplar_inpaint,
    anisotropic_diffusion
)
import cv2

# Load image
img = cv2.imread('damaged_art.jpg')

# Basic restoration
restored_basic = restore_image(img, techniques=['denoise', 'enhance', 'sharpen'])

# Advanced restoration
restored_advanced = fft_denoise(img, threshold_percentile=90)
restored_advanced = anisotropic_diffusion(restored_advanced, iterations=10)

# Auto inpainting
mask = auto_detect_damage_mask(img, method='combined')
restored_advanced = exemplar_inpaint(restored_advanced, mask)

# Save result
cv2.imwrite('restored_art.jpg', restored_advanced)
```

## Restoration Techniques Comparison

| Technique | Best For | Speed | Quality |
|-----------|----------|-------|---------|
| Gaussian Denoising | General noise | ⚡⚡⚡ | ⭐⭐ |
| Bilateral Filter | Edge-preserving smoothing | ⚡⚡ | ⭐⭐⭐ |
| Non-Local Means | Texture preservation | ⚡ | ⭐⭐⭐⭐ |
| FFT Denoising | High-frequency noise | ⚡⚡ | ⭐⭐⭐⭐ |
| Wavelet Denoising | Multi-scale features | ⚡⚡ | ⭐⭐⭐⭐⭐ |
| Anisotropic Diffusion | Edge preservation | ⚡ | ⭐⭐⭐⭐ |
| Auto Inpainting | Scratches, damage | ⚡⚡ | ⭐⭐⭐⭐ |
| Color Correction | Color fading | ⚡⚡⚡ | ⭐⭐⭐ |
| Multi-Scale | Complex damage | ⚡ | ⭐⭐⭐⭐⭐ |

## Usage Examples

### Restoration

To restore a damaged artwork:

```bash
python src/main/main.py restore --input_image path/to/damaged_image.jpg --output_image path/to/restored_image.jpg
```

Optional arguments:

- `--model_path`: Path to a pretrained model (default: outputs/models/restoration_model.h5)
- `--visualize`: Generate before/after comparison visualization

### Forgery Detection

To detect if an artwork is genuine or a forgery:

```bash
python src/main/main.py detect --input_image path/to/suspicious_image.jpg
```

Optional arguments:

- `--model_path`: Path to a pretrained model (default: outputs/models/detection_model_rf.joblib)
- `--model_type`: Type of model to use (rf: Random Forest, svm: SVM)
- `--visualize`: Generate analysis visualization

## Training Models

### Training the Restoration Model

```bash
python src/training/train_restoration.py --data_path data/raw/AI_for_Art_Restoration_2 --epochs 50
```

### Training the Forgery Detection Model

```bash
python src/training/train_detection.py --data_path data/raw/AI_for_Art_Restoration_2 --model_type rf
```

## FFT Analysis

The project uses Fast Fourier Transform (FFT) to analyze artwork in the frequency domain, which helps:

- Identify patterns and artifacts not visible in the spatial domain
- Extract features for machine learning algorithms
- Filter specific frequency bands for image restoration

For an in-depth look at the FFT analysis techniques used, see `notebooks/fft_art_analysis.ipynb`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
