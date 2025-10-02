# ArtifactVision: Art Restoration and Forgery Detection

A comprehensive computer vision project for artifact restoration and forgery detection using classical image processing, FFT analysis, wavelets, and machine learning.

## Project Overview

ArtifactVision provides a complete toolkit for artwork restoration and analysis:

1. **Restore damaged artwork** using both classical and advanced image processing techniques
2. **Detect art forgeries** by analyzing frequency domain features
3. **Interactive experimentation** with Jupyter notebooks for real-time parameter tuning

The project implements multiple restoration approaches:
- **Classical Methods**: Denoising, enhancement, sharpening, inpainting
- **Advanced Techniques**: FFT filtering, wavelet transforms, anisotropic diffusion, automatic damage detection
- **Deep Learning**: (Coming soon) U-Net and GAN-based restoration

## Features

### Classical Image Restoration
- **Denoising**: Gaussian, bilateral filtering, non-local means
- **Enhancement**: CLAHE, histogram equalization
- **Damage Removal**: Morphological operations, scratch removal
- **Inpainting**: OpenCV Telea and Navier-Stokes methods
- **Sharpening**: Unsharp masking with threshold control

### Advanced Image Restoration
- **FFT-Based Filtering**: Frequency domain noise removal and periodic noise elimination
- **Wavelet Denoising**: Multi-scale decomposition using PyWavelets
- **Automatic Damage Detection**: Edge-based, brightness-based, and combined methods
- **Advanced Inpainting**: Exemplar-based with auto-detected masks
- **Anisotropic Diffusion**: Edge-preserving smoothing
- **Color Correction**: White balance and histogram matching
- **Multi-Scale Processing**: Gaussian pyramid-based restoration

### Interactive Tools
- Real-time parameter tuning with ipywidgets
- Side-by-side comparison (Damaged | Restored | Ground Truth)
- Quality metrics (PSNR, SSIM) with improvement tracking
- Multiple interactive notebooks for experimentation

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
