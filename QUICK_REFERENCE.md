# Advanced Restoration Quick Reference

## 🚀 Quick Start Guide

### Installation
```bash
cd "d:\R&D Project\image_processing"
pip install -e .
pip install PyWavelets  # Optional, for wavelet denoising
```

---

## 📖 Function Reference

### Import Statement
```python
from src.basics.advanced_restoration import (
    fft_denoise,
    remove_periodic_noise,
    wavelet_denoise,
    auto_detect_damage_mask,
    exemplar_inpaint,
    anisotropic_diffusion,
    unsharp_mask,
    color_correction,
    multi_scale_restoration
)
```

---

## 🔧 Function Signatures & Usage

### 1. FFT Denoising
```python
fft_denoise(image, threshold_percentile=90)
```
**Purpose**: Remove noise using frequency domain filtering  
**Parameters**:
- `image`: Input image (BGR or grayscale)
- `threshold_percentile`: Higher = more aggressive (70-99)

**Example**:
```python
restored = fft_denoise(damaged_img, threshold_percentile=92)
```

**Best For**: High-frequency noise, speckle noise  
**Speed**: ⚡⚡ Medium  
**Quality**: ⭐⭐⭐⭐

---

### 2. Periodic Noise Removal
```python
remove_periodic_noise(image, filter_type='notch', d0=30)
```
**Purpose**: Eliminate periodic patterns (scan lines, moire)  
**Parameters**:
- `image`: Input image
- `filter_type`: 'notch' or 'bandstop'
- `d0`: Filter radius

**Example**:
```python
cleaned = remove_periodic_noise(img, filter_type='notch', d0=30)
```

**Best For**: Scan artifacts, repetitive patterns  
**Speed**: ⚡⚡ Medium  
**Quality**: ⭐⭐⭐⭐

---

### 3. Wavelet Denoising
```python
wavelet_denoise(image, wavelet='db1', level=1, threshold_scale=1.0)
```
**Purpose**: Multi-scale denoising using wavelet transform  
**Parameters**:
- `image`: Input image
- `wavelet`: Wavelet family ('haar', 'db1', 'db2', 'db4', 'sym2', 'coif1')
- `level`: Decomposition levels (1-4)
- `threshold_scale`: Threshold multiplier

**Example**:
```python
restored = wavelet_denoise(img, wavelet='db1', level=2, threshold_scale=1.0)
```

**Best For**: Preserving textures, complex noise  
**Speed**: ⚡⚡ Medium  
**Quality**: ⭐⭐⭐⭐⭐  
**Requires**: `pip install PyWavelets`

---

### 4. Auto Damage Detection
```python
auto_detect_damage_mask(image, method='edges', **kwargs)
```
**Purpose**: Automatically find damaged areas  
**Parameters**:
- `image`: Input image
- `method`: 'edges', 'brightness', or 'combined'
- `low_thresh`: Brightness threshold low (default: 20 or 50)
- `high_thresh`: Brightness threshold high (default: 150 or 235)
- `dilate_iter`: Dilation iterations for edges (default: 2)

**Returns**: Binary mask (255 = damaged, 0 = good)

**Example**:
```python
# Edge detection
mask = auto_detect_damage_mask(img, method='edges', low_thresh=50, high_thresh=150)

# Brightness detection  
mask = auto_detect_damage_mask(img, method='brightness', low_thresh=20, high_thresh=235)

# Combined (recommended)
mask = auto_detect_damage_mask(img, method='combined')
```

**Best For**: Scratches, cracks, bright/dark damage  
**Speed**: ⚡⚡⚡ Fast  
**Quality**: ⭐⭐⭐⭐

---

### 5. Advanced Inpainting
```python
exemplar_inpaint(image, mask, patch_size=9)
```
**Purpose**: Fill damaged areas using exemplar-based method  
**Parameters**:
- `image`: Input image
- `mask`: Binary mask (255 = damaged)
- `patch_size`: Size of patches (3-15, odd numbers)

**Example**:
```python
mask = auto_detect_damage_mask(img, method='combined')
restored = exemplar_inpaint(img, mask, patch_size=9)
```

**Best For**: Large damaged areas, scratches  
**Speed**: ⚡⚡ Medium  
**Quality**: ⭐⭐⭐⭐

---

### 6. Anisotropic Diffusion
```python
anisotropic_diffusion(image, iterations=10, kappa=50, gamma=0.1)
```
**Purpose**: Edge-preserving smoothing  
**Parameters**:
- `image`: Input image
- `iterations`: Number of iterations (5-30)
- `kappa`: Conduction coefficient (20-100, controls edge preservation)
- `gamma`: Diffusion rate (0.05-0.25)

**Example**:
```python
smoothed = anisotropic_diffusion(img, iterations=10, kappa=50, gamma=0.1)
```

**Best For**: Noise removal while keeping edges sharp  
**Speed**: ⚡ Slow (10+ iterations)  
**Quality**: ⭐⭐⭐⭐⭐

**Tips**:
- Higher `kappa` = preserve more edges
- More `iterations` = smoother (but slower)
- `gamma` around 0.1 is usually good

---

### 7. Unsharp Mask
```python
unsharp_mask(image, sigma=1.0, strength=1.5, threshold=0)
```
**Purpose**: Professional sharpening  
**Parameters**:
- `image`: Input image
- `sigma`: Gaussian blur sigma (0.5-3.0)
- `strength`: Sharpening strength (0.5-3.0)
- `threshold`: Minimum contrast to sharpen (0-20)

**Example**:
```python
sharpened = unsharp_mask(img, sigma=1.0, strength=1.5, threshold=0)
```

**Best For**: Enhancing details, final polish  
**Speed**: ⚡⚡⚡ Fast  
**Quality**: ⭐⭐⭐⭐

**Tips**:
- `sigma=1.0, strength=1.5` is a good starting point
- Use `threshold>0` to avoid sharpening noise
- Don't over-sharpen! (strength > 2.0 can look unnatural)

---

### 8. Color Correction
```python
color_correction(image, reference=None, method='histogram_matching')
```
**Purpose**: Fix color cast, match color palette  
**Parameters**:
- `image`: Input image to correct
- `reference`: Reference image (or None for white balance)
- `method`: 'histogram_matching' or 'white_balance'

**Example**:
```python
# Auto white balance
corrected = color_correction(img, method='white_balance')

# Match to reference
corrected = color_correction(damaged_img, reference=undamaged_img, method='histogram_matching')
```

**Best For**: Color fading, yellow tint, color cast  
**Speed**: ⚡⚡⚡ Fast  
**Quality**: ⭐⭐⭐

---

### 9. Multi-Scale Restoration
```python
multi_scale_restoration(image, scales=3)
```
**Purpose**: Process image at multiple resolutions  
**Parameters**:
- `image`: Input image
- `scales`: Number of pyramid levels (2-5)

**Example**:
```python
restored = multi_scale_restoration(img, scales=3)
```

**Best For**: Complex damage, multiple noise types  
**Speed**: ⚡ Slow  
**Quality**: ⭐⭐⭐⭐⭐

---

## 🎯 Recommended Pipelines

### Pipeline 1: Quick Restoration (Fast)
```python
result = img.copy()
result = fft_denoise(result, threshold_percentile=90)
result = unsharp_mask(result, sigma=1.0, strength=1.3)
```
**Time**: ~1-2 seconds  
**Quality**: Good

---

### Pipeline 2: Balanced Restoration (Medium)
```python
result = img.copy()
result = fft_denoise(result, threshold_percentile=92)
result = anisotropic_diffusion(result, iterations=8, kappa=50, gamma=0.1)
mask = auto_detect_damage_mask(result, method='combined')
result = exemplar_inpaint(result, mask, patch_size=7)
result = color_correction(result, method='white_balance')
result = unsharp_mask(result, sigma=1.0, strength=1.3)
```
**Time**: ~5-10 seconds  
**Quality**: Very Good

---

### Pipeline 3: Maximum Quality (Slow)
```python
result = img.copy()

# Stage 1: Multi-scale denoising
result = wavelet_denoise(result, wavelet='db1', level=2, threshold_scale=1.0)
result = fft_denoise(result, threshold_percentile=93)

# Stage 2: Edge-preserving smoothing
result = anisotropic_diffusion(result, iterations=15, kappa=50, gamma=0.1)

# Stage 3: Damage repair
mask = auto_detect_damage_mask(result, method='combined')
result = exemplar_inpaint(result, mask, patch_size=9)

# Stage 4: Multi-scale processing
result = multi_scale_restoration(result, scales=3)

# Stage 5: Color and detail enhancement
result = color_correction(result, method='white_balance')
result = unsharp_mask(result, sigma=1.0, strength=1.5, threshold=5)
```
**Time**: ~20-30 seconds  
**Quality**: Excellent

---

## 🎨 Parameter Tuning Guide

### For Light Noise
- `fft_denoise`: threshold_percentile=85-90
- `wavelet_denoise`: threshold_scale=0.8-1.0
- `anisotropic_diffusion`: iterations=5-8

### For Heavy Noise
- `fft_denoise`: threshold_percentile=92-95
- `wavelet_denoise`: threshold_scale=1.2-1.5
- `anisotropic_diffusion`: iterations=10-20

### For Sharp Edges (Art with Lines)
- `anisotropic_diffusion`: kappa=70-100 (higher = preserve more)
- `unsharp_mask`: strength=1.0-1.3 (moderate)

### For Soft Images (Portraits)
- `anisotropic_diffusion`: kappa=30-50 (lower = more smoothing)
- `unsharp_mask`: strength=1.5-2.0 (stronger)

---

## 💡 Tips & Tricks

### Memory Management
Large images (>2000x2000) may be slow. Consider:
```python
# Resize for processing
h, w = img.shape[:2]
scale = 1000 / max(h, w)
if scale < 1:
    small = cv2.resize(img, None, fx=scale, fy=scale)
    restored_small = pipeline(small)
    restored = cv2.resize(restored_small, (w, h))
```

### Batch Processing
```python
import os
import glob

input_dir = "damaged_images/"
output_dir = "restored_images/"

for img_path in glob.glob(os.path.join(input_dir, "*.jpg")):
    img = cv2.imread(img_path)
    restored = your_pipeline(img)
    
    filename = os.path.basename(img_path)
    cv2.imwrite(os.path.join(output_dir, filename), restored)
    print(f"Processed: {filename}")
```

### Comparing Results
```python
def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10((255.0 ** 2) / mse)

psnr = calculate_psnr(restored, ground_truth)
print(f"PSNR: {psnr:.2f} dB")
```

---

## 🐛 Troubleshooting

### "PyWavelets not installed"
```bash
pip install PyWavelets
```

### Image looks over-sharpened
- Reduce `unsharp_mask` strength
- Add threshold to prevent sharpening noise

### Colors look wrong
- Apply `color_correction` with white_balance
- Or use histogram_matching with good reference

### Edges are blurred
- Increase `kappa` in anisotropic_diffusion
- Use lower threshold_percentile in fft_denoise

### Processing is too slow
- Reduce `anisotropic_diffusion` iterations
- Skip `multi_scale_restoration`
- Use Quick Pipeline instead of Maximum Quality

---

## 📚 Learn More

- **Basic Tutorial**: `notebooks/image_restoration_tutorial.ipynb`
- **Advanced Tutorial**: `notebooks/advanced_restoration_techniques.ipynb`
- **Full Documentation**: `ADVANCED_RESTORATION_SUMMARY.md`

---

**Quick Reference Version**: 1.0  
**Last Updated**: October 2, 2025
