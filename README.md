# ArtifactVision: Art Restoration and Forgery Detection

A computer vision project for artifact restoration and forgery detection using OpenCV, FFT analysis, and machine learning.

## Project Overview

ArtifactVision uses frequency domain analysis and machine learning to:

1. **Restore damaged artwork** using advanced image processing and deep learning
2. **Detect art forgeries** by analyzing frequency domain features

The project leverages Fast Fourier Transform (FFT) to extract features from the frequency domain of artwork images, which can reveal patterns not visible in the spatial domain.

## Directory Structure

```
image_processing/
├── data/
│   ├── raw/                      # Raw artwork dataset
│   │   └── AI_for_Art_Restoration_2/
│   │       ├── paired_dataset_art/
│   │       │   ├── damaged/      # Damaged artwork images
│   │       │   └── undamaged/    # Original/undamaged artwork images
│   └── processed/                # Processed data for model training
├── notebooks/
│   ├── explore_datasets.ipynb    # Data exploration notebook
│   └── fft_art_analysis.ipynb    # FFT analysis of art images
├── outputs/
│   ├── figures/                  # Output visualizations
│   └── models/                   # Saved model files
└── src/
    ├── data/                     # Data loading and preprocessing
    │   └── dataset.py            # Dataset class
    ├── exp/                      # Experimental code
    │   └── fft_pipeline.py       # FFT analysis pipeline
    ├── main/                     # Main application code
    │   └── main.py               # Command-line interface
    ├── models/                   # ML models
    │   ├── detection_model.py    # Forgery detection model
    │   └── restoration_model.py  # Image restoration model
    ├── training/                 # Training scripts
    │   ├── train_detection.py    # Train forgery detection model
    │   └── train_restoration.py  # Train restoration model
    └── utils/                    # Utility functions
        ├── feature_extraction.py # Feature extraction utilities
        └── visualization.py      # Visualization utilities
```

## Installation

1. Clone the repository:

```
git clone https://github.com/yourusername/artifact-vision.git
cd artifact-vision
```

2. Install dependencies:

```
pip install -r requirements.txt
```

## Usage

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
