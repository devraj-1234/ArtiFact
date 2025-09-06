"""
detection_model.py
Art forgery/fake detection using machine learning.
"""

import os
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# Import our feature extraction utilities
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.feature_extraction import extract_fft_features, extract_fft_bands, extract_texture_features


class ArtForgeryDetector:
    """Art forgery detection model using ML and FFT features."""
    
    def __init__(self, model_type='rf', model_path=None):
        """
        Initialize the model.
        
        Args:
            model_type (str): Type of model to use ('rf' for Random Forest, 'svm' for SVM)
            model_path (str): Path to load pre-trained model
        """
        self.model_type = model_type
        self.model = self._build_model()
        self.feature_names = [
            'fft_mean', 'fft_std', 'fft_skew', 'fft_kurtosis',
            'band1_mean', 'band1_std', 'band2_mean', 'band2_std', 'band3_mean', 'band3_std',
            'texture_contrast', 'texture_dissimilarity', 'texture_homogeneity', 
            'texture_energy', 'texture_correlation'
        ]
        
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"Loaded model from {model_path}")
    
    def _build_model(self):
        """
        Build the detection model.
        
        Returns:
            sklearn Pipeline: ML pipeline with preprocessing and model
        """
        if self.model_type == 'rf':
            model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                random_state=42
            )
        elif self.model_type == 'svm':
            model = SVC(
                kernel='rbf',
                probability=True,
                C=10,
                gamma='scale',
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        
        return pipeline
    
    def extract_features(self, image):
        """
        Extract features from an image for forgery detection.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Feature vector
        """
        # Extract FFT features
        fft_magnitude = extract_fft_features(image, normalize=True)
        
        # Calculate statistical features from FFT magnitude
        fft_mean = np.mean(fft_magnitude)
        fft_std = np.std(fft_magnitude)
        fft_skew = np.mean(((fft_magnitude - fft_mean) / fft_std) ** 3) if fft_std > 0 else 0
        fft_kurtosis = np.mean(((fft_magnitude - fft_mean) / fft_std) ** 4) - 3 if fft_std > 0 else 0
        
        # Extract frequency band features
        band_features = extract_fft_bands(image, bands=3)
        
        # Extract texture features
        texture_features = extract_texture_features(image)
        
        # Combine all features
        features = np.hstack([
            fft_mean, fft_std, fft_skew, fft_kurtosis,
            band_features,
            texture_features
        ])
        
        return features.reshape(1, -1)
    
    def train(self, X_train, y_train):
        """
        Train the forgery detection model.
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training labels (0 for real, 1 for fake)
            
        Returns:
            float: Training accuracy
        """
        self.model.fit(X_train, y_train)
        return self.model.score(X_train, y_train)
    
    def predict(self, image):
        """
        Predict if an image is a forgery.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            tuple: (prediction, probability) where prediction is 0 for real, 1 for fake
        """
        features = self.extract_features(image)
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0][1]  # Probability of being fake
        
        return prediction, probability
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test labels
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        y_pred = self.model.predict(X_test)
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        return results
    
    def save_model(self, path):
        """
        Save model to disk.
        
        Args:
            path (str): Path to save model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")
    
    def get_feature_importance(self):
        """
        Get feature importance from the model.
        
        Returns:
            dict: Feature names and their importance scores
        """
        if self.model_type != 'rf':
            raise ValueError("Feature importance is only available for Random Forest models")
        
        # Get the classifier from the pipeline
        classifier = self.model.named_steps['classifier']
        importances = classifier.feature_importances_
        
        # Create a dictionary mapping feature names to importance scores
        importance_dict = dict(zip(self.feature_names, importances))
        
        return importance_dict
