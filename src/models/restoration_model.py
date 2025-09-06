"""
restoration_model.py
Artifact restoration model using deep learning.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


def unet_model(input_size=(256, 256, 3)):
    """
    U-Net model for image restoration.
    
    Args:
        input_size (tuple): Input image dimensions (height, width, channels)
    
    Returns:
        keras.models.Model: U-Net model
    """
    inputs = Input(input_size)
    
    # Encoder (downsampling path)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # Bottom
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    
    # Decoder (upsampling path)
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)
    
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)
    
    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)
    
    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
    
    # Output layer
    outputs = Conv2D(3, 1, activation='sigmoid')(conv9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model


class ArtifactRestorer:
    """Artifact restoration model class."""
    
    def __init__(self, input_shape=(256, 256, 3), model_path=None):
        """
        Initialize the model.
        
        Args:
            input_shape (tuple): Input shape (height, width, channels)
            model_path (str): Path to load pre-trained model weights
        """
        self.input_shape = input_shape
        self.model = self._build_model()
        
        if model_path and os.path.exists(model_path):
            self.model.load_weights(model_path)
            print(f"Loaded model from {model_path}")
    
    def _build_model(self):
        """
        Build the restoration model.
        
        Returns:
            keras.models.Model: Compiled model
        """
        model = unet_model(self.input_shape)
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='mean_squared_error',
            metrics=['mae']
        )
        return model
    
    def train(self, x_train, y_train, x_val, y_val, batch_size=8, epochs=50, save_path=None):
        """
        Train the model.
        
        Args:
            x_train (numpy.ndarray): Training images (damaged)
            y_train (numpy.ndarray): Target images (undamaged)
            x_val (numpy.ndarray): Validation images (damaged)
            y_val (numpy.ndarray): Validation target images (undamaged)
            batch_size (int): Batch size
            epochs (int): Number of epochs
            save_path (str): Path to save model weights
            
        Returns:
            dict: Training history
        """
        callbacks = []
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            checkpoint = ModelCheckpoint(
                save_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
            callbacks.append(checkpoint)
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        history = self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return history.history
    
    def restore(self, image):
        """
        Restore a damaged image.
        
        Args:
            image (numpy.ndarray): Damaged image
            
        Returns:
            numpy.ndarray: Restored image
        """
        # Resize to expected input shape
        original_shape = image.shape
        resized = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
        
        # Normalize
        normalized = resized / 255.0
        
        # Predict
        predicted = self.model.predict(np.expand_dims(normalized, axis=0))[0]
        
        # Denormalize
        restored = np.clip(predicted * 255.0, 0, 255).astype(np.uint8)
        
        # Resize back to original dimensions if needed
        if original_shape[:2] != self.input_shape[:2]:
            restored = cv2.resize(restored, (original_shape[1], original_shape[0]))
        
        return restored
    
    def save_model(self, path):
        """
        Save model weights.
        
        Args:
            path (str): Path to save model weights
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_weights(path)
        print(f"Model saved to {path}")
