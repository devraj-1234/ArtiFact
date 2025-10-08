"""
U-Net architecture for image-to-image restoration

U-Net is a convolutional neural network designed for image segmentation
and restoration tasks. It has an encoder-decoder architecture with skip
connections that preserve spatial information.

References:
- Original U-Net paper: https://arxiv.org/abs/1505.04597
- Adapted for art restoration tasks
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import cv2


def build_unet(input_shape=(256, 256, 3), num_filters=64):
    """
    Build U-Net architecture for image restoration.
    
    Architecture:
        - Encoder: Downsampling path with 4 levels
        - Bottleneck: Deepest layer with most features
        - Decoder: Upsampling path with skip connections
        - Output: Restored RGB image
    
    Args:
        input_shape: Tuple (H, W, C) for input images
        num_filters: Base number of filters (doubles at each level)
    
    Returns:
        tf.keras.Model: Compiled U-Net model
    """
    inputs = keras.Input(shape=input_shape, name='damaged_image')
    
    # ===== ENCODER (Downsampling) =====
    # Block 1: 256x256 -> 128x128
    conv1 = layers.Conv2D(num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = layers.Conv2D(num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Block 2: 128x128 -> 64x64
    conv2 = layers.Conv2D(num_filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = layers.Conv2D(num_filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Block 3: 64x64 -> 32x32
    conv3 = layers.Conv2D(num_filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = layers.Conv2D(num_filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Block 4: 32x32 -> 16x16
    conv4 = layers.Conv2D(num_filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = layers.Conv2D(num_filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # ===== BOTTLENECK =====
    # Block 5: 16x16 (deepest layer)
    conv5 = layers.Conv2D(num_filters * 16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = layers.Conv2D(num_filters * 16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = layers.Dropout(0.5)(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    
    # ===== DECODER (Upsampling with skip connections) =====
    # Block 6: 16x16 -> 32x32
    up6 = layers.Conv2DTranspose(num_filters * 8, 2, strides=(2, 2), padding='same')(conv5)
    up6 = layers.concatenate([up6, conv4], axis=3)  # Skip connection
    conv6 = layers.Conv2D(num_filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up6)
    conv6 = layers.Conv2D(num_filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    
    # Block 7: 32x32 -> 64x64
    up7 = layers.Conv2DTranspose(num_filters * 4, 2, strides=(2, 2), padding='same')(conv6)
    up7 = layers.concatenate([up7, conv3], axis=3)  # Skip connection
    conv7 = layers.Conv2D(num_filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up7)
    conv7 = layers.Conv2D(num_filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = layers.BatchNormalization()(conv7)
    
    # Block 8: 64x64 -> 128x128
    up8 = layers.Conv2DTranspose(num_filters * 2, 2, strides=(2, 2), padding='same')(conv7)
    up8 = layers.concatenate([up8, conv2], axis=3)  # Skip connection
    conv8 = layers.Conv2D(num_filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up8)
    conv8 = layers.Conv2D(num_filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = layers.BatchNormalization()(conv8)
    
    # Block 9: 128x128 -> 256x256
    up9 = layers.Conv2DTranspose(num_filters, 2, strides=(2, 2), padding='same')(conv8)
    up9 = layers.concatenate([up9, conv1], axis=3)  # Skip connection
    conv9 = layers.Conv2D(num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up9)
    conv9 = layers.Conv2D(num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = layers.BatchNormalization()(conv9)
    
    # ===== OUTPUT =====
    # Final 1x1 convolution to get RGB output
    outputs = layers.Conv2D(3, 1, activation='sigmoid', name='restored_image')(conv9)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name='unet_restoration')
    
    return model


class UNetRestorer:
    """
    Wrapper class for U-Net restoration model.
    Handles image preprocessing, inference, and postprocessing.
    """
    
    def __init__(self, model_path=None, input_size=256):
        """
        Initialize U-Net restorer.
        
        Args:
            model_path: Path to saved model weights (optional)
            input_size: Size to resize images to (must match training)
        """
        self.input_size = input_size
        self.model = build_unet(input_shape=(input_size, input_size, 3))
        
        if model_path:
            self.model.load_weights(model_path)
            print(f'✅ Loaded model weights from {model_path}')
        else:
            print('⚠️ Model initialized but not trained yet')
    
    def preprocess_image(self, image):
        """
        Preprocess image for model input.
        
        Args:
            image: BGR image from cv2.imread()
        
        Returns:
            Preprocessed image ready for model
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Store original size for later
        self.original_size = (image.shape[1], image.shape[0])
        
        # Resize to model input size
        image_resized = cv2.resize(image_rgb, (self.input_size, self.input_size))
        
        # Normalize to [0, 1]
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        return image_batch
    
    def postprocess_image(self, output):
        """
        Postprocess model output back to displayable image.
        
        Args:
            output: Model output (batch of images)
        
        Returns:
            BGR image ready for cv2.imwrite()
        """
        # Remove batch dimension
        image = output[0]
        
        # Denormalize from [0, 1] to [0, 255]
        image = (image * 255).astype(np.uint8)
        
        # Resize back to original size
        image_resized = cv2.resize(image, self.original_size)
        
        # Convert RGB back to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR)
        
        return image_bgr
    
    def restore(self, image):
        """
        Restore a damaged image.
        
        Args:
            image: Input image (BGR format from cv2.imread())
        
        Returns:
            Restored image (BGR format)
        """
        # Preprocess
        input_batch = self.preprocess_image(image)
        
        # Inference
        output_batch = self.model.predict(input_batch, verbose=0)
        
        # Postprocess
        restored = self.postprocess_image(output_batch)
        
        return restored
    
    def restore_from_path(self, image_path, output_path=None):
        """
        Restore image from file path.
        
        Args:
            image_path: Path to damaged image
            output_path: Path to save restored image (optional)
        
        Returns:
            Restored image
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f'Could not load image from {image_path}')
        
        # Restore
        restored = self.restore(image)
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, restored)
            print(f'✅ Saved restored image to {output_path}')
        
        return restored
    
    def compile_model(self, learning_rate=1e-4):
        """
        Compile model with loss and optimizer.
        
        Args:
            learning_rate: Learning rate for Adam optimizer
        """
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',  # Mean Squared Error for pixel-level similarity
            metrics=['mae', self._psnr_metric]
        )
        print('✅ Model compiled successfully')
    
    @staticmethod
    def _psnr_metric(y_true, y_pred):
        """Calculate PSNR as a metric during training"""
        return tf.image.psnr(y_true, y_pred, max_val=1.0)
    
    def summary(self):
        """Print model architecture summary"""
        return self.model.summary()


def perceptual_loss(y_true, y_pred):
    """
    Perceptual loss using VGG16 features.
    Measures similarity in feature space rather than pixel space.
    
    More effective for restoration than simple MSE.
    """
    # Load VGG16 without top layers
    vgg = keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
    vgg.trainable = False
    
    # Use intermediate layers for feature extraction
    feature_extractor = keras.Model(
        inputs=vgg.input,
        outputs=[vgg.get_layer('block3_conv3').output]
    )
    
    # Extract features
    true_features = feature_extractor(y_true)
    pred_features = feature_extractor(y_pred)
    
    # Compute MSE in feature space
    return tf.reduce_mean(tf.square(true_features - pred_features))


def combined_loss(y_true, y_pred, alpha=0.5):
    """
    Combined loss: pixel loss + perceptual loss
    
    Args:
        y_true: Ground truth images
        y_pred: Predicted images
        alpha: Weight for perceptual loss (0-1)
    
    Returns:
        Combined loss value
    """
    pixel_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    percept_loss = perceptual_loss(y_true, y_pred)
    
    return (1 - alpha) * pixel_loss + alpha * percept_loss
