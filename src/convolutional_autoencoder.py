import src.constants as cn  # type: ignore
import src.util as util  # type: ignore
from src.abstract_autoencoder import AbstractAutoencoder  # type: ignore

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras # type: ignore
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.datasets import mnist  # type: ignore
from sklearn.decomposition import PCA # type:ignore
from sklearn.preprocessing import StandardScaler # type:ignore
import pandas as pd # type: ignore
import os
from typing import List, Tuple, Any


BASE_PATH = os.path.join(cn.MODEL_DIR, "convolutional_autoencoder")

# Prepare the data
X_TRAIN, LABEL_TRAIN, X_TEST, LABEL_TEST, CLASS_NAMES = util.getPklMNIST()
# Reshape to add channel dimension (28, 28, 1) for CNN
X_TRAIN = np.expand_dims(X_TRAIN, -1)
X_TEST = np.expand_dims(X_TEST, -1)

print(f"Training data shape: {X_TRAIN.shape}")
print(f"Test data shape: {X_TEST.shape}")

class ConvolutionalAutoencoder(AbstractAutoencoder):

    def __init__(self, image_shape: List[int],
            hidden_dims: List[int] = [128, 64, 32],
            num_detector: int=2, base_path: str=BASE_PATH,
            is_delete_serializations: bool=True):
        """Initializes the convolutional autoencoder.

        Args:
            image_shape (List[int]): Shape of the input images (height, width, channels)
            hidden_dims (List[int]): 3 dimensions of the hidden layers.
                The last layer is the bottleneck (encoding) layer.
            num_detector (int, optional): Number of convolutional detectors. Defaults to 2.
            base_path (str, optional): Base path for model serialization. Defaults to BASE_PATH.
            is_delete_serializations (bool, optional): Whether to delete existing serializations. Defaults to
        """
        if len(image_shape) < 2 or len(image_shape) > 3:
            raise ValueError("image_shape must be of length 2 or 3")
        if len(image_shape) == 2:
            image_shape = [image_shape[0], image_shape[1], 1]
        self.image_shape = image_shape
        if len(hidden_dims) != 3:
            raise ValueError("hidden_dims must be a list of 3 integers")
        self.hidden_dims = hidden_dims
        self.image_size = np.prod(image_shape)
        self.num_detector = num_detector
        super().__init__(base_path=base_path,
                is_delete_serializations=is_delete_serializations)

    def _build(self) -> Tuple[keras.Model, keras.Model, keras.Model, dict]:
        """Builds the convolutional autoencoder model.

        Returns:
            keras.Model: The constructed autoencoder model.
        """
        # Check if the model already exists
        history_dct: dict = {}
        # No existing model. Must build a new one.
        input_img = keras.Input(shape=self.image_shape)
        """
        # Encoder
        encoded = layers.Conv2D(num_detector, (3, 3), activation='relu', padding='same')(input_img)
        encoded = layers.Reshape((num_detector*int(self.image_size/self.image_shape[-1]),))(encoded)
        encoded = layers.Dense(self.hidden_dims[0], activation='relu')(encoded)
        encoded = layers.Dense(self.hidden_dims[1], activation='relu')(encoded)
        encoded = layers.Dense(self.hidden_dims[2], activation='relu')(encoded)
        # Decoder
        decoded = layers.Dense(self.hidden_dims[1], activation='relu')(encoded)
        decoded = layers.Dense(self.hidden_dims[0], activation='relu')(decoded)
        decoded = layers.Dense(int(self.image_size*int(num_detector/self.image_shape[-1])), activation='relu')(decoded)
        decoded = layers.Reshape((*self.image_shape[:-1], num_detector))(decoded)
        decoded = layers.Conv2D(1, (3, 3), activation='relu', padding='same')(decoded)
        #
        encoded = layers.Conv2D(self.image_shape[-1], (3, 3), activation='relu', padding='same')(input_img)  # 28×28×num_detector
        encoded = layers.Reshape((self.image_size,))(encoded)
        encoded = layers.Dense(self.hidden_dims[0], activation='relu')(encoded)
        encoded = layers.Dense(self.hidden_dims[1], activation='relu')(encoded)
        encoded = layers.Dense(self.hidden_dims[2], activation='relu')(encoded)
        # Decoder
        decoded = layers.Dense(self.hidden_dims[1], activation='relu')(encoded)
        decoded = layers.Dense(self.hidden_dims[0], activation='relu')(decoded)
        decoded = layers.Dense(self.image_size, activation='relu')(decoded)
        decoded = layers.Reshape((self.image_shape))(decoded)
        decoded = layers.Conv2D(self.image_shape[-1], (3, 3), activation='relu', padding='same')(decoded)  # 28×28×1
        """
        #
        # No existing model. Must build a new one.
        input_img = keras.Input(shape=self.image_shape)  # e.g., (28, 28, 1)
        spatial_size = self.image_shape[0] * self.image_shape[1]
        # Encoder
        """
        encoded = layers.Conv2D(self.num_detector, (3, 3), activation='relu', padding='same')(input_img)
        encoded = layers.MaxPooling2D((2, 2), padding='same')(encoded)
        encoded = layers.Reshape((self.num_detector*spatial_size//4,))(encoded)
        encoded = layers.Dense(self.hidden_dims[0], activation='relu')(encoded)
        encoded = layers.Dense(self.hidden_dims[1], activation='relu')(encoded)
        encoded = layers.Dense(self.hidden_dims[2], activation='relu')(encoded)
        # Decoder
        decoded = layers.Dense(self.hidden_dims[1], activation='relu')(encoded)
        decoded = layers.Dense(self.hidden_dims[0], activation='relu')(decoded)
        decoded = layers.Dense(spatial_size*self.num_detector//4, activation='relu')(decoded)
        decoded = layers.Reshape((self.image_shape[0]//2, self.image_shape[1]//2, self.num_detector))(decoded)
        decoded = layers.Conv2DTranspose(2*self.num_detector, (3, 3), strides=(2, 2), activation='relu', padding='same')(decoded)
        decoded = layers.Conv2D(self.image_shape[-1], (3, 3), activation='relu', padding='same')(decoded)
        """
        """
        decoded = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='sigmoid', padding='same')(encoded)  # 24x24x128
        decoded = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='sigmoid', padding='same')(decoded)  # 48x48x64
        decoded = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='sigmoid', padding='same')(decoded)  # 96x96x32
        decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(decoded)  # 96x96x3

        decoded = layers.Conv2D(128, (3, 3), activation='sigmoid', padding='same')(encoded)  # 12x12x128
        decoded = layers.UpSampling2D((2, 2))(decoded)  # 24x24x128

        decoded = layers.Conv2D(64, (3, 3), activation='sigmoid', padding='same')(decoded)  # 24x24x64
        decoded = layers.UpSampling2D((2, 2))(decoded)  # 48x48x64

        decoded = layers.Conv2D(32, (3, 3), activation='sigmoid', padding='same')(decoded)  # 48x48x32
        decoded = layers.UpSampling2D((2, 2))(decoded)  # 96x96x32

        decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(decoded)  
        encoded = layers.Reshape((self.num_detector*144,))(encoded)  # 12*12=144
        decoded = layers.Dense(self.hidden_dims[1], activation='sigmoid')(encoded)
        decoded = layers.Dense(self.hidden_dims[0], activation='sigmoid')(decoded)
        decoded = layers.Dense(self.image_size, activation='sigmoid')(decoded)
        decoded = layers.Reshape(self.image_shape)(decoded)  # 12*12=144
        """
        # Downsampling
        encoded = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)  # 96x96x32
        encoded = layers.MaxPooling2D((2, 2), padding='same')(encoded)  # 48x48x32
        encoded = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)  # 48x48x64
        encoded = layers.MaxPooling2D((2, 2), padding='same')(encoded)  # 24x24x64
        encoded = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)  # 24x24x128
        encoded = layers.MaxPooling2D((2, 2), padding='same')(encoded)  # 12x12x128

        # Bottleneck
        encoded = layers.Conv2D(self.num_detector, (3, 3), activation='relu', padding='same')(encoded)  # 12x12xnum_detector

        # Decoder - progressively upsample with transposed convolutions
        decoded = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu', padding='same')(encoded)  # 24x24x128
        decoded = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(decoded)  # 48x48x64
        decoded = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(decoded)  # 96x96x32
        decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(decoded)  # 96x96x3
        # Create and compile the models
        encoder = keras.Model(input_img, encoded, name="encoder")
        encoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
        decoder = keras.Model(encoded, decoded, name="decoder")
        decoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
        autoencoder = keras.Model(input_img, decoded, name="autoencoder")
        autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
        #
        return autoencoder, encoder, decoder, history_dct
    
    def predict(self, image_arr: np.ndarray,
                predictor_type: str = "autoencoder") -> np.ndarray:
        """Generates reconstructed images from the autoencoder.

        Args:
            image_arr (np.ndarray): array of images
            predictor_type (str, optional):
                Type of predictor to use: "autoencoder", "encoder", or "decoder".
                Defaults to "autoencoder".
        Returns:
            np.ndarray: array of reconstructed images
        """
        if predictor_type in ["autoencoder", "encoder"]:
            if len(image_arr.shape) == 3:
                image_arr = np.expand_dims(image_arr, -1)
            if image_arr.shape[1:] != tuple(self.image_shape):
                raise ValueError(f"Input image shape {image_arr.shape[1:]} != image shape {self.image_shape}")
        elif image_arr[0].shape[0] != self.hidden_dims[-1]:
            raise ValueError(f"Input encoded shape {image_arr[0].shape} != encoding shape {self.hidden_dims[-1]}")
        reconstructed = super().predict(image_arr, predictor_type=predictor_type)
        return reconstructed

    @property
    def compression_factor(self) -> float:
        return float(self.image_size / self.hidden_dims[-1])