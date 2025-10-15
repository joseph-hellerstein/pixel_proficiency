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
X_TRAIN, LABEL_TRAIN, X_TEST, LABEL_TEST = util.getPklMNIST()
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
        result = self.deserializeAll(BASE_PATH)
        history_dct: dict = {}
        if result is not None:
            print("Loading existing model and history...")
            return result.autoencoder, result.encoder, result.decoder, result.history_dct  # type: ignore
        # No existing model. Must build a new one.
        input_img = keras.Input(shape=self.image_shape)  # e.g., (28, 28, 1)
        # Encoder
        encoded = layers.Conv2D(self.num_detector, (3, 3), activation='relu', padding='same')(input_img)  # 28×28×num_detector
        encoded = layers.Reshape((self.num_detector*self.image_size,))(encoded)
        encoded = layers.Dense(self.hidden_dims[0], activation='relu')(encoded)
        encoded = layers.Dense(self.hidden_dims[1], activation='relu')(encoded)
        encoded = layers.Dense(self.hidden_dims[2], activation='relu')(encoded)
        # Decoder
        decoded = layers.Dense(self.hidden_dims[1], activation='relu')(encoded)
        decoded = layers.Dense(self.hidden_dims[0], activation='relu')(decoded)
        decoded = layers.Dense(self.image_size*self.num_detector, activation='relu')(decoded)
        decoded = layers.Reshape((*self.image_shape[:-1], self.num_detector))(decoded)
        decoded = layers.Conv2D(1, (3, 3), activation='relu', padding='same')(decoded)  # 28×28×1
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