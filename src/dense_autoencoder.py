# Defines a dense autoencoder using TensorFlow/Keras to
# compress and reconstruct images from the MNIST dataset.

"""
Issues
1. Losses are much higher than in notebook
"""

from src.abstract_autoencoder import AbstractAutoencoder  # type: ignore

import matplotlib.pyplot as plt#  type: ignore
import numpy as np#  type: ignore
import src.constants as cn  # type: ignore
import os
import pandas as pd#  type: ignore
from tensorflow import keras #  type: ignore
from tensorflow.keras.datasets import mnist   # type: ignore
from typing import Tuple, List, Any

# FIXME: fit must flatten


class DenseAutoencoder(AbstractAutoencoder):
    def __init__(self, encode_dims: List[int], base_path: str=cn.MODEL_DIR,
            is_delete_serializations: bool=True):
        """Initializes the dense autoencoder.

        Args:
            encode_dims (List[int]): _description_
        """
        self.encode_dims = encode_dims
        self.num_hidden_layer = len(encode_dims) - 1
        self.autoencoder, self.encoder, self.decoder, self.history_dct = self._build()
        super().__init__(base_path=base_path, is_delete_serializations=is_delete_serializations)

    def context_dct(self) -> dict:
        # Describes the parameters used to build the model.
        context_dct = {
            'encode_dims': self.encode_dims,
        }
        return context_dct

    def _build(self) -> Tuple[keras.Model, keras.Model, keras.Model, dict]:
        """Builds the autoencoder, encoder, and decoder models.

        Returns:
            Tuple[keras.Model, keras.Model, keras.Model]: The autoencoder, encoder, and decoder models.
        """
        # Keep this here since layers is also used as a variable name below
        from tensorflow.keras import layers #  type: ignore
        # Input layer
        input_img = keras.Input(shape=(self.encode_dims[0],))
        # Encoder
        encoded = None
        for idx in range(self.num_hidden_layer):
            if idx == 0:
                encoded = layers.Dense(self.encode_dims[1], activation='relu')(input_img) # type: ignore
            else:
                encoded = layers.Dense(self.encode_dims[idx+1], activation='relu')(encoded) # type: ignore
        # Decoder
        decode_dims = list(self.encode_dims)
        decode_dims.reverse()
        decoded = None
        for idx in range(self.num_hidden_layer):
            if idx == 0:
                decoded = layers.Dense(decode_dims[1], activation='relu')(encoded) # type: ignore
            else:
                decoded = layers.Dense(decode_dims[idx+1], activation='relu')(decoded) # type: ignore
        # Create the autoencoder model
        autoencoder = keras.Model(input_img, decoded)
        # Create encoder model (for extracting encoded representations)
        encoder = keras.Model(input_img, encoded)
        # Create the decoder model
        encoded_input = keras.Input(shape=(self.encode_dims[-1],))
        layers = []
        for idim in range(self.num_hidden_layer, 1, -1):
            layers.append(autoencoder.layers[-idim])
        decoder_layers = list(autoencoder.layers)[-(self.num_hidden_layer):]
        decoder_layer = encoded_input
        for layer in decoder_layers:
            decoder_layer = layer(decoder_layer)
        # Create decoder model
        decoder = keras.Model(encoded_input, decoder_layer)
        # Compile the autoencoder
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        return autoencoder, encoder, decoder, {}

    def _flatten(self, arr: np.ndarray) -> np.ndarray:
        """Flattens the input images.
        Args:
            x (np.ndarray): Input images (not flattened)

        Returns:
            np.ndarray: Flattened images
        """
        self.image_shape = np.shape(arr[0])
        size = np.prod(self.image_shape)
        num_image = np.shape(arr)[0]
        x_flat = arr.reshape(num_image, size).astype('float32')
        return x_flat
    
    def _unflatten(self, arr: np.ndarray) -> np.ndarray:
        """Flattens the input images.
        Args:
            x (np.ndarray): Input images (not flattened)

        Returns:
            np.ndarray: Flattened images
        """
        result_shape = np.zeros(len(self.image_shape) + 1)
        result_shape[0] = np.shape(arr)[0]
        result_shape[1:] = self.image_shape
        result = np.reshape(arr, result_shape.astype(int))
        return result.astype('float32')

    def fit(self, 
            x_train: np.ndarray,
            num_epoch: int,
            batch_size: int,
            validation_data: np.ndarray,
            verbose: int=1) -> None:
        """Trains the autoencoder.
        Args:
            x_train (np.ndarray): Training data (not flattened)
            num_epoch (int): Number of training epochs
            batch_size (int): Size of training batches
            validation_data (np.ndarray): Validation data (not flattened)
            verbose (int, optional): Verbosity level. Defaults to 1.
        """
        # Flatten each image to a vector
        x_flat = self._flatten(x_train)
        test_flat = self._flatten(validation_data)
        super().fit(x_flat, num_epoch, batch_size, test_flat, verbose=verbose)

    
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
            image_arr = self._flatten(image_arr)
        predicted_arr = super().predict(image_arr, predictor_type=predictor_type)
        if predictor_type in ["autoencoder", "decoder"]:
            reconstructed_arr = self._unflatten(predicted_arr)
        else:
            reconstructed_arr = predicted_arr
        return reconstructed_arr
    
    @property
    def compression_factor(self) -> float:
        # Calculates the ratio between the original image size and the bottleneck size.
        # The bottleneck is determined by the last filter size and the downsampling factor.
        return self.encode_dims[0]/ self.encode_dims[-1]

    @classmethod
    def doAnimalExperiments(cls, encode_dims: List[int], batch_size: int, base_path: str=cn.MODEL_DIR):
        dae = cls(encode_dims, is_delete_serializations=True,
                base_path=base_path)
        cls.runAnimalExperiment(dae, batch_size, dae.context_dct())