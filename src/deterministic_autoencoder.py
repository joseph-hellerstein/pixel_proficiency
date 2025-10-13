# Defines a deterministic autoencoder using TensorFlow/Keras to
# compress and reconstruct images from the MNIST dataset.

"""
Issues
1. Losses are much higher than in notebook
"""

import matplotlib.pyplot as plt#  type: ignore
import numpy as np#  type: ignore
import pandas as pd#  type: ignore
from tensorflow import keras #  type: ignore
from tensorflow.keras.datasets import mnist   # type: ignore
from typing import Tuple, List, Any

NORMALIZATION_FACTOR = 255.0


class DeterministicAutoencoder(object):
    def __init__(self, encode_dims: List[int]):
        """Initializes the deterministic autoencoder.

        Args:
            encode_dims (List[int]): _description_
        """
        self.encode_dims = encode_dims
        self.num_hidden_layer = len(encode_dims) - 1
        self.compression_factor = self.encode_dims[0] / self.encode_dims[-1]
        self.autoencoder, self.encoder, self.decoder = self._build()
        self.history: Any = None # Filled in by fit() method
        self.image_shape: Any = None # Filled in by fit() method

    def _build(self) -> Tuple[keras.Model, keras.Model, keras.Model]:
        """Builds the autoencoder, encoder, and decoder models.

        Returns:
            Tuple[keras.Model, keras.Model, keras.Model]: The autoencoder, encoder, and decoder models.
        """
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
        return autoencoder, encoder, decoder

    def summarizeModel(self) -> None:
        self.autoencoder.summary()

    def _flatten(self, arr: np.ndarray, normalization_factor: float=NORMALIZATION_FACTOR) -> np.ndarray:
        """Flattens the input images and normalizes pixel values to [0, 1].
        Args:
            x (np.ndarray): Input images (not flattened)

        Returns:
            np.ndarray: Flattened images
        """
        self.image_shape = np.shape(arr[0])
        size = np.prod(self.image_shape)
        num_image = np.shape(arr)[0]
        x_flat = arr.reshape(num_image, size).astype('float32')/normalization_factor
        return x_flat
    
    def _unflatten(self, arr: np.ndarray, mult_factor: float=1.0) -> np.ndarray:
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
        return result.astype('float32')*mult_factor

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
        """
        # Flatten each image to a vector
        x_flat = self._flatten(x_train)
        test_flat = self._flatten(validation_data)
        self.history = self.autoencoder.fit(x_flat, x_flat, epochs=num_epoch, batch_size=batch_size, shuffle=True,
                validation_data=(test_flat, test_flat), verbose=verbose)

    def plot(self, x_test: np.ndarray) -> None:
        """Plots the training history and the reconstructed images.

        Args:
            x_test (np.ndarray): array of test images (not flattened)
        """
        """ # Plot training history
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()"""
        # Generate predictions
        x_flatten = self._flatten(x_test)
        x_predict = self.autoencoder.predict(x_flatten)
        x_plot = self._unflatten(x_predict)
        # Visualize results
        num_display = 10  # Number of images to display
        plt.figure(figsize=(20, 4))
        for i in range(num_display):
            # Original images
            ax = plt.subplot(2, num_display, i + 1)
            ax.imshow(x_test[i], cmap='gray')
            ax.set_title("Original")
            plt.axis('off')
            # Reconstructed images
            ax = plt.subplot(2, num_display, i + 1 + num_display)
            ax.imshow(x_plot[i], cmap='gray')
            ax.set_title("Reconstructed")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

        # Print compression statistics
        """ print(f"\nOriginal image size: 784 pixels")
        print(f"Encoded representation size: {encoding_dim} values")
        print(f"Compression ratio: {784/encoding_dim:.1f}:1")
        print(f"Final training loss: {history.history['loss'][-1]:.4f}")
        print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}") """

    def serialize(self, path: str) -> None:
        """Serializes the model

        Args:
            path (str): Path to save the model.

        """
        data = {
            'encode_dims': [self.encode_dims],
            'num_hidden_layers': [self.num_hidden_layer],
            'compression_factor': [self.compression_factor]
        }
        self.autoencoder.save(path)
        # encoder.save('mnist_encoder.h5')
        # decoder.save('mnist_decoder.h5')