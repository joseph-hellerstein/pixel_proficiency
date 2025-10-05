# Defines a deterministic autoencoder using TensorFlow/Keras to
# compress and reconstruct images from the MNIST dataset.

import matplotlib.pyplot as plt#  type: ignore
import numpy as np#  type: ignore
import pandas as pd#  type: ignore
from tensorflow import keras #  type: ignore
from tensorflow.keras import layers #  type: ignore
from tensorflow.keras.datasets import mnist   # type: ignore
from typing import Tuple, List, cast


class DeterministicAutoencoder(object):
    def __init__(self, encode_dims: List[int]):
        """Initializes the deterministic autoencoder.

        Args:
            encode_dims (List[int]): _description_
        """
        self.encode_dims = encode_dims
        self.num_hidden_layers = len(encode_dims) - 1
        self.compression_factor = self.encode_dims[0] / self.encode_dims[-1]
        self.autoencoder, self.encoder, self.decoder = self._build()
        self.history = None # Filled in by fit() method

    def _build(self) -> Tuple[keras.Model, keras.Model, keras.Model]:
        """Builds the autoencoder, encoder, and decoder models.

        Returns:
            Tuple[keras.Model, keras.Model, keras.Model]: The autoencoder, encoder, and decoder models.
        """
        # Input layer
        input_img = keras.Input(shape=(self.encode_dims[0],))
        # Encoder
        encoded = None
        for idx, dim in enumerate(self.encode_dims[:-1]):
            if idx == 0:
                encoded = layers.Dense(self.encode_dims[1], activation='relu')(input_img) # type: ignore
            else:
                encoded = layers.Dense(encode_dims[idx+1], activation='relu')(encoded) # type: ignore
        # Decoder
        decode_dims = list(self.encode_dims)
        decode_dims.reverse()
        decoded = None
        for idx, dim in enumerate(decode_dims):
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
        for idim in range(self.num_hidden_layers, 1, -1):
            layers.append(self.autoencoder.layers[-idim])
        decoder_layer = layers[-1](encoded_input)
        for ilayer in range(1, self.num_hidden_layers):
            decoder_layer = layers[ilayer](decoder_layer)
        # Create decoder model
        decoder = keras.Model(encoded_input, decoder_layer)
        # Compile the autoencoder
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        return autoencoder, encoder, decoder

    def summarizeModel(self) -> None:
        self.autoencoder.summary()

    def fit(self, 
            X_train: np.ndarray,
            y_train: np.ndarray,
            epochs: int,
            batch_size: int,
            validation_data: Tuple[np.ndarray, np.ndarray],
            verbose: int=1) -> None:
        """Trains the autoencoder.
        Args:
            X_train (np.ndarray): Training data
            y_train (np.ndarray): Training labels
            epochs (int): Number of training epochs
            batch_size (int): Size of training batches
        """
        self.history = self.autoencoder.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True,
                validation_data=validation_data, verbose=verbose)
        
    def plot(self) -> None:
        keras.utils.plot_model(self.autoencoder, show_shapes=True, to_file='autoencoder.png') # type: ignore


        # Plot training history
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Generate predictions
        encoded_imgs = self.encoder.predict(X_TEST)
        decoded_imgs = self.autoencoder.predict(X_TEST)

        # Visualize results
        plt.subplot(1, 2, 2)
        n = 10  # Number of images to display
        plt.figure(figsize=(20, 4))

        for i in range(n):
            # Original images
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(X_TEST[i].reshape(28, 28), cmap='gray')
            plt.title("Original")
            plt.axis('off')

            # Reconstructed images
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
            plt.title("Reconstructed")
            plt.axis('off')

        plt.tight_layout()
        plt.show()

        # Print compression statistics
        print(f"\nOriginal image size: 784 pixels")
        print(f"Encoded representation size: {encoding_dim} values")
        print(f"Compression ratio: {784/encoding_dim:.1f}:1")
        print(f"Final training loss: {history.history['loss'][-1]:.4f}")
        print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

    def serialize(self, path: str) -> None:
        """Serializes the model

        Args:
            path (str): Path to save the model.

        """
        data = {
            'encode_dims': [self.encode_dims],
            'num_hidden_layers': [self.num_hidden_layers],
            'compression_factor': [self.compression_factor]
        }
        self.autoencoder.save(path)
        # encoder.save('mnist_encoder.h5')
        # decoder.save('mnist_decoder.h5')