# Defines an abstract base class for autoencoders.

import collections
import json
import matplotlib.pyplot as plt#  type: ignore
import numpy as np#  type: ignore
import os
import pandas as pd#  type: ignore
from tensorflow import keras #  type: ignore
from tensorflow.keras.datasets import mnist   # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from typing import Tuple, List, Any, Union, Optional


class AbstractAutoencoder(object):

    def __init__(self, base_path: str, is_delete_serializations: bool=True):
        """Initializes the abstract autoencoder.
        Args:
            base_path (str): Base path for model serialization.
            is_delete_serializations (bool, optional): Whether to delete existing serializations. Defaults to True.
        """
        self.base_path = base_path
        self.autoencoder, self.encoder, self.decoder, self.history_dct =  \
                self._build()
        if is_delete_serializations:
            self.deleteSerializations(base_path=base_path)
        else:
            self.deserializeAll(base_path=base_path)

    @property
    def compression_factor(self) -> float:
        raise NotImplementedError("Subclasses must implement compression_factor property")
    
    def deleteSerializations(self, base_path: Optional[str] = None) -> None:
        """Deletes the serialized models and history files.

        Args:
            base_path (str, optional): Base path for model serialization.
                                        If None, uses self.base_path. Defaults to None.
        """
        if base_path is None:
            base_path = self.base_path
        autoencoder_path, encoder_path, decoder_path, history_path  = \
                self._makeSerializationPaths(base_path)
        for path in [autoencoder_path, encoder_path, decoder_path, history_path]:
            if os.path.exists(path):
                os.remove(path)

    def _build(self) -> Tuple[keras.Model, keras.Model, keras.Model, dict]:
        """Builds the autoencoder, encoder, and decoder models.

        Returns:
            autoencoder, encoder, decoder models and history dictionary
        """
        raise NotImplementedError("Subclasses must implement _build method")

    def summarizeModel(self) -> None:
        self.autoencoder.summary()

    def fit(self, 
            x_train: np.ndarray,
            num_epoch: int,
            batch_size: int,
            validation_data: np.ndarray,
            verbose: int=1) -> None:
        # Train the autoencoder
        self.history = self.autoencoder.fit(x_train, x_train,
                epochs=num_epoch,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(validation_data, validation_data),
                verbose=verbose)
        self.history_dct = self.history.history
        self.serializeAll()
    
    def summary(self) -> None:
        # Prints a summary of the autoencoder model.
        self.autoencoder.summary()
    
    def predict(self, image_arr: np.ndarray) -> np.ndarray:
        """Generates reconstructed images from the autoencoder.

        Args:
            image_arr (np.ndarray): array of images
        Returns:
            np.ndarray: array of reconstructed images
        """
        raise NotImplementedError("Subclasses must implement predict method")

    def plot(self, x_test: np.ndarray) -> None:
        """Plots the training history and the reconstructed images.

        Args:
            x_test (np.ndarray): array of test images
        """
        # Generate predictions
        x_plot = self.predict(x_test)
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

    @staticmethod
    def serializeModel(path: str, model: keras.Model) -> None:
        """Serializes the model

        Args:
            path (str): Path to save the model.

        """
        model.save(path)
    
    @staticmethod
    def deserializeModel(path: str):
        """Deserializes the model

        Args:
            path (str): Path to save the model.

        """
        model = load_model(path, compile=False)
        model.compile(optimizer='adam', loss='mse')
        return model

    @staticmethod
    def deserializeHistory(path: str):
        """Deserializes the training history

        Args:
            path (str): Path to save the history.

        """
        with open(path, 'r') as f:
            history_dict = json.load(f)
        return history_dict
    
    @staticmethod
    def _makeSerializationPaths(base_path: str) -> Tuple[str, str, str, str]:
        """Generates paths for model and history serialization.

        Args:
            base_path (str): Base path for saving the autoencoder,
                                encoder, decoder, and history.

        """
        autoencoder_path = f"{base_path}_autoencoder.keras"
        encoder_path = f"{base_path}_encoder.keras"
        decoder_path = f"{base_path}_decoder.keras"
        history_path = f"{base_path}_history.json"
        return autoencoder_path, encoder_path, decoder_path, history_path

    def serializeAll(self, base_path: Optional[str] = None) -> None:
        """Serializes the model and training history

        Args:
            base_path (str): Path to save the model.

        """
        if base_path is None:
            base_path = self.base_path
        #
        autoencoder_path, encoder_path, decoder_path, history_path  = \
                self._makeSerializationPaths(base_path)
        self.serializeModel(autoencoder_path, self.autoencoder)
        self.serializeModel(encoder_path, self.encoder)
        self.serializeModel(decoder_path, self.decoder)
        with open(history_path, 'w') as f:
            json.dump(self.history_dct, f)

    DeserializeResult = collections.namedtuple(
        'DeserializeResult', ['autoencoder', 'encoder', 'decoder', 'history_dct'])
    @classmethod
    def deserializeAll(cls, base_path: str) -> Union[None, DeserializeResult]:
        """Deserializes the model and training history

        Args:
            base_path (str): Path to save the model.

        """
        autoencoder_path, encoder_path, decoder_path, history_path = \
            cls._makeSerializationPaths(base_path)
        if not (os.path.exists(autoencoder_path)
                and os.path.exists(encoder_path)
                and os.path.exists(decoder_path)
                and os.path.exists(history_path)):
            return None
        autoencoder = cls.deserializeModel(autoencoder_path)
        encoder = cls.deserializeModel(encoder_path)
        decoder = cls.deserializeModel(decoder_path)
        history_dct = cls.deserializeHistory(history_path)
        return cls.DeserializeResult(autoencoder, encoder, decoder, history_dct)