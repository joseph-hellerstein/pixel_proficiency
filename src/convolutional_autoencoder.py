'''Autoencoder that uses convolutional layers for autoencoding images. Images are 2D with a channel dimension.'''

from src.abstract_autoencoder import AbstractAutoencoder  # type: ignore
from src import constants as cn  # type: ignore

import numpy as np
from tensorflow import keras # type: ignore
from tensorflow.keras import layers # type: ignore
from typing import List, Tuple, Union, Any, Optional


class ConvolutionalAutoencoder(AbstractAutoencoder):

    def __init__(self,
            image_shape: Optional[Union[Tuple[int], List[int]]]=None,
            filter_sizes: Optional[List[int]] = None,
            base_path: str=cn.MODEL_DIR,
            is_delete_serializations: bool=True,
            activation: str='sigmoid',
            is_early_stopping: bool = True,
            is_verbose: bool = False,
            dropout_rate: float = 0.4):
        """Initializes the convolutional autoencoder.

        Args:
            image_shape (List[int]): Shape of the input images (height, width, channels)
            filter_sizes (List[int]): dimensions of the filters. Last one is the filter for the bottleneck
            base_path (str, optional): Base path for model serialization. Defaults to BASE_PATH.
            is_delete_serializations (bool, optional): Whether to delete existing serializations. Defaults to
            activation (str, optional): Activation function to use in the layers. Defaults to 'sigmoid'.
            is_early_stopping (bool, optional): Whether to use early stopping during training. Defaults to True.
            is_verbose (bool, optional): Whether to print verbose output during training. Defaults to False.
            dropout_rate (float, optional): Dropout rate to use in the layers. Defaults to 0.4.
        """
        # Check for null constructor for deserialization
        if image_shape is None:
            return
        # Handle default
        if filter_sizes is None:
            filter_sizes = [32, 64, 128, 32]
        # ConvolutionalAutoencoder specific state
        self.filter_sizes = filter_sizes
        # Configure image shape
        if len(image_shape) < 2 or len(image_shape) > 3:
            raise ValueError("image_shape must be of length 2 or 3")
        if len(image_shape) == 2:
            image_shape = [image_shape[0], image_shape[1], 1]
        if len(filter_sizes) == 0:
            raise ValueError("filter_sizes must have at least one element")
        self.image_size = np.prod(image_shape)
        super().__init__(image_shape, base_path=base_path, is_delete_serialization=is_delete_serializations,
                activation=activation, is_early_stopping=is_early_stopping, is_verbose=is_verbose,
                dropout_rate=dropout_rate)

    #### Internal methods
    def _build(self) -> Tuple[keras.Model, keras.Model, keras.Model, dict]:
        """Builds the convolutional autoencoder model.

        Returns:
            keras.Model, keras.Model, keras.Model, dict: The autoencoder, encoder, decoder models, and history dictionary.
        """
        history_dct: dict = {}
        input_img = keras.Input(shape=self.image_shape)
        """ # Validate the network implied by filter_sizes
        reduction_factor = self._calculateSizeReductionFromDownsampling()
        if (self.image_shape[0]*self.image_shape[1]) % reduction_factor != 0:
            raise ValueError(f"Image size {self.image_shape[0]}x{self.image_shape[1]} is not compatible with "
                    f"the reduction factor {reduction_factor} implied by filter_sizes {self.filter_sizes}") """
        # Downsampling
        encoded: Any = None
        for idx, filter_size in enumerate(self.filter_sizes):
            if idx == 0:
                input = input_img
            else:
                input = encoded
            encoded = layers.Conv2D(filter_size, (3, 3), activation=self.activation, padding='same')(input)
            layers.Dropout(self.dropout_rate)
            if idx < len(self.filter_sizes) - 1:
                encoded = layers.MaxPooling2D((2, 2), padding='same')(encoded)
        # Upsampling
        decoded: Any = None
        decoded_sizes = self.filter_sizes[:-1][::-1]  # reverse order except last
        for idx, filter_size in enumerate(decoded_sizes):
            if idx == 0:
                input = encoded
            else:
                input = decoded
            decoded = layers.Conv2DTranspose(filter_size, (3, 3), strides=(2, 2), activation=self.activation,
                    padding='same')(input)
            layers.Dropout(self.dropout_rate)
        decoded = layers.Conv2D(self.image_shape[-1], (3, 3), activation=self.activation, padding='same')(decoded)
        # Create and compile the models
        encoder = keras.Model(input_img, encoded, name="encoder")
        encoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
        decoder = keras.Model(encoded, decoded, name="decoder")
        decoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
        autoencoder = keras.Model(input_img, decoded, name="autoencoder")
        autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
        #
        return autoencoder, encoder, decoder, history_dct
    
    def _calculateSizeReductionFromDownsampling(self) -> int:
        reduction_factor = 4**(len(self.filter_sizes) - 1)
        return reduction_factor

    ###### API
    
    @property
    def compression_factor(self) -> float:
        # Calculates the ratio between the original image size and the bottleneck size.
        # The bottleneck is determined by the last filter size and the downsampling factor.
        reduction_factor = self._calculateSizeReductionFromDownsampling()
        return float(self.image_size / reduction_factor) * self.filter_sizes[-1]
    
    @property
    def context_dct(self) -> dict:
        # Describes the parameters used to build the model.
        context_dct = {
            'image_shape': self.image_shape,
            'filter_sizes': self.filter_sizes,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
        }
        return context_dct

    @classmethod 
    def doAnimalExperiments(cls, filter_sizes: List[int], batch_size: int, base_path: str=cn.MODEL_DIR,
            num_epoch: int=1000, is_stopping_early: bool = True, is_verbose: bool = True,
            )-> AbstractAutoencoder.ExperimentResult:
        cae = cls(cn.ANIMALS_IMAGE_SHAPE, filter_sizes, is_delete_serializations=True,
                base_path=base_path, is_early_stopping=is_stopping_early,
                is_verbose=is_verbose)
        return cae.runAnimalExperiment(batch_size=batch_size, num_epoch=num_epoch)

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
        reconstructed = super().predict(image_arr, predictor_type=predictor_type)
        return reconstructed

    def serialize(self, base_path: Optional[str] = None, **kwargs) -> None:
        """Serializes the model and training history

        Args:
            base_path (str): Path to save the model.
        """
        if len(kwargs) > 0:
            raise ValueError("DenseAutoencoder.serialize does not accept additional keyword arguments.")
        super().serialize(base_path=base_path,
            filter_sizes=self.filter_sizes,
        )