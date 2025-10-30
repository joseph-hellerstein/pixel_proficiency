# Defines a dense autoencoder using TensorFlow/Keras to


from src.abstract_autoencoder import AbstractAutoencoder  # type: ignore

import matplotlib.pyplot as plt  #  type: ignore
import numpy as np     # type: ignore
import src.constants as cn  # type: ignore
from tensorflow import keras  #  type: ignore
from typing import Tuple, List, Optional, Union

MAX_EPOCH = 1000

# FIXME: Constructor uses image_shape so can use flatten and reshape. Should abstract know image shape?
# Should encode dimensions exclude the input dimension?

class DenseAutoencoder(AbstractAutoencoder):
    def __init__(self,
            image_shape: Union[Tuple[int], List[int]],
            encode_dims: List[int],
            base_path: str=cn.MODEL_DIR,
            is_delete_serializations: bool=True,
            activation: str='sigmoid',
            is_early_stopping: bool = True,
            is_verbose: bool = False,
            dropout_rate: float = 0.4):
        """Initialize the dense autoencoder.

        Args:
            image_shape (Union[Tuple[int], List[int]]): Shape of the input images (height, width, channels)
            encode_dims (List[int]): List of integers representing the dimensions of the encoding layers.
                If the first element does not match the flattened image size, it will be prepended.
            base_path (str, optional): Base path for model serialization. Defaults to BASE_PATH.
            is_delete_serializations (bool, optional): Whether to delete existing serializations. Defaults to True.
            activation (str, optional): Activation function to use in the layers. Defaults to 'sigmoid'.
            is_early_stopping (bool, optional): Whether to use early stopping during training. Defaults to True.
            is_verbose (bool, optional): Whether to print verbose output during training. Defaults to False.
            dropout_rate (float, optional): Dropout rate to use in the layers. Defaults to 0.4.
        """
        self.image_shape = image_shape
        self.encode_dims = encode_dims
        first_dim = int(np.prod(image_shape))
        if encode_dims[0] != first_dim:
            encode_dims.insert(0, first_dim)
        self.num_hidden_layer = len(encode_dims) - 1
        self.dropout_rate = dropout_rate
        super().__init__(base_path=base_path, is_delete_serializations=is_delete_serializations,
                activation=activation, is_early_stopping=is_early_stopping, is_verbose=is_verbose)

    def context_dct(self) -> dict:
        # Describes the parameters used to build the model.
        context_dct = {
            'encode_dims': self.encode_dims,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
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
        input_img = keras.Input(shape=self.image_shape, name="input_layer")
        encoded = layers.Flatten()(input_img)
        # Encoder
        for idx in range(self.num_hidden_layer):
            encoded = layers.Dense(self.encode_dims[idx+1], activation=self.activation)(encoded) # type: ignore
            encoded = layers.Dropout(self.dropout_rate)(encoded)
        # Decoder
        decode_dims = list(self.encode_dims)
        decode_dims.reverse()
        decoded = None
        for idx in range(self.num_hidden_layer):
            if idx == 0:
                decoded = layers.Dense(decode_dims[1], activation=self.activation)(encoded) # type: ignore
                layers.Dropout(self.dropout_rate)
            else:
                decoded = layers.Dense(decode_dims[idx+1], activation=self.activation)(decoded) # type: ignore
                layers.Dropout(self.dropout_rate)
        decoded = layers.Reshape(self.image_shape)(decoded)
        # Create the autoencoder model
        autoencoder = keras.Model(input_img, decoded)
        # Create encoder model (for extracting encoded representations)
        encoder = keras.Model(input_img, encoded)
        # Create the decoder model
        # Create the decoder model by rebuilding layers to match decode_dims
        encoded_input = keras.Input(shape=(self.encode_dims[-1],), name='encoded input')
        x = encoded_input
        # decode_dims is reversed encode_dims; skip the first element (bottleneck) and expand back
        for units in decode_dims[1:]:
            x = layers.Dense(units, activation=self.activation)(x)
            x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Reshape(self.image_shape)(x)
        decoder = keras.Model(encoded_input, x)
        # Compile the autoencoder
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        return autoencoder, encoder, decoder, {}

    def fit(self, 
            x_train: np.ndarray,
            num_epoch: int,
            batch_size: int,
            validation_data: np.ndarray,
            is_verbose: Optional[bool]=None) -> None:
        """Trains the autoencoder.
        Args:
            x_train (np.ndarray): Training data (not flattened)
            num_epoch (int): Number of training epochs
            batch_size (int): Size of training batches
            validation_data (np.ndarray): Validation data (not flattened)
            verbose (int, optional): Verbosity level. Defaults to 1.
        """
        if is_verbose is None:
            is_verbose = self.is_verbose
        # Flatten each image to a vector
        #x_flat = self._flatten(x_train)
        #test_flat = self._flatten(validation_data)
        super().fit(x_train, num_epoch, batch_size, validation_data, is_verbose=is_verbose)

    
    def oldpredict(self, image_arr: np.ndarray,
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
    def doAnimalExperiments(cls, encode_dims: List[int], batch_size: int, base_path: str=cn.MODEL_DIR,
            num_epoch: int=MAX_EPOCH, is_verbose: bool = True,
            is_stopping_early: bool = True) -> None:
        dae = cls(cn.ANIMALS_IMAGE_SHAPE, encode_dims, is_delete_serializations=True,
                base_path=base_path, is_early_stopping=is_stopping_early, is_verbose=is_verbose)
        cls.runAnimalExperiment(dae, batch_size, dae.context_dct(), num_epoch=num_epoch)