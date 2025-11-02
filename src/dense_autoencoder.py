# Defines a dense autoencoder using TensorFlow/Keras to


from src.abstract_autoencoder import AbstractAutoencoder  # type: ignore

import matplotlib.pyplot as plt  #  type: ignore
import numpy as np     # type: ignore
import src.constants as cn  # type: ignore
from tensorflow import keras  #  type: ignore
from typing import Tuple, List, Optional, Union

MAX_EPOCH = 1000


class DenseAutoencoder(AbstractAutoencoder):
    def __init__(self,
            image_shape: Optional[Union[Tuple[int], List[int]]]=None,
            encode_dims: Optional[List[int]]=None,
            base_path: str=cn.MODEL_DIR,
            is_delete_serialization: bool=True,
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
        # Check for null constructor for deserialization
        if image_shape is None or encode_dims is None:
            return
        # State specific to DenseAutoencoder
        self.encode_dims = encode_dims
        #
        first_dim = int(np.prod(image_shape))
        if encode_dims[0] != first_dim:
            encode_dims.insert(0, first_dim)
        self.num_hidden_layer = len(encode_dims) - 1
        super().__init__(image_shape, base_path=base_path, is_delete_serialization=is_delete_serialization,
                activation=activation, is_early_stopping=is_early_stopping, is_verbose=is_verbose,
                dropout_rate=dropout_rate)

    #### Internal methods
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
    
    ###### API
    
    @property
    def context_dct(self) -> dict:
        # Describes the parameters used to build the model.
        context_dct = {
            'encode_dims': self.encode_dims,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
        }
        return context_dct

    @property
    def compression_factor(self) -> float:
        # Calculates the ratio between the original image size and the bottleneck size.
        # The bottleneck is determined by the last filter size and the downsampling factor.
        return self.encode_dims[0]/ self.encode_dims[-1]

    @classmethod
    def doAnimalExperiments(cls, encode_dims: List[int], batch_size: int, base_path: str=cn.MODEL_DIR,
            num_epoch: int=MAX_EPOCH, is_verbose: bool = True,
            is_early_stopping: bool = True,
            ) -> AbstractAutoencoder.ExperimentResult:
        dae = cls(cn.ANIMALS_IMAGE_SHAPE, encode_dims, is_delete_serialization=True,
                base_path=base_path, is_early_stopping=is_early_stopping, is_verbose=is_verbose)
        return dae.runAnimalExperiment(batch_size=batch_size, num_epoch=num_epoch)

    def serialize(self, base_path: Optional[str] = None, **kwargs) -> None:
        """Serializes the model and training history

        Args:
            base_path (str): Path to save the model.
        """
        if len(kwargs) > 0:
            raise ValueError("DenseAutoencoder.serialize does not accept additional keyword arguments.")
        super().serialize(
            base_path=base_path,
            image_shape=self.image_shape,
            encode_dims=self.encode_dims,
            is_delete_serialization=self.is_delete_serialization,
            dropout_rate=self.dropout_rate
        )