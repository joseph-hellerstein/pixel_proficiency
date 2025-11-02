# Defines an abstract base class for autoencoders.
import src.constants as cn  # type: ignore
import src.util as util  # type: ignore

import collections
from deepdiff import DeepDiff
import json
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau #  type: ignore
import matplotlib.pyplot as plt#  type: ignore
import matplotlib.cm as cm
import numpy as np#  type: ignore
import os
import pandas as pd#  type: ignore
from tensorflow import keras #  type: ignore
from tensorflow.keras.models import load_model # type: ignore
from typing import Tuple, List, Optional, Union


MAX_EPOCH = 1000   # Maximum number of epochs for training
METADATA_FILE = "metadata.json"

DeserializeResult = collections.namedtuple(
        'DeserializeResult', ['autoencoder', 'encoder', 'decoder', 'history_dct'])

class AbstractAutoencoder(object):
    ExperimentResult = collections.namedtuple('ExperimentResult',
            ['batch_size', 'history', 'base_path', 'context_str', 'autoencoder'])

    def __init__(self,
            image_shape: Union[Tuple[int], List[int]],
            base_path: str,
            is_delete_serialization: bool=True,
            activation: str='sigmoid',
            is_early_stopping: bool = True,
            is_verbose: bool = False,
            dropout_rate: float=0.4,
            batch_size: int = 128,
            ):
        """Initializes the abstract autoencoder.
        Args:
            base_path (str): Base path for model serialization.
            is_delete_serializations (bool, optional): Whether to delete existing serializations. Defaults to True.
            self.activation (str, optional): Activation function to use in the model. Defaults to 'sigmoid'.
            is_early_stopping (bool, optional): Whether to use early stopping during training. Defaults to True.
            is_verbose (bool, optional): Whether to print verbose messages. Defaults to True.
            dropout_rate (float, optional): Dropout rate to use in the model. Defaults to 0.4.
            batch_size (int, optional): Batch size to use during training. Defaults to 128.
        """
        # Common state for all autoencoders
        self.image_shape = image_shape
        self.base_path = base_path
        self.is_delete_serialization = is_delete_serialization
        self.activation = activation
        self.is_early_stopping = is_early_stopping
        self.is_verbose = is_verbose
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        #
        self.png_pat = os.path.join(base_path, "%s.png") # pattern for naming plots
        self.autoencoder, self.encoder, self.decoder, self.history_dct =  \
                self._deserializeAllModels(base_path=base_path)
        self.is_fit = False
        if self.is_delete_serialization or self.autoencoder is None:
            self.print("...deleting existing serializations (if any) and rebuilding the model.")
            self.deleteSerializations(base_path=base_path)
            self.autoencoder, self.encoder, self.decoder, self.history_dct =  \
                    self._build()
        else:
            self.print("...loaded existing serializations.")
            self.is_fit = True

    @property
    def compression_factor(self) -> float:
        raise NotImplementedError("Subclasses must implement compression_factor property")
    
    @property
    def context_dct(self) -> dict:
        # Describes the parameters used to build the model.
        raise NotImplementedError("Subclasses must implement context_dct property")
    
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

    def fit(self, 
            x_train: np.ndarray,
            num_epoch: int,
            validation_data: np.ndarray,
            batch_size: Optional[int]=None,
            is_verbose: Optional[bool]=None,
            ) -> None:
        """Fits the autoencoder model to the training data. Checkpoint the model.
        Normalizes the data to [0, 1] before training.

        Args:
            x_train (np.ndarray): Training data.
            num_epoch (int): Number of epochs to train.
            batch_size (int): Batch size for training.
            validation_data (np.ndarray): Validation data.
            verbose (int, optional): Verbosity level. Defaults to 1.
        """
        # Default parameter values
        if batch_size is None:
            batch_size = self.batch_size
        if is_verbose is None:
            is_verbose = self.is_verbose
        if is_verbose:
            verbose = 1
        else:
            verbose = 0
        # Train the autoencoder
        if self.is_fit:
            print("Model is already fit. Skipping training.")
            return
        # Create a ModelCheckpoint callback
        callbacks: list = [
            ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,        # reduce LR by half
                    patience=5,        # after 5 epochs without improvement
                    min_lr=1e-7,
                    verbose=verbose
                )
        ]
        if self.is_early_stopping:
            callbacks.extend(
                [ModelCheckpoint(
                    os.path.join(cn.MODEL_DIR, 'best_autoencoder.keras'),
                    monitor='val_loss',
                    save_best_only=True,
                    mode='min',
                    verbose=verbose,
                ),
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    # Use min_delta 0 since we don't know the right units to detect improvement
                    min_delta=0,  # type: ignore
                    verbose=verbose
                ),
                ] 
            )
        # Normalize the data
        x_train_nrml = self.normalizeImages(x_train)
        x_validation_nrml = self.normalizeImages(validation_data)
        self.history = self.autoencoder.fit(
            x_train_nrml, x_train_nrml,
            epochs=num_epoch,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(x_validation_nrml, x_validation_nrml),
            callbacks=callbacks,
            verbose=verbose
        )
        self.history_dct = self.history.history
        # Load the best model after training
        self._serializeAllModels()
    
    def summarize(self) -> None:
        # Prints a summary of the autoencoder model.
        if self.is_verbose:
            self.autoencoder.summary()
    
    def predict(self, image_arr: np.ndarray,
            predictor_type: str = "autoencoder") -> np.ndarray:
        """Generates reconstructed images from the autoencoder.

        Args:
            image_arr (np.ndarray): array of images structured as expected by the model
            predictor_type (str, optional):
                Type of predictor to use: "autoencoder", "encoder", or "decoder".
                Defaults to "autoencoder".
        Returns:
            np.ndarray: array of reconstructed images
        """
        # Prepare the input
        if predictor_type in ["autoencoder", "encoder"]:
            image_arr_nrml = self.normalizeImages(image_arr)
        else:
            image_arr_nrml = image_arr
        # Make predictions
        if predictor_type == "autoencoder":
            predicted_nrml = self.autoencoder.predict(image_arr_nrml)
        elif predictor_type == "encoder":
            predicted_nrml = self.encoder.predict(image_arr_nrml)
        elif predictor_type == "decoder":
            predicted_nrml = self.decoder.predict(image_arr_nrml)
        else:
            raise ValueError(f"Unknown predictor type: {predictor_type}")
        # Prepare the output
        if predictor_type in ["autoencoder", "decoder"]:
            predicted_arr = self.denormalizeImages(predicted_nrml)
        else:
            predicted_arr = predicted_nrml
        #
        return predicted_arr

    def plot(self, x_original_arr: np.ndarray,
            x_predicted_arr: Optional[np.ndarray]=None,
            png_path: Optional[str]=None,
            is_plot:bool = True) -> None:
        """Plots the training history and the reconstructed images.

        Args:
            x_original_arr (np.ndarray): array of test images
            x_predicted_arr (np.ndarray, optional): array of predicted images.
                                                    If None, predictions are generated from x_original_arr.
                                                    Defaults to None.
            png_path (str, optional): Path to save the plot. If None, the plot is not saved. Defaults to None.
        """
        # Generate predictions
        if x_predicted_arr is None:
            x_predicted_arr = self.predict(x_original_arr)
        else:
            x_predicted_arr = x_predicted_arr.copy()
        x_predicted_arr = x_predicted_arr.astype('uint8')
        # Visualize results
        fig, axes = plt.subplots(2, 10, figsize=(20, 4))
        for i in range(10):
            # Original images
            ax = axes[0, i]
            ax.imshow(x_original_arr[i], cmap='gray')
            ax.set_title("Original")
            plt.axis('off')
            # Reconstructed images
            ax = axes[1, i]
            ax.imshow(x_predicted_arr[i], cmap='gray')
            ax.set_title("Reconstructed")
            plt.axis('off')
        plt.tight_layout()
        if png_path is not None:
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
        if is_plot:
            plt.show()
        else:
            plt.close()

        # Print compression statistics
        """ print(f"\nOriginal image size: 784 pixels")
        print(f"Encoded representation size: {encoding_dim} values")
        print(f"Compression ratio: {784/encoding_dim:.1f}:1")
        print(f"Final training loss: {history.history['loss'][-1]:.4f}")
        print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}") """

    @staticmethod
    def _serializeModel(path: str, model: keras.Model) -> None:
        """Serializes the model

        Args:
            path (str): Path to save the model.

        """
        model.save(path)
    
    @staticmethod
    def _deserializeModel(path: str):
        """Deserializes the model

        Args:
            path (str): Path to save the model.

        """
        model = load_model(path, compile=False)
        model.compile(optimizer='adam', loss='mse')
        return model

    @staticmethod
    def _deserializeHistory(path: str):
        """Deserializes the training history

        Args:
            path (str): Path to save the history.

        """
        with open(path, 'r') as f:
            history_dict = json.load(f)
        return history_dict
    
    @staticmethod
    def _makeSerializationPaths(base_path: str, serialize_dir=cn.MODEL_DIR) -> Tuple[str, str, str, str]:
        """Generates paths for model and history serialization.

        Args:
            base_path (str): Base path for saving the autoencoder,
                                encoder, decoder, and history.
            serialize_dir (str): Directory to save the models.
        Returns:
            Tuple[str, str, str, str]: Paths for autoencoder, encoder, decoder, and history.

        """
        autoencoder_path = f"{base_path}_autoencoder.keras"
        encoder_path = f"{base_path}_encoder.keras"
        decoder_path = f"{base_path}_decoder.keras"
        history_path = f"{base_path}_history.json"
        return autoencoder_path, encoder_path, decoder_path, history_path

    def _serializeAllModels(self, base_path: Optional[str] = None) -> None:
        """Serializes the model and training history

        Args:
            base_path (str): Path to save the model.

        """
        if base_path is None:
            base_path = self.base_path
        #
        autoencoder_path, encoder_path, decoder_path, history_path  = \
                self._makeSerializationPaths(base_path)
        self._serializeModel(autoencoder_path, self.autoencoder)
        self._serializeModel(encoder_path, self.encoder)
        self._serializeModel(decoder_path, self.decoder)
        with open(history_path, 'w') as f:
            json.dump(self.history_dct, f)
    
    @classmethod
    def _deserializeAllModels(cls, base_path: str) -> DeserializeResult:
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
            return DeserializeResult(None, None, None, dict())
        autoencoder = cls._deserializeModel(autoencoder_path)
        encoder = cls._deserializeModel(encoder_path)
        decoder = cls._deserializeModel(decoder_path)
        history_dct = cls._deserializeHistory(history_path)
        return DeserializeResult(autoencoder, encoder, decoder, history_dct)

    def makeAnimalBasePath(self, batch_size: Optional[int]=None) -> str:
        """Creates a base path for animal experiments.

        Args:
            batch_size (int): The batch size used for training.

        Returns:
            str: The base path for the animal experiments.
        """
        if batch_size is None:
            batch_size = self.batch_size
        # Creates a base path for animal experiments.
        full_context_dct = dict(self.context_dct)
        full_context_dct['batch_size'] = batch_size
        full_context_dct['autoencoder'] = str(self.__class__).split('.')[-1][:-2]
        base_path = os.path.join(self.base_path, "animals_" + str(full_context_dct))
        # Make base_path a file string
        for char in "'{}[] ":
            base_path = base_path.replace(char, "")
        base_path = base_path.replace(":", "-")
        base_path = base_path.replace(",", "__")
        # Make base_path a file string
        base_path = base_path.replace(" ", "_")
        return base_path
    
    def serialize(self, base_path: Optional[str] = None, **kwargs) -> None:
        """Serializes the model and training history

        Args:
            base_path (str): Path to save the model.
            kwargs: Additional keyword arguments for serialization.

        """
        if base_path is None:
            base_path = self.base_path
        self._serializeAllModels(base_path=base_path)
        metadata_path = f"{base_path}_{METADATA_FILE}"
        metadata_dct = {
            'image_shape': self.image_shape,
            'base_path': self.base_path,
            'is_delete_serialization': self.is_delete_serialization,
            'activation': self.activation,
            'is_early_stopping': self.is_early_stopping,
            'is_verbose': self.is_verbose,
            'dropout_rate': self.dropout_rate,
            # computed
            'png_pat': self.png_pat,
            'is_fit': self.is_fit,
        }
        metadata_dct.update(kwargs)
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dct, f)

    @classmethod
    def deserialize(cls, base_path: str) -> 'AbstractAutoencoder':
        """Deserializes the model and training history

        Args:
            base_path (str): Path to save the model.

        """
        context_path = f"{base_path}_{METADATA_FILE}"
        if not os.path.exists(context_path):
            raise ValueError(f"Context file does not exist: {context_path}")
        with open(context_path, 'r') as f:
            context_dct = json.load(f)
        autoencoder = cls()  # type: ignore
        for key, value in context_dct.items():
            setattr(autoencoder, key, value)
        # Models
        autoencoder.autoencoder, autoencoder.encoder, autoencoder.decoder, autoencoder.history_dct =  \
                cls._deserializeAllModels(base_path=base_path)
        # Return
        return autoencoder

    def normalizeImages(self, image_arr: np.ndarray) -> np.ndarray:
        """Normalizes image pixel values to the range [0, 1].

        Args:
            image_arr (np.ndarray): array of images

        Returns:
            np.ndarray: normalized array of images
        """
        return image_arr.astype('float32') / 255.0

    def denormalizeImages(self, image_arr: np.ndarray) -> np.ndarray:
        """Denormalizes image pixel values from the range [0, 1] to [0, 255].

        Args:
            image_arr (np.ndarray): array of normalized images

        Returns:
            np.ndarray: denormalized array of images
        """
        return (image_arr * 255.0).astype(np.uint8)

    def plotEncoded(self, x_test: np.ndarray, x_label: np.ndarray,
            max_num_point: int= 100,
            lim: Optional[List[float]] = None,
            is_plot: bool = True) -> None:
        """Plots the encoded labels in 2D space. If the encoded
        space has more than 2 dimensions, just uses the first two dimensions

        Args:
            x_test (np.ndarray): array of test images
            x_label (np.ndarray): array of test labels
            max_num_point (int, optional): Maximum number of points to plot. Defaults to 100.
        """
        colors = cm.viridis(np.linspace(0, 1, np.max(x_label) + 1)) # type: ignore
        labels = np.sort(np.unique(x_label))
        if x_test.shape[0] > max_num_point:
            perm_arr = np.random.permutation(x_test.shape[0])
            perm_arr = perm_arr[0:max_num_point]
        else:
            perm_arr = np.arange(x_test.shape[0])
        x_test = x_test[perm_arr]
        x_label = x_label[perm_arr]
        if self.encoder.output_shape[1] < 2:
            raise ValueError(f"Encoder output dimension is {self.encoder.output_shape[1]} < 2.")
        # Generate encoded representations
        encoded_arr = self.predict(x_test, predictor_type="encoder")
        # Create a scatter plot of the encoded representations
        plt.figure(figsize=(8, 6))
        for label in labels:
            mask = (x_label == label)
            plt.scatter(encoded_arr[mask, 0], encoded_arr[mask, 1], color=colors[label])
        if lim is not None:
            plt.ylim(lim)
            plt.xlim(lim)
        plt.title('Encoded Digits in 2D Space')
        plt.xlabel('Encoded Dimension 1')
        plt.ylabel('Encoded Dimension 2')
        plt.grid(True)
        str_labels = [str(l) for l in labels]
        plt.legend(str_labels)
        if is_plot:
            plt.show()
        else:
            plt.close()

    def runAnimalExperiment(self, batch_size: Optional[int]=None, num_epoch: int=MAX_EPOCH,
            ) -> 'AbstractAutoencoder.ExperimentResult':
        """Run an experiment on the animal dataset.

        Args:
            batch_size (int): The batch size to use for training.
            num_epoch (int, optional): Number of epochs to train. Defaults to MAX_EPOCH.

        Returns:
            ExperimentResult: The result of the experiment.
        """
        # Default parameter values
        if batch_size is None:
            batch_size = self.batch_size
        # Summary
        self.summarize()
        #
        """ full_context_dct = dict(context_dct)
        full_context_dct['batch_size'] = batch_size
        full_context_dct['autoencoder'] = str(autoencoder.__class__).split('.')[-1][:-2]
        base_path = os.path.join(autoencoder.base_path, "animals_" + str(full_context_dct)) """
        """ for char in "'{}[] ":
            base_path = base_path.replace(char, "")
        base_path = base_path.replace(":", "-")
        base_path = base_path.replace(",", "__") """
        base_path = self.makeAnimalBasePath(batch_size=batch_size)
        x_animals_train, _, x_animals_test, __, ___ = util.getPklAnimals(is_verbose=self.is_verbose)
        self.fit(x_animals_train, num_epoch=num_epoch, batch_size=batch_size,
                validation_data=x_animals_test)
        self.serialize(base_path=base_path)
        self.plot(x_animals_test,
                png_path=base_path + ".png",
                is_plot=False,
        )
        self.serialize(base_path=base_path)
        return self.ExperimentResult(batch_size=batch_size,
                base_path=base_path,
                history=self.history_dct,
                context_str=util.dictToStr(self.context_dct),
                autoencoder=self,
                )

    def print(self, message: str):
        """Prints a message with the class name as prefix.

        Args:
            message (str): Message to print.
        """
        if self.is_verbose:
            print(f"[{self.__class__.__name__}] {message}")

    def __eq__(self, other: 'AbstractAutoencoder', tolerance: float=1e-8) -> bool:  # type: ignore
        """
        Check if two Keras models are identical in architecture and weights.
        """
        # Check architecture
        dd = DeepDiff(self.context_dct, other.context_dct)
        if dd:
            print("Context architectures dictionaries differ:")
            print(dd)
            return False
        # Check weights
        weights1 = self.autoencoder.get_weights()
        weights2 = other.autoencoder.get_weights()
        if len(weights1) != len(weights2):
            print("Models have different number of weight arrays")
            return False
        for idx, (w1, w2) in enumerate(zip(weights1, weights2)):
            if not np.allclose(w1, w2, atol=tolerance):
                print(f"Weight array {idx} differs")
                return False
        # If we reach this point, the models are equivalent
        return True