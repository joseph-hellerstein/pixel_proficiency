# Defines an abstract base class for autoencoders.
import src.constants as cn  # type: ignore
import src.util as util  # type: ignore

import collections
import json
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau #  type: ignore
import matplotlib.pyplot as plt#  type: ignore
import matplotlib.cm as cm
import numpy as np#  type: ignore
import os
import pandas as pd#  type: ignore
import shutil
from tensorflow import keras #  type: ignore
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
        self.png_pat = os.path.join(base_path, "%s.png") # pattern for naming plots
        self.autoencoder, self.encoder, self.decoder, self.history_dct =  \
                self.deserializeAll(base_path=base_path)
        self.is_fit = False
        if is_delete_serializations or self.autoencoder is None:
            self.deleteSerializations(base_path=base_path)
            self.autoencoder, self.encoder, self.decoder, self.history_dct =  \
                    self._build()
        else:
            self.is_fit = True

    @property
    def compression_factor(self) -> float:
        raise NotImplementedError("Subclasses must implement compression_factor property")
    
    #property
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
            batch_size: int,
            validation_data: np.ndarray,
            verbose: int=1) -> None:
        """Fits the autoencoder model to the training data. Checkpoint the model.
        Normalizes the data to [0, 1] before training.

        Args:
            x_train (np.ndarray): Training data.
            num_epoch (int): Number of epochs to train.
            batch_size (int): Batch size for training.
            validation_data (np.ndarray): Validation data.
            verbose (int, optional): Verbosity level. Defaults to 1.
        """
        # Train the autoencoder
        if self.is_fit:
            print("Model is already fit. Skipping training.")
            return
        # Create a ModelCheckpoint callback
        callbacks = [
            ModelCheckpoint(
                'best_autoencoder.keras',
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                # Use min_delta 0 since we don't know the right units to detect improvement
                min_delta=0,  # type: ignore
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,        # reduce LR by half
                patience=5,        # after 5 epochs without improvement
                min_lr=1e-7,
                verbose=1
            )
        ]
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
        self.serializeAll()
    
    def summarize(self) -> None:
        # Prints a summary of the autoencoder model.
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

    DeserializeResult = collections.namedtuple(
        'DeserializeResult', ['autoencoder', 'encoder', 'decoder', 'history_dct'])
    @classmethod
    def deserializeAll(cls, base_path: str) -> DeserializeResult:
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
            return cls.DeserializeResult(None, None, None, dict())
        autoencoder = cls.deserializeModel(autoencoder_path)
        encoder = cls.deserializeModel(encoder_path)
        decoder = cls.deserializeModel(decoder_path)
        history_dct = cls.deserializeHistory(history_path)
        return cls.DeserializeResult(autoencoder, encoder, decoder, history_dct)
    
    def plotEncoded(self, x_test: np.ndarray, x_label: np.ndarray,
            max_num_point: int= 100,
            lim: Optional[List[float]] = None) -> None:
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
            plt.scatter(encoded_arr[mask, 0], encoded_arr[mask, 1], c=colors[label])
        if lim is not None:
            plt.ylim(lim)
            plt.xlim(lim)
        plt.title('Encoded Digits in 2D Space')
        plt.xlabel('Encoded Dimension 1')
        plt.ylabel('Encoded Dimension 2')
        plt.grid(True)
        str_labels = [str(l) for l in labels]
        plt.legend(str_labels)
        plt.show()

    ExperimentResult = collections.namedtuple('ExperimentResult',
            ['batch_size', 'history', 'base_path', 'context_str'])
    @classmethod
    def runAnimalExperiment(cls,
            autoencoder: 'AbstractAutoencoder',
            batch_size: int,
            context_dct: dict,
            ) -> ExperimentResult:
        """Run an experiment on the animal dataset.

        Args:
            autoencoder (AbstractAutoencoder): The autoencoder to use.
            batch_size (int): The batch size to use for training.

        Returns:
            ExperimentResult: The result of the experiment.
        """
        autoencoder.summarize()
        #
        full_context_dct = dict(context_dct)
        full_context_dct['batch_size'] = batch_size
        full_context_dct['autoencoder'] = str(autoencoder.__class__).split('.')[-1][:-2]
        x_animals_train, _, x_animals_test, __, __ = util.getPklAnimals()
        autoencoder.fit(x_animals_train, num_epoch=1, batch_size=batch_size,
                validation_data=x_animals_test, verbose=1)
        base_path = os.path.join(autoencoder.base_path, "animals_" + str(full_context_dct))
        for char in "'{}[] ":
            base_path = base_path.replace(char, "")
        base_path = base_path.replace(":", "-")
        base_path = base_path.replace(",", "__")
        autoencoder.serializeAll(base_path=base_path)
        autoencoder.plot(x_animals_test,
                png_path=base_path + ".png",
                is_plot=False,
        )
        return cls.ExperimentResult(batch_size=batch_size,
                base_path=base_path,
                history=autoencoder.history_dct,
                context_str=util.dictToStr(autoencoder.context_dct()))