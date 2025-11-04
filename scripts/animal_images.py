'''Considers different parameterizations for animal images.'''

from src.dense_autoencoder import DenseAutoencoder  # type: ignore
from src.convolutional_autoencoder import ConvolutionalAutoencoder  # type: ignore
import src.util as util  # type: ignore
import src.constants as cn # type: ignore

import os

if False:
    X_MNIST_TRAIN, LABEL_MNIST_TRAIN, X_MNIST_TEST, LABEL_MNIST_TEST, MNIST_CLASS_NAMES = util.getPklMNIST()
    # Example run for dense autoencoder
    encode_dims = [784, 128, 64, 16]
    image_shape = [28, 28]
    dae = DenseAutoencoder(image_shape, encode_dims, is_delete_serialization=True, is_verbose=True,
                is_early_stopping=False)
    dae.fit(X_MNIST_TRAIN, num_epoch=1000, batch_size=128, validation_data=X_MNIST_TEST)
    dae.summarize()
    dae.plot(X_MNIST_TEST)
    base_path = dae.makeAnimalBasePath(batch_size=128)
    base_path = os.path.join(cn.EXPERIMENT_DIR, base_path)
    dae.serialize(base_path=base_path)
if True:
    X_MNIST_TRAIN, LABEL_MNIST_TRAIN, X_MNIST_TEST, LABEL_MNIST_TEST, MNIST_CLASS_NAMES = util.getPklMNIST()
    # Example run for dense autoencoder
    filter_sizes = [128, 64, 16]
    image_shape = [28, 28]
    cae = ConvolutionalAutoencoder(image_shape, filter_sizes, is_delete_serialization=True, is_verbose=True,
                is_early_stopping=False)
    cae.fit(X_MNIST_TRAIN, num_epoch=1000, batch_size=128, validation_data=X_MNIST_TEST)
    cae.summarize()
    cae.plot(X_MNIST_TEST)
    base_path = cae.makeBasePath(batch_size=128, data_name="mnist")
    base_path = os.path.join(cn.EXPERIMENT_DIR, base_path)
    cae.serialize(base_path=base_path)
if False:
    # Deserialize and plot
    base_path = "animals_image_shape-96__96__3__filter_sizes-256__128__64__activation-sigmoid__dropout_rate-0.4__batch_size-128__autoencoder-ConvolutionalAutoencoder"
    base_path = os.path.join(cn.EXPERIMENT_DIR, base_path)
    cae = ConvolutionalAutoencoder.deserialize(base_path=base_path)
    x_animals_train, _, x_animals_test, __, ___ = util.getPklAnimals()
    cae.plot(x_animals_test, is_plot=True)

# Dense autoencoders
if False:
    for filter2 in [32, 64]:
        for filter3 in [8, 16]:
                encode_dims = [96*96*3, 512, 128, filter2, filter3]
                print("***")
                print(f"***Dense Autoencoder Animal Experiments: {encode_dims}")
                print("***")
                DenseAutoencoder.doAnimalExperiments(encode_dims=encode_dims, batch_size=128, is_early_stopping=True,
                        is_verbose=True)
# Convolutional autoencoders
if False:
    filter_sizes_list = [
            [256, 128, 64],
            [256, 128, 32],
            [256, 64, 32],
            [256, 64, 16],
            [128, 64, 32],
            [128, 64, 16],
            [128, 32, 16],
            ]
    for filter_sizes in filter_sizes_list:
            print("***")
            print(f"***Convolutional Autoencoder Animal Experiments: {filter_sizes}")
            print("***")
            ConvolutionalAutoencoder.doAnimalExperiments(filter_sizes=filter_sizes, batch_size=128,
                    is_stopping_early=True,
                    is_verbose=True)