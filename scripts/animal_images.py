'''Considers different parameterizations for animal images.'''

from src.dense_autoencoder import DenseAutoencoder  # type: ignore
from src.convolutional_autoencoder import ConvolutionalAutoencoder  # type: ignore

# Dense autoencoders
if True:
    for filter2 in [32, 64]:
        for filter3 in [8, 16]:
                encode_dims = [96*96*3, 512, 128, filter2, filter3]
                print("***")
                print(f"***Dense Autoencoder Animal Experiments: {encode_dims}")
                print("***")
                DenseAutoencoder.doAnimalExperiments(encode_dims=encode_dims, batch_size=128, is_stopping_early=False,
                        is_verbose=True)
# Convolutional autoencoders
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