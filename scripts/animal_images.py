'''Considers different parameterizations for animal images.'''

from src.dense_autoencoder import DenseAutoencoder  # type: ignore
from src.convolutional_autoencoder import ConvolutionalAutoencoder  # type: ignore

# Dense autoencoders
print("Dense Autoencoder Animal Experiments: 256, 64, 16")
DenseAutoencoder.doAnimalExperiments(encode_dims=[96*96*3, 128, 32, 8], batch_size=128) 
DenseAutoencoder.doAnimalExperiments(encode_dims=[96*96*3, 128, 32, 16], batch_size=128) 
DenseAutoencoder.doAnimalExperiments(encode_dims=[96*96*3, 128, 64, 8], batch_size=128) 
DenseAutoencoder.doAnimalExperiments(encode_dims=[96*96*3, 128, 64, 16], batch_size=128) 
DenseAutoencoder.doAnimalExperiments(encode_dims=[96*96*3, 256, 32, 8], batch_size=128) 
DenseAutoencoder.doAnimalExperiments(encode_dims=[96*96*3, 256, 32, 16], batch_size=128) 
DenseAutoencoder.doAnimalExperiments(encode_dims=[96*96*3, 256, 64, 8], batch_size=128) 
DenseAutoencoder.doAnimalExperiments(encode_dims=[96*96*3, 256, 64, 16], batch_size=128) 