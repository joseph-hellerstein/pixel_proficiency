# Defines a deterministic autoencoder using TensorFlow/Keras to
# compress and reconstruct images from the MNIST dataset.

import numpy as np#  type: ignore
import matplotlib.pyplot as plt#  type: ignore
from tensorflow import keras #  type: ignore
from tensorflow.keras import layers#  type: ignore
from tensorflow.keras.datasets import mnist   # type: ignore
from sklearn.decomposition import PCA  # type: ignore
from sklearn.preprocessing import StandardScaler # type:ignore#  type: ignore
import pandas as pd#  type: ignore
import os#  type: ignore
from typing import Tuple

class DeterministicAutoencoder(object):
    def __init__(self):
        pass
# Define autoencoder architecture
encoding_dim = 16
print(f"Compression factor is {784/encoding_dim}")

# Input layer
input_img = keras.Input(shape=(784,))

# Encoder
encoded = layers.Dense(128, activation='relu')(input_img)
encoded = layers.Dense(64, activation='relu')(encoded)
encoded = layers.Dense(encoding_dim, activation='relu')(encoded)

# Decoder
decoded = layers.Dense(64, activation='relu')(encoded)
decoded = layers.Dense(128, activation='relu')(decoded)
decoded = layers.Dense(784, activation='sigmoid')(decoded)

# Create the autoencoder model
autoencoder = keras.Model(input_img, decoded)

# Create encoder model (for extracting encoded representations)
encoder = keras.Model(input_img, encoded)

# Create decoder model
encoded_input = keras.Input(shape=(encoding_dim,))
decoder_layer1 = autoencoder.layers[-3]
decoder_layer2 = autoencoder.layers[-2]
decoder_layer3 = autoencoder.layers[-1]
decoder = keras.Model(encoded_input,
                     decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Display model architecture
autoencoder.summary()

# Train the autoencoder
history = autoencoder.fit(X_TRAIN, X_TRAIN,
                         epochs=15,
                         batch_size=256,
                         shuffle=True,
                         validation_data=(X_TEST, X_TEST),
                         verbose=1)

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Generate predictions
encoded_imgs = encoder.predict(X_TEST)
decoded_imgs = autoencoder.predict(X_TEST)

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

# Optional: Save the trained model
# autoencoder.save('mnist_autoencoder.h5')
# encoder.save('mnist_encoder.h5')
# decoder.save('mnist_decoder.h5')