import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras # type: ignore
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.datasets import mnist  # type: ignore
from sklearn.decomposition import PCA # type:ignore
from sklearn.preprocessing import StandardScaler # type:ignore
import pandas as pd # type: ignore
import os
from typing import Tuple
from torchvision import datasets # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import json

import src.constants as cn  # type: ignore
import src.util as util  # type: ignore

AUTOENCODER_PATH = os.path.join(cn.EXPERIMENT_DIR, "convolutional_autoencoder.keras")
ENCODER_PATH = os.path.join(cn.EXPERIMENT_DIR, "convolutional_encoder.keras")
HISTORY_PATH = os.path.join(cn.EXPERIMENT_DIR, "convolutional_history.json")

def recoverData(prefix: str) -> np.ndarray:
    num_row = cn.NUM_TRAIN if prefix == 'train' else cn.NUM_TEST
    ffiles = [f for f in os.listdir(cn.DATA_DIR) if prefix in f]
    full_arr = np.zeros((num_row, 28, 28))
    for ffile in ffiles:
        extract = ffile.split("_")[1]
        irow = int(extract.split(".")[0])
        path = os.path.join(cn.DATA_DIR, ffile)
        arr = loaded_int = np.loadtxt(path, delimiter=',').astype(int)
        full_arr[irow] = arr
    return full_arr

def getData(is_flatten: bool = True):
    # Recover the data
    X_TRAIN = recoverData('train')
    X_TEST = recoverData('test')
    # Normalize pixel values to [0, 1] range
    X_TRAIN = X_TRAIN.astype('float32') / 255.0
    X_TEST = X_TEST.astype('float32') / 255.0
    # Flatten the images (28x28 -> 784)
    if is_flatten:
        X_TRAIN = X_TRAIN.reshape((len(X_TRAIN), 28 * 28))
        X_TEST = X_TEST.reshape((len(X_TEST), 28 * 28))
    return X_TRAIN, X_TEST

def deserializeModel(path: str):
    """Deserializes the model

    Args:
        path (str): Path to save the model.

    """
    model = load_model(path, compile=False)
    model.compile(optimizer='adam', loss='mse')
    return model

def deserializeHistory(path: str):
    """Deserializes the training history

    Args:
        path (str): Path to save the history.

    """
    with open(path, 'r') as f:
        history_dict = json.load(f)
    return history_dict

def serializeHistory(history, path: str):
    """Serializes the training history

    Args:
        history: Training history object from model.fit
        path (str): Path to save the history.

    """
    with open(path, 'w') as f:
        json.dump(history.history, f)

# Prepare the data
X_TRAIN, X_TEST = util.getPklMNIST()
# Reshape to add channel dimension (28, 28, 1) for CNN
X_TRAIN = np.expand_dims(X_TRAIN, -1)
X_TEST = np.expand_dims(X_TEST, -1)

print(f"Training data shape: {X_TRAIN.shape}")
print(f"Test data shape: {X_TEST.shape}")

num_detector = 2
encoded_dim = 2
def _build():
    """Builds the convolutional autoencoder model.

    Returns:
        keras.Model: The constructed autoencoder model.
    """
    # Check if the model already exists
    if os.path.exists(AUTOENCODER_PATH) and os.path.exists(ENCODER_PATH) and os.path.exists(HISTORY_PATH):
        print("Loading existing model and history...")
        autoencoder = deserializeModel(AUTOENCODER_PATH)
        encoder = deserializeModel(ENCODER_PATH)
        history_dict = deserializeHistory(HISTORY_PATH)
        return autoencoder, encoder, history_dict
    # Define the Autoencoder
    # Define convolutional autoencoder architecture
    input_img = keras.Input(shape=(28, 28, 1))

    # Encoder
    encoded = layers.Conv2D(num_detector, (3, 3), activation='relu', padding='same')(input_img)  # 28×28×num_detector
    encoded = layers.Reshape((num_detector*28*28,))(encoded)
    encoded = layers.Dense(128, activation='relu')(encoded)
    encoded = layers.Dense(64, activation='relu')(encoded)
    encoded = layers.Dense(encoded_dim, activation='relu')(encoded)

    # Decoder
    decoded = layers.Dense(64, activation='relu')(encoded)
    decoded = layers.Dense(128, activation='relu')(decoded)
    decoded = layers.Dense(28*28*num_detector, activation='relu')(decoded)
    decoded = layers.Reshape((28, 28, num_detector))(decoded)
    decoded = layers.Conv2D(1, (3, 3), activation='relu', padding='same')(decoded)  # 28×28×1


    # Create the autoencoder model
    autoencoder = keras.Model(input_img, decoded)

    # Create encoder model (for extracting encoded representations)
    encoder = keras.Model(input_img, encoded)

    # Compile the autoencoder
    #autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mae'])
    autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Display model architecture
    autoencoder.summary()

    # Train the autoencoder
    history = autoencoder.fit(X_TRAIN, X_TRAIN,
            epochs=200,
            #batch_size=128,
            batch_size=512,
            shuffle=True,
            validation_data=(X_TEST, X_TEST),
            verbose=1)
    autoencoder.save(AUTOENCODER_PATH)
    encoder.save(ENCODER_PATH)
    serializeHistory(history, HISTORY_PATH)
    return autoencoder, encoder, history.history

autoencoder, encoder, history_dct = _build()

# Plot training history
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history_dct['loss'], label='Training Loss')
plt.plot(history_dct['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history_dct['mae'], label='Training MAE')
plt.plot(history_dct['val_mae'], label='Validation MAE')
plt.title('Mean Absolute Error')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

# Generate predictions
encoded_imgs = encoder.predict(X_TEST)
decoded_imgs = autoencoder.predict(X_TEST)

print(f"Encoded representation shape: {encoded_imgs.shape}")

# Visualize some encoded feature maps
if False:   
    plt.subplot(1, 3, 3)
    plt.imshow(encoded_imgs[0, :, :, 0], cmap='viridis')
    plt.title('Encoded Feature Map (Channel 0)')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

# Visualize original vs reconstructed images
n = 10  # Number of images to display
plt.figure(figsize=(20, 6))

for i in range(n):
    # Original images
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(X_TEST[i].reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # Reconstructed images
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')

    # Difference (error map)
    ax = plt.subplot(3, n, i + 1 + 2*n)
    diff = np.abs(X_TEST[i] - decoded_imgs[i])
    plt.imshow(diff.reshape(28, 28), cmap='hot')
    plt.title("Difference")
    plt.axis('off')

plt.tight_layout()
plt.show()

# Calculate reconstruction error statistics
mse = np.mean((X_TEST - decoded_imgs) ** 2)
mae = np.mean(np.abs(X_TEST - decoded_imgs))

print(f"\nReconstruction Statistics:")
print(f"Mean Squared Error: {mse:.6f}")
print(f"Mean Absolute Error: {mae:.6f}")
print(f"Final training loss: {history_dct['loss'][-1]:.4f}")
print(f"Final validation loss: {history_dct['val_loss'][-1]:.4f}")

# Analyze compression
original_size = 28 * 28 * 1  # 784 parameters per image
compression_ratio = original_size / encoded_dim

print(f"\nCompression Analysis:")
print(f"Original image size: {original_size} pixels")
print(f"Encoded representation: {encoded_dim}")
print(f"Compression ratio: {compression_ratio}")

# Optional: Visualize learned filters from the first convolutional layer
def visualize_conv_filters(model, layer_idx=1, num_filters=16):
    """Visualize the learned convolutional filters"""
    return
    filters = model.layers[layer_idx].get_weights()[0]

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle(f'Learned Filters from Layer {layer_idx}')

    for i in range(min(num_filters, 16)):
        ax = axes[i//4, i%4]
        ax.imshow(filters[:, :, 0, i], cmap='viridis')
        ax.set_title(f'Filter {i+1}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# Visualize learned filters
visualize_conv_filters(autoencoder)