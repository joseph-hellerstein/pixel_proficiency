import src.constants as cn
from collections import namedtuple
import numpy as np
import os
import pickle
from typing import Tuple
from tensorflow.keras.datasets import mnist  # type: ignore
from torchvision import datasets # type: ignore
from typing import List

TrainTestData = namedtuple('TrainTestData', ['x_train', 'label_train', 'x_test', 'label_test', 'class_names'])
# x_train: np.ndarray (training data)
# label_train: np.ndarray (training labels)
# x_test: np.ndarray (test data)
# label_test: np.ndarray (test labels)
# class_names: List[str] (class names)

def getMNISTData(prefix: str) -> np.ndarray:
    """Recovers MNIST image data from the specified prefix.

    Args:
        prefix (str): The prefix to filter image files.

    Returns:
        np.ndarray: The recovered image data.
    """
    num_row = cn.NUM_TRAIN if prefix == 'train' else cn.NUM_TEST
    ffiles = [f for f in os.listdir(cn.DATA_DIR) if prefix in f]
    full_arr = np.zeros((num_row, 28, 28))
    for ffile in ffiles:
        extract = ffile.split("_")[1]
        irow = int(extract.split(".")[0])
        path = os.path.join(cn.DATA_DIR, ffile)
        arr = np.loadtxt(path, delimiter=',').astype(int)
        full_arr[irow] = arr
    return full_arr.astype('int64')

def getMNISTTTData() -> Tuple[np.ndarray, np.ndarray]:
    """Recovers MNIST image data for training and testing.

    Returns:
        np.ndarray: training data
        np.ndarray: test data
    """
    # Recover the data
    x_train = getMNISTData('train').astype('float32')/255.0
    x_test = getMNISTData('test').astype('float32')/255.0
    return x_train, x_test

def getProcessedMNISTData() -> Tuple[np.ndarray, np.ndarray]:
    """Recovers processed MNIST image data from the specified prefix.

    Args:
        prefix (str): The prefix to filter processed image files.

    Returns:
        np.ndarray: training data
        np.ndarray: test data
    """
    # Recover the data
    x_train, x_test = getMNISTTTData()
    # Flatten the images (28x28 -> 784)
    x_train = x_train.reshape((len(x_train), 28 * 28))
    x_test = x_test.reshape((len(x_test), 28 * 28))
    return x_train, x_test

def pklMNIST(x_train: np.ndarray, x_test: np.ndarray) -> None:
    """Saves the MNIST data as pickle files.

    Args:
        x_train (np.ndarray): training data
        x_test (np.ndarray): test data
    """
    """ with open(cn.MNIST_TRAIN_PATH, 'wb') as f:
        pickle.dump(x_train, f)
    with open(cn.MNIST_TEST_PATH, 'wb') as f:
        pickle.dump(x_test, f) """

def unpklMNIST() -> Tuple[np.ndarray, np.ndarray]:
    """Recovers MNIST image data from pickle files.

    Returns:
        np.ndarray: training data
        np.ndarray: test data
    """
    with open(cn.MNIST_TRAIN_PATH, 'rb') as f:
        x_train = pickle.load(f)
    with open(cn.MNIST_TEST_PATH, 'rb') as f:
        x_test = pickle.load(f)
    return x_train, x_test

def getPklMNIST() -> TrainTestData:
    """Recovers MNIST image data from pickle files, or pickles the data if not present.

    Returns:
        TrainTestData: A named tuple containing training and test data and labels.
    """
    if not os.path.exists(cn.MNIST_PATH):
        print("***Pickling MNIST data...")
        (x_train, label_train), (x_test, label_test) = mnist.load_data()
        data = (x_train, label_train, x_test, label_test)
        with open(cn.MNIST_PATH, 'wb') as f:
            pickle.dump(data, f)
    else:
        print("***Unpickling MNIST data...")
        with open(cn.MNIST_PATH, 'rb') as f:
            x_train, label_train, x_test, label_test = pickle.load(f)
    class_names = [str(i) for i in range(10)]
    return TrainTestData(x_train, label_train, x_test, label_test, class_names)

def getPklAnimals() -> TrainTestData:
    """Recovers Animals image data from pickle files, or pickles the data if not present.

    Returns:
        TrainTestData: A named tuple containing training and test data and labels.
    """
    ##
    data_dir = "/Users/jlheller/home/Technical/repos/pixel_proficiency/data/animals"
    def getData(data_type: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        dataset = datasets.STL10(root=data_dir, split=data_type, download=True)
        class_names = dataset.classes  # List of class names
        images = []
        labels = []
        for img, label in dataset:
            images.append(np.array(img))
            labels.append(label)
        images_array = np.stack(images)  # Shape: (N, 96, 96, 3)
        labels_array = np.array(labels)  # Shape: (N,)
        return images_array, labels_array, class_names
    ##
    if not os.path.exists(cn.ANIMALS_PATH):
        print("***Pickling Animals data...")
        train_image_arr, train_label_arr, class_names = getData('train')
        test_image_arr, test_label_arr, class_names = getData('test')
        data = (train_image_arr, train_label_arr, test_image_arr, test_label_arr, class_names)
        with open(cn.ANIMALS_PATH, 'wb') as f:
            pickle.dump(data, f)
    else:
        print("***Unpickling Animals data...")
        with open(cn.ANIMALS_PATH, 'rb') as f:  # type: ignore
            train_image_arr, train_label_arr, test_image_arr, test_label_arr, class_names = pickle.load(f)
    return TrainTestData(train_image_arr, train_label_arr, test_image_arr, test_label_arr, class_names)