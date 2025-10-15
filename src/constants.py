import os

NUM_TRAIN = 60000
NUM_TEST = 10000

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(PROJECT_DIR, "tests")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
TEST = os.path.join(DATA_DIR, "data")
MNIST_TRAIN_PATH = os.path.join(DATA_DIR, "mnist_train.pkl")
MNIST_TEST_PATH = os.path.join(DATA_DIR, "mnist_test.pkl")
MNIST_PATH = os.path.join(DATA_DIR, "mnist.pkl")
ANIMALS_PATH = os.path.join(DATA_DIR, "animals.pkl")