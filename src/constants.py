import os

DATA_DIR = "/Users/jlheller/home/Technical/repos/pixel_proficiency/data"
NUM_TRAIN = 60000
NUM_TEST = 10000

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(PROJECT_DIR, "tests")
TEST = os.path.join(DATA_DIR, "data")