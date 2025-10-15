import unittest
import src.util as util  # type: ignore
import numpy as np


IGNORE_TEST = False
IS_PLOT = False


#############################
# Tests
#############################
class TestUtil(unittest.TestCase):

    def setUp(self):
        pass

    def testRecoverImages(self):
        if IGNORE_TEST:
            self.skipTest("Ignoring test as per flag.")
        arr = util.getMNISTData('test')
        self.assertEqual(arr.shape, (10000, 28, 28))
        self.assertEqual(arr.dtype, 'int64')

    def testPklMNIST(self):
        if IGNORE_TEST:
            self.skipTest("Ignoring test as per flag.")
        train_test_data = util.getPklMNIST()
        self.assertEqual(train_test_data.x_train.shape, (60000, 28, 28))
        self.assertEqual(train_test_data.x_test.shape, (10000, 28, 28))
        self.assertEqual(train_test_data.label_train.shape, (60000,))
        self.assertEqual(train_test_data.label_test.shape, (10000,))
        self.assertEqual(train_test_data.x_train.dtype, 'uint8')
        self.assertEqual(train_test_data.x_test.dtype, 'uint8')
        self.assertEqual(train_test_data.label_train.dtype, 'uint8')
        self.assertEqual(train_test_data.label_test.dtype, 'uint8')

    def testPklAnimals(self):
        if IGNORE_TEST:
            self.skipTest("Ignoring test as per flag.")
        train_test_data = util.getPklAnimals()
        self.assertEqual(len(train_test_data.class_names), len(np.unique(train_test_data.label_train)))
        self.assertEqual(train_test_data.x_train.shape[0], 5000)
        self.assertEqual(train_test_data.x_test.shape[0], 8000)
        self.assertEqual(train_test_data.label_train.shape[0], 5000)
        self.assertEqual(train_test_data.label_test.shape[0], 8000)
        self.assertEqual(train_test_data.x_train.dtype, 'uint8')
        self.assertEqual(train_test_data.x_test.dtype, 'uint8')
        self.assertEqual(train_test_data.label_train.dtype, 'int64')
        self.assertEqual(train_test_data.label_test.dtype, 'int64')


if __name__ == '__main__':
    unittest.main()