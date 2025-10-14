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
        #if IGNORE_TEST:
        #    self.skipTest("Ignoring test as per flag.")
        mnist_data = util.getPklMNIST()
        import pdb; pdb.set_trace()
        self.assertEqual(mnist_data.x_train.shape, (60000, 28, 28))
        self.assertEqual(mnist_data.x_test.shape, (10000, 28, 28))
        self.assertEqual(mnist_data.label_train.shape, (60000,))
        self.assertEqual(mnist_data.label_test.shape, (10000,))
        self.assertEqual(mnist_data.x_train.dtype, 'float32')
        self.assertEqual(mnist_data.x_test.dtype, 'float32')
        self.assertEqual(mnist_data.label_train.dtype, 'uint8')
        self.assertEqual(mnist_data.label_test.dtype, 'uint8')
        

if __name__ == '__main__':
    unittest.main()