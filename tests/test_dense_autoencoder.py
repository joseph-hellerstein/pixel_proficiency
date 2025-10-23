import unittest
import src.constants as cn  # type: ignore
from dense_autoencoder import DenseAutoencoder  # type: ignore
import src.util as util  # type: ignore

import numpy as np
import os


IGNORE_TEST = True
IS_PLOT = False
X_TRAIN, LABEL_TRAIN, X_TEST, LABEL_TEST, CLASS_NAMES = util.getPklMNIST()
#X_TRAIN = np.reshape(X_TRAIN, (np.shape(X_TRAIN)[0], 28, 28))
#X_TEST = np.reshape(X_TEST, (np.shape(X_TEST)[0], 28, 28))


#############################
# Tests
#############################
class TestDeterministicAutoencoder(unittest.TestCase):

    def setUp(self):
        pass

    def testConstructor(self):
        if IGNORE_TEST:
            return
        encode_dims = [784, 128, 64, 32]
        dae = DenseAutoencoder(encode_dims)
        self.assertEqual(dae.encode_dims, encode_dims)
        self.assertEqual(dae.num_hidden_layer, 3)
        self.assertEqual(dae.compression_factor, 784/32)
        self.assertIsNotNone(dae.autoencoder)
        self.assertIsNotNone(dae.encoder)
        self.assertIsNotNone(dae.decoder)
        self.assertEqual(len(dae.history_dct), 0)

    def testFlattenUnflatten(self):
        if IGNORE_TEST:
            return
        encode_dims = [784, 128, 64, 32]
        dae = DenseAutoencoder(encode_dims)
        arr = X_TRAIN[0:10]
        flat = dae._flatten(arr)
        self.assertEqual(flat.shape, (10, 784))
        unflat = dae._unflatten(flat)
        self.assertEqual(unflat.shape, (10, 28, 28))
        for i in range(10):
            for j in range(28):
                for k in range(28):
                    self.assertAlmostEqual(arr[i][j][k], unflat[i][j][k], places=5)

    def testFitPlot(self):
        #if IGNORE_TEST:
        #    return
        encode_dims = [784, 128, 64, 32]
        encode_dims = [784, 256, 128, 16]
        dae = DenseAutoencoder(encode_dims, is_delete_serializations=False)
        dae.fit(X_TRAIN, num_epoch=1000, batch_size=128, validation_data=X_TEST, verbose=1)
        dae.summarize()
        dae.plot(X_TEST)
        self.assertIn('loss', dae.history_dct)
        self.assertIn('val_loss', dae.history_dct)

if __name__ == '__main__':
    unittest.main()