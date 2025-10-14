import unittest
import src.constants as cn  # type: ignore
from src.deterministic_autoencoder import DeterministicAutoencoder  # type: ignore
import src.util as util  # type: ignore

import numpy as np
import os


IGNORE_TEST = True
IS_PLOT = False
X_TRAIN, LABEL_TRAIN, X_TEST, LABEL_TEST = util.getPklMNIST()
X_TRAIN = np.reshape(X_TRAIN, (np.shape(X_TRAIN)[0], 28, 28))
X_TEST = np.reshape(X_TEST, (np.shape(X_TEST)[0], 28, 28))


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
        dae = DeterministicAutoencoder(encode_dims)
        self.assertEqual(dae.encode_dims, encode_dims)
        self.assertEqual(dae.num_hidden_layer, 3)
        self.assertEqual(dae.compression_factor, 784/32)
        self.assertIsNotNone(dae.autoencoder)
        self.assertIsNotNone(dae.encoder)
        self.assertIsNotNone(dae.decoder)
        self.assertIsNone(dae.history)

    def testFlattenUnflatten(self):
        if IGNORE_TEST:
            return
        encode_dims = [784, 128, 64, 32]
        dae = DeterministicAutoencoder(encode_dims)
        arr = X_TRAIN[0:10]
        flat = dae._flatten(arr)
        self.assertEqual(flat.shape, (10, 784))
        unflat = dae._unflatten(flat, mult_factor=255.0)
        self.assertEqual(unflat.shape, (10, 28, 28))
        for i in range(10):
            for j in range(28):
                for k in range(28):
                    self.assertAlmostEqual(arr[i][j][k], unflat[i][j][k], places=5)

    def testFitPlot(self):
        #if IGNORE_TEST:
        #    return
        encode_dims = [784, 128, 64, 8]
        dae = DeterministicAutoencoder(encode_dims)
        dae.fit(X_TRAIN, num_epoch=10, batch_size=256, validation_data=X_TEST, verbose=1)
        dae.summarizeModel()
        dae.plot(X_TEST)
        self.assertIsNotNone(dae.history)
        self.assertIn('loss', dae.history.history)
        self.assertIn('val_loss', dae.history.history)
        

if __name__ == '__main__':
    unittest.main()