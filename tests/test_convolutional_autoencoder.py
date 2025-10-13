import unittest
import src.constants as cn  # type: ignore
from src.convolutional_autoencoder import ConvolutionalAutoencoder  # type: ignore
import src.util as util  # type: ignore

import numpy as np
import os

IGNORE_TEST = True
IS_PLOT = False
IMAGE_SHAPE = [28,28,1]
ENCODE_DIMS = [128, 64, 2]
# Prepare the data
X_TRAIN, X_TEST = util.getPklMNIST()
# Reshape to add channel dimension (28, 28, 1) for CNN
X_TRAIN = np.expand_dims(X_TRAIN, -1)
X_TEST = np.expand_dims(X_TEST, -1)

# FIXME: Test encoder, decoder
# FIXME: Plot digits in 2 space


#############################
# Tests
#############################
class TestConvolutionalAutoencoder(unittest.TestCase):

    def setUp(self):
        if IGNORE_TEST:
            return
        self.encode_dims = ENCODE_DIMS
        self.cae = ConvolutionalAutoencoder(image_shape=IMAGE_SHAPE,
                hidden_dims=self.encode_dims,
                is_delete_serializations=True)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(self.cae.hidden_dims, self.encode_dims)
        self.assertIsNotNone(self.cae.autoencoder)
        self.assertIsNotNone(self.cae.encoder)
        self.assertIsNotNone(self.cae.decoder)
        self.assertTrue(len(self.cae.history_dct) == 0)

    def testFitPlot(self):
        if IGNORE_TEST:
            return
        self.cae.fit(X_TRAIN, num_epoch=500, batch_size=256, validation_data=X_TEST, verbose=1)
        self.cae.summarizeModel()
        self.cae.plot(X_TEST)
        self.assertIsNotNone(self.cae.history)
        self.assertIn('loss', self.cae.history.history)
        # Evaluate serialization
        cae2 = ConvolutionalAutoencoder(image_shape=IMAGE_SHAPE,
                hidden_dims=self.encode_dims,
                is_delete_serializations=False)
        self.cae.plot(X_TEST)
        self.assertIsNotNone(cae2.autoencoder)
        self.assertIsNotNone(cae2.encoder)
        self.assertIsNotNone(cae2.decoder)
        self.assertTrue(len(cae2.history_dct) != 0)

    def testDecoderEncoder(self):
        #if IGNORE_TEST:
        #    return
        cae = ConvolutionalAutoencoder(image_shape=IMAGE_SHAPE,
                hidden_dims=ENCODE_DIMS,
                is_delete_serializations=False)
        import pdb; pdb.set_trace()
        cae.plot(X_TEST)
if __name__ == '__main__':
    unittest.main()