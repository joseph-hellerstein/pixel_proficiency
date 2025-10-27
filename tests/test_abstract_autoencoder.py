import unittest
from src.convolutional_autoencoder import ConvolutionalAutoencoder  # type: ignore
import src.util as util  # type: ignore
import src.constants as cn  # type: ignore
from typing import cast

import numpy as np
import os

IGNORE_TEST = False
IS_PLOT = False
MNIST_SHAPE = [28,28,1]
FILTER_SIZES = [64, 8, 1]
# Prepare the data
X_MNIST_TRAIN, LABEL_MNIST_TRAIN, X_MNIST_TEST, LABEL_MNIST_TEST, CLASS_MNIST_NAMES = util.getPklMNIST()
X_ANIMALS_TRAIN, LABEL_ANIMALS_TRAIN, X_ANIMALS_TEST, LABEL_ANIMALS_TEST, CLASS_ANIMALS_NAMES = util.getPklAnimals()
ANIMALS_SHAPE = list(np.shape(X_ANIMALS_TRAIN[0]))
MNIST_SHAPE = list(np.shape(X_MNIST_TRAIN[0]))
# Reshape to add channel dimension (28, 28, 1) for CNN
X_MNIST_TRAIN_FLAT = np.expand_dims(X_MNIST_TRAIN, -1)
X_MNIST_TEST_FLAT = np.expand_dims(X_MNIST_TEST, -1)
NUM_EPOCH = 2
if IGNORE_TEST:
    VERBOSE = 1
else:
    VERBOSE = 0


#############################
# Tests
#############################
class TestConvolutionalAutoencoder(unittest.TestCase):

    def setUp(self):
        self.filter_sizes = FILTER_SIZES
        self.cae = ConvolutionalAutoencoder(image_shape=MNIST_SHAPE,
                filter_sizes=self.filter_sizes,
                is_delete_serializations=True)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertEqual(self.cae.filter_sizes, self.filter_sizes)
        self.assertIsNotNone(self.cae.autoencoder)
        self.assertIsNotNone(self.cae.encoder)
        self.assertIsNotNone(self.cae.decoder)
        self.assertTrue(len(self.cae.history_dct) == 0)

    def testFitMNIST(self):
        if IGNORE_TEST or not IS_PLOT:
            return
        self.cae.summarize()
        self.cae.fit(X_MNIST_TRAIN_FLAT, num_epoch=NUM_EPOCH, batch_size=512, validation_data=X_MNIST_TEST_FLAT,
                is_verbose=IGNORE_TEST)
        self.cae.plot(X_MNIST_TEST_FLAT, is_plot=IS_PLOT)
        self.assertIsNotNone(self.cae.history)
        self.assertIn('loss', self.cae.history.history)

    def testFitAnimals(self):
        if IGNORE_TEST or not IS_PLOT:
            return
        cae = ConvolutionalAutoencoder(image_shape=[96, 96, 3],
                filter_sizes=[256, 128, 64],
                is_delete_serializations=True)
        cae.summarize()
        cae.fit(X_ANIMALS_TRAIN, num_epoch=10, batch_size=128, validation_data=X_ANIMALS_TEST,
                is_verbose=IGNORE_TEST)
        cae.plot(X_ANIMALS_TEST, is_plot=IS_PLOT)
        self.assertIsNotNone(cae.history_dct)
        self.assertIn('loss', cae.history_dct)

    def testDecoderEncoder(self):
        # Check that running the encoder and decoder in sequence is the same as the autoencoder.
        if IGNORE_TEST:
            return
        cae = ConvolutionalAutoencoder(image_shape=ANIMALS_SHAPE,
                filter_sizes=FILTER_SIZES,
                is_delete_serializations=True)
        cae.summarize()
        cae.fit(X_ANIMALS_TRAIN, num_epoch=NUM_EPOCH, batch_size=128, validation_data=X_ANIMALS_TEST,
                is_verbose=IGNORE_TEST)
        autoencoder_prediction = cae.predict(X_ANIMALS_TEST, predictor_type="autoencoder")
        encoder_prediction = cae.predict(X_ANIMALS_TEST, predictor_type="encoder")
        decoder_prediction = cae.predict(encoder_prediction, predictor_type="decoder")
        self.assertTrue(np.all(autoencoder_prediction.shape == X_ANIMALS_TEST.shape))
        self.assertTrue(np.all(decoder_prediction.shape == X_ANIMALS_TEST.shape))
        mse = np.sum((decoder_prediction - autoencoder_prediction)**2/autoencoder_prediction.size)
        self.assertAlmostEqual(mse, 0.0, places=5)

    def testPlotEncodedLabels(self):
        if IGNORE_TEST:
            return
        cae = ConvolutionalAutoencoder(image_shape=MNIST_SHAPE,
                filter_sizes=[64, 32, 2],
                is_delete_serializations=True)
        cae.fit(X_MNIST_TRAIN_FLAT, num_epoch=NUM_EPOCH, batch_size=128, validation_data=X_MNIST_TEST_FLAT,
                is_verbose=IGNORE_TEST)
        cae.plotEncoded(X_MNIST_TEST_FLAT, LABEL_MNIST_TEST, max_num_point=300, lim=[-10, 150],
                is_plot=IS_PLOT)

    def testDoAnimalExperiments(self):
        if IGNORE_TEST:
            return
        filter_sizes = [64, 32, 8]
        ConvolutionalAutoencoder.doAnimalExperiments(filter_sizes=filter_sizes, batch_size=128,
                base_path=cn.TEST_DIR, num_epoch=NUM_EPOCH)
        ffiles = os.listdir(cn.TEST_DIR)
        true = any(["animals_" in f for f in ffiles])
        self.assertTrue(true)
        # Ensure that generated files exist
        file_list = []
        for size in filter_sizes:
            ffiles = [f for f in ffiles if (str(size) in f and "ConvolutionalAutoencoder" in f)]
            file_list.append(ffiles)
        selected_files = file_list[0]
        for files in file_list[1:]:
            selected_files = set(selected_files).intersection(set(files))
        self.assertTrue(len(selected_files) > 0)

if __name__ == '__main__':
    unittest.main()