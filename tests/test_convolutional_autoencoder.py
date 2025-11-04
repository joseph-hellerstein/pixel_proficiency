import unittest
from src.convolutional_autoencoder import ConvolutionalAutoencoder  # type: ignore
import src.util as util  # type: ignore
import src.constants as cn  # type: ignore

import numpy as np
import os

IGNORE_TEST = False
IS_PLOT = False
IMAGE_SHAPE = [28,28,1]
FILTER_SIZES = [64, 8, 1]
# Prepare the data
X_MNIST_TRAIN, LABEL_MNIST_TRAIN, X_MNIST_TEST, LABEL_MNIST_TEST, CLASS_MNIST_NAMES = util.getPklMNIST()
X_ANIMALS_TRAIN, LABEL_ANIMALS_TRAIN, X_ANIMALS_TEST, LABEL_ANIMALS_TEST, CLASS_ANIMALS_NAMES = util.getPklAnimals()
# Reshape to add channel dimension (28, 28, 1) for CNN
X_MNIST_TRAIN = np.expand_dims(X_MNIST_TRAIN, -1)
X_MNIST_TEST = np.expand_dims(X_MNIST_TEST, -1)
if IGNORE_TEST:
    VERBOSE = 1
else:
    VERBOSE = 0
NUM_EPOCH = 2


#############################
# Tests
#############################
class TestConvolutionalAutoencoder(unittest.TestCase):

    def setUp(self):
        self.filter_sizes = FILTER_SIZES
        self.cae = ConvolutionalAutoencoder(image_shape=IMAGE_SHAPE,
                filter_sizes=self.filter_sizes,
                is_delete_serialization=True)

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
        self.cae.fit(X_MNIST_TRAIN, num_epoch=NUM_EPOCH, batch_size=512, validation_data=X_MNIST_TEST,
                is_verbose=IGNORE_TEST)
        self.cae.plot(X_MNIST_TEST, is_plot=IS_PLOT)
        self.assertIsNotNone(self.cae.history)
        self.assertIn('loss', self.cae.history.history)
        print(f"Compression factor: {self.cae.compression_factor}")
        # Evaluate serialization
        cae2 = ConvolutionalAutoencoder(image_shape=IMAGE_SHAPE,
                filter_sizes=[64, 32, 8],
                is_delete_serialization=True)
        cae2.plot(X_MNIST_TEST, is_plot=IS_PLOT)
        self.assertIsNotNone(cae2.autoencoder)
        self.assertIsNotNone(cae2.encoder)
        self.assertIsNotNone(cae2.decoder)
        self.assertTrue(len(cae2.history_dct) != 0)

    def testFitAnimals(self):
        if IGNORE_TEST:
            return
        cae = ConvolutionalAutoencoder(image_shape=[96, 96, 3],
                #filter_sizes=[64, 32, 8],
                #filter_sizes=[32, 128, 256, 16],
                filter_sizes=[256, 128, 64],
                is_delete_serialization=True)
        cae.summarize()
        cae.fit(X_ANIMALS_TRAIN, num_epoch=NUM_EPOCH, batch_size=128, validation_data=X_ANIMALS_TEST,
                is_verbose=IGNORE_TEST)
        cae.plot(X_ANIMALS_TEST, is_plot=IS_PLOT)
        self.assertIsNotNone(cae.history_dct)
        self.assertIn('loss', cae.history_dct)

    def testEncoderDecoder1(self):
        if IGNORE_TEST:
            return
        cae = ConvolutionalAutoencoder(image_shape=IMAGE_SHAPE,
                filter_sizes=FILTER_SIZES,
                is_delete_serialization=True)
        prediction_arr = cae.predict(X_MNIST_TEST, predictor_type="encoder")
        x_test = cae.predict(prediction_arr, predictor_type="decoder")
        cae.plot(x_original_arr=X_MNIST_TEST, x_predicted_arr=x_test, is_plot=IS_PLOT)

    def testDecoderEncoder(self):
        if IGNORE_TEST:
            return
        cae = ConvolutionalAutoencoder(image_shape=IMAGE_SHAPE,
                filter_sizes=FILTER_SIZES,
                is_delete_serialization=True)
        autoencoder_prediction = cae.predict(X_MNIST_TEST, predictor_type="autoencoder")
        encoder_prediction = cae.predict(X_MNIST_TEST, predictor_type="encoder")
        decoder_prediction = cae.predict(encoder_prediction, predictor_type="decoder")
        self.assertTrue(np.all(autoencoder_prediction.shape == X_MNIST_TEST.shape))
        self.assertTrue(np.all(decoder_prediction.shape == X_MNIST_TEST.shape))

    def testPlotEncodedLabels(self):
        if IGNORE_TEST:
            return
        cae = ConvolutionalAutoencoder(image_shape=IMAGE_SHAPE,
                filter_sizes=FILTER_SIZES,
                is_delete_serialization=True)
        cae.plotEncoded(X_MNIST_TEST, LABEL_MNIST_TEST, max_num_point=300, lim=[-10, 500], is_plot=IS_PLOT)

    def testDoAnimalExperiments(self):
        if IGNORE_TEST:
            return
        filter_sizes = [64, 32, 8]
        ConvolutionalAutoencoder.doAnimalExperiments(filter_sizes=filter_sizes, batch_size=128,
                base_path=cn.TEST_DIR, num_epoch=NUM_EPOCH,
                is_stopping_early=True, is_verbose=IGNORE_TEST)
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
    
    def testSerializeDeserializeContext(self):
        if IGNORE_TEST:
            return
        self.cae.summarize()
        self.cae.fit(X_MNIST_TRAIN, num_epoch=1, batch_size=512, validation_data=X_MNIST_TEST,
                is_verbose=IGNORE_TEST)
        path = os.path.join(cn.TEST_DIR, "cae_test_context")
        self.cae.serialize(path)
        cae = ConvolutionalAutoencoder.deserialize(path)
        cae.plot(X_MNIST_TEST, is_plot=IS_PLOT)
        self.assertIsNotNone(cae.history_dct)

    def testBug(self):
        if IGNORE_TEST:
            return
        ConvolutionalAutoencoder.doAnimalExperiments(filter_sizes=[64, 32, 8], batch_size=128,
                base_path=cn.TEST_DIR, num_epoch=NUM_EPOCH)

if __name__ == '__main__':
    unittest.main()