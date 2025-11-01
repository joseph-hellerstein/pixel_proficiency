import unittest
import src.constants as cn  # type: ignore
from dense_autoencoder import DenseAutoencoder  # type: ignore
import src.util as util  # type: ignore

import os


IGNORE_TEST = False
IS_PLOT = False
X_TRAIN, LABEL_TRAIN, X_TEST, LABEL_TEST, CLASS_NAMES = util.getPklMNIST()
NUM_EPOCH = 2
if IGNORE_TEST:
    VERBOSE = 1
else:
    VERBOSE = 0


#############################
# Tests
#############################
class TestDeterministicAutoencoder(unittest.TestCase):

    def setUp(self):
        pass

    def testConstructor(self):
        if IGNORE_TEST:
            return
        encode_dims = [96*96*3, 128, 64, 32]
        dae = DenseAutoencoder(cn.ANIMALS_IMAGE_SHAPE, encode_dims)
        self.assertEqual(dae.encode_dims, encode_dims)
        self.assertEqual(dae.num_hidden_layer, 3)
        self.assertEqual(dae.compression_factor, (96*96*3)/32)
        self.assertIsNotNone(dae.autoencoder)
        self.assertIsNotNone(dae.encoder)
        self.assertIsNotNone(dae.decoder)
        self.assertEqual(len(dae.history_dct), 0)

    def testFitPlot(self):
        if IGNORE_TEST:
            return
        encode_dims = [256, 128, 16]
        dae = DenseAutoencoder(cn.MNIST_IMAGE_SHAPE, encode_dims,
                is_delete_serialization=True, is_verbose=IGNORE_TEST)
        dae.fit(X_TRAIN, num_epoch=NUM_EPOCH, batch_size=128, validation_data=X_TEST, is_verbose=IGNORE_TEST)
        dae.summarize()
        dae.plot(X_TEST, is_plot=IS_PLOT)
        self.assertIn('loss', dae.history_dct)
        self.assertIn('val_loss', dae.history_dct)

    def testSerializeDeserialize(self):
        if IGNORE_TEST:
            return
        encode_dims = [256, 128, 16]
        dae1 = DenseAutoencoder(cn.MNIST_IMAGE_SHAPE, encode_dims,
                is_delete_serialization=True, is_verbose=IGNORE_TEST)
        dae1.fit(X_TRAIN, num_epoch=NUM_EPOCH, batch_size=128, validation_data=X_TEST, is_verbose=IGNORE_TEST)
        path = os.path.join(cn.TEST_DIR, "dae_test_context")
        dae1.serialize(base_path=path)
        dae2 = DenseAutoencoder.deserialize(base_path=path)
        dae2.summarize()
        dae2.plot(X_TEST, is_plot=IS_PLOT)
        self.assertIn('loss', dae2.history_dct)

    def testDoAnimalExperiments(self):
        if IGNORE_TEST:
            return
        DenseAutoencoder.doAnimalExperiments(encode_dims=[96*96*3, 512, 128, 64], batch_size=128,
                num_epoch=NUM_EPOCH, base_path=cn.TEST_DIR, is_early_stopping=True, is_verbose=IGNORE_TEST)

if __name__ == '__main__':
    unittest.main()