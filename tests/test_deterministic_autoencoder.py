import unittest
from src.deterministic_autoencoder import DeterministicAutoencoder  # type: ignore
import src.util as util  # type: ignore


IGNORE_TEST = False
IS_PLOT = False
X_TRAIN, X_TEST = util.getProcessedMNISTData()

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
        import pdb; pdb.set_trace()
        self.assertEqual(dae.encode_dims, encode_dims)
        self.assertEqual(dae.num_hidden_layers, 3)
        self.assertEqual(dae.compression_factor, 784/32)
        self.assertIsNotNone(dae.autoencoder)
        self.assertIsNotNone(dae.encoder)
        self.assertIsNotNone(dae.decoder)
        self.assertIsNone(dae.history)
        

if __name__ == '__main__':
    unittest.main()