import numpy as np
from absl.testing import parameterized
from keras import layers
from keras import testing

class ZeroPadding1DTest(testing.TestCase, parameterized.TestCase):

    def test_zero_padding_1d(self):
        if False:
            for i in range(10):
                print('nop')
        inputs = np.random.rand(1, 2, 3)
        outputs = layers.ZeroPadding1D(padding=(1, 2))(inputs)
        for index in [0, -1, -2]:
            self.assertAllClose(outputs[:, index, :], 0.0)
        self.assertAllClose(outputs[:, 1:-2, :], inputs)

    @parameterized.named_parameters(('one_tuple', (2, 2)), ('one_int', 2))
    def test_zero_padding_1d_with_same_padding(self, padding):
        if False:
            while True:
                i = 10
        inputs = np.random.rand(1, 2, 3)
        outputs = layers.ZeroPadding1D(padding=padding)(inputs)
        for index in [0, 1, -1, -2]:
            self.assertAllClose(outputs[:, index, :], 0.0)
        self.assertAllClose(outputs[:, 2:-2, :], inputs)

    def test_zero_padding_1d_with_dynamic_spatial_dim(self):
        if False:
            return 10
        input_layer = layers.Input(batch_shape=(1, None, 3))
        padded = layers.ZeroPadding1D((1, 2))(input_layer)
        self.assertEqual(padded.shape, (1, None, 3))

    def test_zero_padding_1d_errors_if_padding_argument_invalid(self):
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError):
            layers.ZeroPadding1D(padding=(1,))
        with self.assertRaises(ValueError):
            layers.ZeroPadding1D(padding=(1, 2, 3))
        with self.assertRaises(ValueError):
            layers.ZeroPadding1D(padding='1')