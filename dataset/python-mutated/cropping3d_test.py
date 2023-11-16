import numpy as np
import pytest
from absl.testing import parameterized
from keras import layers
from keras import ops
from keras import testing

class Cropping3DTest(testing.TestCase, parameterized.TestCase):

    @parameterized.product(({'dim1_cropping': (1, 2), 'dim1_expected': (1, 5)}, {'dim1_cropping': (0, 2), 'dim1_expected': (0, 5)}, {'dim1_cropping': (1, 0), 'dim1_expected': (1, 7)}), ({'dim2_cropping': (3, 4), 'dim2_expected': (3, 5)}, {'dim2_cropping': (0, 4), 'dim2_expected': (0, 5)}, {'dim2_cropping': (3, 0), 'dim2_expected': (3, 9)}), ({'dim3_cropping': (5, 6), 'dim3_expected': (5, 7)}, {'dim3_cropping': (0, 6), 'dim3_expected': (0, 7)}, {'dim3_cropping': (5, 0), 'dim3_expected': (5, 13)}), ({'data_format': 'channels_first'}, {'data_format': 'channels_last'}))
    @pytest.mark.requires_trainable_backend
    def test_cropping_3d(self, dim1_cropping, dim2_cropping, dim3_cropping, data_format, dim1_expected, dim2_expected, dim3_expected):
        if False:
            for i in range(10):
                print('nop')
        if data_format == 'channels_first':
            inputs = np.random.rand(3, 5, 7, 9, 13)
            expected_output = ops.convert_to_tensor(inputs[:, :, dim1_expected[0]:dim1_expected[1], dim2_expected[0]:dim2_expected[1], dim3_expected[0]:dim3_expected[1]])
        else:
            inputs = np.random.rand(3, 7, 9, 13, 5)
            expected_output = ops.convert_to_tensor(inputs[:, dim1_expected[0]:dim1_expected[1], dim2_expected[0]:dim2_expected[1], dim3_expected[0]:dim3_expected[1], :])
        cropping = (dim1_cropping, dim2_cropping, dim3_cropping)
        self.run_layer_test(layers.Cropping3D, init_kwargs={'cropping': cropping, 'data_format': data_format}, input_data=inputs, expected_output=expected_output)

    @parameterized.product(({'cropping': ((2, 2), (2, 2), (2, 2)), 'expected': ((2, 5), (2, 7), (2, 11))}, {'cropping': (2, 2, 2), 'expected': ((2, 5), (2, 7), (2, 11))}, {'cropping': 2, 'expected': ((2, 5), (2, 7), (2, 11))}), ({'data_format': 'channels_first'}, {'data_format': 'channels_last'}))
    @pytest.mark.requires_trainable_backend
    def test_cropping_3d_with_same_cropping(self, cropping, data_format, expected):
        if False:
            print('Hello World!')
        if data_format == 'channels_first':
            inputs = np.random.rand(3, 5, 7, 9, 13)
            expected_output = ops.convert_to_tensor(inputs[:, :, expected[0][0]:expected[0][1], expected[1][0]:expected[1][1], expected[2][0]:expected[2][1]])
        else:
            inputs = np.random.rand(3, 7, 9, 13, 5)
            expected_output = ops.convert_to_tensor(inputs[:, expected[0][0]:expected[0][1], expected[1][0]:expected[1][1], expected[2][0]:expected[2][1], :])
        self.run_layer_test(layers.Cropping3D, init_kwargs={'cropping': cropping, 'data_format': data_format}, input_data=inputs, expected_output=expected_output)

    def test_cropping_3d_with_dynamic_spatial_dim(self):
        if False:
            while True:
                i = 10
        input_layer = layers.Input(batch_shape=(1, 7, None, 13, 5))
        cropped = layers.Cropping3D(((1, 2), (3, 4), (5, 6)))(input_layer)
        self.assertEqual(cropped.shape, (1, 4, None, 2, 5))

    @parameterized.product(({'cropping': ((3, 6), (0, 0), (0, 0))}, {'cropping': ((0, 0), (5, 8), (0, 0))}, {'cropping': ((0, 0), (0, 0), (7, 6))}), ({'data_format': 'channels_first'}, {'data_format': 'channels_last'}))
    def test_cropping_3d_errors_if_cropping_more_than_available(self, cropping, data_format):
        if False:
            for i in range(10):
                print('nop')
        input_layer = layers.Input(batch_shape=(3, 7, 9, 13, 5))
        with self.assertRaises(ValueError):
            layers.Cropping3D(cropping=cropping, data_format=data_format)(input_layer)

    def test_cropping_3d_errors_if_cropping_argument_invalid(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(ValueError):
            layers.Cropping3D(cropping=(1,))
        with self.assertRaises(ValueError):
            layers.Cropping3D(cropping=(1, 2))
        with self.assertRaises(ValueError):
            layers.Cropping3D(cropping=(1, 2, 3, 4))
        with self.assertRaises(ValueError):
            layers.Cropping3D(cropping='1')
        with self.assertRaises(ValueError):
            layers.Cropping3D(cropping=((1, 2), (3, 4), (5, 6, 7)))
        with self.assertRaises(ValueError):
            layers.Cropping3D(cropping=((1, 2), (3, 4), (5, -6)))
        with self.assertRaises(ValueError):
            layers.Cropping3D(cropping=((1, 2), (3, 4), '5'))

    @parameterized.product(({'cropping': ((8, 1), (1, 1), (1, 1))}, {'cropping': ((1, 1), (10, 1), (1, 1))}, {'cropping': ((1, 1), (1, 1), (14, 1))}), ({'data_format': 'channels_first'}, {'data_format': 'channels_last'}))
    def test_cropping_3d_with_excessive_cropping(self, cropping, data_format):
        if False:
            return 10
        if data_format == 'channels_first':
            shape = (3, 5, 7, 9, 13)
            input_layer = layers.Input(batch_shape=shape)
        else:
            shape = (3, 7, 9, 13, 5)
            input_layer = layers.Input(batch_shape=shape)
        expected_error_msg = 'Values in `cropping` argument should be greater than the'
        with self.assertRaisesRegex(ValueError, expected_error_msg):
            layers.Cropping3D(cropping=cropping, data_format=data_format)(input_layer)