import numpy as np
import pytest
from keras import constraints
from keras import layers
from keras import regularizers
from keras import testing

class GroupNormalizationTest(testing.TestCase):

    @pytest.mark.requires_trainable_backend
    def test_groupnorm(self):
        if False:
            i = 10
            return i + 15
        self.run_layer_test(layers.GroupNormalization, init_kwargs={'gamma_regularizer': regularizers.L2(0.01), 'beta_regularizer': regularizers.L2(0.01)}, input_shape=(3, 4, 32), expected_output_shape=(3, 4, 32), expected_num_trainable_weights=2, expected_num_non_trainable_weights=0, expected_num_seed_generators=0, expected_num_losses=2, supports_masking=True)
        self.run_layer_test(layers.GroupNormalization, init_kwargs={'groups': 4, 'gamma_constraint': constraints.UnitNorm(), 'beta_constraint': constraints.UnitNorm()}, input_shape=(3, 4, 4), expected_output_shape=(3, 4, 4), expected_num_trainable_weights=2, expected_num_non_trainable_weights=0, expected_num_seed_generators=0, expected_num_losses=0, supports_masking=True)

    def test_undefined_dim_error(self):
        if False:
            i = 10
            return i + 15
        inputs = layers.Input(shape=(2, 2, 2, None))
        layer = layers.GroupNormalization()
        with self.assertRaisesRegex(ValueError, 'input tensor should have a defined dimension but the layer received an input with shape'):
            _ = layer(inputs)

    def test_groups_bigger_than_dim_error(self):
        if False:
            while True:
                i = 10
        inputs = np.ones(shape=(2, 2, 2, 4))
        layer = layers.GroupNormalization(groups=5)
        with self.assertRaisesRegex(ValueError, 'cannot be more than the number of channels'):
            _ = layer(inputs)

    def test_groups_not_a_multiple_of_dim_error(self):
        if False:
            print('Hello World!')
        inputs = np.ones(shape=(2, 2, 2, 4))
        layer = layers.GroupNormalization(groups=3)
        with self.assertRaisesRegex(ValueError, 'must be a multiple of the number of channels'):
            _ = layer(inputs)

    def test_groups_instance_norm(self):
        if False:
            return 10
        instance_norm_layer_1 = layers.GroupNormalization(groups=-1, axis=-1, scale=False, center=False)
        instance_norm_layer_2 = layers.GroupNormalization(groups=4, axis=-1, scale=False, center=False)
        inputs = np.array([[[-1.0, 1.0, 0, 2.0], [1.0, 3.0, -4, -2.0]]])
        outputs_1 = instance_norm_layer_1(inputs)
        outputs_2 = instance_norm_layer_2(inputs)
        self.assertAllClose(outputs_1, outputs_2)

    def test_correctness_instance_norm(self):
        if False:
            return 10
        instance_norm_layer = layers.GroupNormalization(groups=4, axis=-1, scale=False, center=False)
        inputs = np.array([[[-1.0, 1.0, 0, 2.0], [1.0, 3.0, -4, -2.0]]])
        expected_instance_norm_output = np.array([[[-1.0, -1.0, 1.0, 1.0], [1.0, 1.0, -1.0, -1.0]]])
        self.assertAllClose(instance_norm_layer(inputs), expected_instance_norm_output, atol=0.001)

    def test_correctness_1d(self):
        if False:
            return 10
        layer_with_1_group = layers.GroupNormalization(groups=1, axis=-1, scale=False, center=False)
        layer_with_2_groups = layers.GroupNormalization(groups=2, axis=1, scale=False, center=False)
        inputs = np.array([[-1.0, -1.0, 1.0, 1.0, 2.0, 2.0, 0, -2.0]])
        expected_output_1_group = np.array([[-0.898, -0.898, 0.539, 0.539, 1.257, 1.257, -0.18, -1.616]])
        self.assertAllClose(layer_with_1_group(inputs), expected_output_1_group, atol=0.001)
        expected_output_2_groups = np.array([[-1.0, -1.0, 1.0, 1.0, 0.904, 0.904, -0.301, -1.507]])
        self.assertAllClose(layer_with_2_groups(inputs), expected_output_2_groups, atol=0.001)

    def test_correctness_2d(self):
        if False:
            while True:
                i = 10
        layer_with_1_group = layers.GroupNormalization(groups=1, axis=-1, scale=False, center=False)
        layer_with_2_groups = layers.GroupNormalization(groups=2, axis=2, scale=False, center=False)
        inputs = np.array([[[-1.0, -1.0, 2.0, 2.0], [1.0, 1.0, 0, -2.0]]])
        expected_output_1_group = np.array([[[-0.898, -0.898, 1.257, 1.257], [0.539, 0.539, -0.18, -1.616]]])
        self.assertAllClose(layer_with_1_group(inputs), expected_output_1_group, atol=0.001)
        expected_output_2_groups = np.array([[[-1.0, -1.0, 0.904, 0.904], [1.0, 1.0, -0.301, -1.507]]])
        self.assertAllClose(layer_with_2_groups(inputs), expected_output_2_groups, atol=0.001)

    def test_broadcasting_2d_channels_first(self):
        if False:
            return 10
        x = np.arange(16).reshape((1, 4, 2, 2)).astype('float32')
        x = layers.GroupNormalization(groups=2, axis=1)(x)
        self.assertAllClose(x, np.array([[[[-1.5274, -1.091], [-0.6546, -0.2182]], [[0.2182, 0.6546], [1.091, 1.5274]], [[-1.5274, -1.091], [-0.6546, -0.2182]], [[0.2182, 0.6546], [1.091, 1.5274]]]]), atol=0.001)