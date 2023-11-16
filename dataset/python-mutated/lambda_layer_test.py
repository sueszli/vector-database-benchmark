import numpy as np
import pytest
from keras import layers
from keras import ops
from keras import testing

class LambdaTest(testing.TestCase):

    @pytest.mark.requires_trainable_backend
    def test_lambda_basics(self):
        if False:
            return 10
        self.run_layer_test(layers.Lambda, init_kwargs={'function': ops.square}, input_shape=(2, 3), expected_output_shape=(2, 3), expected_num_trainable_weights=0, expected_num_non_trainable_weights=0, expected_num_seed_generators=0, expected_num_losses=0, supports_masking=False, custom_objects={'square': ops.square})
        self.run_layer_test(layers.Lambda, init_kwargs={'function': ops.square, 'mask': ops.ones((2, 3))}, input_shape=(2, 3, 4), expected_output_shape=(2, 3, 4), expected_num_trainable_weights=0, expected_num_non_trainable_weights=0, expected_num_seed_generators=0, expected_num_losses=0, supports_masking=True, custom_objects={'square': ops.square})

        def stacker(x):
            if False:
                i = 10
                return i + 15
            return ops.concatenate([x, x], axis=1)
        self.run_layer_test(layers.Lambda, init_kwargs={'function': stacker, 'output_shape': (6,)}, input_shape=(2, 3), expected_output_shape=(2, 6), expected_num_trainable_weights=0, expected_num_non_trainable_weights=0, expected_num_seed_generators=0, expected_num_losses=0, supports_masking=False, custom_objects={'stacker': stacker})

        def stacker_shape(s):
            if False:
                i = 10
                return i + 15
            return (s[0], s[1] * 2)
        self.run_layer_test(layers.Lambda, init_kwargs={'function': stacker, 'output_shape': stacker_shape}, input_shape=(2, 3), expected_output_shape=(2, 6), expected_num_trainable_weights=0, expected_num_non_trainable_weights=0, expected_num_seed_generators=0, expected_num_losses=0, supports_masking=False, custom_objects={'stacker': stacker, 'stacker_shape': stacker_shape})

    def test_correctness(self):
        if False:
            while True:
                i = 10
        layer = layers.Lambda(lambda x: x ** 2)
        output = layer(2 * np.ones((2, 3)))
        self.assertAllClose(4 * np.ones((2, 3)), output)
        config = layer.get_config()
        layer = layers.Lambda.from_config(config, safe_mode=False)
        output = layer(2 * np.ones((2, 3)))
        self.assertAllClose(4 * np.ones((2, 3)), output)

    def test_correctness_lambda_shape(self):
        if False:
            for i in range(10):
                print('nop')
        layer = layers.Lambda(lambda x: x ** 2, output_shape=lambda x: x)
        output = layer(2 * np.ones((2, 3)))
        self.assertAllClose(4 * np.ones((2, 3)), output)
        config = layer.get_config()
        layer = layers.Lambda.from_config(config, safe_mode=False)
        output = layer(2 * np.ones((2, 3)))
        self.assertAllClose(4 * np.ones((2, 3)), output)