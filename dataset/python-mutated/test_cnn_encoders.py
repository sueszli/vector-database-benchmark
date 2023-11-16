import itertools
import unittest
from ray.rllib.core.models.base import ENCODER_OUT
from ray.rllib.core.models.configs import CNNEncoderConfig
from ray.rllib.models.utils import get_filter_config
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import framework_iterator, ModelChecker
(_, tf, _) = try_import_tf()
(torch, _) = try_import_torch()

class TestCNNEncoders(unittest.TestCase):

    def test_cnn_encoders(self):
        if False:
            return 10
        'Tests building CNN encoders properly and checks for correct architecture.'
        inputs_dimss = [[480, 640, 3], [480, 640, 1], [240, 320, 3], [240, 320, 1], [96, 96, 3], [96, 96, 1], [84, 84, 3], [84, 84, 1], [64, 64, 3], [64, 64, 1], [42, 42, 3], [42, 42, 1], [10, 10, 3]]
        cnn_activations = [None, 'linear', 'relu']
        cnn_use_layernorms = [False, True]
        cnn_use_biases = [False, True]
        for permutation in itertools.product(inputs_dimss, cnn_activations, cnn_use_layernorms, cnn_use_biases):
            (inputs_dims, cnn_activation, cnn_use_layernorm, cnn_use_bias) = permutation
            filter_specifiers = get_filter_config(inputs_dims)
            print(f'Testing ...\ninput_dims: {inputs_dims}\ncnn_filter_specifiers: {filter_specifiers}\ncnn_activation: {cnn_activation}\ncnn_use_layernorm: {cnn_use_layernorm}\ncnn_use_bias: {cnn_use_bias}\n')
            config = CNNEncoderConfig(input_dims=inputs_dims, cnn_filter_specifiers=filter_specifiers, cnn_activation=cnn_activation, cnn_use_layernorm=cnn_use_layernorm, cnn_use_bias=cnn_use_bias)
            model_checker = ModelChecker(config)
            for fw in framework_iterator(frameworks=('tf2', 'torch')):
                outputs = model_checker.add(framework=fw)
                self.assertEqual(outputs[ENCODER_OUT].shape, (1, config.output_dims[0]))
            model_checker.check()

    def test_cnn_encoders_valid_padding(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests building CNN encoders with valid padding.'
        inputs_dims = [42, 42, 3]
        filter_specifiers = [[16, 4, 2, 'same'], [32, 4, 2], [256, 11, 1, 'valid']]
        config = CNNEncoderConfig(input_dims=inputs_dims, cnn_filter_specifiers=filter_specifiers)
        model_checker = ModelChecker(config)
        for fw in framework_iterator(frameworks=('tf2', 'torch')):
            outputs = model_checker.add(framework=fw)
            self.assertEqual(outputs[ENCODER_OUT].shape, (1, config.output_dims[0]))
        model_checker.check()
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))