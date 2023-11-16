import itertools
import unittest
from ray.rllib.core.models.configs import CNNTransposeHeadConfig
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import framework_iterator, ModelChecker
(_, tf, _) = try_import_tf()
(torch, _) = try_import_torch()

class TestCNNTransposeHeads(unittest.TestCase):

    def test_cnn_transpose_heads(self):
        if False:
            i = 10
            return i + 15
        'Tests building DeConv heads properly and checks for correct architecture.'
        inputs_dimss = [[1], [50]]
        initial_image_dims = [4, 4, 96]
        cnn_transpose_filter_specifierss = [([[48, 4, 2], [24, 4, 2], [3, 4, 2]], [32, 32, 3]), ([[48, 4, 2], [24, 4, 2], [1, 4, 2]], [32, 32, 1]), ([[3, 4, 3]], [12, 12, 3]), ([[1, 7, 3]], [12, 12, 1])]
        cnn_transpose_activations = [None, 'relu', 'silu']
        cnn_transpose_use_layernorms = [False, True]
        cnn_transpose_use_biases = [False, True]
        for permutation in itertools.product(inputs_dimss, cnn_transpose_filter_specifierss, cnn_transpose_activations, cnn_transpose_use_layernorms, cnn_transpose_use_biases):
            (inputs_dims, cnn_transpose_filter_specifiers, cnn_transpose_activation, cnn_transpose_use_layernorm, cnn_transpose_use_bias) = permutation
            (cnn_transpose_filter_specifiers, expected_output_dims) = cnn_transpose_filter_specifiers
            print(f'Testing ...\ninputs_dims: {inputs_dims}\ninitial_image_dims: {initial_image_dims}\ncnn_transpose_filter_specifiers: {cnn_transpose_filter_specifiers}\ncnn_transpose_activation: {cnn_transpose_activation}\ncnn_transpose_use_layernorm: {cnn_transpose_use_layernorm}\ncnn_transpose_use_bias: {cnn_transpose_use_bias}\n')
            config = CNNTransposeHeadConfig(input_dims=inputs_dims, initial_image_dims=initial_image_dims, cnn_transpose_filter_specifiers=cnn_transpose_filter_specifiers, cnn_transpose_activation=cnn_transpose_activation, cnn_transpose_use_layernorm=cnn_transpose_use_layernorm, cnn_transpose_use_bias=cnn_transpose_use_bias)
            model_checker = ModelChecker(config)
            for fw in framework_iterator(frameworks=('tf2', 'torch')):
                outputs = model_checker.add(framework=fw)
                self.assertEqual(outputs.shape, (1,) + tuple(expected_output_dims))
            model_checker.check()
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))