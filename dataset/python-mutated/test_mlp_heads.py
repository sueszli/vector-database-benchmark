import itertools
import unittest
from ray.rllib.core.models.configs import MLPHeadConfig, FreeLogStdMLPHeadConfig
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import framework_iterator, ModelChecker
(_, tf, _) = try_import_tf()
(torch, nn) = try_import_torch()

class TestMLPHeads(unittest.TestCase):

    def test_mlp_heads(self):
        if False:
            return 10
        'Tests building MLP heads properly and checks for correct architecture.'
        inputs_dims_configs = [[1], [50]]
        list_of_hidden_layer_dims = [[], [1], [64, 64], [512, 512]]
        hidden_layer_activations = ['linear', 'relu', 'swish']
        hidden_layer_use_layernorms = [False, True]
        output_dims = [2, 50]
        output_activations = hidden_layer_activations
        hidden_use_biases = [False, True]
        output_use_biases = [False, True]
        free_stds = [False, True]
        for permutation in itertools.product(inputs_dims_configs, list_of_hidden_layer_dims, hidden_layer_activations, hidden_layer_use_layernorms, output_activations, output_dims, hidden_use_biases, output_use_biases, free_stds):
            (inputs_dims, hidden_layer_dims, hidden_layer_activation, hidden_layer_use_layernorm, output_activation, output_dim, hidden_use_bias, output_use_bias, free_std) = permutation
            print(f'Testing ...\ninput_dims: {inputs_dims}\nhidden_layer_dims: {hidden_layer_dims}\nhidden_layer_activation: {hidden_layer_activation}\nhidden_layer_use_layernorm: {hidden_layer_use_layernorm}\noutput_activation: {output_activation}\noutput_dim: {output_dim}\nfree_std: {free_std}\nhidden_use_bias: {hidden_use_bias}\noutput_use_bias: {output_use_bias}\n')
            config_cls = FreeLogStdMLPHeadConfig if free_std else MLPHeadConfig
            config = config_cls(input_dims=inputs_dims, hidden_layer_dims=hidden_layer_dims, hidden_layer_activation=hidden_layer_activation, hidden_layer_use_layernorm=hidden_layer_use_layernorm, hidden_layer_use_bias=hidden_use_bias, output_layer_dim=output_dim, output_layer_activation=output_activation, output_layer_use_bias=output_use_bias)
            model_checker = ModelChecker(config)
            for fw in framework_iterator(frameworks=('tf2', 'torch')):
                outputs = model_checker.add(framework=fw)
                self.assertEqual(outputs.shape, (1, output_dim))
            model_checker.check()
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))