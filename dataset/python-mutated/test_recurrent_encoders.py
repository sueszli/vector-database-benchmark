import itertools
import unittest
from ray.rllib.core.models.base import ENCODER_OUT, STATE_OUT
from ray.rllib.core.models.configs import RecurrentEncoderConfig
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import framework_iterator, ModelChecker
(_, tf, _) = try_import_tf()
(torch, _) = try_import_torch()

class TestRecurrentEncoders(unittest.TestCase):

    def test_gru_encoders(self):
        if False:
            while True:
                i = 10
        'Tests building GRU encoders properly and checks for correct architecture.'
        inputs_dimss = [[1], [100]]
        num_layerss = [1, 4]
        hidden_dims = [128, 256]
        use_biases = [False, True]
        for permutation in itertools.product(inputs_dimss, num_layerss, hidden_dims, use_biases):
            (inputs_dims, num_layers, hidden_dim, use_bias) = permutation
            print(f'Testing ...\ninput_dims: {inputs_dims}\nnum_layers: {num_layers}\nhidden_dim: {hidden_dim}\nuse_bias: {use_bias}\n')
            config = RecurrentEncoderConfig(recurrent_layer_type='gru', input_dims=inputs_dims, num_layers=num_layers, hidden_dim=hidden_dim, use_bias=use_bias)
            model_checker = ModelChecker(config)
            for fw in framework_iterator(frameworks=('tf2', 'torch')):
                outputs = model_checker.add(framework=fw)
                self.assertEqual(outputs[ENCODER_OUT].shape, (1, 1, config.output_dims[0]))
                self.assertEqual(outputs[STATE_OUT]['h'].shape, (1, num_layers, hidden_dim))
            model_checker.check()

    def test_lstm_encoders(self):
        if False:
            return 10
        'Tests building LSTM encoders properly and checks for correct architecture.'
        inputs_dimss = [[1], [100]]
        num_layerss = [1, 3]
        hidden_dims = [16, 128]
        use_biases = [False, True]
        for permutation in itertools.product(inputs_dimss, num_layerss, hidden_dims, use_biases):
            (inputs_dims, num_layers, hidden_dim, use_bias) = permutation
            print(f'Testing ...\ninput_dims: {inputs_dims}\nnum_layers: {num_layers}\nhidden_dim: {hidden_dim}\nuse_bias: {use_bias}\n')
            config = RecurrentEncoderConfig(recurrent_layer_type='lstm', input_dims=inputs_dims, num_layers=num_layers, hidden_dim=hidden_dim, use_bias=use_bias)
            model_checker = ModelChecker(config)
            for fw in framework_iterator(frameworks=('tf2', 'torch')):
                outputs = model_checker.add(framework=fw)
                self.assertEqual(outputs[ENCODER_OUT].shape, (1, 1, config.output_dims[0]))
                self.assertEqual(outputs[STATE_OUT]['h'].shape, (1, num_layers, hidden_dim))
                self.assertEqual(outputs[STATE_OUT]['c'].shape, (1, num_layers, hidden_dim))
            if use_bias is False:
                model_checker.check()
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))