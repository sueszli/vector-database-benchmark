from numpy.testing import assert_almost_equal
import torch
from allennlp.modules import ResidualWithLayerDropout
from allennlp.common.testing import AllenNlpTestCase

class TestResidualWithLayerDropout(AllenNlpTestCase):

    def test_dropout_works_for_training(self):
        if False:
            while True:
                i = 10
        layer_input_tensor = torch.FloatTensor([[2, 1], [-3, -2]])
        layer_output_tensor = torch.FloatTensor([[1, 3], [2, -1]])
        residual_with_layer_dropout = ResidualWithLayerDropout(1)
        residual_with_layer_dropout.train()
        result = residual_with_layer_dropout(layer_input_tensor, layer_output_tensor).data.numpy()
        assert result.shape == (2, 2)
        assert_almost_equal(result, [[2, 1], [-3, -2]])
        result = residual_with_layer_dropout(layer_input_tensor, layer_output_tensor, 1, 1).data.numpy()
        assert result.shape == (2, 2)
        assert_almost_equal(result, [[2, 1], [-3, -2]])
        residual_with_layer_dropout = ResidualWithLayerDropout(0.0)
        residual_with_layer_dropout.train()
        result = residual_with_layer_dropout(layer_input_tensor, layer_output_tensor).data.numpy()
        assert result.shape == (2, 2)
        assert_almost_equal(result, [[2 + 1, 1 + 3], [-3 + 2, -2 - 1]])

    def test_dropout_works_for_testing(self):
        if False:
            for i in range(10):
                print('nop')
        layer_input_tensor = torch.FloatTensor([[2, 1], [-3, -2]])
        layer_output_tensor = torch.FloatTensor([[1, 3], [2, -1]])
        residual_with_layer_dropout = ResidualWithLayerDropout(0.2)
        residual_with_layer_dropout.eval()
        result = residual_with_layer_dropout(layer_input_tensor, layer_output_tensor).data.numpy()
        assert result.shape == (2, 2)
        assert_almost_equal(result, [[2 + 1 * 0.8, 1 + 3 * 0.8], [-3 + 2 * 0.8, -2 - 1 * 0.8]])