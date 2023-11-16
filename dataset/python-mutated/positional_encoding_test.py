import copy
import torch
import numpy as np
from allennlp.common import Params
from allennlp.modules.transformer import SinusoidalPositionalEncoding
from allennlp.common.testing import AllenNlpTestCase

class TestSinusoidalPositionalEncoding(AllenNlpTestCase):

    def setup_method(self):
        if False:
            return 10
        super().setup_method()
        self.params_dict = {'min_timescale': 1.0, 'max_timescale': 10000.0}
        params = Params(copy.deepcopy(self.params_dict))
        self.positional_encoding = SinusoidalPositionalEncoding.from_params(params)

    def test_can_construct_from_params(self):
        if False:
            return 10
        assert self.positional_encoding.min_timescale == self.params_dict['min_timescale']
        assert self.positional_encoding.max_timescale == self.params_dict['max_timescale']

    def test_forward(self):
        if False:
            i = 10
            return i + 15
        tensor2tensor_result = np.asarray([[0.0, 0.0, 1.0, 1.0], [0.841470957, 9.99999902e-05, 0.540302277, 1.0], [0.909297407, 0.00019999998, -0.416146845, 1.0]])
        tensor = torch.zeros([2, 3, 4])
        result = self.positional_encoding(tensor)
        np.testing.assert_almost_equal(result[0].detach().cpu().numpy(), tensor2tensor_result)
        np.testing.assert_almost_equal(result[1].detach().cpu().numpy(), tensor2tensor_result)
        tensor2tensor_result = np.asarray([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0], [0.841470957, 0.00999983307, 9.99999902e-05, 0.540302277, 0.999949992, 1.0, 0.0], [0.909297407, 0.0199986659, 0.00019999998, -0.416146815, 0.999800026, 1.0, 0.0]])
        tensor = torch.zeros([2, 3, 7])
        result = self.positional_encoding(tensor)
        np.testing.assert_almost_equal(result[0].detach().cpu().numpy(), tensor2tensor_result)
        np.testing.assert_almost_equal(result[1].detach().cpu().numpy(), tensor2tensor_result)