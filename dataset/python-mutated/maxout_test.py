from numpy.testing import assert_almost_equal
import pytest
import torch
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.modules import Maxout
from allennlp.nn import InitializerApplicator, Initializer
from allennlp.common.testing import AllenNlpTestCase

class TestMaxout(AllenNlpTestCase):

    def test_init_checks_output_dims_consistency(self):
        if False:
            print('Hello World!')
        with pytest.raises(ConfigurationError):
            Maxout(input_dim=2, num_layers=2, output_dims=[5, 4, 3], pool_sizes=4, dropout=0.0)

    def test_init_checks_pool_sizes_consistency(self):
        if False:
            while True:
                i = 10
        with pytest.raises(ConfigurationError):
            Maxout(input_dim=2, num_layers=2, output_dims=5, pool_sizes=[4, 5, 2], dropout=0.0)

    def test_init_checks_dropout_consistency(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(ConfigurationError):
            Maxout(input_dim=2, num_layers=3, output_dims=5, pool_sizes=4, dropout=[0.2, 0.3])

    def test_forward_gives_correct_output(self):
        if False:
            while True:
                i = 10
        params = Params({'input_dim': 2, 'output_dims': 3, 'pool_sizes': 4, 'dropout': 0.0, 'num_layers': 2})
        maxout = Maxout.from_params(params)
        constant_init = Initializer.from_params(Params({'type': 'constant', 'val': 1.0}))
        initializer = InitializerApplicator([('.*', constant_init)])
        initializer(maxout)
        input_tensor = torch.FloatTensor([[-3, 1]])
        output = maxout(input_tensor).data.numpy()
        assert output.shape == (1, 3)
        assert_almost_equal(output, [[-2, -2, -2]])