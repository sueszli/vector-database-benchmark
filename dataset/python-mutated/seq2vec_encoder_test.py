import pytest
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.modules import Seq2VecEncoder
from allennlp.common.testing import AllenNlpTestCase

class TestSeq2VecEncoder(AllenNlpTestCase):

    def test_from_params_builders_encoder_correctly(self):
        if False:
            for i in range(10):
                print('nop')
        params = Params({'type': 'lstm', 'bidirectional': True, 'num_layers': 3, 'input_size': 5, 'hidden_size': 7})
        encoder = Seq2VecEncoder.from_params(params)
        assert encoder.__class__.__name__ == 'LstmSeq2VecEncoder'
        assert encoder._module.__class__.__name__ == 'LSTM'
        assert encoder._module.num_layers == 3
        assert encoder._module.input_size == 5
        assert encoder._module.hidden_size == 7
        assert encoder._module.bidirectional is True
        assert encoder._module.batch_first is True

    def test_from_params_requires_batch_first(self):
        if False:
            while True:
                i = 10
        params = Params({'type': 'lstm', 'batch_first': False})
        with pytest.raises(ConfigurationError):
            Seq2VecEncoder.from_params(params)