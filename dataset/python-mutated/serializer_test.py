from typing import Iterator, List
import torch
import pytest
import json
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common import Params
from allennlp.common.util import sanitize
from allennlp.data.data_loaders import TensorDict
from allennlp.evaluation import Serializer
from allennlp.evaluation.serializers import SimpleSerializer

class DummyDataLoader:

    def __init__(self, outputs: List[TensorDict]) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self._outputs = outputs

    def __iter__(self) -> Iterator[TensorDict]:
        if False:
            i = 10
            return i + 15
        yield from self._outputs

    def __len__(self):
        if False:
            return 10
        return len(self._outputs)

    def set_target_device(self, _):
        if False:
            print('Hello World!')
        pass

class TestSerializer(AllenNlpTestCase):

    def setup_method(self):
        if False:
            return 10
        super(TestSerializer, self).setup_method()
        self.postprocessor = Serializer.from_params(Params({}))

    def test_postprocessor_default_implementation(self):
        if False:
            i = 10
            return i + 15
        assert self.postprocessor.to_params().params == {'type': 'simple'}
        assert isinstance(self.postprocessor, SimpleSerializer)

    @pytest.mark.parametrize('batch', [{'Do you want ants?': "Because that's how you get ants.", 'testing': torch.tensor([[1, 2, 3]])}, {}, None], ids=['TestBatch', 'EmptyBatch', 'None'])
    @pytest.mark.parametrize('output_dict', [{"You're": ['Not', [['My']], 'Supervisor']}, {}, None], ids=['TestOutput', 'EmptyOutput', 'None'])
    @pytest.mark.parametrize('postprocess_func', [lambda x: {k.upper(): v for (k, v) in x.items()}, None], ids=['PassedFunction', 'NoPassedFunction'])
    def test_simple_postprocessor_call(self, batch, output_dict, postprocess_func):
        if False:
            while True:
                i = 10
        data_loader = DummyDataLoader([])
        if batch is None or output_dict is None:
            with pytest.raises(ValueError):
                self.postprocessor(batch, output_dict, data_loader)
            return
        expected = json.dumps(sanitize({**batch, **(postprocess_func(output_dict) if postprocess_func else output_dict)}))
        result = self.postprocessor(batch, output_dict, data_loader, postprocess_func)
        assert result == expected