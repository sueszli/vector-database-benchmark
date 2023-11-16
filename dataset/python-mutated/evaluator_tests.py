from typing import Iterator, List, Dict
import torch
import pytest
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.data_loaders import TensorDict
from allennlp.models import Model
from allennlp.evaluation import Evaluator
from allennlp.common import Params

class DummyDataLoader:

    def __init__(self, outputs: List[TensorDict]) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self._outputs = outputs

    def __iter__(self) -> Iterator[TensorDict]:
        if False:
            return 10
        yield from self._outputs

    def __len__(self):
        if False:
            return 10
        return len(self._outputs)

    def set_target_device(self, _):
        if False:
            while True:
                i = 10
        pass

class DummyModel(Model):

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        super().__init__(None)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        if False:
            return 10
        return kwargs

class TestEvaluator(AllenNlpTestCase):

    def setup_method(self):
        if False:
            return 10
        self.evaluator = Evaluator.from_params(Params({'batch_postprocessor': 'simple'}))

    def test_evaluate_calculates_average_loss(self):
        if False:
            return 10
        losses = [7.0, 9.0, 8.0]
        outputs = [{'loss': torch.Tensor([loss])} for loss in losses]
        data_loader = DummyDataLoader(outputs)
        metrics = self.evaluator(DummyModel(), data_loader, '')
        assert metrics['loss'] == pytest.approx(8.0)

    def test_evaluate_calculates_average_loss_with_weights(self):
        if False:
            print('Hello World!')
        losses = [7.0, 9.0, 8.0]
        weights = [10, 2, 1.5]
        inputs = zip(losses, weights)
        outputs = [{'loss': torch.Tensor([loss]), 'batch_weight': torch.Tensor([weight])} for (loss, weight) in inputs]
        data_loader = DummyDataLoader(outputs)
        metrics = self.evaluator(DummyModel(), data_loader, 'batch_weight')
        assert metrics['loss'] == pytest.approx((70 + 18 + 12) / 13.5)

    def test_to_params(self):
        if False:
            print('Hello World!')
        assert self.evaluator.to_params() == {'type': 'simple', 'cuda_device': -1, 'batch_postprocessor': {'type': 'simple'}}