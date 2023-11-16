from bigdl.chronos.utils import LazyImport
torch = LazyImport('torch')
AsymWeightLoss = LazyImport('bigdl.chronos.pytorch.loss.AsymWeightLoss')
from unittest import TestCase
import pytest
from ... import op_torch

@op_torch
class TestChronosPytorchLoss(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def tearDown(self):
        if False:
            return 10
        pass

    def test_asym_weight_loss(self):
        if False:
            return 10
        y = torch.rand(100, 10, 2)
        yhat_high = y + 1
        yhat_low = y - 1
        loss = AsymWeightLoss(underestimation_penalty=2)
        assert loss(yhat_high, y) < loss(yhat_low, y)
        loss = AsymWeightLoss(underestimation_penalty=0.5)
        assert loss(yhat_high, y) > loss(yhat_low, y)