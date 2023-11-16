from bigdl.chronos.utils import LazyImport
torch = LazyImport('torch')
LinexLoss = LazyImport('bigdl.chronos.pytorch.loss.LinexLoss')
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
            for i in range(10):
                print('nop')
        pass

    def test_linex_loss(self):
        if False:
            for i in range(10):
                print('nop')
        y = torch.rand(100, 10, 2)
        yhat_high = y + 1
        yhat_low = y - 1
        loss = LinexLoss(1)
        assert loss(yhat_high, y) < loss(yhat_low, y)
        loss = LinexLoss(-1)
        assert loss(yhat_high, y) > loss(yhat_low, y)