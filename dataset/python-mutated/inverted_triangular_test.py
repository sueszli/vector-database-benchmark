from math import isclose
import torch
from allennlp.common.params import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.training.momentum_schedulers import MomentumScheduler
from allennlp.training.optimizers import Optimizer

class InvertedTriangularTest(AllenNlpTestCase):

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        super().setup_method()
        self.model = torch.nn.Sequential(torch.nn.Linear(10, 10))
        self.base_momentum = 0.9

    def _get_optimizer(self):
        if False:
            while True:
                i = 10
        return Optimizer.from_params(model_parameters=self.model.named_parameters(), params=Params({'type': 'sgd', 'lr': 1.0, 'momentum': self.base_momentum}))

    def test_from_params(self):
        if False:
            print('Hello World!')
        optimizer = self._get_optimizer()
        scheduler = MomentumScheduler.from_params(optimizer=optimizer, params=Params({'type': 'inverted_triangular', 'cool_down': 10, 'warm_up': 10}))
        assert scheduler.cool_down == 10
        assert scheduler.warm_up == 10
        assert scheduler.ratio == 10
        assert scheduler.last_epoch == -1

    def test_basic_schedule(self):
        if False:
            return 10
        optimizer = self._get_optimizer()
        scheduler = MomentumScheduler.from_params(optimizer=optimizer, params=Params({'type': 'inverted_triangular', 'cool_down': 6, 'warm_up': 10, 'ratio': 5}))
        assert optimizer.param_groups[0]['momentum'] == self.base_momentum
        scheduler.step()
        assert isclose(optimizer.param_groups[0]['momentum'], self.base_momentum - (self.base_momentum - self.base_momentum / 5) * (1 / 6))
        scheduler.step()
        assert isclose(optimizer.param_groups[0]['momentum'], self.base_momentum - (self.base_momentum - self.base_momentum / 5) * (2 / 6))
        scheduler.last_epoch = 4
        scheduler.step()
        assert isclose(optimizer.param_groups[0]['momentum'], self.base_momentum / 5)
        scheduler.step()
        assert isclose(optimizer.param_groups[0]['momentum'], self.base_momentum / 5 + (self.base_momentum - self.base_momentum / 5) * (1 / 10))
        scheduler.last_epoch = 14
        scheduler.step()
        assert isclose(optimizer.param_groups[0]['momentum'], self.base_momentum)
        scheduler.step()
        assert isclose(optimizer.param_groups[0]['momentum'], self.base_momentum)
        scheduler.step()
        assert isclose(optimizer.param_groups[0]['momentum'], self.base_momentum)