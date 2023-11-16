from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, List, Tuple
import torch
import pytest
from allennlp.common import Lazy, Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.training import Trainer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler, SlantedTriangular
from allennlp.training.optimizers import Optimizer

def is_hat_shaped(learning_rates: List[float]):
    if False:
        for i in range(10):
            print('nop')
    '\n    Check if the list of learning rates is "hat" shaped, i.e.,\n    increases then decreases\n    '
    has_increasing_segment = False
    has_decreasing_segment = False
    for k in range(1, len(learning_rates)):
        delta = learning_rates[k] - learning_rates[k - 1]
        if delta > 1e-08:
            has_increasing_segment = True
            if has_decreasing_segment:
                return False
        elif delta < -1e-08:
            if not has_increasing_segment:
                return False
            has_decreasing_segment = True
        else:
            pass
    return has_increasing_segment and has_decreasing_segment

class SlantedTriangularTest(AllenNlpTestCase):

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        super().setup_method()
        self.model = torch.nn.Sequential(OrderedDict([('lin1', torch.nn.Linear(10, 10)), ('lin2', torch.nn.Linear(10, 10))]))

    def _get_optimizer(self, lr: float=1.0):
        if False:
            return 10
        optimizer_params = Params({'type': 'sgd', 'lr': lr})
        optimizer_params['parameter_groups'] = [[[f'^{m}'], {}] for m in self.model._modules]
        return Optimizer.from_params(model_parameters=self.model.named_parameters(), params=optimizer_params)

    def _run_scheduler_get_lrs(self, params, num_steps_per_epoch):
        if False:
            return 10
        optimizer = self._get_optimizer()
        params['type'] = 'slanted_triangular'
        scheduler = LearningRateScheduler.from_params(optimizer=optimizer, params=Params(deepcopy(params)))
        lrs = []
        batch_num_total = 0
        for epoch in range(params['num_epochs']):
            for _ in range(num_steps_per_epoch):
                batch_num_total += 1
                lrs.append([param_group['lr'] * float(param_group['params'][0].requires_grad) for param_group in optimizer.param_groups[:2]])
                scheduler.step_batch(batch_num_total)
                if params.get('gradual_unfreezing') and epoch == 0:
                    assert scheduler.freezing_current
            scheduler.step(None)
        return lrs

    def test_is_hat_shaped(self):
        if False:
            i = 10
            return i + 15
        assert not is_hat_shaped([0.0] * 10)
        assert not is_hat_shaped([float(k) for k in range(10)])
        assert not is_hat_shaped([float(10 - k) for k in range(10)])
        assert is_hat_shaped([float(k) for k in range(10)] + [float(10 - k) for k in range(10)])
        assert not is_hat_shaped([float(k) for k in range(10)] + [float(10 - k) for k in range(10)] + [float(k) for k in range(10)])

    def test_from_params_in_trainer(self):
        if False:
            for i in range(10):
                print('nop')
        params = Params({'num_epochs': 5, 'learning_rate_scheduler': {'type': 'slanted_triangular', 'gradual_unfreezing': True, 'discriminative_fine_tuning': True, 'decay_factor': 0.5}})
        instances = [1] * 40
        optim = self._get_optimizer()
        trainer = Trainer.from_params(model=self.model, optimizer=Lazy(lambda **kwargs: optim), serialization_dir=self.TEST_DIR, params=params, data_loader=SimpleDataLoader(instances, batch_size=10))
        assert isinstance(trainer._learning_rate_scheduler, SlantedTriangular)
        assert trainer._learning_rate_scheduler.num_epochs == 5
        assert trainer._learning_rate_scheduler.num_steps_per_epoch == 4
        params = Params({'num_epochs': 5, 'learning_rate_scheduler': {'type': 'slanted_triangular', 'num_epochs': 3, 'gradual_unfreezing': True, 'discriminative_fine_tuning': True, 'decay_factor': 0.5}})
        trainer = Trainer.from_params(model=self.model, optimizer=Lazy(lambda **kwargs: optim), serialization_dir=self.TEST_DIR, params=params, data_loader=SimpleDataLoader(instances, batch_size=10))
        assert trainer._learning_rate_scheduler.num_epochs == 3

    def test_from_params(self):
        if False:
            print('Hello World!')
        optim = self._get_optimizer()
        sched = LearningRateScheduler.from_params(optimizer=optim, params=Params({'type': 'slanted_triangular', 'num_epochs': 5, 'num_steps_per_epoch': 10, 'gradual_unfreezing': True, 'discriminative_fine_tuning': True, 'decay_factor': 0.5}))
        assert sched.num_epochs == 5
        assert sched.num_steps_per_epoch == 10
        assert sched.gradual_unfreezing is True
        assert sched.freezing_current is True
        assert len(optim.param_groups) == 3
        assert not optim.param_groups[-1]['params']
        assert optim.param_groups[-2]['lr'] == 1.0 / sched.ratio
        assert optim.param_groups[-3]['lr'] == 0.5 / sched.ratio
        with pytest.raises(ConfigurationError):
            LearningRateScheduler.from_params(optimizer=optim, params=Params({'type': 'slanted_triangular', 'num_epochs': 5}))
            LearningRateScheduler.from_params(optimizer=optim, params=Params({'type': 'slanted_triangular', 'num_steps_epochs': 10}))

    def test_schedules(self):
        if False:
            while True:
                i = 10
        slanted_triangular_cases: List[Tuple[Dict[str, Any], List[Tuple[int, int, float]]]] = [({'num_epochs': 5, 'num_steps_per_epoch': 10, 'gradual_unfreezing': True}, [(0, 1, 0.03125), (0, 0, 0.0), (1, 1, 1.0), (1, 0, 0.0), (9, 1, 0.138888), (9, 0, 0.0), (10, 1, 0.03125), (10, 0, 0.03125), (14, 1, 1.0), (14, 0, 1.0), (49, 1, 0.05815972), (49, 0, 0.05815972)]), ({'num_epochs': 5, 'num_steps_per_epoch': 10, 'discriminative_fine_tuning': True, 'decay_factor': 0.5}, [(0, 1, 0.03125), (0, 0, 0.015625), (5, 1, 1.0), (5, 0, 0.5), (49, 1, 0.052777), (49, 0, 0.026388)]), ({'num_epochs': 5, 'num_steps_per_epoch': 10, 'gradual_unfreezing': True, 'discriminative_fine_tuning': True, 'decay_factor': 0.5}, [(0, 1, 0.03125), (0, 0, 0.0), (1, 1, 1.0), (1, 0, 0.0), (9, 1, 0.138888), (9, 0, 0.0), (10, 1, 0.03125), (10, 0, 0.015625), (14, 1, 1.0), (14, 0, 0.5), (49, 1, 0.0581597222), (49, 0, 0.0290798611)])]
        for (params, lr_checks) in slanted_triangular_cases:
            lrs = self._run_scheduler_get_lrs(params, params['num_steps_per_epoch'])
            for (it, layer, lr) in lr_checks:
                lr_check = round(lr, 5)
                lr = round(lrs[it][layer], 5)
                assert lr == lr_check, f'Learning rate {lr} at iteration {it} at layer {layer} != {lr_check}.'

    def test_schedules_num_steps_per_epoch(self):
        if False:
            return 10
        for gradual_unfreezing in [True, False]:
            for discriminative_fine_tuning in [True, False]:
                for num_actual_steps_per_epoch in [7, 11]:
                    params = {'num_epochs': 5, 'num_steps_per_epoch': 10, 'gradual_unfreezing': gradual_unfreezing, 'discriminative_fine_tuning': discriminative_fine_tuning}
                    lrs = self._run_scheduler_get_lrs(params, num_actual_steps_per_epoch)
                    first_layer_lrs = [rates[0] for rates in lrs]
                    second_layer_lrs = [rates[1] for rates in lrs]
                    if gradual_unfreezing:
                        assert max(first_layer_lrs[:num_actual_steps_per_epoch]) < 1e-08
                        assert min(first_layer_lrs[:num_actual_steps_per_epoch]) > -1e-08
                        assert is_hat_shaped(first_layer_lrs[num_actual_steps_per_epoch:])
                        assert is_hat_shaped(second_layer_lrs[:num_actual_steps_per_epoch])
                        assert is_hat_shaped(second_layer_lrs[num_actual_steps_per_epoch:])
                    else:
                        assert is_hat_shaped(first_layer_lrs)
                        assert is_hat_shaped(second_layer_lrs)