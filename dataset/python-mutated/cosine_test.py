from copy import deepcopy
from typing import Dict, Any
import torch
import pytest
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.optimizers import Optimizer

class CosineWithRestartsTest(AllenNlpTestCase):

    def setup_method(self):
        if False:
            while True:
                i = 10
        super().setup_method()
        self.model = torch.nn.Sequential(torch.nn.Linear(10, 10))
        self.cosine_schedule_cases = [(30, {'t_initial': 30, 't_mul': 1.0}, [(0, 1.0), (15, 0.5000000000000001), (29, 0.0027390523158632996)], [10, 14]), (10, {'t_initial': 1, 't_mul': 2.0}, [(0, 1.0), (1, 1.0), (2, 0.5), (3, 1.0)], [1, 3]), (30, {'t_initial': 1, 't_mul': 1.0}, [(0, 1.0), (15, 1.0), (29, 1.0)], []), (60, {'t_initial': 30, 't_mul': 1.0}, [(0, 1.0), (15, 0.5000000000000001), (29, 0.0027390523158632996), (30, 1.0), (45, 0.5000000000000001), (59, 0.0027390523158632996)], [30, 35]), (60, {'t_initial': 30, 't_mul': 1.0, 'eta_mul': 0.5}, [(0, 1.0), (15, 0.5000000000000001), (29, 0.0027390523158632996), (30, 0.5)], []), (100, {'t_initial': 30, 't_mul': 1.5}, [(0, 1.0), (29, 0.0027390523158632996), (30, 1.0), (74, 0.0012179748700879012)], []), (210, {'t_initial': 30, 't_mul': 2}, [(0, 1.0), (29, 0.0027390523158632996), (30, 1.0), (89, 0.0006852326227130834), (90, 1.0), (209, 0.00017133751222137006)], []), (210, {'t_initial': 30, 't_mul': 2, 'eta_mul': 0.5}, [(0, 1.0), (30, 0.5), (90, 0.25)], [29, 90]), (150, {'t_initial': 30, 't_mul': 1}, [(0, 1.0), (29, 0.0027390523158632996), (30, 1.0), (59, 0.0027390523158632996), (60, 1.0), (89, 0.0027390523158632996), (90, 1.0)], []), (10, {'t_initial': 1, 't_mul': 1, 'eta_mul': 0.5}, [(0, 1.0), (1, 0.5), (2, 0.25)], [])]

    def _get_optimizer(self, lr: float=1.0):
        if False:
            i = 10
            return i + 15
        return Optimizer.from_params(model_parameters=self.model.named_parameters(), params=Params({'type': 'sgd', 'lr': lr}))

    def test_from_params(self):
        if False:
            return 10
        'Make sure `from_params` initializes an instance properly.'
        optim = self._get_optimizer()
        sched = LearningRateScheduler.from_params(optimizer=optim, params=Params({'type': 'cosine', 't_initial': 5}))
        assert sched.t_initial == 5
        assert sched.last_epoch == -1
        assert optim.param_groups[0]['lr'] == 1.0
        with pytest.raises(ConfigurationError):
            LearningRateScheduler.from_params(optimizer=optim, params=Params({'type': 'cosine'}))

    def test_schedules(self):
        if False:
            return 10
        'Make sure the math is correct.'
        for (epochs, params, lr_checks, _) in self.cosine_schedule_cases:
            optimizer = self._get_optimizer()
            params['type'] = 'cosine'
            scheduler = LearningRateScheduler.from_params(optimizer=optimizer, params=Params(params))
            lrs = [optimizer.param_groups[0]['lr']]
            for _ in range(epochs):
                scheduler.step()
                lrs.append(optimizer.param_groups[0]['lr'])
            for (it, lr) in lr_checks:
                assert lrs[it] == pytest.approx(lr), f'Iteration {it}: {lrs[it]} != {lr}'

    def test_schedules_with_save_and_resume(self):
        if False:
            for i in range(10):
                print('nop')
        'Make sure scheduler will resume with the right state.'

        def init_and_restore_scheduler(optimizer: torch.optim.Optimizer, params: Dict[str, Any], state_dict: Dict[str, Any]=None):
            if False:
                print('Hello World!')
            '\n            Initialize a new scheduler and optionally restore its state from\n            a checkpoint.\n            '
            params['type'] = 'cosine'
            scheduler = LearningRateScheduler.from_params(optimizer=optimizer, params=Params(deepcopy(params)))
            if state_dict is not None:
                scheduler.load_state_dict(state_dict)
            return scheduler
        for (epochs, params, lr_checks, checkpoints) in self.cosine_schedule_cases:
            optimizer = self._get_optimizer()
            scheduler = init_and_restore_scheduler(optimizer, params)
            state = scheduler.state_dict()
            lrs = [optimizer.param_groups[0]['lr']]
            for epoch in range(epochs):
                if epoch in checkpoints:
                    scheduler = init_and_restore_scheduler(optimizer, params, state_dict=state)
                scheduler.step(1)
                lrs.append(optimizer.param_groups[0]['lr'])
                state = scheduler.state_dict()
            for (it, lr) in lr_checks:
                assert lrs[it] == pytest.approx(lr), f'Iteration {it}: {lrs[it]} != {lr}'