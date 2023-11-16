from torch.ao.pruning._experimental.data_sparsifier.data_norm_sparsifier import DataNormSparsifier
from torch.ao.pruning._experimental.data_scheduler.base_data_scheduler import BaseDataScheduler
import torch
import torch.nn as nn
from typing import List
from torch.ao.pruning._experimental.data_sparsifier.lightning.callbacks.data_sparsity import PostTrainingDataSparsity, TrainingAwareDataSparsity
from torch.ao.pruning._experimental.data_sparsifier.lightning.callbacks._data_sparstity_utils import _get_valid_name
from torch.ao.pruning._experimental.data_sparsifier.base_data_sparsifier import SUPPORTED_TYPES
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.common_utils import run_tests
import importlib
import unittest
import warnings
import math
from torch.nn.utils.parametrize import is_parametrized

class DummyModel(nn.Module):

    def __init__(self, iC: int, oC: List[int]):
        if False:
            return 10
        super().__init__()
        self.linears = nn.Sequential()
        i = iC
        for (idx, c) in enumerate(oC):
            self.linears.append(nn.Linear(i, c, bias=False))
            if idx < len(oC) - 1:
                self.linears.append(nn.ReLU())
            i = c

def _make_lightning_module(iC: int, oC: List[int]):
    if False:
        print('Hello World!')
    import pytorch_lightning as pl

    class DummyLightningModule(pl.LightningModule):

        def __init__(self, ic: int, oC: List[int]):
            if False:
                i = 10
                return i + 15
            super().__init__()
            self.model = DummyModel(iC, oC)

        def forward(self):
            if False:
                for i in range(10):
                    print('nop')
            pass
    return DummyLightningModule(iC, oC)

class StepSLScheduler(BaseDataScheduler):
    """The sparsity param of each data group is multiplied by gamma every step_size epochs.
    """

    def __init__(self, data_sparsifier, schedule_param='sparsity_level', step_size=1, gamma=2, last_epoch=-1, verbose=False):
        if False:
            print('Hello World!')
        self.gamma = gamma
        self.step_size = step_size
        super().__init__(data_sparsifier, schedule_param, last_epoch, verbose)

    def get_schedule_param(self):
        if False:
            print('Hello World!')
        if not self._get_sp_called_within_step:
            warnings.warn('To get the last learning rate computed by the scheduler, please use `get_last_lr()`.', UserWarning)
        data_groups = self.data_sparsifier.data_groups
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return {name: config[self.schedule_param] for (name, config) in data_groups.items()}
        return {name: config[self.schedule_param] * self.gamma for (name, config) in data_groups.items()}

class TestPostTrainingCallback(TestCase):

    def _check_on_fit_end(self, pl_module, callback, sparsifier_args):
        if False:
            return 10
        'Makes sure that each component of is working as expected while calling the\n        post-training callback.\n        Specifically, check the following -\n            1. sparsifier config is the same as input config\n            2. data sparsifier is correctly attached to the model\n            3. sparsity is achieved after .step()\n            4. non-sparsified values are the same as original values\n        '
        callback.on_fit_end(42, pl_module)
        for (key, value) in sparsifier_args.items():
            assert callback.data_sparsifier.defaults[key] == value
        for (name, param) in pl_module.model.named_parameters():
            valid_name = _get_valid_name(name)
            if type(param) not in SUPPORTED_TYPES:
                assert valid_name not in callback.data_sparsifier.state
                assert valid_name not in callback.data_sparsifier.data_groups
                continue
            assert valid_name in callback.data_sparsifier.data_groups
            assert valid_name in callback.data_sparsifier.state
            mask = callback.data_sparsifier.get_mask(name=valid_name)
            assert 1.0 - mask.float().mean() > 0.0
            sparsified_data = callback.data_sparsifier.get_data(name=valid_name, return_original=False)
            assert torch.all(sparsified_data[sparsified_data != 0] == param[sparsified_data != 0])

    @unittest.skipIf(not importlib.util.find_spec('pytorch_lightning'), 'No pytorch_lightning')
    def test_post_training_callback(self):
        if False:
            return 10
        sparsifier_args = {'sparsity_level': 0.5, 'sparse_block_shape': (1, 4), 'zeros_per_block': 4}
        callback = PostTrainingDataSparsity(DataNormSparsifier, sparsifier_args)
        pl_module = _make_lightning_module(100, [128, 256, 16])
        self._check_on_fit_end(pl_module, callback, sparsifier_args)

class TestTrainingAwareCallback(TestCase):
    """Class to test in-training version of lightning callback
    Simulates model training and makes sure that each hook is doing what is expected
    """

    def _check_on_train_start(self, pl_module, callback, sparsifier_args, scheduler_args):
        if False:
            while True:
                i = 10
        'Makes sure that the data_sparsifier and data_scheduler objects are being created\n        correctly.\n        Basically, confirms that the input args and sparsifier/scheduler args are in-line.\n        '
        callback.on_train_start(42, pl_module)
        assert callback.data_scheduler is not None and callback.data_sparsifier is not None
        for (key, value) in sparsifier_args.items():
            assert callback.data_sparsifier.defaults[key] == value
        for (key, value) in scheduler_args.items():
            assert getattr(callback.data_scheduler, key) == value

    def _simulate_update_param_model(self, pl_module):
        if False:
            while True:
                i = 10
        'This function might not be needed as the model is being copied\n        during train_epoch_end() but good to have if things change in the future\n        '
        for (_, param) in pl_module.model.named_parameters():
            param.data = param + 1

    def _check_on_train_epoch_start(self, pl_module, callback):
        if False:
            print('Hello World!')
        "Basically ensures that the sparsifier's state is correctly being restored.\n        The state_dict() comparison is needed. Consider the flow -\n\n        **Epoch: 1**\n            1. on_train_epoch_start(): Nothing happens (for now)\n            2. on_train_epoch_end():\n                a) the model is copied into the data_sparsifier\n                b) .step() is called\n                c) internally, the state of each layer of the model inside\n                   data sparsifier changes\n\n        **Epoch: 2**\n            1. on_train_epoch_start(): Assume nothing happens\n            2. on_train_epoch_end():\n                a) the model is copied into the data_sparsifier.\n                   But wait! you need the config to attach layer\n                   of the module to the sparsifier. If config is None,\n                   the data_sparsifier uses the default config which we\n                   do not want as the config of each layer changes after\n                   .step()\n\n        Hence, we need to dump and restore the state_dict() everytime because we're\n        copying the model after each epoch.\n        Hence, it is essential to make sure that the sparsifier's state_dict() is being\n        correctly dumped and restored.\n\n        "
        callback.on_train_epoch_start(42, pl_module)
        if callback.data_sparsifier_state_dict is None:
            return
        data_sparsifier_state_dict = callback.data_sparsifier.state_dict()
        container_obj1 = data_sparsifier_state_dict['_container']
        container_obj2 = callback.data_sparsifier_state_dict['_container']
        assert len(container_obj1) == len(container_obj2)
        for (key, value) in container_obj2.items():
            assert key in container_obj1
            assert torch.all(value == container_obj1[key])
        state_obj1 = data_sparsifier_state_dict['state']
        state_obj2 = callback.data_sparsifier_state_dict['state']
        assert len(state_obj1) == len(state_obj2)
        for (key, value) in state_obj2.items():
            assert key in state_obj1
            assert 'mask' in value and 'mask' in state_obj1[key]
            assert torch.all(value['mask'] == state_obj1[key]['mask'])
        data_grp1 = data_sparsifier_state_dict['data_groups']
        data_grp2 = callback.data_sparsifier_state_dict['data_groups']
        assert len(data_grp1) == len(data_grp2)
        for (key, value) in data_grp2.items():
            assert key in data_grp1
            assert value == data_grp1[key]

    def _check_on_train_epoch_end(self, pl_module, callback):
        if False:
            while True:
                i = 10
        'Checks the following -\n        1. sparsity is correctly being achieved after .step()\n        2. scheduler and data_sparsifier sparsity levels are in-line\n        '
        callback.on_train_epoch_end(42, pl_module)
        data_scheduler = callback.data_scheduler
        base_sl = data_scheduler.base_param
        for (name, _) in pl_module.model.named_parameters():
            valid_name = _get_valid_name(name)
            mask = callback.data_sparsifier.get_mask(name=valid_name)
            assert 1.0 - mask.float().mean() > 0
            last_sl = data_scheduler.get_last_param()
            last_epoch = data_scheduler.last_epoch
            log_last_sl = math.log(last_sl[valid_name])
            log_actual_sl = math.log(base_sl[valid_name] * data_scheduler.gamma ** last_epoch)
            assert log_last_sl == log_actual_sl

    def _check_on_train_end(self, pl_module, callback):
        if False:
            print('Hello World!')
        'Confirms that the mask is squashed after the training ends\n        This is achieved by making sure that each parameter in the internal container\n        are not parametrized.\n        '
        callback.on_train_end(42, pl_module)
        for (name, _) in pl_module.model.named_parameters():
            valid_name = _get_valid_name(name)
            assert not is_parametrized(callback.data_sparsifier._continer, valid_name)

    @unittest.skipIf(not importlib.util.find_spec('pytorch_lightning'), 'No pytorch_lightning')
    def test_train_aware_callback(self):
        if False:
            i = 10
            return i + 15
        sparsifier_args = {'sparsity_level': 0.5, 'sparse_block_shape': (1, 4), 'zeros_per_block': 4}
        scheduler_args = {'gamma': 2, 'step_size': 1}
        callback = TrainingAwareDataSparsity(data_sparsifier_class=DataNormSparsifier, data_sparsifier_args=sparsifier_args, data_scheduler_class=StepSLScheduler, data_scheduler_args=scheduler_args)
        pl_module = _make_lightning_module(100, [128, 256, 16])
        self._check_on_train_start(pl_module, callback, sparsifier_args, scheduler_args)
        num_epochs = 5
        for _ in range(0, num_epochs):
            self._check_on_train_epoch_start(pl_module, callback)
            self._simulate_update_param_model(pl_module)
            self._check_on_train_epoch_end(pl_module, callback)
if __name__ == '__main__':
    run_tests()