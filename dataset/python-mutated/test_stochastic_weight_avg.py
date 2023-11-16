import logging
import os
from pathlib import Path
from typing import ContextManager, Optional
from unittest import mock
import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset, RandomIterableDataset
from lightning.pytorch.strategies import Strategy
from lightning.pytorch.strategies.launchers import _MultiProcessingLauncher
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import SWALR
from torch.utils.data import DataLoader
from tests_pytorch.helpers.runif import RunIf

def test_swa_callback_initial_state():
    if False:
        for i in range(10):
            print('nop')
    swa = StochasticWeightAveraging(swa_lrs=0.01, swa_epoch_start=0.1, annealing_epochs=1, annealing_strategy='linear', avg_fn=sum)
    assert swa._swa_lrs == 0.01
    assert swa._swa_epoch_start == 0.1
    assert swa._annealing_epochs == 1
    assert swa._annealing_strategy == 'linear'
    assert swa._avg_fn == sum
    assert swa._average_model is None

class SwaTestModel(BoringModel):

    def __init__(self, batchnorm: bool=True, interval: str='epoch', iterable_dataset: bool=False, crash_on_epoch=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        layers = [nn.Linear(32, 32)]
        if batchnorm:
            layers.append(nn.BatchNorm1d(32))
        layers += [nn.ReLU(), nn.Linear(32, 2)]
        self.layer = nn.Sequential(*layers)
        self.interval = interval
        self.iterable_dataset = iterable_dataset
        self.crash_on_epoch = crash_on_epoch

    def training_step(self, batch, batch_idx):
        if False:
            for i in range(10):
                print('nop')
        if self.crash_on_epoch and self.trainer.current_epoch >= self.crash_on_epoch:
            raise Exception('SWA crash test')
        return super().training_step(batch, batch_idx)

    def train_dataloader(self):
        if False:
            while True:
                i = 10
        dset_cls = RandomIterableDataset if self.iterable_dataset else RandomDataset
        dset = dset_cls(32, 64)
        return DataLoader(dset, batch_size=2)

    def configure_optimizers(self):
        if False:
            return 10
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=1), 'interval': self.interval}}

class SwaTestCallback(StochasticWeightAveraging):
    update_parameters_calls: int = 0
    transfer_weights_calls: int = 0
    first_epoch: Optional[int] = None

    def update_parameters(self, *args, **kwargs):
        if False:
            return 10
        self.update_parameters_calls += 1
        return StochasticWeightAveraging.update_parameters(*args, **kwargs)

    def transfer_weights(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.transfer_weights_calls += 1
        return StochasticWeightAveraging.transfer_weights(*args, **kwargs)

    def on_train_epoch_start(self, trainer, *args):
        if False:
            return 10
        super().on_train_epoch_start(trainer, *args)
        if self.first_epoch is None and (not trainer.fit_loop.restarting):
            self.first_epoch = trainer.current_epoch
        assert trainer.fit_loop._skip_backward == (trainer.current_epoch > self.swa_end)
        if self.swa_start <= trainer.current_epoch:
            assert isinstance(trainer.lr_scheduler_configs[0].scheduler, SWALR)
            assert trainer.lr_scheduler_configs[0].interval == 'epoch'
            assert trainer.lr_scheduler_configs[0].frequency == 1

    def on_train_epoch_end(self, trainer, *args):
        if False:
            for i in range(10):
                print('nop')
        super().on_train_epoch_end(trainer, *args)
        if self.swa_start <= trainer.current_epoch <= self.swa_end:
            swa_epoch = trainer.current_epoch - self.swa_start
            assert self.n_averaged == swa_epoch + 1
            assert self._swa_scheduler is not None
            assert self._swa_scheduler._step_count == swa_epoch + 2
        elif trainer.current_epoch > self.swa_end:
            assert self.n_averaged == self._max_epochs - self.swa_start

    def on_train_end(self, trainer, pl_module):
        if False:
            return 10
        super().on_train_end(trainer, pl_module)
        assert not trainer.fit_loop._skip_backward
        assert trainer.accumulate_grad_batches == 2
        assert trainer.num_training_batches == 5
        if not isinstance(trainer.strategy.launcher, _MultiProcessingLauncher):
            assert trainer.strategy.backward.call_count == (trainer.max_epochs - self.first_epoch) * trainer.limit_train_batches
        first_swa_epoch = max(self.first_epoch, self.swa_start)
        assert self.update_parameters_calls == trainer.max_epochs - first_swa_epoch
        assert self.transfer_weights_calls == 1

def train_with_swa(tmpdir, batchnorm=True, strategy='auto', accelerator='cpu', devices=1, interval='epoch', iterable_dataset=False):
    if False:
        for i in range(10):
            print('nop')
    model = SwaTestModel(batchnorm=batchnorm, interval=interval, iterable_dataset=iterable_dataset)
    swa_start = 2
    max_epochs = 5
    swa_callback = SwaTestCallback(swa_epoch_start=swa_start, swa_lrs=0.1)
    assert swa_callback.update_parameters_calls == 0
    assert swa_callback.transfer_weights_calls == 0
    trainer = Trainer(default_root_dir=tmpdir, enable_progress_bar=False, enable_model_summary=False, max_epochs=max_epochs, limit_train_batches=5, limit_val_batches=0, callbacks=[swa_callback], accumulate_grad_batches=2, strategy=strategy, accelerator=accelerator, devices=devices)
    with _backward_patch(trainer):
        trainer.fit(model)
    assert trainer.lightning_module == model

@RunIf(min_cuda_gpus=2, standalone=True)
def test_swa_callback_ddp(tmpdir):
    if False:
        print('Hello World!')
    train_with_swa(tmpdir, strategy='ddp', accelerator='gpu', devices=2)

@RunIf(min_cuda_gpus=2)
def test_swa_callback_ddp_spawn(tmpdir):
    if False:
        i = 10
        return i + 15
    train_with_swa(tmpdir, strategy='ddp_spawn', accelerator='gpu', devices=2)

@RunIf(skip_windows=True)
def test_swa_callback_ddp_cpu(tmpdir):
    if False:
        while True:
            i = 10
    train_with_swa(tmpdir, strategy='ddp_spawn', accelerator='cpu', devices=2)

@pytest.mark.parametrize('accelerator', [pytest.param('gpu', marks=RunIf(min_cuda_gpus=1)), pytest.param('mps', marks=RunIf(mps=True))])
def test_swa_callback_1_gpu(tmpdir, accelerator):
    if False:
        print('Hello World!')
    train_with_swa(tmpdir, accelerator=accelerator, devices=1)

@pytest.mark.parametrize('batchnorm', [True, False])
@pytest.mark.parametrize('iterable_dataset', [True, False])
def test_swa_callback(tmpdir, batchnorm: bool, iterable_dataset: bool):
    if False:
        print('Hello World!')
    train_with_swa(tmpdir, batchnorm=batchnorm, iterable_dataset=iterable_dataset)

@pytest.mark.parametrize('interval', ['epoch', 'step'])
def test_swa_callback_scheduler_step(tmpdir, interval: str):
    if False:
        return 10
    train_with_swa(tmpdir, interval=interval)

def test_swa_warns(tmpdir, caplog):
    if False:
        for i in range(10):
            print('nop')
    model = SwaTestModel(interval='step')
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, callbacks=StochasticWeightAveraging(swa_lrs=0.01))
    with caplog.at_level(level=logging.INFO), pytest.warns(UserWarning, match='SWA is currently only supported'):
        trainer.fit(model)
    assert 'Swapping scheduler `StepLR` for `SWALR`' in caplog.text

def test_swa_raises():
    if False:
        print('Hello World!')
    with pytest.raises(MisconfigurationException, match='>0 integer or a float between 0 and 1'):
        StochasticWeightAveraging(swa_epoch_start=0, swa_lrs=0.1)
    with pytest.raises(MisconfigurationException, match='>0 integer or a float between 0 and 1'):
        StochasticWeightAveraging(swa_epoch_start=1.5, swa_lrs=0.1)
    with pytest.raises(MisconfigurationException, match='>0 integer or a float between 0 and 1'):
        StochasticWeightAveraging(swa_epoch_start=-1, swa_lrs=0.1)
    with pytest.raises(MisconfigurationException, match='positive float, or a list of positive floats'):
        StochasticWeightAveraging(swa_epoch_start=5, swa_lrs=[0.2, 1])

def test_swa_deepcopy(tmpdir):
    if False:
        return 10
    "Test to ensure SWA Callback doesn't deepcopy dataloaders and datamodule potentially leading to OOM."

    class TestSWA(StochasticWeightAveraging):

        def __init__(self, *args, **kwargs):
            if False:
                while True:
                    i = 10
            super().__init__(*args, **kwargs)
            self.setup_called = False

        def setup(self, trainer, pl_module, stage) -> None:
            if False:
                return 10
            super().setup(trainer, pl_module, stage)
            assert self._average_model.train_dataloader is not pl_module.train_dataloader
            assert self._average_model.train_dataloader.__self__ == self._average_model
            assert self._average_model._trainer is None
            self.setup_called = True
    model = BoringModel()
    swa = TestSWA(swa_lrs=0.01)
    trainer = Trainer(default_root_dir=tmpdir, callbacks=swa, fast_dev_run=True)
    trainer.fit(model, train_dataloaders=DataLoader(RandomDataset(32, 2)))
    assert swa.setup_called

def test_swa_multiple_lrs(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    swa_lrs = [0.123, 0.321]

    class TestModel(BoringModel):

        def __init__(self):
            if False:
                return 10
            super().__init__()
            self.layer1 = torch.nn.Linear(32, 32)
            self.layer2 = torch.nn.Linear(32, 2)
            self.on_train_epoch_start_called = False

        def forward(self, x):
            if False:
                for i in range(10):
                    print('nop')
            x = self.layer1(x)
            x = self.layer2(x)
            return x

        def configure_optimizers(self):
            if False:
                for i in range(10):
                    print('nop')
            params = [{'params': self.layer1.parameters(), 'lr': 0.1}, {'params': self.layer2.parameters(), 'lr': 0.2}]
            return torch.optim.Adam(params)

        def on_train_epoch_start(self):
            if False:
                i = 10
                return i + 15
            optimizer = trainer.optimizers[0]
            assert [pg['lr'] for pg in optimizer.param_groups] == [0.1, 0.2]
            assert [pg['initial_lr'] for pg in optimizer.param_groups] == swa_lrs
            assert [pg['swa_lr'] for pg in optimizer.param_groups] == swa_lrs
            self.on_train_epoch_start_called = True
    model = TestModel()
    swa_callback = StochasticWeightAveraging(swa_lrs=swa_lrs)
    trainer = Trainer(default_root_dir=tmpdir, callbacks=swa_callback, fast_dev_run=1)
    trainer.fit(model)
    assert model.on_train_epoch_start_called

def _swa_resume_training_from_checkpoint(tmpdir, model, resume_model, ddp=False):
    if False:
        while True:
            i = 10
    swa_start = 3
    trainer_kwargs = {'default_root_dir': tmpdir, 'max_epochs': 5, 'accelerator': 'cpu', 'strategy': 'ddp_spawn' if ddp else 'auto', 'devices': 2 if ddp else 1, 'limit_train_batches': 5, 'limit_val_batches': 0, 'accumulate_grad_batches': 2, 'enable_progress_bar': False, 'logger': False}
    trainer = Trainer(callbacks=SwaTestCallback(swa_epoch_start=swa_start, swa_lrs=0.1), **trainer_kwargs)
    with _backward_patch(trainer), pytest.raises(Exception, match='SWA crash test'):
        trainer.fit(model)
    checkpoint_dir = Path(tmpdir) / 'checkpoints'
    checkpoint_files = os.listdir(checkpoint_dir)
    assert len(checkpoint_files) == 1
    ckpt_path = str(checkpoint_dir / checkpoint_files[0])
    trainer = Trainer(callbacks=SwaTestCallback(swa_epoch_start=swa_start, swa_lrs=0.1), **trainer_kwargs)
    with _backward_patch(trainer):
        trainer.fit(resume_model, ckpt_path=ckpt_path)

class CustomSchedulerModel(SwaTestModel):

    def configure_optimizers(self):
        if False:
            for i in range(10):
                print('nop')
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)

        def lr_lambda(current_step: int):
            if False:
                return 10
            return 0.1
        scheduler = LambdaLR(optimizer, lr_lambda, -1)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': self.interval}}

@pytest.mark.parametrize('crash_on_epoch', [1, 3])
def test_swa_resume_training_from_checkpoint(tmpdir, crash_on_epoch):
    if False:
        print('Hello World!')
    model = SwaTestModel(crash_on_epoch=crash_on_epoch)
    resume_model = SwaTestModel()
    _swa_resume_training_from_checkpoint(tmpdir, model, resume_model)

@pytest.mark.parametrize('crash_on_epoch', [1, 3])
def test_swa_resume_training_from_checkpoint_custom_scheduler(tmpdir, crash_on_epoch):
    if False:
        print('Hello World!')
    model = CustomSchedulerModel(crash_on_epoch=crash_on_epoch)
    resume_model = CustomSchedulerModel()
    _swa_resume_training_from_checkpoint(tmpdir, model, resume_model)

@RunIf(skip_windows=True)
def test_swa_resume_training_from_checkpoint_ddp(tmpdir):
    if False:
        print('Hello World!')
    model = SwaTestModel(crash_on_epoch=3)
    resume_model = SwaTestModel()
    _swa_resume_training_from_checkpoint(tmpdir, model, resume_model, ddp=True)

@pytest.mark.parametrize('strategy', [pytest.param('deepspeed', marks=RunIf(deepspeed=True, min_cuda_gpus=1)), pytest.param('fsdp', marks=RunIf(min_cuda_gpus=1, skip_windows=True))])
def test_misconfiguration_error_with_sharded_model(tmpdir, strategy: str):
    if False:
        for i in range(10):
            print('nop')
    model = SwaTestModel()
    swa_callback = SwaTestCallback(swa_epoch_start=2, swa_lrs=0.1)
    trainer = Trainer(default_root_dir=tmpdir, enable_progress_bar=False, max_epochs=5, callbacks=[swa_callback], strategy=strategy, accelerator='gpu', devices=1)
    with pytest.raises(MisconfigurationException, match='SWA does not currently support sharded models'):
        trainer.fit(model)

def _backward_patch(trainer: Trainer) -> ContextManager:
    if False:
        return 10
    return mock.patch.object(Strategy, 'backward', wraps=trainer.strategy.backward)