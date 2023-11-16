import math
import os
import pickle
import re
import time
from argparse import Namespace
from datetime import timedelta
from pathlib import Path
from typing import Union
from unittest import mock
from unittest.mock import Mock, call, patch
import cloudpickle
import lightning.pytorch as pl
import pytest
import torch
import yaml
from lightning.fabric.utilities.cloud_io import _load as pl_load
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.imports import _OMEGACONF_AVAILABLE
from torch import optim
from tests_pytorch.helpers.runif import RunIf
if _OMEGACONF_AVAILABLE:
    from omegaconf import Container, OmegaConf

def test_model_checkpoint_state_key():
    if False:
        while True:
            i = 10
    early_stopping = ModelCheckpoint(monitor='val_loss')
    expected_id = "ModelCheckpoint{'monitor': 'val_loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"
    assert early_stopping.state_key == expected_id

class LogInTwoMethods(BoringModel):

    def training_step(self, batch, batch_idx):
        if False:
            return 10
        out = super().training_step(batch, batch_idx)
        self.log('early_stop_on', out['loss'])
        return out

    def on_validation_epoch_end(self):
        if False:
            return 10
        self.log('val_acc', torch.tensor(1.23))

def mock_training_epoch_loop(trainer):
    if False:
        return 10
    calls = {}
    old_get_monitor_value = trainer.fit_loop.epoch_loop._get_monitor_value

    def mock(key):
        if False:
            for i in range(10):
                print('nop')
        value = old_get_monitor_value(key)
        calls[trainer.current_epoch] = {key: value}
        return value
    trainer.fit_loop.epoch_loop._get_monitor_value = mock
    return calls

@pytest.mark.parametrize(('validation_step_none', 'val_dataloaders_none', 'monitor'), [(False, False, 'val_log'), (True, False, 'train_log_epoch'), (False, True, 'val_log')])
@pytest.mark.parametrize('reduce_lr_on_plateau', [False, True])
def test_model_checkpoint_score_and_ckpt(tmpdir, validation_step_none: bool, val_dataloaders_none: bool, monitor: str, reduce_lr_on_plateau: bool):
    if False:
        print('Hello World!')
    'Test that when a model checkpoint is saved, it saves with the correct score appended to ckpt_path and checkpoint\n    data.'
    max_epochs = 3
    limit_train_batches = 5
    limit_val_batches = 7
    (lr, gamma) = (0.1, 2)

    class CustomBoringModel(BoringModel):

        def __init__(self):
            if False:
                while True:
                    i = 10
            super().__init__()
            self.train_log_epochs = torch.randn(max_epochs, limit_train_batches)
            self.val_logs = torch.randn(max_epochs, limit_val_batches)
            self.scores = []

        def training_step(self, batch, batch_idx):
            if False:
                for i in range(10):
                    print('nop')
            log_value = self.train_log_epochs[self.current_epoch, batch_idx]
            self.log('train_log', log_value, on_epoch=True)
            return super().training_step(batch, batch_idx)

        def validation_step(self, batch, batch_idx):
            if False:
                for i in range(10):
                    print('nop')
            log_value = self.val_logs[self.current_epoch, batch_idx]
            self.log('val_log', log_value)
            return super().validation_step(batch, batch_idx)

        def configure_optimizers(self):
            if False:
                for i in range(10):
                    print('nop')
            optimizer = optim.SGD(self.parameters(), lr=lr)
            if reduce_lr_on_plateau:
                lr_scheduler = {'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer), 'monitor': monitor, 'strict': True}
            else:
                lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
            return ([optimizer], [lr_scheduler])

        def on_train_epoch_end(self):
            if False:
                i = 10
                return i + 15
            if 'train' in monitor:
                self.scores.append(self.trainer.logged_metrics[monitor])

        def on_validation_epoch_end(self):
            if False:
                for i in range(10):
                    print('nop')
            if not self.trainer.sanity_checking and 'val' in monitor:
                self.scores.append(self.trainer.logged_metrics[monitor])
    filename = '{' + f'{monitor}' + ':.4f}-{epoch}'
    checkpoint = ModelCheckpoint(dirpath=tmpdir, filename=filename, monitor=monitor, save_top_k=-1)
    model = CustomBoringModel()
    if validation_step_none:
        model.validation_step = None
    if val_dataloaders_none:
        model.val_dataloaders = None
    trainer = Trainer(default_root_dir=tmpdir, callbacks=[checkpoint], limit_train_batches=limit_train_batches, limit_val_batches=limit_val_batches, max_epochs=max_epochs, enable_progress_bar=False)
    calls = mock_training_epoch_loop(trainer)
    trainer.fit(model)
    ckpt_files = list(Path(tmpdir).glob('*.ckpt'))
    assert len(ckpt_files) == len(model.scores) == max_epochs
    for epoch in range(max_epochs):
        score = model.scores[epoch]
        expected_score = getattr(model, f'{monitor}s')[epoch].mean().item()
        assert math.isclose(score, expected_score, abs_tol=1e-05)
        expected_filename = f'{monitor}={score:.4f}-epoch={epoch}.ckpt'
        chk = pl_load(os.path.join(checkpoint.dirpath, expected_filename))
        assert chk['epoch'] == epoch
        assert chk['global_step'] == limit_train_batches * (epoch + 1)
        mc_specific_data = chk['callbacks'][f"ModelCheckpoint{{'monitor': '{monitor}', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}}"]
        assert mc_specific_data['dirpath'] == checkpoint.dirpath
        assert mc_specific_data['monitor'] == monitor
        assert mc_specific_data['current_score'] == score
        if not reduce_lr_on_plateau:
            actual_step_count = chk['lr_schedulers'][0]['_step_count']
            actual_lr = chk['lr_schedulers'][0]['_last_lr'][0]
            assert actual_step_count == epoch + 2
            assert actual_lr == lr * gamma ** (epoch + 1)
        else:
            assert calls[epoch] == {monitor: score}

@pytest.mark.parametrize(('val_check_interval', 'reduce_lr_on_plateau', 'epoch_aligned'), [(0.25, True, True), (0.25, False, True), (0.42, False, False)])
def test_model_checkpoint_score_and_ckpt_val_check_interval(tmpdir, val_check_interval, reduce_lr_on_plateau, epoch_aligned):
    if False:
        print('Hello World!')
    'Test that when a model checkpoint is saved, it saves with the correct score appended to ckpt_path and checkpoint\n    data with val_check_interval.'
    seed_everything(0)
    max_epochs = 3
    limit_train_batches = 12
    limit_val_batches = 7
    (lr, gamma) = (0.1, 2)
    monitor = 'val_log'
    per_val_train_batches = int(limit_train_batches * val_check_interval)
    (per_epoch_val_checks, leftover_train_batches) = divmod(limit_train_batches, per_val_train_batches)

    class CustomBoringModel(BoringModel):

        def __init__(self):
            if False:
                while True:
                    i = 10
            super().__init__()
            self.val_logs = torch.randn(per_epoch_val_checks * max_epochs, limit_val_batches)
            self.val_loop_count = 0
            self.scores = []

        def validation_step(self, batch, batch_idx):
            if False:
                return 10
            log_value = self.val_logs[self.val_loop_count, batch_idx]
            self.log('val_log', log_value)
            return super().validation_step(batch, batch_idx)

        def on_validation_epoch_end(self):
            if False:
                i = 10
                return i + 15
            self.val_loop_count += 1
            self.scores.append(self.trainer.logged_metrics[monitor])

        def configure_optimizers(self):
            if False:
                while True:
                    i = 10
            optimizer = optim.SGD(self.parameters(), lr=lr)
            if reduce_lr_on_plateau:
                lr_scheduler = {'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer), 'monitor': monitor, 'strict': True}
            else:
                lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
            return ([optimizer], [lr_scheduler])
    filename = '{' + f'{monitor}' + ':.4f}-{epoch}'
    checkpoint = ModelCheckpoint(dirpath=tmpdir, filename=filename, monitor=monitor, save_top_k=-1)
    model = CustomBoringModel()
    trainer = Trainer(default_root_dir=tmpdir, callbacks=[checkpoint], limit_train_batches=limit_train_batches, limit_val_batches=limit_val_batches, max_epochs=max_epochs, val_check_interval=val_check_interval, enable_progress_bar=False, num_sanity_val_steps=0)
    calls = mock_training_epoch_loop(trainer)
    trainer.fit(model)

    def _make_assertions(epoch, ix):
        if False:
            return 10
        global_ix = ix + per_epoch_val_checks * epoch
        epoch_end_checkpoint = epoch_aligned and ix == per_epoch_val_checks - 1
        score = model.scores[global_ix]
        expected_score = getattr(model, f'{monitor}s')[global_ix].mean().item()
        expected_filename = f'{monitor}={score:.4f}-epoch={epoch}.ckpt'
        assert math.isclose(score, expected_score, rel_tol=0.0001)
        chk = pl_load(os.path.join(checkpoint.dirpath, expected_filename))
        assert chk['epoch'] == epoch
        expected_global_step = per_val_train_batches * (global_ix + 1) + leftover_train_batches * epoch
        assert chk['global_step'] == expected_global_step
        mc_specific_data = chk['callbacks'][f"ModelCheckpoint{{'monitor': '{monitor}', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}}"]
        assert mc_specific_data['dirpath'] == checkpoint.dirpath
        assert mc_specific_data['monitor'] == monitor
        assert mc_specific_data['current_score'] == score
        if not reduce_lr_on_plateau:
            actual_step_count = chk['lr_schedulers'][0]['_step_count']
            actual_lr = chk['lr_schedulers'][0]['_last_lr'][0]
            assert actual_step_count == epoch + 1 + epoch_end_checkpoint
            assert actual_lr == lr * gamma ** (epoch + epoch_end_checkpoint)
        return score
    ckpt_files = list(Path(tmpdir).glob('*.ckpt'))
    assert len(ckpt_files) == len(model.scores) == per_epoch_val_checks * max_epochs
    for epoch in range(max_epochs):
        for i in range(per_epoch_val_checks):
            score = _make_assertions(epoch, i)
        if reduce_lr_on_plateau:
            assert calls[epoch] == {monitor: score}

@pytest.mark.parametrize('save_top_k', [-1, 0, 1, 2])
def test_model_checkpoint_with_non_string_input(tmpdir, save_top_k: int):
    if False:
        while True:
            i = 10
    'Test that dirpath=None in checkpoint callback is valid and that ckpt_path is set correctly.'
    model = LogInTwoMethods()
    checkpoint = ModelCheckpoint(monitor='early_stop_on', dirpath=None, filename='{epoch}', save_top_k=save_top_k)
    max_epochs = 2
    trainer = Trainer(default_root_dir=tmpdir, callbacks=[checkpoint], overfit_batches=0.2, max_epochs=max_epochs, logger=False)
    trainer.fit(model)
    assert checkpoint.dirpath == tmpdir / 'checkpoints'
    if save_top_k == -1:
        ckpt_files = os.listdir(checkpoint.dirpath)
        expected_ckpt_files = [f'epoch={i}.ckpt' for i in range(max_epochs)]
        assert len(ckpt_files) == len(expected_ckpt_files) == max_epochs
        assert set(ckpt_files) == set(expected_ckpt_files)

@pytest.mark.parametrize('save_top_k', [-1, 0, 1, 2])
def test_model_checkpoint_to_yaml(tmpdir, save_top_k: int):
    if False:
        return 10
    'Test that None in checkpoint callback is valid and that chkp_path is set correctly.'
    model = LogInTwoMethods()
    checkpoint = ModelCheckpoint(dirpath=tmpdir, monitor='early_stop_on', save_top_k=save_top_k)
    trainer = Trainer(default_root_dir=tmpdir, callbacks=[checkpoint], overfit_batches=0.2, max_epochs=2)
    trainer.fit(model)
    path_yaml = os.path.join(tmpdir, 'best_k_models.yaml')
    checkpoint.to_yaml(path_yaml)
    with open(path_yaml) as fo:
        d = yaml.full_load(fo)
    best_k = dict(checkpoint.best_k_models.items())
    assert d == best_k

@pytest.mark.parametrize(('logger_version', 'expected'), [(None, 'version_0'), (1, 'version_1'), ('awesome', 'awesome')])
def test_model_checkpoint_path(tmpdir, logger_version: Union[None, int, str], expected: str):
    if False:
        return 10
    'Test that "version_" prefix is only added when logger\'s version is an integer.'
    model = LogInTwoMethods()
    logger = TensorBoardLogger(str(tmpdir), version=logger_version)
    trainer = Trainer(default_root_dir=tmpdir, overfit_batches=0.2, max_epochs=2, logger=logger)
    trainer.fit(model)
    ckpt_version = Path(trainer.checkpoint_callback.dirpath).parent.name
    assert ckpt_version == expected

def test_pickling(tmpdir):
    if False:
        i = 10
        return i + 15
    ckpt = ModelCheckpoint(dirpath=tmpdir)
    ckpt_pickled = pickle.dumps(ckpt)
    ckpt_loaded = pickle.loads(ckpt_pickled)
    assert vars(ckpt) == vars(ckpt_loaded)
    ckpt_pickled = cloudpickle.dumps(ckpt)
    ckpt_loaded = cloudpickle.loads(ckpt_pickled)
    assert vars(ckpt) == vars(ckpt_loaded)

class ModelCheckpointTestInvocations(ModelCheckpoint):

    def __init__(self, expected_count, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        self.expected_count = expected_count
        self.state_dict_count = 0

    def on_train_start(self, trainer, pl_module):
        if False:
            print('Hello World!')
        torch.save = Mock(wraps=torch.save)

    def state_dict(self):
        if False:
            for i in range(10):
                print('nop')
        super().state_dict()
        self.state_dict_count += 1

    def on_train_end(self, trainer, pl_module):
        if False:
            return 10
        super().on_train_end(trainer, pl_module)
        assert self.best_model_path
        assert self.best_model_score
        assert self.state_dict_count == self.expected_count
        if trainer.is_global_zero:
            assert torch.save.call_count == self.expected_count
        else:
            assert torch.save.call_count == 0

@RunIf(skip_windows=True)
def test_model_checkpoint_no_extraneous_invocations(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Test to ensure that the model callback saves the checkpoints only once in distributed mode.'
    model = LogInTwoMethods()
    num_epochs = 4
    model_checkpoint = ModelCheckpointTestInvocations(monitor='early_stop_on', expected_count=num_epochs, save_top_k=-1)
    trainer = Trainer(strategy='ddp_spawn', accelerator='cpu', devices=2, default_root_dir=tmpdir, callbacks=[model_checkpoint], max_epochs=num_epochs)
    trainer.fit(model)
    assert trainer.state.finished, f'Training failed with {trainer.state}'

def test_model_checkpoint_format_checkpoint_name(tmpdir, monkeypatch):
    if False:
        print('Hello World!')
    ckpt_name = ModelCheckpoint._format_checkpoint_name('', {'epoch': 3, 'step': 2})
    assert ckpt_name == 'epoch=3-step=2'
    ckpt_name = ModelCheckpoint._format_checkpoint_name(None, {'epoch': 3, 'step': 2}, prefix='test')
    assert ckpt_name == 'test-epoch=3-step=2'
    ckpt_name = ModelCheckpoint._format_checkpoint_name('ckpt', {}, prefix='test')
    assert ckpt_name == 'test-ckpt'
    ckpt_name = ModelCheckpoint._format_checkpoint_name('{epoch:03d}-{acc}', {'epoch': 3, 'acc': 0.03})
    assert ckpt_name == 'epoch=003-acc=0.03'
    ckpt_name = ModelCheckpoint._format_checkpoint_name('{epoch:03d}-{epoch_test:03d}', {'epoch': 3, 'epoch_test': 3})
    assert ckpt_name == 'epoch=003-epoch_test=003'
    monkeypatch.setattr(ModelCheckpoint, 'CHECKPOINT_JOIN_CHAR', '@')
    ckpt_name = ModelCheckpoint._format_checkpoint_name('{epoch},{acc:.5f}', {'epoch': 3, 'acc': 0.03}, prefix='test')
    assert ckpt_name == 'test@epoch=3,acc=0.03000'
    monkeypatch.undo()
    monkeypatch.setattr(ModelCheckpoint, 'CHECKPOINT_EQUALS_CHAR', ':')
    ckpt_name = ModelCheckpoint._format_checkpoint_name('{epoch:03d}-{acc}', {'epoch': 3, 'acc': 0.03})
    assert ckpt_name == 'epoch:003-acc:0.03'
    monkeypatch.undo()
    ckpt_name = ModelCheckpoint(monitor='early_stop_on', dirpath=None).format_checkpoint_name({'epoch': 3, 'step': 2})
    assert ckpt_name == 'epoch=3-step=2.ckpt'
    ckpt_name = ModelCheckpoint(monitor='early_stop_on', dirpath='').format_checkpoint_name({'epoch': 5, 'step': 4})
    assert ckpt_name == 'epoch=5-step=4.ckpt'
    ckpt_name = ModelCheckpoint(monitor='early_stop_on', dirpath='.').format_checkpoint_name({'epoch': 3, 'step': 4})
    assert ckpt_name == str(Path('.').resolve() / 'epoch=3-step=4.ckpt')
    ckpt = ModelCheckpoint(monitor='early_stop_on', dirpath=tmpdir, filename='name')
    ckpt_name = ckpt.format_checkpoint_name({}, ver=3)
    assert ckpt_name == tmpdir / 'name-v3.ckpt'
    ckpt = ModelCheckpoint(monitor='early_stop_on', dirpath=None, filename='{epoch}_{val/loss:.5f}')
    ckpt_name = ckpt.format_checkpoint_name({'epoch': 4, 'val/loss': 0.03})
    assert ckpt_name == 'epoch=4_val/loss=0.03000.ckpt'
    ckpt_name = ModelCheckpoint._format_checkpoint_name('epoch={epoch:03d}-val_acc={val/acc}', {'epoch': 3, 'val/acc': 0.03}, auto_insert_metric_name=False)
    assert ckpt_name == 'epoch=003-val_acc=0.03'
    ckpt_name = ModelCheckpoint._format_checkpoint_name('mAP@0.50={val/mAP@0.50:.4f}', {'val/mAP@0.50': 0.2}, auto_insert_metric_name=False)
    assert ckpt_name == 'mAP@0.50=0.2000'

class ModelCheckpointExtensionTest(ModelCheckpoint):
    FILE_EXTENSION = '.tpkc'

def test_model_checkpoint_file_extension(tmpdir):
    if False:
        print('Hello World!')
    'Test ModelCheckpoint with different file extension.'
    model = LogInTwoMethods()
    model_checkpoint = ModelCheckpointExtensionTest(monitor='early_stop_on', dirpath=tmpdir, save_top_k=1, save_last=True)
    trainer = Trainer(default_root_dir=tmpdir, callbacks=[model_checkpoint], max_steps=1, logger=False)
    trainer.fit(model)
    expected = ['epoch=0-step=1.tpkc', 'last.tpkc']
    assert set(expected) == set(os.listdir(tmpdir))

def test_model_checkpoint_save_last(tmpdir, monkeypatch):
    if False:
        while True:
            i = 10
    'Tests that save_last produces only one last checkpoint.'
    seed_everything()
    model = LogInTwoMethods()
    epochs = 3
    monkeypatch.setattr(ModelCheckpoint, 'CHECKPOINT_NAME_LAST', 'last-{epoch}')
    model_checkpoint = ModelCheckpoint(monitor='early_stop_on', dirpath=tmpdir, save_top_k=-1, save_last=True)
    trainer = Trainer(default_root_dir=tmpdir, callbacks=[model_checkpoint], max_epochs=epochs, limit_train_batches=10, limit_val_batches=10, logger=False)
    trainer.fit(model)
    last_filename = model_checkpoint._format_checkpoint_name(ModelCheckpoint.CHECKPOINT_NAME_LAST, {'epoch': trainer.current_epoch - 1})
    last_filename = last_filename + '.ckpt'
    assert str(tmpdir / last_filename) == model_checkpoint.last_model_path
    assert set(os.listdir(tmpdir)) == set([f'epoch={i}-step={j}.ckpt' for (i, j) in zip(range(epochs), [10, 20, 30])] + [last_filename])
    assert os.path.islink(tmpdir / last_filename)
    assert os.path.realpath(tmpdir / last_filename) == model_checkpoint._last_checkpoint_saved

def test_model_checkpoint_link_checkpoint(tmp_path):
    if False:
        return 10
    'Test that linking a checkpoint works and overwrites an existing link if present.'
    trainer = Mock()
    file = tmp_path / 'file'
    file.touch()
    link = tmp_path / 'link'
    ModelCheckpoint._link_checkpoint(trainer, filepath=str(file), linkpath=str(link))
    assert os.path.islink(link)
    assert os.path.realpath(link) == str(file)
    new_file1 = tmp_path / 'new_file1'
    new_file1.touch()
    ModelCheckpoint._link_checkpoint(trainer, filepath=str(new_file1), linkpath=str(link))
    assert os.path.islink(link)
    assert os.path.realpath(link) == str(new_file1)
    new_file2 = tmp_path / 'new_file2'
    new_file2.touch()
    ModelCheckpoint._link_checkpoint(trainer, filepath=str(new_file2), linkpath=str(link))
    assert os.path.islink(link)
    assert os.path.realpath(link) == str(new_file2)
    folder = tmp_path / 'folder'
    folder.mkdir()
    folder_link = tmp_path / 'folder_link'
    folder_link.mkdir()
    ModelCheckpoint._link_checkpoint(trainer, filepath=str(folder), linkpath=str(folder_link))
    assert os.path.islink(folder_link)
    assert os.path.realpath(folder_link) == str(folder)
    new_folder = tmp_path / 'new_folder'
    new_folder.mkdir()
    ModelCheckpoint._link_checkpoint(trainer, filepath=str(new_folder), linkpath=str(folder_link))
    assert os.path.islink(folder_link)
    assert os.path.realpath(folder_link) == str(new_folder)
    file = tmp_path / 'win_file'
    file.touch()
    link = tmp_path / 'win_link'
    with mock.patch('lightning.pytorch.callbacks.model_checkpoint.os.symlink', Mock(side_effect=OSError)):
        ModelCheckpoint._link_checkpoint(trainer, filepath=str(file), linkpath=str(link))
    assert not os.path.islink(link)
    assert os.path.isfile(link)

def test_invalid_top_k(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Make sure that a MisconfigurationException is raised for a negative save_top_k argument.'
    with pytest.raises(MisconfigurationException, match='.*Must be >= -1'):
        ModelCheckpoint(dirpath=tmpdir, save_top_k=-3)

def test_none_monitor_top_k(tmpdir):
    if False:
        return 10
    'Test that a warning appears for positive top_k with monitor=None.'
    with pytest.raises(MisconfigurationException, match='ModelCheckpoint\\(save_top_k=3, monitor=None\\) is not a valid*'):
        ModelCheckpoint(dirpath=tmpdir, save_top_k=3)
    ModelCheckpoint(dirpath=tmpdir, save_top_k=-1)
    ModelCheckpoint(dirpath=tmpdir, save_top_k=0)
    ModelCheckpoint(dirpath=tmpdir, save_top_k=1)

def test_invalid_every_n_epochs(tmpdir):
    if False:
        while True:
            i = 10
    'Make sure that a MisconfigurationException is raised for a negative every_n_epochs argument.'
    with pytest.raises(MisconfigurationException, match='.*Must be >= 0'):
        ModelCheckpoint(dirpath=tmpdir, every_n_epochs=-3)
    ModelCheckpoint(dirpath=tmpdir, every_n_epochs=0)
    ModelCheckpoint(dirpath=tmpdir, every_n_epochs=1)
    ModelCheckpoint(dirpath=tmpdir, every_n_epochs=2)

def test_invalid_every_n_train_steps(tmpdir):
    if False:
        print('Hello World!')
    'Make sure that a MisconfigurationException is raised for a negative every_n_epochs argument.'
    with pytest.raises(MisconfigurationException, match='.*Must be >= 0'):
        ModelCheckpoint(dirpath=tmpdir, every_n_train_steps=-3)
    ModelCheckpoint(dirpath=tmpdir, every_n_train_steps=0)
    ModelCheckpoint(dirpath=tmpdir, every_n_train_steps=1)
    ModelCheckpoint(dirpath=tmpdir, every_n_epochs=2)

def test_invalid_trigger_combination(tmpdir):
    if False:
        return 10
    'Test that a MisconfigurationException is raised if more than one of every_n_epochs, every_n_train_steps, and\n    train_time_interval are enabled together.'
    with pytest.raises(MisconfigurationException, match='.*Combination of parameters every_n_train_steps'):
        ModelCheckpoint(dirpath=tmpdir, every_n_train_steps=1, every_n_epochs=2)
    with pytest.raises(MisconfigurationException, match='.*Combination of parameters every_n_train_steps'):
        ModelCheckpoint(train_time_interval=timedelta(minutes=1), every_n_epochs=2)
    with pytest.raises(MisconfigurationException, match='.*Combination of parameters every_n_train_steps'):
        ModelCheckpoint(train_time_interval=timedelta(minutes=1), every_n_train_steps=2)
    ModelCheckpoint(dirpath=tmpdir, every_n_train_steps=0, every_n_epochs=3)
    ModelCheckpoint(dirpath=tmpdir, every_n_train_steps=4, every_n_epochs=0)
    ModelCheckpoint(dirpath=tmpdir, every_n_train_steps=0, every_n_epochs=0, train_time_interval=timedelta(minutes=1))

def test_none_every_n_train_steps_val_epochs(tmpdir):
    if False:
        while True:
            i = 10
    checkpoint_callback = ModelCheckpoint(dirpath=tmpdir)
    assert checkpoint_callback.every_n_epochs == 1
    assert checkpoint_callback._every_n_train_steps == 0

def test_model_checkpoint_save_last_none_monitor(tmpdir, caplog):
    if False:
        print('Hello World!')
    'Test that it is possible to save all checkpoints when monitor=None.'
    seed_everything()
    model = LogInTwoMethods()
    epochs = 2
    checkpoint_callback = ModelCheckpoint(monitor=None, dirpath=tmpdir, save_top_k=-1, save_last=True)
    trainer = Trainer(default_root_dir=tmpdir, callbacks=[checkpoint_callback], limit_train_batches=10, limit_val_batches=10, max_epochs=epochs, logger=False)
    trainer.fit(model)
    assert checkpoint_callback.monitor is None
    assert checkpoint_callback.best_model_path == tmpdir / 'epoch=1-step=20.ckpt'
    assert checkpoint_callback.last_model_path == tmpdir / 'last.ckpt'
    assert checkpoint_callback.best_model_score is None
    assert checkpoint_callback.best_k_models == {}
    assert checkpoint_callback.kth_best_model_path == ''
    expected = [f'epoch={i}-step={j}.ckpt' for (i, j) in zip(range(epochs), [10, 20])]
    expected.append('last.ckpt')
    assert set(os.listdir(tmpdir)) == set(expected)
    assert os.path.islink(tmpdir / 'last.ckpt')

@pytest.mark.parametrize('every_n_epochs', list(range(4)))
def test_model_checkpoint_every_n_epochs(tmpdir, every_n_epochs):
    if False:
        i = 10
        return i + 15
    model = LogInTwoMethods()
    epochs = 5
    checkpoint_callback = ModelCheckpoint(dirpath=tmpdir, filename='{epoch}', save_top_k=-1, every_n_epochs=every_n_epochs)
    trainer = Trainer(default_root_dir=tmpdir, callbacks=[checkpoint_callback], max_epochs=epochs, limit_train_batches=1, limit_val_batches=1, logger=False)
    trainer.fit(model)
    expected = [f'epoch={e}.ckpt' for e in range(epochs) if not (e + 1) % every_n_epochs] if every_n_epochs > 0 else []
    assert set(os.listdir(tmpdir)) == set(expected)

def test_ckpt_every_n_train_steps(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Tests that the checkpoints are saved every n training steps.'
    model = LogInTwoMethods()
    every_n_train_steps = 16
    max_epochs = 2
    epoch_length = 64
    checkpoint_callback = ModelCheckpoint(filename='{step}', every_n_epochs=0, every_n_train_steps=every_n_train_steps, dirpath=tmpdir, save_top_k=-1, save_last=False)
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=2, enable_progress_bar=False, callbacks=[checkpoint_callback], logger=False)
    trainer.fit(model)
    expected = [f'step={i}.ckpt' for i in range(every_n_train_steps, max_epochs * epoch_length + 1, every_n_train_steps)]
    assert set(os.listdir(tmpdir)) == set(expected)

@mock.patch('lightning.pytorch.callbacks.model_checkpoint.time')
def test_model_checkpoint_train_time_interval(mock_datetime, tmpdir) -> None:
    if False:
        return 10
    'Tests that the checkpoints are saved at the specified time interval.'
    seconds_per_batch = 7
    start_time = time.monotonic()
    batches_per_epoch = 64
    num_epochs = 2
    max_batches = batches_per_epoch * num_epochs + 1
    mock_datetime.monotonic.side_effect = [start_time + seconds_per_batch * i for i in range(max_batches)]
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, min_epochs=num_epochs, max_epochs=num_epochs, enable_progress_bar=False, callbacks=[ModelCheckpoint(filename='{epoch}-{step}', dirpath=tmpdir, train_time_interval=timedelta(minutes=1), save_top_k=-1, save_last=False)], logger=False)
    trainer.fit(model)
    assert len(os.listdir(tmpdir)) == 14

def test_model_checkpoint_topk_zero(tmpdir):
    if False:
        while True:
            i = 10
    'Test that no checkpoints are saved when save_top_k=0.'
    model = LogInTwoMethods()
    checkpoint_callback = ModelCheckpoint(dirpath=tmpdir, save_top_k=0, save_last=True)
    trainer = Trainer(default_root_dir=tmpdir, callbacks=[checkpoint_callback], max_epochs=2, logger=False)
    trainer.fit(model)
    assert checkpoint_callback.monitor is None
    assert checkpoint_callback.best_model_path == ''
    assert checkpoint_callback.best_model_score is None
    assert checkpoint_callback.best_k_models == {}
    assert checkpoint_callback.kth_best_model_path == ''
    assert os.listdir(tmpdir) == ['last.ckpt']
    assert checkpoint_callback.last_model_path == tmpdir / 'last.ckpt'
    assert not os.path.islink(checkpoint_callback.last_model_path)

def test_model_checkpoint_topk_all(tmpdir):
    if False:
        print('Hello World!')
    'Test that save_top_k=-1 tracks the best models when monitor key is provided.'
    seed_everything(1000)
    epochs = 3
    model = BoringModel()
    checkpoint_callback = ModelCheckpoint(dirpath=tmpdir, filename='{epoch}', monitor='epoch', mode='max', save_top_k=-1)
    trainer = Trainer(default_root_dir=tmpdir, callbacks=[checkpoint_callback], max_epochs=epochs, logger=False, val_check_interval=1.0)
    trainer.fit(model)
    assert checkpoint_callback.monitor == 'epoch'
    assert checkpoint_callback.best_model_path == tmpdir / 'epoch=2.ckpt'
    assert checkpoint_callback.best_model_score == epochs - 1
    assert len(os.listdir(tmpdir)) == len(checkpoint_callback.best_k_models) == epochs
    assert set(checkpoint_callback.best_k_models.keys()) == {str(tmpdir / f'epoch={i}.ckpt') for i in range(epochs)}
    assert checkpoint_callback.kth_best_model_path == tmpdir / 'epoch=0.ckpt'

def test_ckpt_metric_names(tmpdir):
    if False:
        while True:
            i = 10
    model = LogInTwoMethods()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, gradient_clip_val=1.0, overfit_batches=0.2, enable_progress_bar=False, limit_train_batches=0.01, limit_val_batches=0.01, callbacks=[ModelCheckpoint(monitor='early_stop_on', dirpath=tmpdir, filename='{val_loss:.2f}')])
    trainer.fit(model)
    ckpts = os.listdir(tmpdir)
    ckpts = [x for x in ckpts if 'val_loss' in x]
    assert len(ckpts) == 1
    val = re.sub('[^0-9.]', '', ckpts[0])
    assert len(val) > 3

def test_default_checkpoint_behavior(tmpdir):
    if False:
        i = 10
        return i + 15
    seed_everything(1234)
    model = LogInTwoMethods()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=3, enable_progress_bar=False, limit_train_batches=5, limit_val_batches=5, logger=False)
    with patch.object(trainer, 'save_checkpoint', wraps=trainer.save_checkpoint) as save_mock:
        trainer.fit(model)
        results = trainer.test()
    assert len(results) == 1
    save_dir = tmpdir / 'checkpoints'
    save_weights_only = trainer.checkpoint_callback.save_weights_only
    save_mock.assert_has_calls([call(save_dir / 'epoch=0-step=5.ckpt', save_weights_only), call(save_dir / 'epoch=1-step=10.ckpt', save_weights_only), call(save_dir / 'epoch=2-step=15.ckpt', save_weights_only)])
    ckpts = os.listdir(save_dir)
    assert len(ckpts) == 1
    assert ckpts[0] == 'epoch=2-step=15.ckpt'

def test_model_checkpoint_save_last_checkpoint_contents(tmpdir):
    if False:
        return 10
    'Tests that the save_last checkpoint contains the latest information.'
    seed_everything(100)
    model = LogInTwoMethods()
    num_epochs = 3
    model_checkpoint = ModelCheckpoint(monitor='early_stop_on', dirpath=tmpdir, filename='{epoch}', save_top_k=num_epochs, save_last=True)
    trainer = Trainer(default_root_dir=tmpdir, callbacks=[model_checkpoint], max_epochs=num_epochs, limit_train_batches=2, limit_val_batches=2)
    trainer.fit(model)
    path_last_epoch = str(tmpdir / f'epoch={num_epochs - 1}.ckpt')
    path_last = str(tmpdir / 'last.ckpt')
    assert path_last == model_checkpoint.last_model_path
    assert os.path.isfile(path_last_epoch)
    assert os.path.islink(path_last)
    ckpt_last_epoch = torch.load(path_last_epoch)
    ckpt_last = torch.load(path_last)
    assert ckpt_last_epoch['epoch'] == ckpt_last['epoch']
    assert ckpt_last_epoch['global_step'] == ckpt_last['global_step']
    ckpt_id = "ModelCheckpoint{'monitor': 'early_stop_on', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"
    assert ckpt_last['callbacks'][ckpt_id] == ckpt_last_epoch['callbacks'][ckpt_id]
    model_last_epoch = LogInTwoMethods.load_from_checkpoint(path_last_epoch)
    model_last = LogInTwoMethods.load_from_checkpoint(model_checkpoint.last_model_path)
    for (w0, w1) in zip(model_last_epoch.parameters(), model_last.parameters()):
        assert w0.eq(w1).all()

@pytest.mark.parametrize('mode', ['min', 'max'])
def test_checkpointing_with_nan_as_first(tmpdir, mode):
    if False:
        while True:
            i = 10
    monitor = [float('nan')]
    monitor += [5, 7, 8] if mode == 'max' else [8, 7, 5]

    class CurrentModel(LogInTwoMethods):

        def on_validation_epoch_end(self):
            if False:
                for i in range(10):
                    print('nop')
            val_loss = monitor[self.current_epoch]
            self.log('abc', val_loss)
    model = CurrentModel()
    callback = ModelCheckpoint(monitor='abc', mode=mode, save_top_k=1, dirpath=tmpdir)
    trainer = Trainer(callbacks=[callback], default_root_dir=tmpdir, val_check_interval=1.0, max_epochs=len(monitor))
    trainer.save_checkpoint = Mock()
    trainer.fit(model)
    assert trainer.save_checkpoint.call_count == len(monitor)
    assert mode == 'min' and callback.best_model_score == 5 or (mode == 'max' and callback.best_model_score == 8)

def test_checkpoint_repeated_strategy(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'This test validates checkpoint can be called several times without increasing internally its global step if\n    nothing run.'
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath=tmpdir, filename='{epoch:02d}')

    class ExtendedBoringModel(BoringModel):

        def validation_step(self, batch, batch_idx):
            if False:
                print('Hello World!')
            loss = self.step(batch)
            self.log('val_loss', loss)
    model = ExtendedBoringModel()
    trainer_kwargs = {'max_epochs': 1, 'limit_train_batches': 2, 'limit_val_batches': 2, 'limit_test_batches': 2, 'enable_progress_bar': False, 'enable_model_summary': False, 'log_every_n_steps': 1, 'default_root_dir': tmpdir, 'logger': CSVLogger(tmpdir)}
    trainer = Trainer(**trainer_kwargs, callbacks=[checkpoint_callback])
    trainer.fit(model)
    assert set(os.listdir(tmpdir)) == {'epoch=00.ckpt', 'lightning_logs'}
    for idx in range(4):
        trainer = Trainer(**trainer_kwargs)
        trainer.fit(model, ckpt_path=checkpoint_callback.best_model_path)
        trainer.test(ckpt_path=checkpoint_callback.best_model_path, verbose=False)
        assert set(os.listdir(tmpdir)) == {'epoch=00.ckpt', 'lightning_logs'}
    assert set(os.listdir(tmpdir / 'lightning_logs')) == {'version_0'}

def test_checkpoint_repeated_strategy_extended(tmpdir):
    if False:
        i = 10
        return i + 15
    'This test validates checkpoint can be called several times without increasing internally its global step if\n    nothing run.'

    class ExtendedBoringModel(BoringModel):

        def validation_step(self, batch, batch_idx):
            if False:
                print('Hello World!')
            loss = self.step(batch)
            self.log('val_loss', loss)
            return {'val_loss': loss}

    def assert_trainer_init(trainer):
        if False:
            i = 10
            return i + 15
        assert trainer.global_step == 0
        assert trainer.current_epoch == 0

    def get_last_checkpoint(ckpt_dir):
        if False:
            return 10
        last = ckpt_dir.listdir(sort=True)[-1]
        return str(last)

    def assert_checkpoint_content(ckpt_dir):
        if False:
            for i in range(10):
                print('nop')
        chk = pl_load(get_last_checkpoint(ckpt_dir))
        assert chk['epoch'] == epochs - 1
        assert chk['global_step'] == 4

    def assert_checkpoint_log_dir(idx):
        if False:
            i = 10
            return i + 15
        lightning_logs = tmpdir / 'lightning_logs'
        actual = [d.basename for d in lightning_logs.listdir(sort=True)]
        assert actual == [f'version_{i}' for i in range(idx + 1)]
        actual = [d.basename for d in ckpt_dir.listdir()]
        assert len(actual) == epochs, actual
    ckpt_dir = tmpdir / 'checkpoints'
    checkpoint_cb = ModelCheckpoint(dirpath=ckpt_dir, save_top_k=-1)
    epochs = 2
    limit_train_batches = 2
    trainer_config = {'default_root_dir': tmpdir, 'max_epochs': epochs, 'limit_train_batches': limit_train_batches, 'limit_val_batches': 3, 'limit_test_batches': 4, 'callbacks': [checkpoint_cb], 'logger': TensorBoardLogger(tmpdir)}
    trainer = Trainer(**trainer_config)
    assert_trainer_init(trainer)
    model = ExtendedBoringModel()
    trainer.fit(model)
    assert trainer.global_step == epochs * limit_train_batches
    assert trainer.current_epoch == epochs
    assert_checkpoint_log_dir(0)
    assert_checkpoint_content(ckpt_dir)
    trainer.validate(model)
    assert trainer.current_epoch == epochs
    trainer.test(model)
    assert trainer.current_epoch == epochs
    for idx in range(1, 5):
        chk = get_last_checkpoint(ckpt_dir)
        assert_checkpoint_content(ckpt_dir)
        trainer_config['logger'] = TensorBoardLogger(tmpdir)
        trainer = pl.Trainer(**trainer_config)
        assert_trainer_init(trainer)
        model = ExtendedBoringModel()
        trainer.test(model)
        assert_trainer_init(trainer)
        trainer.fit(model, ckpt_path=chk)
        assert trainer.global_step == epochs * limit_train_batches
        assert trainer.current_epoch == epochs
        assert trainer.fit_loop.epoch_progress.current.processed == epochs
        trainer.validate(model)
        assert trainer.global_step == epochs * limit_train_batches
        assert trainer.current_epoch == epochs
        assert trainer.fit_loop.epoch_progress.current.processed == epochs
        trainer.fit(model)
        assert trainer.global_step == epochs * limit_train_batches
        assert trainer.current_epoch == epochs
        assert trainer.fit_loop.epoch_progress.current.processed == epochs
        assert_checkpoint_log_dir(idx)

def test_configure_model_checkpoint(tmpdir):
    if False:
        return 10
    'Test all valid and invalid ways a checkpoint callback can be passed to the Trainer.'
    kwargs = {'default_root_dir': tmpdir}
    callback1 = ModelCheckpoint(monitor='foo')
    callback2 = ModelCheckpoint(monitor='bar')
    trainer = Trainer(enable_checkpointing=False, callbacks=[], **kwargs)
    assert not any((isinstance(c, ModelCheckpoint) for c in trainer.callbacks))
    assert trainer.checkpoint_callback is None
    trainer = Trainer(callbacks=[], **kwargs)
    assert sum((1 for c in trainer.callbacks if isinstance(c, ModelCheckpoint))) == 1
    assert isinstance(trainer.checkpoint_callback, ModelCheckpoint)
    trainer = Trainer(enable_checkpointing=True, callbacks=[callback1], **kwargs)
    assert [c for c in trainer.callbacks if isinstance(c, ModelCheckpoint)] == [callback1]
    assert trainer.checkpoint_callback == callback1
    trainer = Trainer(callbacks=[callback1, callback2], **kwargs)
    assert trainer.checkpoint_callback == callback1
    assert trainer.checkpoint_callbacks == [callback1, callback2]
    with pytest.raises(MisconfigurationException, match='`enable_checkpointing=False` but found `ModelCheckpoint`'):
        Trainer(enable_checkpointing=False, callbacks=[callback1], **kwargs)

def test_val_check_interval_checkpoint_files(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Test correct checkpoint naming when validating/checkpointing multiple times per epoch.'
    model = LogInTwoMethods()
    model_checkpoint = ModelCheckpoint(dirpath=tmpdir, save_top_k=-1, monitor='val_acc', mode='max')
    trainer = Trainer(default_root_dir=tmpdir, val_check_interval=0.2, max_epochs=1, limit_train_batches=10, callbacks=[model_checkpoint], logger=False, enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(model)
    files = {p.basename for p in tmpdir.listdir()}
    assert files == {f'epoch=0-step={s}.ckpt' for s in [2, 4, 6, 8, 10]}

def test_current_score(tmpdir):
    if False:
        print('Hello World!')
    'Check that the current_score value is correct and was saved.'

    class TestModel(BoringModel):

        def training_step(self, *args):
            if False:
                i = 10
                return i + 15
            self.log('foo', (self.current_epoch + 1) / 10)
            return super().training_step(*args)
    model_checkpoint = ModelCheckpoint(dirpath=tmpdir, save_top_k=3, monitor='foo', mode='min')
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=3, limit_train_batches=1, limit_val_batches=1, callbacks=[model_checkpoint], logger=False, enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(TestModel())
    assert model_checkpoint.current_score == 0.3
    ckpts = [torch.load(str(ckpt)) for ckpt in tmpdir.listdir()]
    ckpts = [ckpt['callbacks']["ModelCheckpoint{'monitor': 'foo', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"] for ckpt in ckpts]
    assert sorted((ckpt['current_score'] for ckpt in ckpts)) == [0.1, 0.2, 0.3]

@pytest.mark.parametrize('mode', ['min', 'max'])
def test_current_score_when_nan(tmpdir, mode: str):
    if False:
        while True:
            i = 10
    'Check that ModelCheckpoint handles NaN values correctly.'

    class TestModel(BoringModel):

        def training_step(self, *args):
            if False:
                print('Hello World!')
            self.log('foo', float('nan'))
            return super().training_step(*args)
    model_checkpoint = ModelCheckpoint(dirpath=tmpdir, save_top_k=1, monitor='foo', mode=mode)
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, limit_train_batches=1, limit_val_batches=1, callbacks=[model_checkpoint], logger=False, enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(TestModel())
    expected = float('inf' if mode == 'min' else '-inf')
    assert model_checkpoint.best_model_score == expected
    assert model_checkpoint.current_score == expected

@pytest.mark.parametrize('use_omegaconf', [False, pytest.param(True, marks=RunIf(omegaconf=True))])
def test_hparams_type(tmpdir, use_omegaconf):
    if False:
        print('Hello World!')

    class TestModel(BoringModel):

        def __init__(self, hparams):
            if False:
                while True:
                    i = 10
            super().__init__()
            self.save_hyperparameters(hparams)
    model_checkpoint = ModelCheckpoint(dirpath=tmpdir, save_top_k=1)
    trainer = Trainer(max_epochs=1, default_root_dir=tmpdir, limit_train_batches=1, limit_val_batches=1, callbacks=[model_checkpoint], logger=False, enable_progress_bar=False, enable_model_summary=False)
    hp = {'test_hp_0': 1, 'test_hp_1': 2}
    hp = OmegaConf.create(hp) if use_omegaconf else Namespace(**hp)
    model = TestModel(hp)
    trainer.fit(model)
    ckpt = trainer._checkpoint_connector.dump_checkpoint()
    if use_omegaconf:
        assert isinstance(ckpt[model.CHECKPOINT_HYPER_PARAMS_KEY], Container)
    else:
        ckpt_params_type = type(ckpt[model.CHECKPOINT_HYPER_PARAMS_KEY])
        assert ckpt_params_type is dict

def test_ckpt_version_after_rerun_new_trainer(tmpdir):
    if False:
        while True:
            i = 10
    'Check that previous checkpoints are renamed to have the correct version suffix when new trainer instances are\n    used.'
    epochs = 2
    for i in range(epochs):
        mc = ModelCheckpoint(dirpath=tmpdir, save_top_k=-1, monitor='epoch', filename='{epoch}')
        trainer = Trainer(max_epochs=epochs, limit_train_batches=1, limit_val_batches=1, default_root_dir=tmpdir, callbacks=[mc], logger=False, enable_progress_bar=False, enable_model_summary=False)
        trainer.fit(BoringModel())
        expected = {'epoch=0-v1.ckpt', 'epoch=1-v1.ckpt'} if i else {'epoch=0.ckpt', 'epoch=1.ckpt'}
        assert {Path(f).name for f in mc.best_k_models} == expected
    actual = {f.basename for f in tmpdir.listdir()}
    assert actual == {'epoch=0.ckpt', 'epoch=1.ckpt', 'epoch=0-v1.ckpt', 'epoch=1-v1.ckpt'}

def test_ckpt_version_after_rerun_same_trainer(tmpdir):
    if False:
        while True:
            i = 10
    'Check that previous checkpoints are renamed to have the correct version suffix when the same trainer instance is\n    used.'
    mc = ModelCheckpoint(dirpath=tmpdir, save_top_k=-1, monitor='epoch', filename='test')
    mc.STARTING_VERSION = 9
    trainer = Trainer(max_epochs=2, limit_train_batches=1, limit_val_batches=1, default_root_dir=tmpdir, callbacks=[mc], logger=False, enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(BoringModel())
    trainer.fit_loop.max_epochs = 4
    trainer.fit(BoringModel())
    ckpt_range = range(mc.STARTING_VERSION, trainer.max_epochs + mc.STARTING_VERSION - 1)
    expected = {'test.ckpt', *(f'test-v{i}.ckpt' for i in ckpt_range)}
    assert {Path(f).name for f in mc.best_k_models} == expected
    assert set(os.listdir(tmpdir)) == expected

def test_ckpt_version_counter_disabled_after_rerun_new_trainer(tmpdir):
    if False:
        print('Hello World!')
    'Check that previous checkpoints get overwritten and no suffixes are generated when new trainer instances are\n    used.'
    epochs = 2
    for i in range(epochs):
        mc = ModelCheckpoint(dirpath=tmpdir, save_top_k=-1, save_last=True, monitor='epoch', filename='{epoch}', enable_version_counter=False)
        trainer = Trainer(max_epochs=epochs, limit_train_batches=1, limit_val_batches=1, default_root_dir=tmpdir, callbacks=[mc], logger=False, enable_progress_bar=False, enable_model_summary=False)
        trainer.fit(BoringModel())
        assert {Path(f).name for f in mc.best_k_models} == {'epoch=0.ckpt', 'epoch=1.ckpt'}
        assert Path(mc.last_model_path).name == 'last.ckpt'
    actual = {f.basename for f in tmpdir.listdir()}
    assert actual == {'epoch=0.ckpt', 'epoch=1.ckpt', 'last.ckpt'}

def test_model_checkpoint_mode_options():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(MisconfigurationException, match='`mode` can be .* but got unknown_option'):
        ModelCheckpoint(mode='unknown_option')

def test_check_val_every_n_epochs_top_k_integration(tmpdir):
    if False:
        i = 10
        return i + 15
    model = BoringModel()
    mc = ModelCheckpoint(dirpath=tmpdir, monitor='epoch', save_top_k=-1, filename='{epoch}')
    trainer = Trainer(default_root_dir=tmpdir, limit_train_batches=1, limit_val_batches=1, num_sanity_val_steps=0, max_epochs=5, check_val_every_n_epoch=2, callbacks=mc, enable_model_summary=False, logger=False)
    trainer.fit(model)
    assert set(os.listdir(tmpdir)) == {'epoch=1.ckpt', 'epoch=3.ckpt'}

def test_model_checkpoint_saveload_ckpt(tmpdir):
    if False:
        print('Hello World!')

    def make_assertions(cb_restore, written_ckpt):
        if False:
            i = 10
            return i + 15
        expected_keys = {'dirpath': False, 'best_model_score': False, 'kth_best_model_path': False, 'kth_value': False, 'best_k_models': False, 'last_model_path': False, 'best_model_path': True}
        for (key, should_match) in expected_keys.items():
            if should_match:
                assert getattr(cb_restore, key) == written_ckpt[key]
            else:
                assert getattr(cb_restore, key) != written_ckpt[key]

    class CustomModelCheckpoint(ModelCheckpoint):

        def on_load_checkpoint(self, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            assert self.dirpath is not None
            return super().on_load_checkpoint(*args, **kwargs)
    ckpt = {'best_model_path': 'epoch=10-step=1436.ckpt', 'best_model_score': torch.tensor(2.246), 'best_k_models': {'epoch=10-step=1436.ckpt': torch.tensor(2.246)}, 'kth_best_model_path': 'epoch=10-step=1436.ckpt', 'kth_value': torch.tensor(2.246), 'last_model_path': 'last2245.ckpt'}
    cb_write = ModelCheckpoint(dirpath=tmpdir, save_top_k=-1, save_last=True)
    for (key, val) in ckpt.items():
        setattr(cb_write, key, val)
    written_ckpt = cb_write.state_dict()
    for state in ckpt:
        assert ckpt[state] == written_ckpt[state]
    cb_restore = ModelCheckpoint(dirpath=tmpdir + '/restore', monitor=None, save_top_k=-1, save_last=True)
    with pytest.warns(UserWarning, match='The dirpath has changed from*'):
        cb_restore.load_state_dict(written_ckpt)
    make_assertions(cb_restore, written_ckpt)
    cb_restore = CustomModelCheckpoint()
    cb_restore.setup(Trainer(), BoringModel(), stage='fit')
    with pytest.warns(UserWarning, match='The dirpath has changed from*'):
        cb_restore.load_state_dict(written_ckpt)
    make_assertions(cb_restore, written_ckpt)

def test_resume_training_preserves_old_ckpt_last(tmpdir):
    if False:
        i = 10
        return i + 15
    'Ensures that the last saved checkpoint is not deleted from the previous folder when training is resumed from the\n    old checkpoint.'
    model = BoringModel()
    trainer_kwargs = {'default_root_dir': tmpdir, 'max_epochs': 1, 'limit_train_batches': 3, 'limit_val_batches': 0, 'enable_model_summary': False, 'logger': False}
    mc_kwargs = {'filename': '{step}', 'monitor': 'step', 'mode': 'max', 'save_last': True, 'save_top_k': 2, 'every_n_train_steps': 1}
    trainer = Trainer(**trainer_kwargs, callbacks=ModelCheckpoint(**mc_kwargs))
    trainer.fit(model)
    assert set(os.listdir(tmpdir / 'checkpoints')) == {'last.ckpt', 'step=2.ckpt', 'step=3.ckpt'}
    trainer_kwargs['max_epochs'] += 1
    mc_kwargs['dirpath'] = f'{tmpdir}/new'
    trainer = Trainer(**trainer_kwargs, callbacks=ModelCheckpoint(**mc_kwargs))
    trainer.fit(model, ckpt_path=f'{tmpdir}/checkpoints/step=2.ckpt')
    assert os.path.isfile(f'{tmpdir}/checkpoints/last.ckpt')

def test_save_last_saves_correct_last_model_path(tmpdir):
    if False:
        return 10
    mc = ModelCheckpoint(dirpath=tmpdir, save_last=True)
    mc.CHECKPOINT_NAME_LAST = '{foo}-last'
    trainer = Trainer(callbacks=mc)
    trainer.strategy.connect(BoringModel())
    mc._save_last_checkpoint(trainer, {'foo': torch.tensor(1)})
    expected = 'foo=1-last.ckpt'
    assert os.listdir(tmpdir) == [expected]
    full_path = str(tmpdir / expected)
    ckpt = torch.load(full_path)
    assert ckpt['callbacks'][mc.state_key]['last_model_path'] == full_path

def test_save_last_versioning(tmpdir):
    if False:
        i = 10
        return i + 15
    model = BoringModel()
    for _ in range(2):
        mc = ModelCheckpoint(dirpath=tmpdir, save_top_k=0, save_last=True)
        trainer = Trainer(max_epochs=2, callbacks=mc, limit_train_batches=1, limit_val_batches=0, enable_progress_bar=False, enable_model_summary=False, logger=False)
        trainer.fit(model)
    assert {'last.ckpt', 'last-v1.ckpt'} == set(os.listdir(tmpdir))
    assert all((not os.path.islink(tmpdir / path) for path in set(os.listdir(tmpdir))))

def test_none_monitor_saves_correct_best_model_path(tmpdir):
    if False:
        print('Hello World!')
    mc = ModelCheckpoint(dirpath=tmpdir, monitor=None)
    trainer = Trainer(callbacks=mc)
    trainer.strategy.connect(BoringModel())
    mc._save_none_monitor_checkpoint(trainer, {})
    expected = 'epoch=0-step=0.ckpt'
    assert os.listdir(tmpdir) == [expected]
    full_path = str(tmpdir / expected)
    ckpt = torch.load(full_path)
    assert ckpt['callbacks'][mc.state_key]['best_model_path'] == full_path

def test_last_global_step_saved():
    if False:
        return 10
    model_checkpoint = ModelCheckpoint(save_top_k=0, save_last=False, monitor='foo')
    trainer = Mock()
    monitor_candidates = {'foo': torch.tensor(123)}
    model_checkpoint._save_topk_checkpoint(trainer, monitor_candidates)
    model_checkpoint._save_last_checkpoint(trainer, monitor_candidates)
    assert model_checkpoint._last_global_step_saved == 0

@pytest.mark.parametrize('every_n_epochs', [0, 5])
def test_save_last_every_n_epochs_interaction(tmpdir, every_n_epochs):
    if False:
        while True:
            i = 10
    'Test that `save_last` ignores `every_n_epochs`.'
    mc = ModelCheckpoint(every_n_epochs=every_n_epochs, save_last=True, save_top_k=0, save_on_train_epoch_end=True)
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=2, callbacks=mc, limit_train_batches=1, limit_val_batches=0, enable_progress_bar=False, enable_model_summary=False, logger=False)
    model = BoringModel()
    with patch.object(trainer, 'save_checkpoint') as save_mock:
        trainer.fit(model)
    assert mc.last_model_path
    assert save_mock.call_count == trainer.max_epochs

def test_train_epoch_end_ckpt_with_no_validation():
    if False:
        return 10
    trainer = Trainer(val_check_interval=0.5)
    trainer.fit_loop.epoch_loop.val_loop._max_batches = [0]
    assert trainer.checkpoint_callback._should_save_on_train_epoch_end(trainer)
    trainer.fit_loop.epoch_loop.val_loop._max_batches = [1]
    assert not trainer.checkpoint_callback._should_save_on_train_epoch_end(trainer)
    trainer.val_check_interval = 0.8
    assert not trainer.checkpoint_callback._should_save_on_train_epoch_end(trainer)

@pytest.mark.parametrize('same_resume_folder', [True, False])
def test_resume_and_old_checkpoint_files_remain(same_resume_folder, tmp_path):
    if False:
        i = 10
        return i + 15
    "Test that checkpoints saved in the resume-folder won't be deleted under the save-top-k mechanism."
    model = BoringModel()
    trainer_kwargs = {'default_root_dir': tmp_path, 'limit_train_batches': 10, 'limit_val_batches': 0, 'enable_progress_bar': False, 'enable_model_summary': False, 'logger': False}
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    new_dirpath = first if same_resume_folder else second
    callback = ModelCheckpoint(dirpath=first, monitor='step', mode='max', save_top_k=2, every_n_train_steps=2)
    trainer = Trainer(callbacks=callback, max_steps=5, **trainer_kwargs)
    trainer.fit(model)
    assert set(os.listdir(first)) == {'epoch=0-step=2.ckpt', 'epoch=0-step=4.ckpt'}
    callback = ModelCheckpoint(dirpath=new_dirpath, monitor='step', mode='max', save_top_k=2, every_n_train_steps=2)
    trainer = Trainer(callbacks=callback, max_steps=8, **trainer_kwargs)
    trainer.fit(model, ckpt_path=str(first / 'epoch=0-step=4.ckpt'))
    if same_resume_folder:
        assert set(os.listdir(first)) == {'epoch=0-step=4.ckpt', 'epoch=0-step=6.ckpt', 'epoch=0-step=8.ckpt'}
    else:
        assert set(os.listdir(first)) == {'epoch=0-step=2.ckpt', 'epoch=0-step=4.ckpt'}
        assert set(os.listdir(second)) == {'epoch=0-step=6.ckpt', 'epoch=0-step=8.ckpt'}