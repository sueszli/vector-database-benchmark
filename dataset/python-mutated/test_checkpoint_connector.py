import os
from unittest import mock
from unittest.mock import Mock
import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.migration.utils import _set_version

def test_preloaded_checkpoint_lifecycle(tmpdir):
    if False:
        print('Hello World!')
    'Tests that the preloaded checkpoint contents gets cleared from memory when it is not required anymore.'
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, max_steps=1)
    trainer.fit(model)
    connector = trainer._checkpoint_connector
    assert not connector._ckpt_path
    assert not connector._loaded_checkpoint
    connector.resume_start()
    assert not connector._ckpt_path
    assert not connector._loaded_checkpoint
    connector.resume_end()
    assert not connector._ckpt_path
    assert not connector._loaded_checkpoint
    ckpt_path = trainer.checkpoint_callback.best_model_path
    trainer = Trainer(default_root_dir=tmpdir, max_steps=2)
    connector = trainer._checkpoint_connector
    connector.resume_start(ckpt_path)
    assert connector._ckpt_path == ckpt_path
    assert connector._loaded_checkpoint
    assert isinstance(connector._loaded_checkpoint, dict)
    trainer.state.fn = TrainerFn.FITTING
    connector.resume_end()
    assert connector._ckpt_path == ckpt_path
    assert not connector._loaded_checkpoint

@mock.patch('lightning.fabric.plugins.environments.slurm.SLURMEnvironment.detect', return_value=True)
def test_hpc_restore_attempt(_, tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Test that restore() attempts to restore the hpc_ckpt with highest priority.'
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, max_steps=1, enable_checkpointing=False, logger=False)
    trainer.fit(model)
    hpc_ckpt_path = tmpdir / 'hpc_ckpt_3.ckpt'
    trainer.save_checkpoint(hpc_ckpt_path)
    assert os.listdir(tmpdir) == ['hpc_ckpt_3.ckpt']
    for param in model.parameters():
        torch.nn.init.constant_(param, 0)
    trainer = Trainer(default_root_dir=tmpdir, max_steps=2, enable_checkpointing=False, logger=False)
    trainer.fit(model)
    for param in model.parameters():
        assert param.abs().sum() > 0
        torch.nn.init.constant_(param, 0)
    trainer = Trainer(default_root_dir=tmpdir, max_steps=3)
    with pytest.raises(FileNotFoundError, match='Checkpoint file not found: not existing'):
        trainer.fit(model, ckpt_path='not existing')

def test_hpc_max_ckpt_version(tmpdir):
    if False:
        while True:
            i = 10
    'Test that the _CheckpointConnector is able to find the hpc checkpoint file with the highest version.'
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, max_steps=1)
    trainer.fit(model)
    trainer.save_checkpoint(tmpdir / 'hpc_ckpt.ckpt')
    trainer.save_checkpoint(tmpdir / 'hpc_ckpt_0.ckpt')
    trainer.save_checkpoint(tmpdir / 'hpc_ckpt_3.ckpt')
    trainer.save_checkpoint(tmpdir / 'hpc_ckpt_33.ckpt')
    assert trainer._checkpoint_connector._hpc_resume_path == str(tmpdir / 'hpc_ckpt_33.ckpt')
    assert trainer._checkpoint_connector._CheckpointConnector__max_ckpt_version_in_folder(tmpdir) == 33
    assert trainer._checkpoint_connector._CheckpointConnector__max_ckpt_version_in_folder(tmpdir / 'not' / 'existing') is None

def test_ckpt_for_fsspec():
    if False:
        print('Hello World!')
    'Test that the _CheckpointConnector is able to write to fsspec file systems.'
    model = BoringModel()
    trainer = Trainer(default_root_dir='memory://test_ckpt_for_fsspec', limit_train_batches=1, limit_val_batches=1, max_epochs=1)
    trainer.fit(model)
    trainer.save_checkpoint('memory://test_ckpt_for_fsspec/hpc_ckpt.ckpt')
    trainer.save_checkpoint('memory://test_ckpt_for_fsspec/hpc_ckpt_0.ckpt')
    trainer.save_checkpoint('memory://test_ckpt_for_fsspec/hpc_ckpt_3.ckpt')
    trainer.save_checkpoint('memory://test_ckpt_for_fsspec/hpc_ckpt_33.ckpt')
    assert trainer._checkpoint_connector._hpc_resume_path == 'memory://test_ckpt_for_fsspec/hpc_ckpt_33.ckpt'
    assert trainer._checkpoint_connector._CheckpointConnector__max_ckpt_version_in_folder('memory://test_ckpt_for_fsspec') == 33
    assert trainer._checkpoint_connector._CheckpointConnector__max_ckpt_version_in_folder('memory://not_existing') is None

def test_loops_restore(tmpdir):
    if False:
        return 10
    'Test that required loop state_dict is loaded correctly by checkpoint connector.'
    model = BoringModel()
    checkpoint_callback = ModelCheckpoint(dirpath=tmpdir, save_last=True)
    trainer_args = {'default_root_dir': tmpdir, 'max_epochs': 1, 'limit_train_batches': 1, 'limit_val_batches': 1, 'logger': False, 'callbacks': [checkpoint_callback], 'num_sanity_val_steps': 0}
    trainer = Trainer(**trainer_args)
    trainer.fit(model)
    ckpt_path = str(tmpdir / 'last.ckpt')
    trainer = Trainer(**trainer_args)
    trainer.strategy.connect(model)
    trainer_fns = list(TrainerFn)
    for fn in trainer_fns:
        trainer_fn = getattr(trainer, f'{fn.value}_loop')
        trainer_fn.load_state_dict = mock.Mock()
    for fn in trainer_fns:
        trainer.state.fn = fn
        trainer._checkpoint_connector.resume_start(ckpt_path)
        trainer._checkpoint_connector.restore_loops()
        trainer_loop = getattr(trainer, f'{fn.value}_loop')
        trainer_loop.load_state_dict.assert_called()
        trainer_loop.load_state_dict.reset_mock()
        for fn2 in trainer_fns:
            if fn2 != fn:
                trainer_loop2 = getattr(trainer, f'{fn2.value}_loop')
                trainer_loop2.load_state_dict.assert_not_called()

def test_stateful_trainer_ckpt_path_support(tmp_path):
    if False:
        i = 10
        return i + 15
    "Tests support for the pattern used by NeMo's experiment manager."
    model = BoringModel()
    ckpt_data = {'state_dict': model.state_dict(), 'optimizer_states': {}, 'lr_schedulers': {}}
    _set_version(ckpt_data, '2.0.0')
    ckpt_path = tmp_path / 'foo.ckpt'
    torch.save(ckpt_data, ckpt_path)
    model_checkpoint = Mock(spec=ModelCheckpoint)
    last_path = tmp_path / 'last.ckpt'
    torch.save(ckpt_data, last_path)
    model_checkpoint._find_last_checkpoints.return_value = {last_path}
    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=True, callbacks=model_checkpoint)
    trainer.ckpt_path = ckpt_path
    trainer.fit(model)
    assert trainer.ckpt_path == ckpt_path
    assert trainer._checkpoint_connector._user_managed
    with pytest.warns(UserWarning, match='trainer.ckpt_path =.*but then you passed'):
        trainer.fit(model, ckpt_path='last')
    assert trainer.ckpt_path == last_path
    assert not trainer._checkpoint_connector._user_managed
    best_path = tmp_path / 'best.ckpt'
    torch.save(ckpt_data, best_path)
    model_checkpoint.best_model_path = best_path
    trainer.ckpt_path = ckpt_path
    trainer.test()
    assert trainer.ckpt_path == ckpt_path
    trainer.ckpt_path = None
    assert trainer._checkpoint_connector._ckpt_path is None
    assert not trainer._checkpoint_connector._user_managed
    trainer.test()
    assert trainer.ckpt_path == best_path