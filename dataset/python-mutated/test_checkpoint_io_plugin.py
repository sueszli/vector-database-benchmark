import os
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, Mock
import torch
from lightning.fabric.plugins import CheckpointIO, TorchCheckpointIO
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.plugins.io.async_plugin import AsyncCheckpointIO
from lightning.pytorch.strategies import SingleDeviceStrategy

class CustomCheckpointIO(CheckpointIO):

    def save_checkpoint(self, checkpoint: Dict[str, Any], path: _PATH, storage_options: Optional[Any]=None) -> None:
        if False:
            print('Hello World!')
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: _PATH, storage_options: Optional[Any]=None) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        return torch.load(path)

    def remove_checkpoint(self, path: _PATH) -> None:
        if False:
            for i in range(10):
                print('nop')
        os.remove(path)

def test_checkpoint_plugin_called(tmpdir):
    if False:
        while True:
            i = 10
    'Ensure that the custom checkpoint IO plugin and torch checkpoint IO plugin is called when saving/loading.'
    checkpoint_plugin = CustomCheckpointIO()
    checkpoint_plugin = MagicMock(wraps=checkpoint_plugin, spec=CustomCheckpointIO)
    ck = ModelCheckpoint(dirpath=tmpdir, save_last=True)
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, accelerator='cpu', strategy=SingleDeviceStrategy('cpu', checkpoint_io=checkpoint_plugin), callbacks=ck, max_epochs=2, limit_train_batches=1, limit_val_batches=0, limit_test_batches=1)
    trainer.fit(model)
    ckpt_files = {fn.name for fn in Path(tmpdir).glob('*.ckpt')}
    assert ckpt_files == {'epoch=1-step=2.ckpt', 'last.ckpt'}
    assert trainer.checkpoint_callback.best_model_path == tmpdir / 'epoch=1-step=2.ckpt'
    assert trainer.checkpoint_callback.last_model_path == tmpdir / 'last.ckpt'
    assert checkpoint_plugin.save_checkpoint.call_count == 2
    assert checkpoint_plugin.remove_checkpoint.call_count == 1
    trainer.test(model, ckpt_path=ck.last_model_path)
    checkpoint_plugin.load_checkpoint.assert_called_with(tmpdir / 'last.ckpt')
    checkpoint_plugin.reset_mock()
    ck = ModelCheckpoint(dirpath=tmpdir, save_last=True)
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, accelerator='cpu', strategy=SingleDeviceStrategy('cpu'), plugins=[checkpoint_plugin], callbacks=ck, max_epochs=2, limit_train_batches=1, limit_val_batches=0, limit_test_batches=1)
    trainer.fit(model)
    ckpt_files = {fn.name for fn in Path(tmpdir).glob('*.ckpt')}
    assert ckpt_files == {'epoch=1-step=2.ckpt', 'last.ckpt', 'epoch=1-step=2-v1.ckpt', 'last-v1.ckpt'}
    assert trainer.checkpoint_callback.best_model_path == tmpdir / 'epoch=1-step=2-v1.ckpt'
    assert trainer.checkpoint_callback.last_model_path == tmpdir / 'last-v1.ckpt'
    assert checkpoint_plugin.save_checkpoint.call_count == 2
    assert checkpoint_plugin.remove_checkpoint.call_count == 1
    trainer.test(model, ckpt_path=ck.last_model_path)
    checkpoint_plugin.load_checkpoint.assert_called_once()
    checkpoint_plugin.load_checkpoint.assert_called_with(tmpdir / 'last-v1.ckpt')

def test_async_checkpoint_plugin(tmpdir):
    if False:
        print('Hello World!')
    'Ensure that the custom checkpoint IO plugin and torch checkpoint IO plugin is called when async saving and\n    loading.'
    checkpoint_plugin = AsyncCheckpointIO()
    checkpoint_plugin.save_checkpoint = Mock(wraps=checkpoint_plugin.save_checkpoint)
    checkpoint_plugin.remove_checkpoint = Mock(wraps=checkpoint_plugin.remove_checkpoint)

    class CustomBoringModel(BoringModel):

        def on_fit_start(self):
            if False:
                for i in range(10):
                    print('nop')
            base_ckpt_io = self.trainer.strategy.checkpoint_io.checkpoint_io
            base_ckpt_io.save_checkpoint = Mock(wraps=base_ckpt_io.save_checkpoint)
            base_ckpt_io.remove_checkpoint = Mock(wraps=base_ckpt_io.remove_checkpoint)
    ck = ModelCheckpoint(dirpath=tmpdir, save_top_k=2, monitor='step', mode='max')
    model = CustomBoringModel()
    trainer = Trainer(default_root_dir=tmpdir, plugins=[checkpoint_plugin], callbacks=ck, max_epochs=3, limit_train_batches=1, limit_val_batches=0, enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(model)
    assert checkpoint_plugin.save_checkpoint.call_count == 3
    assert checkpoint_plugin.remove_checkpoint.call_count == 1
    base_ckpt_io = trainer.strategy.checkpoint_io.checkpoint_io
    assert base_ckpt_io.save_checkpoint.call_count == 3
    assert base_ckpt_io.remove_checkpoint.call_count == 1

def test_multi_wrapped_checkpoint_io_initialization():
    if False:
        print('Hello World!')
    base_ckpt_io = TorchCheckpointIO()
    wrap_ckpt = AsyncCheckpointIO(base_ckpt_io)
    ckpt_io = AsyncCheckpointIO(wrap_ckpt)
    assert ckpt_io.checkpoint_io is wrap_ckpt
    assert ckpt_io.checkpoint_io.checkpoint_io is base_ckpt_io
    assert ckpt_io._base_checkpoint_io_configured is True
    assert ckpt_io.checkpoint_io._base_checkpoint_io_configured is True
    wrap_ckpt = AsyncCheckpointIO()
    ckpt_io = AsyncCheckpointIO(wrap_ckpt)
    trainer = Trainer(accelerator='cpu', plugins=[ckpt_io])
    trainer.strategy.checkpoint_io
    assert ckpt_io.checkpoint_io is wrap_ckpt
    assert isinstance(ckpt_io.checkpoint_io.checkpoint_io, TorchCheckpointIO)
    assert ckpt_io._base_checkpoint_io_configured is True
    assert ckpt_io.checkpoint_io._base_checkpoint_io_configured is True