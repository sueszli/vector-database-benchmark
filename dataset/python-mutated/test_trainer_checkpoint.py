import os
from unittest import mock
from unittest.mock import ANY, Mock
import lightning.pytorch as pl
import pytest
import torch
from lightning.fabric.plugins import TorchCheckpointIO, XLACheckpointIO
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos.boring_classes import BoringModel

def test_finetuning_with_ckpt_path(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'This test validates that generated ModelCheckpoint is pointing to the right best_model_path during test.'
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath=tmpdir, filename='{epoch:02d}', save_top_k=-1)

    class ExtendedBoringModel(BoringModel):

        def configure_optimizers(self):
            if False:
                print('Hello World!')
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.001)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            return ([optimizer], [lr_scheduler])

        def validation_step(self, batch, batch_idx):
            if False:
                print('Hello World!')
            loss = self.step(batch)
            self.log('val_loss', loss, on_epoch=True, prog_bar=True)
    model = ExtendedBoringModel()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, limit_train_batches=12, limit_val_batches=6, limit_test_batches=12, callbacks=[checkpoint_callback], logger=False)
    trainer.fit(model)
    assert os.listdir(tmpdir) == ['epoch=00.ckpt']
    best_model_paths = [checkpoint_callback.best_model_path]
    for idx in range(3, 6):
        trainer = pl.Trainer(default_root_dir=tmpdir, max_epochs=idx, limit_train_batches=12, limit_val_batches=12, limit_test_batches=12, enable_progress_bar=False)
        trainer.fit(model, ckpt_path=best_model_paths[-1])
        trainer.test()
        best_model_paths.append(trainer.checkpoint_callback.best_model_path)
    for (idx, best_model_path) in enumerate(best_model_paths):
        if idx == 0:
            assert best_model_path.endswith(f'epoch=0{idx}.ckpt')
        else:
            assert f'epoch={idx + 1}' in best_model_path

def test_trainer_save_checkpoint_storage_options(tmpdir, xla_available):
    if False:
        return 10
    'This test validates that storage_options argument is properly passed to ``CheckpointIO``'
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, enable_checkpointing=False)
    trainer.fit(model)
    instance_path = tmpdir + '/path.ckpt'
    instance_storage_options = 'my instance storage options'
    with mock.patch('lightning.fabric.plugins.io.torch_io.TorchCheckpointIO.save_checkpoint') as io_mock:
        trainer.save_checkpoint(instance_path, storage_options=instance_storage_options)
        io_mock.assert_called_with(ANY, instance_path, storage_options=instance_storage_options)
        trainer.save_checkpoint(instance_path)
        io_mock.assert_called_with(ANY, instance_path, storage_options=None)
    checkpoint_mock = Mock()
    with mock.patch.object(trainer.strategy, 'save_checkpoint') as save_mock, mock.patch.object(trainer._checkpoint_connector, 'dump_checkpoint', return_value=checkpoint_mock) as dump_mock:
        trainer.save_checkpoint(instance_path, True)
        dump_mock.assert_called_with(True)
        save_mock.assert_called_with(checkpoint_mock, instance_path, storage_options=None)
        trainer.save_checkpoint(instance_path, False, instance_storage_options)
        dump_mock.assert_called_with(False)
        save_mock.assert_called_with(checkpoint_mock, instance_path, storage_options=instance_storage_options)
    torch_checkpoint_io = TorchCheckpointIO()
    with pytest.raises(TypeError, match=f"`Trainer.save_checkpoint\\(..., storage_options=...\\)` with `storage_options` arg is not supported for `{torch_checkpoint_io.__class__.__name__}`. Please implement your custom `CheckpointIO` to define how you'd like to use `storage_options`."):
        torch_checkpoint_io.save_checkpoint({}, instance_path, storage_options=instance_storage_options)
    xla_checkpoint_io = XLACheckpointIO()
    with pytest.raises(TypeError, match=f"`Trainer.save_checkpoint\\(..., storage_options=...\\)` with `storage_options` arg is not supported for `{xla_checkpoint_io.__class__.__name__}`. Please implement your custom `CheckpointIO` to define how you'd like to use `storage_options`."):
        xla_checkpoint_io.save_checkpoint({}, instance_path, storage_options=instance_storage_options)