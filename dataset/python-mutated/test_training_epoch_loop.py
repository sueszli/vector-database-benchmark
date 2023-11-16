import logging
from unittest.mock import Mock, patch
import pytest
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.trainer.trainer import Trainer

def test_no_val_on_train_epoch_loop_restart(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    "Test that training validation loop doesn't get triggered at the beginning of a restart."
    trainer_kwargs = {'max_epochs': 1, 'limit_train_batches': 1, 'limit_val_batches': 1, 'num_sanity_val_steps': 0, 'enable_checkpointing': False}
    trainer = Trainer(**trainer_kwargs)
    model = BoringModel()
    trainer.fit(model)
    ckpt_path = str(tmpdir / 'last.ckpt')
    trainer.save_checkpoint(ckpt_path)
    trainer_kwargs['max_epochs'] = 2
    trainer = Trainer(**trainer_kwargs)
    with patch.object(trainer.fit_loop.epoch_loop.val_loop, '_evaluation_step', wraps=trainer.fit_loop.epoch_loop.val_loop._evaluation_step) as step_mock:
        trainer.fit(model, ckpt_path=ckpt_path)
    assert step_mock.call_count == 1

@pytest.mark.parametrize(('min_epochs', 'min_steps', 'current_epoch', 'global_step', 'early_stop', 'epoch_loop_done', 'raise_info_msg'), [(None, None, 1, 4, True, True, False), (None, None, 1, 10, True, True, False), (4, None, 1, 4, False, False, True), (4, 2, 1, 4, False, False, True), (4, None, 1, 10, False, True, False), (4, 3, 1, 3, False, False, True), (4, 10, 1, 10, False, True, False), (None, 4, 1, 4, True, True, False)])
def test_should_stop_early_stopping_conditions_not_met(caplog, min_epochs, min_steps, current_epoch, global_step, early_stop, epoch_loop_done, raise_info_msg):
    if False:
        print('Hello World!')
    'Test that checks that info message is logged when users sets `should_stop` but min conditions are not met.'
    trainer = Trainer(min_epochs=min_epochs, min_steps=min_steps, limit_val_batches=0)
    trainer.fit_loop.max_batches = 10
    trainer.should_stop = True
    trainer.fit_loop.epoch_loop.automatic_optimization.optim_progress.optimizer.step.total.completed = global_step
    trainer.fit_loop.epoch_loop.batch_progress.current.ready = global_step
    trainer.fit_loop.epoch_progress.current.completed = current_epoch - 1
    message = f'min_epochs={min_epochs}` or `min_steps={min_steps}` has not been met. Training will continue'
    with caplog.at_level(logging.INFO, logger='lightning.pytorch.loops'):
        assert trainer.fit_loop.epoch_loop.done is epoch_loop_done
    assert (message in caplog.text) is raise_info_msg
    assert trainer.fit_loop._can_stop_early is early_stop

@pytest.mark.parametrize(('min_epochs', 'min_steps', 'val_count'), [(3, None, 3), (None, 3, 2)])
def test_should_stop_triggers_validation_once(min_epochs, min_steps, val_count, tmp_path):
    if False:
        print('Hello World!')
    'Regression test for issue #15708.\n\n    Test that the request for `should_stop=True` only triggers validation when Trainer is allowed to stop\n    (min_epochs/steps is satisfied).\n\n    '
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmp_path, num_sanity_val_steps=0, limit_val_batches=2, limit_train_batches=2, max_epochs=3, min_epochs=min_epochs, min_steps=min_steps, enable_model_summary=False, enable_checkpointing=False)
    trainer.should_stop = True
    trainer.fit_loop.epoch_loop.val_loop.run = Mock()
    trainer.fit(model)
    assert trainer.fit_loop.epoch_loop.val_loop.run.call_count == val_count

def test_training_loop_dataloader_iter_multiple_dataloaders(tmp_path):
    if False:
        print('Hello World!')
    trainer = Trainer(default_root_dir=tmp_path, limit_train_batches=3, limit_val_batches=0, max_epochs=1, enable_model_summary=False, enable_checkpointing=False, logger=False, devices=1)

    class MyModel(BoringModel):
        batch_start_ins = []
        step_outs = []
        batch_end_ins = []

        def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
            if False:
                for i in range(10):
                    print('nop')
            self.batch_start_ins.append((batch, batch_idx, dataloader_idx))

        def training_step(self, dataloader_iter):
            if False:
                for i in range(10):
                    print('nop')
            self.step_outs.append(next(dataloader_iter))

        def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
            if False:
                return 10
            self.batch_end_ins.append((batch, batch_idx, dataloader_idx))
    model = MyModel()
    trainer.fit(model, {'a': [0, 1], 'b': [2, 3]})
    assert model.batch_start_ins == [(None, 0, 0)] + model.step_outs[:-1]
    assert model.step_outs == [({'a': 0, 'b': 2}, 0, 0), ({'a': 1, 'b': 3}, 1, 0)]
    assert model.batch_end_ins == model.step_outs

def test_no_batch_idx_gradient_accumulation():
    if False:
        while True:
            i = 10
    'Regression test for an issue where excluding the batch_idx from training_step would disable gradient\n    accumulation.'

    class MyModel(BoringModel):
        last_batch_idx = -1

        def training_step(self, batch):
            if False:
                return 10
            return self.step(batch)

        def optimizer_step(self, epoch, batch_idx, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            assert batch_idx in (1, 3)
            self.last_batch_idx = batch_idx
            return super().optimizer_step(epoch, batch_idx, *args, **kwargs)
    trainer = Trainer(fast_dev_run=4, accumulate_grad_batches=2, limit_val_batches=0)
    model = MyModel()
    trainer.fit(model)
    assert model.last_batch_idx == 3