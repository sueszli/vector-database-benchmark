"""Tests the evaluation loop."""
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.core.module import LightningModule
from lightning.pytorch.trainer.states import RunningStage
from torch import Tensor
from tests_pytorch.helpers.deterministic_model import DeterministicModel

def test__eval_step__flow(tmpdir):
    if False:
        print('Hello World!')
    'Tests that only training_step can be used.'

    class TestModel(DeterministicModel):

        def training_step(self, batch, batch_idx):
            if False:
                while True:
                    i = 10
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx
            self.training_step_called = True
            return acc

        def validation_step(self, batch, batch_idx):
            if False:
                for i in range(10):
                    print('nop')
            self.validation_step_called = True
            if batch_idx == 0:
                out = ['1', 2, torch.tensor(2)]
            if batch_idx > 0:
                out = {'something': 'random'}
            return out

        def backward(self, loss):
            if False:
                print('Hello World!')
            return LightningModule.backward(self, loss)
    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, limit_train_batches=2, limit_val_batches=2, max_epochs=2, log_every_n_steps=1, enable_model_summary=False)
    trainer.fit(model)
    assert model.validation_step_called
    trainer.state.stage = RunningStage.TRAINING
    kwargs = {'batch': next(iter(model.train_dataloader())), 'batch_idx': 0}
    train_step_out = trainer.fit_loop.epoch_loop.automatic_optimization.run(trainer.optimizers[0], 0, kwargs)
    assert isinstance(train_step_out['loss'], Tensor)
    assert train_step_out['loss'].item() == 171
    opt_closure = trainer.fit_loop.epoch_loop.automatic_optimization._make_closure(kwargs, trainer.optimizers[0], 0)
    opt_closure_result = opt_closure()
    assert opt_closure_result.item() == 171

def test__eval_step__epoch_end__flow(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Tests that only training_step can be used.'

    class TestModel(DeterministicModel):

        def training_step(self, batch, batch_idx):
            if False:
                print('Hello World!')
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx
            self.training_step_called = True
            return acc

        def validation_step(self, batch, batch_idx):
            if False:
                print('Hello World!')
            self.validation_step_called = True
            if batch_idx == 0:
                out = ['1', 2, torch.tensor(2)]
                self.out_a = out
            if batch_idx > 0:
                out = {'something': 'random'}
                self.out_b = out
            return out

        def backward(self, loss):
            if False:
                for i in range(10):
                    print('nop')
            return LightningModule.backward(self, loss)
    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, limit_train_batches=2, limit_val_batches=2, max_epochs=2, log_every_n_steps=1, enable_model_summary=False)
    trainer.fit(model)
    assert model.validation_step_called