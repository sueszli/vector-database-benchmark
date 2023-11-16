"""Tests to ensure that the training loop works with a dict (1.0)"""
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.core.module import LightningModule
from tests_pytorch.helpers.deterministic_model import DeterministicModel

def test__training_step__flow_dict(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Tests that only training_step can be used.'

    class TestModel(DeterministicModel):

        def training_step(self, batch, batch_idx):
            if False:
                for i in range(10):
                    print('nop')
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx
            self.training_step_called = True
            return {'loss': acc, 'random_things': [1, 'a', torch.tensor(2)]}

        def backward(self, loss):
            if False:
                print('Hello World!')
            return LightningModule.backward(self, loss)
    model = TestModel()
    model.val_dataloader = None
    trainer = Trainer(default_root_dir=tmpdir, limit_train_batches=2, limit_val_batches=2, max_epochs=2, log_every_n_steps=1, enable_model_summary=False)
    trainer.fit(model)
    assert model.training_step_called

def test__training_step__tr_batch_end__flow_dict(tmpdir):
    if False:
        return 10
    'Tests that only training_step can be used.'

    class TestModel(DeterministicModel):

        def training_step(self, batch, batch_idx):
            if False:
                for i in range(10):
                    print('nop')
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx
            self.training_step_called = True
            self.out = {'loss': acc, 'random_things': [1, 'a', torch.tensor(2)]}
            return self.out

        def on_train_batch_end(self, tr_step_output, *_):
            if False:
                while True:
                    i = 10
            assert self.count_num_graphs(tr_step_output) == 0

        def backward(self, loss):
            if False:
                print('Hello World!')
            return LightningModule.backward(self, loss)
    model = TestModel()
    model.val_dataloader = None
    trainer = Trainer(default_root_dir=tmpdir, limit_train_batches=2, limit_val_batches=2, max_epochs=2, log_every_n_steps=1, enable_model_summary=False)
    trainer.fit(model)
    assert model.training_step_called

def test__training_step__epoch_end__flow_dict(tmpdir):
    if False:
        while True:
            i = 10
    'Tests that only training_step can be used.'

    class TestModel(DeterministicModel):

        def training_step(self, batch, batch_idx):
            if False:
                print('Hello World!')
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx
            self.training_step_called = True
            return {'loss': acc, 'random_things': [1, 'a', torch.tensor(2)], 'batch_idx': batch_idx}

        def backward(self, loss):
            if False:
                print('Hello World!')
            return LightningModule.backward(self, loss)
    model = TestModel()
    model.val_dataloader = None
    trainer = Trainer(default_root_dir=tmpdir, limit_train_batches=2, limit_val_batches=2, max_epochs=2, log_every_n_steps=1, enable_model_summary=False)
    trainer.fit(model)
    assert model.training_step_called

def test__training_step__batch_end__epoch_end__flow_dict(tmpdir):
    if False:
        print('Hello World!')
    'Tests that only training_step can be used.'

    class TestModel(DeterministicModel):

        def training_step(self, batch, batch_idx):
            if False:
                for i in range(10):
                    print('nop')
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx
            self.training_step_called = True
            self.out = {'loss': acc, 'random_things': [1, 'a', torch.tensor(2)], 'batch_idx': batch_idx}
            return self.out

        def on_train_batch_end(self, tr_step_output, *_):
            if False:
                i = 10
                return i + 15
            assert self.count_num_graphs(tr_step_output) == 0

        def backward(self, loss):
            if False:
                i = 10
                return i + 15
            return LightningModule.backward(self, loss)
    model = TestModel()
    model.val_dataloader = None
    trainer = Trainer(default_root_dir=tmpdir, limit_train_batches=2, limit_val_batches=2, max_epochs=2, log_every_n_steps=1, enable_model_summary=False)
    trainer.fit(model)
    assert model.training_step_called