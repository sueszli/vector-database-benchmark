"""Tests to ensure that the behaviours related to multiple optimizers works."""
import lightning.pytorch as pl
import pytest
import torch
from lightning.pytorch.demos.boring_classes import BoringModel

class MultiOptModel(BoringModel):

    def configure_optimizers(self):
        if False:
            i = 10
            return i + 15
        opt_a = torch.optim.SGD(self.layer.parameters(), lr=0.001)
        opt_b = torch.optim.SGD(self.layer.parameters(), lr=0.001)
        return (opt_a, opt_b)

def test_multiple_optimizers_automatic_optimization_raises():
    if False:
        print('Hello World!')
    'Test that multiple optimizers in automatic optimization is not allowed.'

    class TestModel(BoringModel):

        def training_step(self, batch, batch_idx, optimizer_idx):
            if False:
                while True:
                    i = 10
            return super().training_step(batch, batch_idx)
    model = TestModel()
    model.automatic_optimization = True
    trainer = pl.Trainer()
    with pytest.raises(RuntimeError, match='Remove the `optimizer_idx` argument from `training_step`'):
        trainer.fit(model)

    class TestModel(BoringModel):

        def configure_optimizers(self):
            if False:
                i = 10
                return i + 15
            return (torch.optim.Adam(self.parameters()), torch.optim.Adam(self.parameters()))
    model = TestModel()
    model.automatic_optimization = True
    trainer = pl.Trainer()
    with pytest.raises(RuntimeError, match='multiple optimizers is only supported with manual optimization'):
        trainer.fit(model)

def test_multiple_optimizers_manual(tmpdir):
    if False:
        for i in range(10):
            print('nop')

    class TestModel(MultiOptModel):

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__()
            self.automatic_optimization = False

        def training_step(self, batch, batch_idx):
            if False:
                print('Hello World!')
            self.training_step_called = True
            (opt_a, opt_b) = self.optimizers()
            loss_1 = self.step(batch[0])
            self.manual_backward(loss_1)
            opt_a.step()
            opt_a.zero_grad()
            loss_2 = self.step(batch[0])
            self.manual_backward(loss_2)
            opt_b.step()
            opt_b.zero_grad()
    model = TestModel()
    model.val_dataloader = None
    trainer = pl.Trainer(default_root_dir=tmpdir, limit_train_batches=2, max_epochs=1, log_every_n_steps=1, enable_model_summary=False)
    trainer.fit(model)
    assert model.training_step_called