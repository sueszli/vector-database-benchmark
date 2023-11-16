import pytest
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from tests_pytorch.helpers.runif import RunIf

class TrainerGetModel(BoringModel):

    def on_fit_start(self):
        if False:
            i = 10
            return i + 15
        assert self == self.trainer.lightning_module

    def on_fit_end(self):
        if False:
            return 10
        assert self == self.trainer.lightning_module

def test_get_model(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Tests that `trainer.lightning_module` extracts the model correctly.'
    model = TrainerGetModel()
    limit_train_batches = 2
    trainer = Trainer(default_root_dir=tmpdir, limit_train_batches=limit_train_batches, limit_val_batches=2, max_epochs=1)
    trainer.fit(model)

@RunIf(skip_windows=True)
def test_get_model_ddp_cpu(tmpdir):
    if False:
        i = 10
        return i + 15
    'Tests that `trainer.lightning_module` extracts the model correctly when using ddp on cpu.'
    model = TrainerGetModel()
    limit_train_batches = 2
    trainer = Trainer(default_root_dir=tmpdir, limit_train_batches=limit_train_batches, limit_val_batches=2, max_epochs=1, accelerator='cpu', devices=2, strategy='ddp_spawn')
    trainer.fit(model)

@pytest.mark.parametrize('accelerator', [pytest.param('gpu', marks=RunIf(min_cuda_gpus=1)), pytest.param('mps', marks=RunIf(mps=True))])
def test_get_model_gpu(tmpdir, accelerator):
    if False:
        return 10
    'Tests that `trainer.lightning_module` extracts the model correctly when using GPU.'
    model = TrainerGetModel()
    limit_train_batches = 2
    trainer = Trainer(default_root_dir=tmpdir, limit_train_batches=limit_train_batches, limit_val_batches=2, max_epochs=1, accelerator=accelerator, devices=1)
    trainer.fit(model)