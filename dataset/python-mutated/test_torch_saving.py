import os
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from tests_pytorch.helpers.runif import RunIf

def test_model_torch_save(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Test to ensure torch save does not fail for model and trainer.'
    model = BoringModel()
    num_epochs = 1
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=num_epochs)
    temp_path = os.path.join(tmpdir, 'temp.pt')
    trainer.fit(model)
    torch.save(trainer.model, temp_path)
    torch.save(trainer, temp_path)
    trainer = torch.load(temp_path)

@RunIf(skip_windows=True)
def test_model_torch_save_ddp_cpu(tmpdir):
    if False:
        i = 10
        return i + 15
    'Test to ensure torch save does not fail for model and trainer using cpu ddp.'
    model = BoringModel()
    num_epochs = 1
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=num_epochs, strategy='ddp_spawn', accelerator='cpu', devices=2, logger=False)
    temp_path = os.path.join(tmpdir, 'temp.pt')
    trainer.fit(model)
    torch.save(trainer.model, temp_path)
    torch.save(trainer, temp_path)

@RunIf(min_cuda_gpus=2)
def test_model_torch_save_ddp_cuda(tmpdir):
    if False:
        return 10
    'Test to ensure torch save does not fail for model and trainer using gpu ddp.'
    model = BoringModel()
    num_epochs = 1
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=num_epochs, strategy='ddp_spawn', accelerator='gpu', devices=2)
    temp_path = os.path.join(tmpdir, 'temp.pt')
    trainer.fit(model)
    torch.save(trainer.model, temp_path)
    torch.save(trainer, temp_path)