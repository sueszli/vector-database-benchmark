import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from torch.utils.data import Dataset

class RandomDatasetA(Dataset):

    def __init__(self, size, length):
        if False:
            i = 10
            return i + 15
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        if False:
            for i in range(10):
                print('nop')
        return torch.zeros(1)

    def __len__(self):
        if False:
            while True:
                i = 10
        return self.len

class RandomDatasetB(Dataset):

    def __init__(self, size, length):
        if False:
            for i in range(10):
                print('nop')
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        if False:
            for i in range(10):
                print('nop')
        return torch.ones(1)

    def __len__(self):
        if False:
            while True:
                i = 10
        return self.len

@pytest.mark.parametrize('seq_type', [tuple, list])
def test_multiple_eval_dataloaders_seq(tmpdir, seq_type):
    if False:
        i = 10
        return i + 15

    class TestModel(BoringModel):

        def validation_step(self, batch, batch_idx, dataloader_idx):
            if False:
                for i in range(10):
                    print('nop')
            if dataloader_idx == 0:
                assert batch.sum() == 0
            elif dataloader_idx == 1:
                assert batch.sum() == 11
            else:
                raise Exception('should only have two dataloaders')

        def val_dataloader(self):
            if False:
                i = 10
                return i + 15
            dl1 = torch.utils.data.DataLoader(RandomDatasetA(32, 64), batch_size=11)
            dl2 = torch.utils.data.DataLoader(RandomDatasetB(32, 64), batch_size=11)
            return seq_type((dl1, dl2))
    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, limit_train_batches=2, limit_val_batches=2, max_epochs=1, log_every_n_steps=1, enable_model_summary=False)
    trainer.fit(model)