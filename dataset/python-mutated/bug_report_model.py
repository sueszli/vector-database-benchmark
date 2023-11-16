import os
import torch
from lightning.pytorch import LightningModule, Trainer
from torch.utils.data import DataLoader, Dataset

class RandomDataset(Dataset):

    def __init__(self, size, length):
        if False:
            for i in range(10):
                print('nop')
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        if False:
            while True:
                i = 10
        return self.data[index]

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.len

class BoringModel(LightningModule):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        if False:
            return 10
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        if False:
            while True:
                i = 10
        loss = self(batch).sum()
        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        if False:
            print('Hello World!')
        loss = self(batch).sum()
        self.log('valid_loss', loss)

    def test_step(self, batch, batch_idx):
        if False:
            for i in range(10):
                print('nop')
        loss = self(batch).sum()
        self.log('test_loss', loss)

    def configure_optimizers(self):
        if False:
            i = 10
            return i + 15
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)

def run():
    if False:
        for i in range(10):
            print('nop')
    train_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    val_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    test_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    model = BoringModel()
    trainer = Trainer(default_root_dir=os.getcwd(), limit_train_batches=1, limit_val_batches=1, limit_test_batches=1, num_sanity_val_steps=0, max_epochs=1, enable_model_summary=False)
    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)
    trainer.test(model, dataloaders=test_data)
if __name__ == '__main__':
    run()