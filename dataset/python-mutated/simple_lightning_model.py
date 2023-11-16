from __future__ import annotations
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import random_split, DataLoader
from torchmetrics.functional import accuracy
from torchvision.datasets import MNIST
from torchvision import transforms
import nni
from .simple_torch_model import SimpleTorchModel

class SimpleLightningModel(pl.LightningModule):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.model = SimpleTorchModel()

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if False:
            print('Hello World!')
        (x, y) = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def evaluate(self, batch, stage=None):
        if False:
            while True:
                i = 10
        (x, y) = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, 'multiclass', num_classes=10)
        if stage:
            self.log(f'default', acc, prog_bar=False)
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_acc', acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        if False:
            return 10
        self.evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        if False:
            i = 10
            return i + 15
        self.evaluate(batch, 'test')

    def configure_optimizers(self):
        if False:
            while True:
                i = 10
        optimizer = nni.trace(torch.optim.SGD)(self.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
        scheduler_dict = {'scheduler': nni.trace(ExponentialLR)(optimizer, 0.1), 'interval': 'epoch'}
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}

class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str='./'):
        if False:
            while True:
                i = 10
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        if False:
            for i in range(10):
                print('nop')
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str | None=None):
        if False:
            print('Hello World!')
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            (self.mnist_train, self.mnist_val) = random_split(mnist_full, [55000, 5000])
        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
        if stage == 'predict' or stage is None:
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        if False:
            while True:
                i = 10
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        if False:
            print('Hello World!')
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        if False:
            i = 10
            return i + 15
        return DataLoader(self.mnist_test, batch_size=32)

    def predict_dataloader(self):
        if False:
            i = 10
            return i + 15
        return DataLoader(self.mnist_predict, batch_size=32)