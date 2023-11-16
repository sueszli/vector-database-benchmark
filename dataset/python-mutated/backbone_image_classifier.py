"""MNIST backbone image classifier example.

To run: python backbone_image_classifier.py --trainer.max_epochs=50

"""
from os import path
from typing import Optional
import torch
from lightning.pytorch import LightningDataModule, LightningModule, cli_lightning_logo
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.mnist_datamodule import MNIST
from lightning.pytorch.utilities.imports import _TORCHVISION_AVAILABLE
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
if _TORCHVISION_AVAILABLE:
    from torchvision import transforms
DATASETS_PATH = path.join(path.dirname(__file__), '..', '..', 'Datasets')

class Backbone(torch.nn.Module):
    """
    >>> Backbone()  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Backbone(
      (l1): Linear(...)
      (l2): Linear(...)
    )
    """

    def __init__(self, hidden_dim=128):
        if False:
            return 10
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, 10)

    def forward(self, x):
        if False:
            return 10
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        return torch.relu(self.l2(x))

class LitClassifier(LightningModule):
    """
    >>> LitClassifier(Backbone())  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    LitClassifier(
      (backbone): ...
    )
    """

    def __init__(self, backbone: Optional[Backbone]=None, learning_rate: float=0.0001):
        if False:
            while True:
                i = 10
        super().__init__()
        self.save_hyperparameters(ignore=['backbone'])
        if backbone is None:
            backbone = Backbone()
        self.backbone = backbone

    def forward(self, x):
        if False:
            print('Hello World!')
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        if False:
            return 10
        (x, y) = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if False:
            i = 10
            return i + 15
        (x, y) = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('valid_loss', loss, on_step=True)

    def test_step(self, batch, batch_idx):
        if False:
            return 10
        (x, y) = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        if False:
            print('Hello World!')
        (x, y) = batch
        return self(x)

    def configure_optimizers(self):
        if False:
            print('Hello World!')
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

class MyDataModule(LightningDataModule):

    def __init__(self, batch_size: int=32):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        dataset = MNIST(DATASETS_PATH, train=True, download=True, transform=transforms.ToTensor())
        self.mnist_test = MNIST(DATASETS_PATH, train=False, download=True, transform=transforms.ToTensor())
        (self.mnist_train, self.mnist_val) = random_split(dataset, [55000, 5000], generator=torch.Generator().manual_seed(42))
        self.batch_size = batch_size

    def train_dataloader(self):
        if False:
            while True:
                i = 10
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        if False:
            print('Hello World!')
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        if False:
            i = 10
            return i + 15
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        if False:
            print('Hello World!')
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

def cli_main():
    if False:
        while True:
            i = 10
    cli = LightningCLI(LitClassifier, MyDataModule, seed_everything_default=1234, save_config_kwargs={'overwrite': True}, run=False)
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(ckpt_path='best', datamodule=cli.datamodule)
    predictions = cli.trainer.predict(ckpt_path='best', datamodule=cli.datamodule)
    print(predictions[0])
if __name__ == '__main__':
    cli_lightning_logo()
    cli_main()