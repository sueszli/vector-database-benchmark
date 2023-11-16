from fastai.data.load import DataLoader
import os, torch
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from lightning import LightningModule

class LitModel(LightningModule):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        if False:
            return 10
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        if False:
            while True:
                i = 10
        (x, y) = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return {'loss': loss}

    def configure_optimizers(self):
        if False:
            i = 10
            return i + 15
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):
        if False:
            print('Hello World!')
        dataset = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
        return DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True)

    def validation_step(self, batch, batch_idx):
        if False:
            while True:
                i = 10
        (x, y) = batch
        y_hat = self(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_epoch_end(self, outputs):
        if False:
            i = 10
            return i + 15
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        print(avg_loss)
        return {'val_loss': avg_loss}

    def val_dataloader(self):
        if False:
            for i in range(10):
                print('nop')
        dataset = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())
        loader = DataLoader(dataset, batch_size=32, num_workers=4)
        return loader