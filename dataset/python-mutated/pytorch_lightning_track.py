import os
import lightning.pytorch as pl
from aim.pytorch_lightning import AimLogger
from torch import nn, optim, utils
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

class LitAutoEncoder(pl.LightningModule):

    def __init__(self, encoder, decoder):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        if False:
            print('Hello World!')
        (x, y) = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        if False:
            for i in range(10):
                print('nop')
        (x, y) = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        if False:
            for i in range(10):
                print('nop')
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer
autoencoder = LitAutoEncoder(encoder, decoder)
dataset = MNIST(os.getcwd(), train=True, download=True, transform=ToTensor())
(train_dataset, val_dataset) = utils.data.random_split(dataset, [55000, 5000])
test_dataset = MNIST(os.getcwd(), train=False, download=True, transform=ToTensor())
train_loader = utils.data.DataLoader(train_dataset)
val_loader = utils.data.DataLoader(val_dataset)
test_loader = utils.data.DataLoader(test_dataset)
aim_logger = AimLogger(experiment='example_experiment', train_metric_prefix='train_', test_metric_prefix='test_', val_metric_prefix='val_')
trainer = pl.Trainer(limit_train_batches=100, max_epochs=5, logger=aim_logger)
trainer.fit(model=autoencoder, train_dataloaders=train_loader, val_dataloaders=val_loader)
trainer.test(dataloaders=test_loader)