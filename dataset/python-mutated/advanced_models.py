import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.core.module import LightningModule
from torch.utils.data import DataLoader
from tests_pytorch import _PATH_DATASETS
from tests_pytorch.helpers.datasets import MNIST, AverageDataset, TrialMNIST

class Generator(nn.Module):

    def __init__(self, latent_dim: int, img_shape: tuple):
        if False:
            print('Hello World!')
        super().__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            if False:
                for i in range(10):
                    print('nop')
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(*block(latent_dim, 128, normalize=False), *block(128, 256), *block(256, 512), *block(512, 1024), nn.Linear(1024, int(np.prod(img_shape))), nn.Tanh())

    def forward(self, z):
        if False:
            for i in range(10):
                print('nop')
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

class Discriminator(nn.Module):

    def __init__(self, img_shape: tuple):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.model = nn.Sequential(nn.Linear(int(np.prod(img_shape)), 512), nn.LeakyReLU(0.2, inplace=True), nn.Linear(512, 256), nn.LeakyReLU(0.2, inplace=True), nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, img):
        if False:
            while True:
                i = 10
        img_flat = img.view(img.size(0), -1)
        return self.model(img_flat)

class BasicGAN(LightningModule):
    """Implements a basic GAN for the purpose of illustrating multiple optimizers."""

    def __init__(self, hidden_dim: int=128, learning_rate: float=0.001, b1: float=0.5, b2: float=0.999, **kwargs):
        if False:
            print('Hello World!')
        super().__init__()
        self.automatic_optimization = False
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.b1 = b1
        self.b2 = b2
        mnist_shape = (1, 28, 28)
        self.generator = Generator(latent_dim=self.hidden_dim, img_shape=mnist_shape)
        self.discriminator = Discriminator(img_shape=mnist_shape)
        self.generated_imgs = None
        self.last_imgs = None
        self.example_input_array = torch.rand(2, self.hidden_dim)

    def forward(self, z):
        if False:
            for i in range(10):
                print('nop')
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        if False:
            while True:
                i = 10
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx):
        if False:
            while True:
                i = 10
        (imgs, _) = batch
        self.last_imgs = imgs
        (optimizer1, optimizer2) = self.optimizers()
        self.toggle_optimizer(optimizer1)
        z = torch.randn(imgs.shape[0], self.hidden_dim)
        z = z.type_as(imgs)
        self.generated_imgs = self(z)
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)
        g_loss = self.adversarial_loss(self.discriminator(self.generated_imgs), valid)
        self.log('g_loss', g_loss, prog_bar=True, logger=True)
        self.manual_backward(g_loss)
        optimizer1.step()
        optimizer1.zero_grad()
        self.untoggle_optimizer(optimizer1)
        self.toggle_optimizer(optimizer2)
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)
        real_loss = self.adversarial_loss(self.discriminator(imgs), valid)
        fake = torch.zeros(imgs.size(0), 1)
        fake = fake.type_as(fake)
        fake_loss = self.adversarial_loss(self.discriminator(self.generated_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        self.log('d_loss', d_loss, prog_bar=True, logger=True)
        self.manual_backward(d_loss)
        optimizer2.step()
        optimizer2.zero_grad()
        self.untoggle_optimizer(optimizer2)

    def configure_optimizers(self):
        if False:
            return 10
        lr = self.learning_rate
        b1 = self.b1
        b2 = self.b2
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return ([opt_g, opt_d], [])

    def train_dataloader(self):
        if False:
            return 10
        return DataLoader(TrialMNIST(root=_PATH_DATASETS, train=True, download=True), batch_size=16)

class ParityModuleRNN(LightningModule):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.rnn = nn.LSTM(10, 20, batch_first=True)
        self.linear_out = nn.Linear(in_features=20, out_features=5)
        self.example_input_array = torch.rand(2, 3, 10)
        self._loss = []

    def forward(self, x):
        if False:
            return 10
        (seq, last) = self.rnn(x)
        return self.linear_out(seq)

    def training_step(self, batch, batch_nb):
        if False:
            return 10
        (x, y) = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self._loss.append(loss.item())
        return {'loss': loss}

    def configure_optimizers(self):
        if False:
            i = 10
            return i + 15
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def train_dataloader(self):
        if False:
            i = 10
            return i + 15
        return DataLoader(AverageDataset(), batch_size=30)

class ParityModuleMNIST(LightningModule):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.c_d1 = nn.Linear(in_features=28 * 28, out_features=128)
        self.c_d1_bn = nn.BatchNorm1d(128)
        self.c_d1_drop = nn.Dropout(0.3)
        self.c_d2 = nn.Linear(in_features=128, out_features=10)
        self.example_input_array = torch.rand(2, 1, 28, 28)
        self._loss = []

    def forward(self, x):
        if False:
            return 10
        x = x.view(x.size(0), -1)
        x = self.c_d1(x)
        x = torch.tanh(x)
        x = self.c_d1_bn(x)
        x = self.c_d1_drop(x)
        x = self.c_d2(x)
        return x

    def training_step(self, batch, batch_nb):
        if False:
            while True:
                i = 10
        (x, y) = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self._loss.append(loss.item())
        return {'loss': loss}

    def configure_optimizers(self):
        if False:
            for i in range(10):
                print('nop')
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def train_dataloader(self):
        if False:
            i = 10
            return i + 15
        return DataLoader(MNIST(root=_PATH_DATASETS, train=True, download=True), batch_size=128, num_workers=1)