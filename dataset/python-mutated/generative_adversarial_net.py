"""To run this template just do: python generative_adversarial_net.py.

After a few epochs, launch TensorBoard to see the images being generated at every batch:

tensorboard --logdir default

"""
from argparse import ArgumentParser, Namespace
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import cli_lightning_logo
from lightning.pytorch.core import LightningModule
from lightning.pytorch.demos.mnist_datamodule import MNISTDataModule
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.utilities.imports import _TORCHVISION_AVAILABLE
if _TORCHVISION_AVAILABLE:
    import torchvision

class Generator(nn.Module):
    """
    >>> Generator(img_shape=(1, 8, 8))  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Generator(
      (model): Sequential(...)
    )
    """

    def __init__(self, latent_dim: int=100, img_shape: tuple=(1, 28, 28)):
        if False:
            print('Hello World!')
        super().__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            if False:
                while True:
                    i = 10
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
        return img.view(img.size(0), *self.img_shape)

class Discriminator(nn.Module):
    """
    >>> Discriminator(img_shape=(1, 28, 28))  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Discriminator(
      (model): Sequential(...)
    )
    """

    def __init__(self, img_shape):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.model = nn.Sequential(nn.Linear(int(np.prod(img_shape)), 512), nn.LeakyReLU(0.2, inplace=True), nn.Linear(512, 256), nn.LeakyReLU(0.2, inplace=True), nn.Linear(256, 1))

    def forward(self, img):
        if False:
            i = 10
            return i + 15
        img_flat = img.view(img.size(0), -1)
        return self.model(img_flat)

class GAN(LightningModule):
    """
    >>> GAN(img_shape=(1, 8, 8))  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    GAN(
      (generator): Generator(
        (model): Sequential(...)
      )
      (discriminator): Discriminator(
        (model): Sequential(...)
      )
    )
    """

    def __init__(self, img_shape: tuple=(1, 28, 28), lr: float=0.0002, b1: float=0.5, b2: float=0.999, latent_dim: int=100):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.generator = Generator(latent_dim=self.hparams.latent_dim, img_shape=img_shape)
        self.discriminator = Discriminator(img_shape=img_shape)
        self.validation_z = torch.randn(8, self.hparams.latent_dim)
        self.example_input_array = torch.zeros(2, self.hparams.latent_dim)

    def forward(self, z):
        if False:
            for i in range(10):
                print('nop')
        return self.generator(z)

    @staticmethod
    def adversarial_loss(y_hat, y):
        if False:
            print('Hello World!')
        return F.binary_cross_entropy_with_logits(y_hat, y)

    def training_step(self, batch):
        if False:
            while True:
                i = 10
        (imgs, _) = batch
        (opt_g, opt_d) = self.optimizers()
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)
        self.toggle_optimizer(opt_g)
        g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()
        self.untoggle_optimizer(opt_g)
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)
        self.toggle_optimizer(opt_d)
        real_loss = self.adversarial_loss(self.discriminator(imgs), valid)
        fake = torch.zeros(imgs.size(0), 1)
        fake = fake.type_as(imgs)
        fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()
        self.untoggle_optimizer(opt_d)
        self.log_dict({'d_loss': d_loss, 'g_loss': g_loss})

    def configure_optimizers(self):
        if False:
            while True:
                i = 10
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return (opt_g, opt_d)

    def on_train_epoch_end(self):
        if False:
            while True:
                i = 10
        z = self.validation_z.type_as(self.generator.model[0].weight)
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        for logger in self.loggers:
            logger.experiment.add_image('generated_images', grid, self.current_epoch)

def main(args: Namespace) -> None:
    if False:
        while True:
            i = 10
    model = GAN(lr=args.lr, b1=args.b1, b2=args.b2, latent_dim=args.latent_dim)
    dm = MNISTDataModule()
    trainer = Trainer(accelerator='gpu', devices=1)
    trainer.fit(model, dm)
if __name__ == '__main__':
    cli_lightning_logo()
    parser = ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of second order momentum of gradient')
    parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
    args = parser.parse_args()
    main(args)