from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from feature_model import FeaturesNet
from original_models import Discriminator, Generator

class LitModel(pl.LightningModule):

    def __init__(self, use_adversarial: bool=True, lr: float=0.0004, b1: float=0.9, b2: float=0.999, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.use_adversarial = use_adversarial
        self.generator = Generator(input_channels=3, output_channels=3)
        self.discriminator = Discriminator()
        self.featuresnet = FeaturesNet()

    def forward(self, input, **kwargs):
        if False:
            print('Hello World!')
        output = self.generator(input)
        return output

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        if False:
            i = 10
            return i + 15
        input = batch['input']
        target = batch['target']
        if optimizer_idx == 0:
            output = self.forward(input)
            loss_dict = self.compute_losses(output, target)
            loss = loss_dict['loss']
            self.log_dict(loss_dict)
            if batch_idx == 0:
                batch.update({'output': output})
                self.log_images(batch)
        elif optimizer_idx == 1:
            real_pred = self.discriminator(target)
            real = torch.ones_like(real_pred)
            output = self(input)
            real_loss = F.mse_loss(real_pred, real)
            fake_pred = self.discriminator(output)
            fake = torch.zeros_like(fake_pred)
            fake_loss = F.mse_loss(fake_pred, fake)
            loss = (real_loss + fake_loss) / 2
        return loss

    def log_images(self, images, n_log: int=None, prefix: str=''):
        if False:
            i = 10
            return i + 15
        for (k, v) in images.items():
            if isinstance(v, torch.Tensor):
                if n_log is not None:
                    v = v[:n_log]
                self.logger.experiment.add_images(prefix + k, v, self.current_epoch)

    def configure_optimizers(self):
        if False:
            print('Hello World!')
        lr = self.lr
        b1 = self.b1
        b2 = self.b2
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2), weight_decay=1e-05)
        opts = [opt_g]
        if self.use_adversarial:
            opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2), weight_decay=1e-05)
            opts.append(opt_d)
        return opts

    def validation_step(self, batch, batch_idx):
        if False:
            while True:
                i = 10
        batch = {k: v for (k, v) in batch.items() if isinstance(v, torch.Tensor)}
        input = batch['input']
        output = self.forward(input)
        batch.update({'output': output})
        val_loss_dict = self.compute_losses(**batch)
        self.log_dict(val_loss_dict)
        self.log_images(batch, n_log=1, prefix='val_')
        return val_loss_dict['loss']

    def compute_losses(self, output, target, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        perception_loss = F.l1_loss(self.featuresnet(target), self.featuresnet(output)) if self.featuresnet else 0
        reconstruction_loss = F.mse_loss(target, output)
        loss = perception_loss + reconstruction_loss
        loss_dict = {'perception': perception_loss, 'reconstruction': reconstruction_loss}
        if self.use_adversarial:
            real_pred = self.discriminator(output)
            real = torch.ones_like(real_pred)
            adversarial_loss = F.mse_loss(real_pred, real)
            loss_dict.update({'adversarial': adversarial_loss})
            loss = loss + adversarial_loss
        loss_dict.update({'loss': loss})
        return loss_dict