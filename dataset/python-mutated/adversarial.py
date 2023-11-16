"""
Part of the implementation is borrowed and modified from LaMa, publicly available at
https://github.com/saic-mdal/lama
"""
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseAdversarialLoss:

    def pre_generator_step(self, real_batch: torch.Tensor, fake_batch: torch.Tensor, generator: nn.Module, discriminator: nn.Module):
        if False:
            return 10
        '\n        Prepare for generator step\n        :param real_batch: Tensor, a batch of real samples\n        :param fake_batch: Tensor, a batch of samples produced by generator\n        :param generator:\n        :param discriminator:\n        :return: None\n        '

    def pre_discriminator_step(self, real_batch: torch.Tensor, fake_batch: torch.Tensor, generator: nn.Module, discriminator: nn.Module):
        if False:
            return 10
        '\n        Prepare for discriminator step\n        :param real_batch: Tensor, a batch of real samples\n        :param fake_batch: Tensor, a batch of samples produced by generator\n        :param generator:\n        :param discriminator:\n        :return: None\n        '

    def generator_loss(self, real_batch: torch.Tensor, fake_batch: torch.Tensor, discr_real_pred: torch.Tensor, discr_fake_pred: torch.Tensor, mask: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if False:
            i = 10
            return i + 15
        '\n        Calculate generator loss\n        :param real_batch: Tensor, a batch of real samples\n        :param fake_batch: Tensor, a batch of samples produced by generator\n        :param discr_real_pred: Tensor, discriminator output for real_batch\n        :param discr_fake_pred: Tensor, discriminator output for fake_batch\n        :param mask: Tensor, actual mask, which was at input of generator when making fake_batch\n        :return: total generator loss along with some values that might be interesting to log\n        '
        raise NotImplementedError

    def discriminator_loss(self, real_batch: torch.Tensor, fake_batch: torch.Tensor, discr_real_pred: torch.Tensor, discr_fake_pred: torch.Tensor, mask: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if False:
            print('Hello World!')
        '\n        Calculate discriminator loss and call .backward() on it\n        :param real_batch: Tensor, a batch of real samples\n        :param fake_batch: Tensor, a batch of samples produced by generator\n        :param discr_real_pred: Tensor, discriminator output for real_batch\n        :param discr_fake_pred: Tensor, discriminator output for fake_batch\n        :param mask: Tensor, actual mask, which was at input of generator when making fake_batch\n        :return: total discriminator loss along with some values that might be interesting to log\n        '
        raise NotImplementedError

    def interpolate_mask(self, mask, shape):
        if False:
            print('Hello World!')
        assert mask is not None
        assert self.allow_scale_mask or shape == mask.shape[-2:]
        if shape != mask.shape[-2:] and self.allow_scale_mask:
            if self.mask_scale_mode == 'maxpool':
                mask = F.adaptive_max_pool2d(mask, shape)
            else:
                mask = F.interpolate(mask, size=shape, mode=self.mask_scale_mode)
        return mask

def make_r1_gp(discr_real_pred, real_batch):
    if False:
        print('Hello World!')
    if torch.is_grad_enabled():
        grad_real = torch.autograd.grad(outputs=discr_real_pred.sum(), inputs=real_batch, create_graph=True)[0]
        grad_penalty = (grad_real.view(grad_real.shape[0], -1).norm(2, dim=1) ** 2).mean()
    else:
        grad_penalty = 0
    real_batch.requires_grad = False
    return grad_penalty

class NonSaturatingWithR1(BaseAdversarialLoss):

    def __init__(self, gp_coef=5, weight=1, mask_as_fake_target=False, allow_scale_mask=False, mask_scale_mode='nearest', extra_mask_weight_for_gen=0, use_unmasked_for_gen=True, use_unmasked_for_discr=True):
        if False:
            for i in range(10):
                print('nop')
        self.gp_coef = gp_coef
        self.weight = weight
        assert use_unmasked_for_gen or not use_unmasked_for_discr
        assert use_unmasked_for_discr or not mask_as_fake_target
        self.use_unmasked_for_gen = use_unmasked_for_gen
        self.use_unmasked_for_discr = use_unmasked_for_discr
        self.mask_as_fake_target = mask_as_fake_target
        self.allow_scale_mask = allow_scale_mask
        self.mask_scale_mode = mask_scale_mode
        self.extra_mask_weight_for_gen = extra_mask_weight_for_gen

    def generator_loss(self, real_batch: torch.Tensor, fake_batch: torch.Tensor, discr_real_pred: torch.Tensor, discr_fake_pred: torch.Tensor, mask=None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if False:
            i = 10
            return i + 15
        fake_loss = F.softplus(-discr_fake_pred)
        if self.mask_as_fake_target and self.extra_mask_weight_for_gen > 0 or not self.use_unmasked_for_gen:
            mask = self.interpolate_mask(mask, discr_fake_pred.shape[-2:])
            if not self.use_unmasked_for_gen:
                fake_loss = fake_loss * mask
            else:
                pixel_weights = 1 + mask * self.extra_mask_weight_for_gen
                fake_loss = fake_loss * pixel_weights
        return (fake_loss.mean() * self.weight, dict())

    def pre_discriminator_step(self, real_batch: torch.Tensor, fake_batch: torch.Tensor, generator: nn.Module, discriminator: nn.Module):
        if False:
            return 10
        real_batch.requires_grad = True

    def discriminator_loss(self, real_batch: torch.Tensor, fake_batch: torch.Tensor, discr_real_pred: torch.Tensor, discr_fake_pred: torch.Tensor, mask=None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if False:
            for i in range(10):
                print('nop')
        real_loss = F.softplus(-discr_real_pred)
        grad_penalty = make_r1_gp(discr_real_pred, real_batch) * self.gp_coef
        fake_loss = F.softplus(discr_fake_pred)
        if not self.use_unmasked_for_discr or self.mask_as_fake_target:
            mask = self.interpolate_mask(mask, discr_fake_pred.shape[-2:])
            fake_loss = fake_loss * mask
            if self.mask_as_fake_target:
                fake_loss = fake_loss + (1 - mask) * F.softplus(-discr_fake_pred)
        sum_discr_loss = real_loss + grad_penalty + fake_loss
        metrics = dict(discr_real_out=discr_real_pred.mean(), discr_fake_out=discr_fake_pred.mean(), discr_real_gp=grad_penalty)
        return (sum_discr_loss.mean(), metrics)