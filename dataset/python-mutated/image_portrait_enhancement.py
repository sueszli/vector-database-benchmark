import math
import os.path as osp
from typing import Any, Dict, List, Union
import torch
import torch.nn.functional as F
from torch import autograd, nn
from modelscope.metainfo import Models
from modelscope.models.base import Tensor, TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from .gpen import Discriminator, FullGenerator
from .losses.losses import IDLoss, L1Loss
logger = get_logger()
__all__ = ['ImagePortraitEnhancement']

@MODELS.register_module(Tasks.image_portrait_enhancement, module_name=Models.gpen)
class ImagePortraitEnhancement(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'initialize the face enhancement model from the `model_dir` path.\n\n        Args:\n            model_dir (str): the model path.\n        '
        super().__init__(model_dir, *args, **kwargs)
        self.size = 256
        self.style_dim = 512
        self.n_mlp = 8
        self.mean_path_length = 0
        self.accum = 0.5 ** (32 / (10 * 1000))
        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')
        self.l1_loss = L1Loss()
        self.id_loss = IDLoss(f'{model_dir}/arcface/model_ir_se50.pth', self._device)
        self.generator = FullGenerator(self.size, self.style_dim, self.n_mlp, isconcat=True).to(self._device)
        self.g_ema = FullGenerator(self.size, self.style_dim, self.n_mlp, isconcat=True).to(self._device)
        self.discriminator = Discriminator(self.size).to(self._device)
        if self.size == 512:
            self.load_pretrained(model_dir)

    def load_pretrained(self, model_dir):
        if False:
            print('Hello World!')
        g_path = f'{model_dir}/{ModelFile.TORCH_MODEL_FILE}'
        g_dict = torch.load(g_path, map_location=torch.device('cpu'))
        self.generator.load_state_dict(g_dict)
        self.g_ema.load_state_dict(g_dict)
        d_path = f'{model_dir}/net_d.pt'
        d_dict = torch.load(d_path, map_location=torch.device('cpu'))
        self.discriminator.load_state_dict(d_dict)
        logger.info('load model done.')

    def accumulate(self):
        if False:
            while True:
                i = 10
        par1 = dict(self.g_ema.named_parameters())
        par2 = dict(self.generator.named_parameters())
        for k in par1.keys():
            par1[k].data.mul_(self.accum).add_(1 - self.accum, par2[k].data)

    def requires_grad(self, model, flag=True):
        if False:
            return 10
        for p in model.parameters():
            p.requires_grad = flag

    def d_logistic_loss(self, real_pred, fake_pred):
        if False:
            i = 10
            return i + 15
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)
        return real_loss.mean() + fake_loss.mean()

    def d_r1_loss(self, real_pred, real_img):
        if False:
            print('Hello World!')
        (grad_real,) = autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)
        grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
        return grad_penalty

    def g_nonsaturating_loss(self, fake_pred, fake_img=None, real_img=None, input_img=None):
        if False:
            print('Hello World!')
        loss = F.softplus(-fake_pred).mean()
        loss_l1 = self.l1_loss(fake_img, real_img)
        (loss_id, __, __) = self.id_loss(fake_img, real_img, input_img)
        loss_id = 0
        loss += 1.0 * loss_l1 + 1.0 * loss_id
        return loss

    def g_path_regularize(self, fake_img, latents, mean_path_length, decay=0.01):
        if False:
            for i in range(10):
                print('nop')
        noise = torch.randn_like(fake_img) / math.sqrt(fake_img.shape[2] * fake_img.shape[3])
        (grad,) = autograd.grad(outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True)
        path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))
        path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)
        path_penalty = (path_lengths - path_mean).pow(2).mean()
        return (path_penalty, path_mean.detach(), path_lengths)

    @torch.no_grad()
    def _evaluate_postprocess(self, input: Tensor, target: Tensor) -> Dict[str, list]:
        if False:
            i = 10
            return i + 15
        (preds, _) = self.generator(input)
        preds = list(torch.split(preds, 1, 0))
        targets = list(torch.split(target, 1, 0))
        preds = [((pred.data * 0.5 + 0.5) * 255.0).squeeze(0).type(torch.uint8).permute(1, 2, 0).cpu().numpy() for pred in preds]
        targets = [((target.data * 0.5 + 0.5) * 255.0).squeeze(0).type(torch.uint8).permute(1, 2, 0).cpu().numpy() for target in targets]
        return {'pred': preds, 'target': targets}

    def _train_forward_d(self, input: Tensor, target: Tensor) -> Tensor:
        if False:
            while True:
                i = 10
        self.requires_grad(self.generator, False)
        self.requires_grad(self.discriminator, True)
        (preds, _) = self.generator(input)
        fake_pred = self.discriminator(preds)
        real_pred = self.discriminator(target)
        d_loss = self.d_logistic_loss(real_pred, fake_pred)
        return d_loss

    def _train_forward_d_r1(self, input: Tensor, target: Tensor) -> Tensor:
        if False:
            return 10
        input.requires_grad = True
        target.requires_grad = True
        real_pred = self.discriminator(target)
        r1_loss = self.d_r1_loss(real_pred, target)
        return r1_loss

    def _train_forward_g(self, input: Tensor, target: Tensor) -> Tensor:
        if False:
            print('Hello World!')
        self.requires_grad(self.generator, True)
        self.requires_grad(self.discriminator, False)
        (preds, _) = self.generator(input)
        fake_pred = self.discriminator(preds)
        g_loss = self.g_nonsaturating_loss(fake_pred, preds, target, input)
        return g_loss

    def _train_forward_g_path(self, input: Tensor, target: Tensor) -> Tensor:
        if False:
            i = 10
            return i + 15
        (fake_img, latents) = self.generator(input, return_latents=True)
        (path_loss, self.mean_path_length, path_lengths) = self.g_path_regularize(fake_img, latents, self.mean_path_length)
        return path_loss

    @torch.no_grad()
    def _inference_forward(self, input: Tensor) -> Dict[str, Tensor]:
        if False:
            print('Hello World!')
        return {'outputs': (self.generator(input)[0] * 0.5 + 0.5).clamp(0, 1)}

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, Union[list, Tensor]]:
        if False:
            return 10
        'return the result by the model\n\n        Args:\n            input (Dict[str, Tensor]): the preprocessed data\n\n        Returns:\n            Dict[str, Union[list, Tensor]]: results\n        '
        for (key, value) in input.items():
            input[key] = input[key].to(self._device)
        if 'target' in input:
            return self._evaluate_postprocess(**input)
        else:
            return self._inference_forward(**input)