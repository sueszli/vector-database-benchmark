import numbers
import os.path as osp
from typing import Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from modelscope.metainfo import Models
from modelscope.models.base import Tensor, TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
logger = get_logger()
__all__ = ['AdaIntImageColorEnhance']
try:
    from modelscope.ops.ailut import ailut_transform
except ImportError:
    raise ImportError('The model [AdaInt] requires cuda extension to be installed.')

class BasicBlock(nn.Sequential):
    """The basic block module (Conv+LeakyReLU[+InstanceNorm]).
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, norm=False):
        if False:
            for i in range(10):
                print('nop')
        body = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1), nn.LeakyReLU(0.2)]
        if norm:
            body.append(nn.InstanceNorm2d(out_channels, affine=True))
        super(BasicBlock, self).__init__(*body)

class TPAMIBackbone(nn.Sequential):
    """The 5-layer CNN backbone module in [TPAMI 3D-LUT]
        (https://github.com/HuiZeng/Image-Adaptive-3DLUT).

    Args:
        pretrained (bool, optional): [ignored].
        input_resolution (int, optional): Resolution for pre-downsampling. Default: 256.
        extra_pooling (bool, optional): Whether to insert an extra pooling layer
            at the very end of the module to reduce the number of parameters of
            the subsequent module. Default: False.
    """

    def __init__(self, pretrained=False, input_resolution=256, extra_pooling=False):
        if False:
            i = 10
            return i + 15
        body = [BasicBlock(3, 16, stride=2, norm=True), BasicBlock(16, 32, stride=2, norm=True), BasicBlock(32, 64, stride=2, norm=True), BasicBlock(64, 128, stride=2, norm=True), BasicBlock(128, 128, stride=2), nn.Dropout(p=0.5)]
        if extra_pooling:
            body.append(nn.AdaptiveAvgPool2d(2))
        super().__init__(*body)
        self.input_resolution = input_resolution
        self.out_channels = 128 * (4 if extra_pooling else 64)

    def forward(self, imgs):
        if False:
            return 10
        imgs = F.interpolate(imgs, size=(self.input_resolution,) * 2, mode='bilinear', align_corners=False)
        return super().forward(imgs).view(imgs.shape[0], -1)

class Res18Backbone(nn.Module):
    """The ResNet-18 backbone.

    Args:
        pretrained (bool, optional): Whether to use the torchvison pretrained weights.
            Default: True.
        input_resolution (int, optional): Resolution for pre-downsampling. Default: 224.
        extra_pooling (bool, optional): [ignore].
    """

    def __init__(self, pretrained=True, input_resolution=224, extra_pooling=False):
        if False:
            i = 10
            return i + 15
        super().__init__()
        net = torchvision.models.resnet18(pretrained=pretrained)
        net.fc = nn.Identity()
        self.net = net
        self.input_resolution = input_resolution
        self.out_channels = 512

    def forward(self, imgs):
        if False:
            i = 10
            return i + 15
        imgs = F.interpolate(imgs, size=(self.input_resolution,) * 2, mode='bilinear', align_corners=False)
        return self.net(imgs).view(imgs.shape[0], -1)

class LUTGenerator(nn.Module):
    """The LUT generator module (mapping h).

    Args:
        n_colors (int): Number of input color channels.
        n_vertices (int): Number of sampling points along each lattice dimension.
        n_feats (int): Dimension of the input image representation vector.
        n_ranks (int): Number of ranks in the mapping h (or the number of basis LUTs).
    """

    def __init__(self, n_colors, n_vertices, n_feats, n_ranks) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.weights_generator = nn.Linear(n_feats, n_ranks)
        self.basis_luts_bank = nn.Linear(n_ranks, n_colors * n_vertices ** n_colors, bias=False)
        self.n_colors = n_colors
        self.n_vertices = n_vertices
        self.n_feats = n_feats
        self.n_ranks = n_ranks

    def init_weights(self):
        if False:
            for i in range(10):
                print('nop')
        'Init weights for models.\n\n        For the mapping f (`backbone`) and h (`lut_generator`), we follow the initialization in\n            [TPAMI 3D-LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT).\n\n        '
        nn.init.ones_(self.weights_generator.bias)
        tmp1 = torch.meshgrid(*[torch.arange(self.n_vertices) for _ in range(self.n_colors)])
        tmp2 = [torch.zeros(self.n_colors, *(self.n_vertices,) * self.n_colors) for _ in range(self.n_ranks - 1)]
        identity_lut = torch.stack([torch.stack(tmp1, dim=0).div(self.n_vertices - 1).flip(0), *tmp2], dim=0).view(self.n_ranks, -1)
        self.basis_luts_bank.weight.data.copy_(identity_lut.t())

    def forward(self, x):
        if False:
            while True:
                i = 10
        weights = self.weights_generator(x)
        luts = self.basis_luts_bank(weights)
        luts = luts.view(x.shape[0], -1, *(self.n_vertices,) * self.n_colors)
        return (weights, luts)

    def regularizations(self, smoothness, monotonicity):
        if False:
            return 10
        basis_luts = self.basis_luts_bank.weight.t().view(self.n_ranks, self.n_colors, *(self.n_vertices,) * self.n_colors)
        (tv, mn) = (0, 0)
        diff = basis_luts[:, :, :-1, ...] - basis_luts[:, :, 1:, ...]
        tv += torch.square(diff).sum(0).mean()
        mn += F.relu(diff).sum(0).mean()
        diff = basis_luts[:, :, :, :-1, :] - basis_luts[:, :, :, 1:, :]
        tv += torch.square(diff).sum(0).mean()
        mn += F.relu(diff).sum(0).mean()
        diff = basis_luts[:, :, :, :, :-1] - basis_luts[:, :, :, :, 1:]
        tv += torch.square(diff).sum(0).mean()
        mn += F.relu(diff).sum(0).mean()
        reg_smoothness = smoothness * tv
        reg_monotonicity = monotonicity * mn
        return (reg_smoothness, reg_monotonicity)

class AdaInt(nn.Module):
    """The Adaptive Interval Learning (AdaInt) module (mapping g).

    It consists of a single fully-connected layer and some post-process operations.

    Args:
        n_colors (int): Number of input color channels.
        n_vertices (int): Number of sampling points along each lattice dimension.
        n_feats (int): Dimension of the input image representation vector.
        adaint_share (bool, optional): Whether to enable Share-AdaInt. Default: False.
    """

    def __init__(self, n_colors, n_vertices, n_feats, adaint_share=False) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        repeat_factor = n_colors if not adaint_share else 1
        self.intervals_generator = nn.Linear(n_feats, (n_vertices - 1) * repeat_factor)
        self.n_colors = n_colors
        self.n_vertices = n_vertices
        self.adaint_share = adaint_share

    def init_weights(self):
        if False:
            i = 10
            return i + 15
        'Init weights for models.\n\n        We use all-zero and all-one initializations for its weights and bias, respectively.\n        '
        nn.init.zeros_(self.intervals_generator.weight)
        nn.init.ones_(self.intervals_generator.bias)

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        'Forward function for AdaInt module.\n\n        Args:\n            x (tensor): Input image representation, shape (b, f).\n        Returns:\n            Tensor: Sampling coordinates along each lattice dimension, shape (b, c, d).\n        '
        x = x.view(x.shape[0], -1)
        intervals = self.intervals_generator(x).view(x.shape[0], -1, self.n_vertices - 1)
        if self.adaint_share:
            intervals = intervals.repeat_interleave(self.n_colors, dim=1)
        intervals = intervals.softmax(-1)
        vertices = F.pad(intervals.cumsum(-1), (1, 0), 'constant', 0)
        return vertices

@MODELS.register_module(Tasks.image_color_enhancement, module_name=Models.adaint)
class AdaIntImageColorEnhance(TorchModel):
    """Adaptive-Interval 3D Lookup Table for real-time image enhancement.

    Args:
        n_ranks (int, optional): Number of ranks in the mapping h
            (or the number of basis LUTs). Default: 3.
        n_vertices (int, optional): Number of sampling points along
            each lattice dimension. Default: 33.
        en_adaint (bool, optional): Whether to enable AdaInt. Default: True.
        en_adaint_share (bool, optional): Whether to enable Share-AdaInt.
            Only used when `en_adaint` is True. Default: False.
        backbone (str, optional): Backbone architecture to use. Can be either 'tpami'
            or 'res18'. Default: 'tpami'.
        pretrained (bool, optional): Whether to use ImageNet-pretrained weights.
            Only used when `backbone` is 'res18'. Default: None.
        n_colors (int, optional): Number of input color channels. Default: 3.
        recons_loss (dict, optional): Config for pixel-wise reconstruction loss.
        train_cfg (dict, optional): Config for training. Default: None.
        test_cfg (dict, optional): Config for testing. Default: None.
    """

    def __init__(self, n_ranks=3, n_vertices=33, en_adaint=True, en_adaint_share=False, backbone='tpami', pretrained=False, n_colors=3, *args, **kwargs):
        if False:
            while True:
                i = 10
        super(AdaIntImageColorEnhance, self).__init__()
        assert backbone.lower() in ['tpami', 'res18']
        self.backbone = dict(tpami=TPAMIBackbone, res18=Res18Backbone)[backbone.lower()](pretrained, extra_pooling=en_adaint)
        self.lut_generator = LUTGenerator(n_colors, n_vertices, self.backbone.out_channels, n_ranks)
        if en_adaint:
            self.adaint = AdaInt(n_colors, n_vertices, self.backbone.out_channels, en_adaint_share)
        else:
            uniform_vertices = torch.arange(n_vertices).div(n_vertices - 1).repeat(n_colors, 1)
            self.register_buffer('uniform_vertices', uniform_vertices.unsqueeze(0))
        self.n_ranks = n_ranks
        self.n_colors = n_colors
        self.n_vertices = n_vertices
        self.en_adaint = en_adaint
        self.backbone_name = backbone.lower()
        self.init_weights()

    def init_weights(self):
        if False:
            return 10
        'Init weights for models.\n\n        For the mapping f (`backbone`) and h (`lut_generator`), we follow the initialization in\n            [TPAMI 3D-LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT).\n        For the mapping g (`adaint`), we use all-zero and all-one initializations for its weights\n        and bias, respectively.\n        '

        def special_initilization(m):
            if False:
                while True:
                    i = 10
            classname = m.__class__.__name__
            if 'Conv' in classname:
                nn.init.xavier_normal_(m.weight.data)
            elif 'InstanceNorm' in classname:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
        if self.backbone_name not in ['res18']:
            self.apply(special_initilization)
        self.lut_generator.init_weights()
        if self.en_adaint:
            self.adaint.init_weights()

    def __forward(self, imgs):
        if False:
            print('Hello World!')
        'The real implementation of model forward.\n\n        Args:\n            img (Tensor): Input image, shape (b, c, h, w).\n        Returns:\n            tuple(Tensor, Tensor, Tensor):\n                Output image, LUT weights, Sampling Coordinates.\n        '
        codes = self.backbone(imgs)
        (weights, luts) = self.lut_generator(codes)
        if self.en_adaint:
            vertices = self.adaint(codes)
        else:
            vertices = self.uniform_vertices
        outs = ailut_transform(imgs, luts, vertices)
        return (outs, weights, vertices)

    def _evaluate_postprocess(self, src: Tensor, target: Tensor) -> Dict[str, list]:
        if False:
            print('Hello World!')
        (preds, _, _) = self.__forward(src)
        preds = list(torch.split(preds, 1, 0))
        targets = list(torch.split(target, 1, 0))
        preds = [(pred.data * 255.0).squeeze(0).type(torch.uint8).permute(1, 2, 0).cpu().numpy() for pred in preds]
        targets = [(target.data * 255.0).squeeze(0).type(torch.uint8).permute(1, 2, 0).cpu().numpy() for target in targets]
        return {'pred': preds, 'target': targets}

    def _inference_forward(self, src: Tensor) -> Dict[str, Tensor]:
        if False:
            for i in range(10):
                print('nop')
        return {'outputs': self.__forward(src)[0].clamp(0, 1)}

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, Union[list, Tensor]]:
        if False:
            return 10
        'return the result by the model\n\n        Args:\n            input (Dict[str, Tensor]): the preprocessed data\n\n        Returns:\n            Dict[str, Union[list, Tensor]]: results\n        '
        if 'target' in input:
            return self._evaluate_postprocess(**input)
        else:
            return self._inference_forward(**input)

    def regularizations(self, smoothness, monotonicity):
        if False:
            i = 10
            return i + 15
        return self.lut_generator.regularizations(smoothness, monotonicity)

    @classmethod
    def _instantiate(cls, **kwargs):
        if False:
            return 10
        model_path = osp.join(kwargs['model_dir'], ModelFile.TORCH_MODEL_FILE)
        model = cls(**kwargs)
        model = model._load_pretrained(model, model_path, strict=False, param_key='state_dict')
        if model.training:
            model.train()
        else:
            model.eval()
        return model