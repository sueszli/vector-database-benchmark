import collections.abc
import math
import warnings
from itertools import repeat
import torch
import torchvision
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    if False:
        while True:
            i = 10
    'Initialize network weights.\n\n    Args:\n        module_list (list[nn.Module] | nn.Module): Modules to be initialized.\n        scale (float): Scale initialized weights, especially for residual\n            blocks. Default: 1.\n        bias_fill (float): The value to fill bias. Default: 0\n        kwargs (dict): Other arguments for initialization function.\n    '
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)

def make_layer(basic_block, num_basic_block, **kwarg):
    if False:
        return 10
    'Make layers by stacking the same blocks.\n\n    Args:\n        basic_block (nn.module): nn.module class for basic block.\n        num_basic_block (int): number of blocks.\n\n    Returns:\n        nn.Sequential: Stacked blocks in nn.Sequential.\n    '
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        if False:
            print('Hello World!')
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        if False:
            return 10
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        if False:
            while True:
                i = 10
        m = []
        if scale & scale - 1 == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    if False:
        while True:
            i = 10
    "Warp an image or feature map with optical flow.\n\n    Args:\n        x (Tensor): Tensor with size (n, c, h, w).\n        flow (Tensor): Tensor with size (n, h, w, 2), normal value.\n        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.\n        padding_mode (str): 'zeros' or 'border' or 'reflection'.\n            Default: 'zeros'.\n        align_corners (bool): Before pytorch 1.3, the default value is\n            align_corners=True. After pytorch 1.3, the default value is\n            align_corners=False. Here, we use the True as default.\n\n    Returns:\n        Tensor: Warped image or feature map.\n    "
    assert x.size()[-2:] == flow.size()[1:3]
    (_, _, h, w) = x.size()
    (grid_y, grid_x) = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()
    grid.requires_grad = False
    vgrid = grid + flow
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)
    return output

def resize_flow(flow, size_type, sizes, interp_mode='bilinear', align_corners=False):
    if False:
        print('Hello World!')
    "Resize a flow according to ratio or shape.\n\n    Args:\n        flow (Tensor): Precomputed flow. shape [N, 2, H, W].\n        size_type (str): 'ratio' or 'shape'.\n        sizes (list[int | float]): the ratio for resizing or the final output\n            shape.\n            1) The order of ratio should be [ratio_h, ratio_w]. For\n            downsampling, the ratio should be smaller than 1.0 (i.e., ratio\n            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,\n            ratio > 1.0).\n            2) The order of output_size should be [out_h, out_w].\n        interp_mode (str): The mode of interpolation for resizing.\n            Default: 'bilinear'.\n        align_corners (bool): Whether align corners. Default: False.\n\n    Returns:\n        Tensor: Resized flow.\n    "
    (_, _, flow_h, flow_w) = flow.size()
    if size_type == 'ratio':
        (output_h, output_w) = (int(flow_h * sizes[0]), int(flow_w * sizes[1]))
    elif size_type == 'shape':
        (output_h, output_w) = (sizes[0], sizes[1])
    else:
        raise ValueError(f'Size type should be ratio or shape, but got type {size_type}.')
    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(input=input_flow, size=(output_h, output_w), mode=interp_mode, align_corners=align_corners)
    return resized_flow

def pixel_unshuffle(x, scale):
    if False:
        for i in range(10):
            print('nop')
    ' Pixel unshuffle.\n\n    Args:\n        x (Tensor): Input feature with shape (b, c, hh, hw).\n        scale (int): Downsample ratio.\n\n    Returns:\n        Tensor: the pixel unshuffled feature.\n    '
    (b, c, hh, hw) = x.size()
    out_channel = c * scale ** 2
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)