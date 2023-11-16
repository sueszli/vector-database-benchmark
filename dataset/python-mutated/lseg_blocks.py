import torch
import torch.nn as nn
from .lseg_vit import _make_pretrained_clip_vitl16_384, forward_vit

def _make_encoder(backbone, features, use_pretrained=True, groups=1, expand=False, exportable=True, hooks=None, use_vit_only=False, use_readout='ignore', enable_attention_hooks=False):
    if False:
        print('Hello World!')
    if backbone == 'clip_vitl16_384':
        (clip_pretrained, pretrained) = _make_pretrained_clip_vitl16_384(use_pretrained, hooks=hooks, use_readout=use_readout, enable_attention_hooks=enable_attention_hooks)
        scratch = _make_scratch([256, 512, 1024, 1024], features, groups=groups, expand=expand)
    else:
        raise NotImplementedError(f"Backbone '{backbone}' not implemented")
    return (clip_pretrained, pretrained, scratch)

def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    if False:
        return 10
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    if expand is True:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8
    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    return scratch

class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        if False:
            while True:
                i = 10
        'Init.\n\n        Args:\n            scale_factor (float): scaling\n            mode (str): interpolation mode\n        '
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        if False:
            while True:
                i = 10
        'Forward pass.\n\n        Args:\n            x (tensor): input\n\n        Returns:\n            tensor: interpolated data\n        '
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return x

class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features):
        if False:
            while True:
                i = 10
        'Init.\n\n        Args:\n            features (int): number of features\n        '
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if False:
            print('Hello World!')
        'Forward pass.\n\n        Args:\n            x (tensor): input\n\n        Returns:\n            tensor: output\n        '
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x

class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(self, features):
        if False:
            return 10
        'Init.\n\n        Args:\n            features (int): number of features\n        '
        super(FeatureFusionBlock, self).__init__()
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        if False:
            print('Hello World!')
        'Forward pass.\n\n        Returns:\n            tensor: output\n        '
        output = xs[0]
        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])
        output = self.resConfUnit2(output)
        output = nn.functional.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)
        return output

class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn):
        if False:
            return 10
        'Init.\n\n        Args:\n            features (int): number of features\n        '
        super().__init__()
        self.bn = bn
        self.groups = 1
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=not self.bn, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=not self.bn, groups=self.groups)
        if self.bn is True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        'Forward pass.\n\n        Args:\n            x (tensor): input\n\n        Returns:\n            tensor: output\n        '
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn is True:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn is True:
            out = self.bn2(out)
        if self.groups > 1:
            out = self.conv_merge(out)
        return self.skip_add.add(out, x)

class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block."""

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True):
        if False:
            while True:
                i = 10
        'Init.\n\n        Args:\n            features (int): number of features\n        '
        super(FeatureFusionBlock_custom, self).__init__()
        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = 1
        self.expand = expand
        out_features = features
        if self.expand is True:
            out_features = features // 2
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)
        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        if False:
            for i in range(10):
                print('nop')
        'Forward pass.\n\n        Returns:\n            tensor: output\n        '
        output = xs[0]
        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
        output = self.resConfUnit2(output)
        output = nn.functional.interpolate(output, scale_factor=2, mode='bilinear', align_corners=self.align_corners)
        output = self.out_conv(output)
        return output