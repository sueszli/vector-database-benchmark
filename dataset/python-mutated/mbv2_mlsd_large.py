import os
import sys
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F

class BlockTypeA(nn.Module):

    def __init__(self, in_c1, in_c2, out_c1, out_c2, upscale=True):
        if False:
            print('Hello World!')
        super(BlockTypeA, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_c2, out_c2, kernel_size=1), nn.BatchNorm2d(out_c2), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_c1, out_c1, kernel_size=1), nn.BatchNorm2d(out_c1), nn.ReLU(inplace=True))
        self.upscale = upscale

    def forward(self, a, b):
        if False:
            for i in range(10):
                print('nop')
        b = self.conv1(b)
        a = self.conv2(a)
        if self.upscale:
            b = F.interpolate(b, scale_factor=2.0, mode='bilinear', align_corners=True)
        return torch.cat((a, b), dim=1)

class BlockTypeB(nn.Module):

    def __init__(self, in_c, out_c):
        if False:
            return 10
        super(BlockTypeB, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_c, in_c, kernel_size=3, padding=1), nn.BatchNorm2d(in_c), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU())

    def forward(self, x):
        if False:
            return 10
        x = self.conv1(x) + x
        x = self.conv2(x)
        return x

class BlockTypeC(nn.Module):

    def __init__(self, in_c, out_c):
        if False:
            while True:
                i = 10
        super(BlockTypeC, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_c, in_c, kernel_size=3, padding=5, dilation=5), nn.BatchNorm2d(in_c), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_c, in_c, kernel_size=3, padding=1), nn.BatchNorm2d(in_c), nn.ReLU())
        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1)

    def forward(self, x):
        if False:
            print('Hello World!')
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

def _make_divisible(v, divisor, min_value=None):
    if False:
        return 10
    '\n    This function is taken from the original tf repo.\n    It ensures that all layers have a channel number that is divisible by 8\n    It can be seen here:\n    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py\n    :param v:\n    :param divisor:\n    :param min_value:\n    :return:\n    '
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        if False:
            return 10
        self.channel_pad = out_planes - in_planes
        self.stride = stride
        if stride == 2:
            padding = 0
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False), nn.BatchNorm2d(out_planes), nn.ReLU6(inplace=True))
        self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        if False:
            return 10
        if self.stride == 2:
            x = F.pad(x, (0, 1, 0, 1), 'constant', 0)
        for module in self:
            if not isinstance(module, nn.MaxPool2d):
                x = module(x)
        return x

class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        if False:
            print('Hello World!')
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):

    def __init__(self, pretrained=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        MobileNet V2 main class\n        Args:\n            num_classes (int): Number of classes\n            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount\n            inverted_residual_setting: Network structure\n            round_nearest (int): Round the number of channels in each layer to be a multiple of this number\n            Set to 1 to turn off rounding\n            block: Module specifying inverted residual building block for mobilenet\n        '
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        width_mult = 1.0
        round_nearest = 8
        inverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1]]
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError('inverted_residual_setting should be non-empty or a 4-element list, got {}'.format(inverted_residual_setting))
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(4, input_channel, stride=2)]
        for (t, c, n, s) in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        self.features = nn.Sequential(*features)
        self.fpn_selected = [1, 3, 6, 10, 13]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        if pretrained:
            self._load_pretrained_model()

    def _forward_impl(self, x):
        if False:
            while True:
                i = 10
        fpn_features = []
        for (i, f) in enumerate(self.features):
            if i > self.fpn_selected[-1]:
                break
            x = f(x)
            if i in self.fpn_selected:
                fpn_features.append(x)
        (c1, c2, c3, c4, c5) = fpn_features
        return (c1, c2, c3, c4, c5)

    def forward(self, x):
        if False:
            while True:
                i = 10
        return self._forward_impl(x)

    def _load_pretrained_model(self):
        if False:
            for i in range(10):
                print('nop')
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/mobilenet_v2-b0353104.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for (k, v) in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

class MobileV2_MLSD_Large(nn.Module):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(MobileV2_MLSD_Large, self).__init__()
        self.backbone = MobileNetV2(pretrained=False)
        self.block15 = BlockTypeA(in_c1=64, in_c2=96, out_c1=64, out_c2=64, upscale=False)
        self.block16 = BlockTypeB(128, 64)
        self.block17 = BlockTypeA(in_c1=32, in_c2=64, out_c1=64, out_c2=64)
        self.block18 = BlockTypeB(128, 64)
        self.block19 = BlockTypeA(in_c1=24, in_c2=64, out_c1=64, out_c2=64)
        self.block20 = BlockTypeB(128, 64)
        self.block21 = BlockTypeA(in_c1=16, in_c2=64, out_c1=64, out_c2=64)
        self.block22 = BlockTypeB(128, 64)
        self.block23 = BlockTypeC(64, 16)

    def forward(self, x):
        if False:
            while True:
                i = 10
        (c1, c2, c3, c4, c5) = self.backbone(x)
        x = self.block15(c4, c5)
        x = self.block16(x)
        x = self.block17(c3, x)
        x = self.block18(x)
        x = self.block19(c2, x)
        x = self.block20(x)
        x = self.block21(c1, x)
        x = self.block22(x)
        x = self.block23(x)
        x = x[:, 7:, :, :]
        return x