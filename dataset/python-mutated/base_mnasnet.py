from nni.retiarii import basic_unit
import nni.retiarii.nn.pytorch as nn
import warnings
import torch
import torch.nn as torch_nn
import torch.nn.functional as F
from nni.retiarii import model_wrapper
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
_BN_MOMENTUM = 1 - 0.9997
_FIRST_DEPTH = 32
_MOBILENET_V2_FILTERS = [16, 24, 32, 64, 96, 160, 320]
_MOBILENET_V2_NUM_LAYERS = [1, 2, 3, 4, 3, 3, 1]

class _ResidualBlock(nn.Module):

    def __init__(self, net):
        if False:
            print('Hello World!')
        super().__init__()
        self.net = net

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        return self.net(x) + x

class _InvertedResidual(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride, expansion_factor, skip, bn_momentum=0.1):
        if False:
            print('Hello World!')
        super(_InvertedResidual, self).__init__()
        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        mid_ch = in_ch * expansion_factor
        self.apply_residual = skip and in_ch == out_ch and (stride == 1)
        self.layers = nn.Sequential(nn.Conv2d(in_ch, mid_ch, 1, bias=False), nn.BatchNorm2d(mid_ch, momentum=bn_momentum), nn.ReLU(inplace=True), nn.Conv2d(mid_ch, mid_ch, kernel_size, padding=kernel_size // 2, stride=stride, groups=mid_ch, bias=False), nn.BatchNorm2d(mid_ch, momentum=bn_momentum), nn.ReLU(inplace=True), nn.Conv2d(mid_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch, momentum=bn_momentum))

    def forward(self, input):
        if False:
            while True:
                i = 10
        if self.apply_residual:
            ret = self.layers(input) + input
        else:
            ret = self.layers(input)
        return ret

def _stack_inverted_residual(in_ch, out_ch, kernel_size, skip, stride, exp_factor, repeats, bn_momentum):
    if False:
        i = 10
        return i + 15
    ' Creates a stack of inverted residuals. '
    assert repeats >= 1
    first = _InvertedResidual(in_ch, out_ch, kernel_size, stride, exp_factor, skip, bn_momentum=bn_momentum)
    remaining = []
    for _ in range(1, repeats):
        remaining.append(_InvertedResidual(out_ch, out_ch, kernel_size, 1, exp_factor, skip, bn_momentum=bn_momentum))
    return nn.Sequential(first, *remaining)

def _stack_normal_conv(in_ch, out_ch, kernel_size, skip, dconv, stride, repeats, bn_momentum):
    if False:
        while True:
            i = 10
    assert repeats >= 1
    stack = []
    for i in range(repeats):
        s = stride if i == 0 else 1
        if dconv:
            modules = [nn.Conv2d(in_ch, in_ch, kernel_size, padding=kernel_size // 2, stride=s, groups=in_ch, bias=False), nn.BatchNorm2d(in_ch, momentum=bn_momentum), nn.ReLU(inplace=True), nn.Conv2d(in_ch, out_ch, 1, padding=0, stride=1, bias=False), nn.BatchNorm2d(out_ch, momentum=bn_momentum)]
        else:
            modules = [nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, stride=s, bias=False), nn.ReLU(inplace=True), nn.BatchNorm2d(out_ch, momentum=bn_momentum)]
        if skip and in_ch == out_ch and (s == 1):
            stack.append(_ResidualBlock(nn.Sequential(*modules)))
        else:
            stack += modules
        in_ch = out_ch
    return stack

def _round_to_multiple_of(val, divisor, round_up_bias=0.9):
    if False:
        return 10
    ' Asymmetric rounding to make `val` divisible by `divisor`. With default\n    bias, will round up, unless the number is no more than 10% greater than the\n    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. '
    assert 0.0 < round_up_bias < 1.0
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor

def _get_depths(depths, alpha):
    if False:
        while True:
            i = 10
    ' Scales tensor depths as in reference MobileNet code, prefers rouding up\n    rather than down. '
    return [_round_to_multiple_of(depth * alpha, 8) for depth in depths]

@model_wrapper
class MNASNet(nn.Module):
    """ MNASNet, as described in https://arxiv.org/pdf/1807.11626.pdf. This
    implements the B1 variant of the model.
    >>> model = MNASNet(1000, 1.0)
    >>> x = torch.rand(1, 3, 224, 224)
    >>> y = model(x)
    >>> y.dim()
    1
    >>> y.nelement()
    1000
    """
    _version = 2

    def __init__(self, alpha, depths, convops, kernel_sizes, num_layers, skips, num_classes=1000, dropout=0.2):
        if False:
            print('Hello World!')
        super().__init__()
        assert alpha > 0.0
        assert len(depths) == len(convops) == len(kernel_sizes) == len(num_layers) == len(skips) == 7
        self.alpha = alpha
        self.num_classes = num_classes
        depths = _get_depths([_FIRST_DEPTH] + depths, alpha)
        base_filter_sizes = [16, 24, 40, 80, 96, 192, 320]
        exp_ratios = [3, 3, 3, 6, 6, 6, 6]
        strides = [1, 2, 2, 2, 1, 2, 1]
        layers = [nn.Conv2d(3, depths[0], 3, padding=1, stride=2, bias=False), nn.BatchNorm2d(depths[0], momentum=_BN_MOMENTUM), nn.ReLU(inplace=True)]
        count = 0
        for (filter_size, exp_ratio, stride) in zip(base_filter_sizes, exp_ratios, strides):
            ph = nn.Placeholder(label=f'mutable_{count}', **{'kernel_size_options': [1, 3, 5], 'n_layer_options': [1, 2, 3, 4], 'op_type_options': ['__mutated__.base_mnasnet.RegularConv', '__mutated__.base_mnasnet.DepthwiseConv', '__mutated__.base_mnasnet.MobileConv'], 'skip_options': ['identity', 'no'], 'n_filter_options': [int(filter_size * x) for x in [0.75, 1.0, 1.25]], 'exp_ratio': exp_ratio, 'stride': stride, 'in_ch': depths[0] if count == 0 else None})
            layers.append(ph)
            'if conv == "mconv":\n                # MNASNet blocks: stacks of inverted residuals.\n                layers.append(_stack_inverted_residual(prev_depth, depth, ks, skip,\n                                                       stride, exp_ratio, repeat, _BN_MOMENTUM))\n            else:\n                # Normal conv and depth-separated conv\n                layers += _stack_normal_conv(prev_depth, depth, ks, skip, conv == "dconv",\n                                             stride, repeat, _BN_MOMENTUM)'
            count += 1
            if count >= 2:
                break
        layers += [nn.Conv2d(depths[7], 1280, 1, padding=0, stride=1, bias=False), nn.BatchNorm2d(1280, momentum=_BN_MOMENTUM), nn.ReLU(inplace=True)]
        self.layers = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(1280, num_classes))
        self._initialize_weights()

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        x = self.layers(x)
        x = x.mean([2, 3])
        x = F.relu(x)
        return self.classifier(x)

    def _initialize_weights(self):
        if False:
            while True:
                i = 10
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch_nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch_nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                torch_nn.init.ones_(m.weight)
                torch_nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                torch_nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='sigmoid')
                torch_nn.init.zeros_(m.bias)

def test_model(model):
    if False:
        for i in range(10):
            print('nop')
    model(torch.randn(2, 3, 224, 224))
BN_MOMENTUM = 1 - 0.9997

class RegularConv(nn.Module):

    def __init__(self, kernel_size, in_ch, out_ch, skip, exp_ratio, stride):
        if False:
            print('Hello World!')
        super().__init__()
        self.kernel_size = kernel_size
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.skip = skip
        self.exp_ratio = exp_ratio
        self.stride = stride
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, stride=stride, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_ch, momentum=BN_MOMENTUM)

    def forward(self, x):
        if False:
            print('Hello World!')
        out = self.bn(self.relu(self.conv(x)))
        if self.skip == 'identity':
            out = out + x
        return out

class DepthwiseConv(nn.Module):

    def __init__(self, kernel_size, in_ch, out_ch, skip, exp_ratio, stride):
        if False:
            while True:
                i = 10
        super().__init__()
        self.kernel_size = kernel_size
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.skip = skip
        self.exp_ratio = exp_ratio
        self.stride = stride
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size, padding=kernel_size // 2, stride=stride, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 1, padding=0, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch, momentum=BN_MOMENTUM)

    def forward(self, x):
        if False:
            print('Hello World!')
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.skip == 'identity':
            out = out + x
        return out

class MobileConv(nn.Module):

    def __init__(self, kernel_size, in_ch, out_ch, skip, exp_ratio, stride):
        if False:
            return 10
        super().__init__()
        self.kernel_size = kernel_size
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.skip = skip
        self.exp_ratio = exp_ratio
        self.stride = stride
        mid_ch = in_ch * exp_ratio
        self.layers = nn.Sequential(nn.Conv2d(in_ch, mid_ch, 1, bias=False), nn.BatchNorm2d(mid_ch, momentum=BN_MOMENTUM), nn.ReLU(inplace=True), nn.Conv2d(mid_ch, mid_ch, kernel_size, padding=(kernel_size - 1) // 2, stride=stride, groups=mid_ch, bias=False), nn.BatchNorm2d(mid_ch, momentum=BN_MOMENTUM), nn.ReLU(inplace=True), nn.Conv2d(mid_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch, momentum=BN_MOMENTUM))

    def forward(self, x):
        if False:
            while True:
                i = 10
        out = self.layers(x)
        if self.skip == 'identity':
            out = out + x
        return out
ir_module = _InvertedResidual(16, 16, 3, 1, 1, True)