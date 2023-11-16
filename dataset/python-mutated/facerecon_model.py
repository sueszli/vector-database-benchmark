import os
from typing import Any, Callable, List, Optional, Type, Union
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import lr_scheduler
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

def filter_state_dict(state_dict, remove_name='fc'):
    if False:
        for i in range(10):
            print('nop')
    new_state_dict = {}
    for key in state_dict:
        if remove_name in key:
            continue
        new_state_dict[key] = state_dict[key]
    return new_state_dict

def get_scheduler(optimizer, opt):
    if False:
        print('Hello World!')
    'Return a learning rate scheduler\n\n    Parameters:\n        optimizer          -- the optimizer of the network\n        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．\u3000\n                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine\n\n    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.\n    See https://pytorch.org/docs/stable/optim.html for more details.\n    '
    if opt.lr_policy == 'linear':

        def lambda_rule(epoch):
            if False:
                for i in range(10):
                    print('nop')
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_epochs, gamma=0.2)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def define_net_recon(net_recon, use_last_fc=False, init_path=None):
    if False:
        print('Hello World!')
    return ReconNetWrapper(net_recon, use_last_fc=use_last_fc, init_path=init_path)

class ReconNetWrapper(nn.Module):
    fc_dim = 257

    def __init__(self, net_recon, use_last_fc=False, init_path=None):
        if False:
            for i in range(10):
                print('nop')
        super(ReconNetWrapper, self).__init__()
        self.use_last_fc = use_last_fc
        if net_recon not in func_dict:
            return NotImplementedError('network [%s] is not implemented', net_recon)
        (func, last_dim) = func_dict[net_recon]
        backbone = func(use_last_fc=use_last_fc, num_classes=self.fc_dim)
        if init_path and os.path.isfile(init_path):
            state_dict = filter_state_dict(torch.load(init_path, map_location='cpu'))
            backbone.load_state_dict(state_dict)
            print('loading init net_recon %s from %s' % (net_recon, init_path))
        self.backbone = backbone
        if not use_last_fc:
            self.final_layers = nn.ModuleList([conv1x1(last_dim, 80, bias=True), conv1x1(last_dim, 64, bias=True), conv1x1(last_dim, 80, bias=True), conv1x1(last_dim, 3, bias=True), conv1x1(last_dim, 27, bias=True), conv1x1(last_dim, 2, bias=True), conv1x1(last_dim, 1, bias=True)])
            for m in self.final_layers:
                nn.init.constant_(m.weight, 0.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        x = self.backbone(x)
        if not self.use_last_fc:
            output = []
            for layer in self.final_layers:
                output.append(layer(x))
            x = torch.flatten(torch.cat(output, dim=1), 1)
        return x
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2']
model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth', 'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth', 'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth', 'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth', 'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth', 'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth', 'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth', 'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth', 'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth'}

def conv3x3(in_planes: int, out_planes: int, stride: int=1, groups: int=1, dilation: int=1) -> nn.Conv2d:
    if False:
        for i in range(10):
            print('nop')
    '3x3 convolution with padding'
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int=1, bias: bool=False) -> nn.Conv2d:
    if False:
        i = 10
        return i + 15
    '1x1 convolution'
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes: int, planes: int, stride: int=1, downsample: Optional[nn.Module]=None, groups: int=1, base_width: int=64, dilation: int=1, norm_layer: Optional[Callable[..., nn.Module]]=None) -> None:
        if False:
            print('Hello World!')
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        if False:
            return 10
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, inplanes: int, planes: int, stride: int=1, downsample: Optional[nn.Module]=None, groups: int=1, base_width: int=64, dilation: int=1, norm_layer: Optional[Callable[..., nn.Module]]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        if False:
            return 10
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], num_classes: int=1000, zero_init_residual: bool=False, use_last_fc: bool=False, groups: int=1, width_per_group: int=64, replace_stride_with_dilation: Optional[List[bool]]=None, norm_layer: Optional[Callable[..., nn.Module]]=None) -> None:
        if False:
            print('Hello World!')
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.use_last_fc = use_last_fc
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if self.use_last_fc:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int, stride: int=1, dilate: bool=False) -> nn.Sequential:
        if False:
            while True:
                i = 10
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        if False:
            i = 10
            return i + 15
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        if self.use_last_fc:
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        if False:
            i = 10
            return i + 15
        return self._forward_impl(x)

def _resnet(arch: str, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], pretrained: bool, progress: bool, **kwargs: Any) -> ResNet:
    if False:
        while True:
            i = 10
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet18(pretrained: bool=False, progress: bool=True, **kwargs: Any) -> ResNet:
    if False:
        i = 10
        return i + 15
    'ResNet-18 model from\n    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.\n\n    Args:\n        pretrained (bool): If True, returns a model pre-trained on ImageNet\n        progress (bool): If True, displays a progress bar of the download to stderr\n    '
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)

def resnet34(pretrained: bool=False, progress: bool=True, **kwargs: Any) -> ResNet:
    if False:
        for i in range(10):
            print('nop')
    'ResNet-34 model from\n    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.\n\n    Args:\n        pretrained (bool): If True, returns a model pre-trained on ImageNet\n        progress (bool): If True, displays a progress bar of the download to stderr\n    '
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)

def resnet50(pretrained: bool=False, progress: bool=True, **kwargs: Any) -> ResNet:
    if False:
        for i in range(10):
            print('nop')
    'ResNet-50 model from\n    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.\n\n    Args:\n        pretrained (bool): If True, returns a model pre-trained on ImageNet\n        progress (bool): If True, displays a progress bar of the download to stderr\n    '
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)

def resnet101(pretrained: bool=False, progress: bool=True, **kwargs: Any) -> ResNet:
    if False:
        while True:
            i = 10
    'ResNet-101 model from\n    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.\n\n    Args:\n        pretrained (bool): If True, returns a model pre-trained on ImageNet\n        progress (bool): If True, displays a progress bar of the download to stderr\n    '
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)

def resnet152(pretrained: bool=False, progress: bool=True, **kwargs: Any) -> ResNet:
    if False:
        for i in range(10):
            print('nop')
    'ResNet-152 model from\n    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.\n\n    Args:\n        pretrained (bool): If True, returns a model pre-trained on ImageNet\n        progress (bool): If True, displays a progress bar of the download to stderr\n    '
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)

def resnext50_32x4d(pretrained: bool=False, progress: bool=True, **kwargs: Any) -> ResNet:
    if False:
        i = 10
        return i + 15
    'ResNeXt-50 32x4d model from\n    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.\n\n    Args:\n        pretrained (bool): If True, returns a model pre-trained on ImageNet\n        progress (bool): If True, displays a progress bar of the download to stderr\n    '
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)

def resnext101_32x8d(pretrained: bool=False, progress: bool=True, **kwargs: Any) -> ResNet:
    if False:
        for i in range(10):
            print('nop')
    'ResNeXt-101 32x8d model from\n    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.\n\n    Args:\n        pretrained (bool): If True, returns a model pre-trained on ImageNet\n        progress (bool): If True, displays a progress bar of the download to stderr\n    '
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)

def wide_resnet50_2(pretrained: bool=False, progress: bool=True, **kwargs: Any) -> ResNet:
    if False:
        for i in range(10):
            print('nop')
    'Wide ResNet-50-2 model from\n    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.\n\n    The model is the same as ResNet except for the bottleneck number of channels\n    which is twice larger in every block. The number of channels in outer 1x1\n    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048\n    channels, and in Wide ResNet-50-2 has 2048-1024-2048.\n\n    Args:\n        pretrained (bool): If True, returns a model pre-trained on ImageNet\n        progress (bool): If True, displays a progress bar of the download to stderr\n    '
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)

def wide_resnet101_2(pretrained: bool=False, progress: bool=True, **kwargs: Any) -> ResNet:
    if False:
        print('Hello World!')
    'Wide ResNet-101-2 model from\n    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.\n\n    The model is the same as ResNet except for the bottleneck number of channels\n    which is twice larger in every block. The number of channels in outer 1x1\n    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048\n    channels, and in Wide ResNet-50-2 has 2048-1024-2048.\n\n    Args:\n        pretrained (bool): If True, returns a model pre-trained on ImageNet\n        progress (bool): If True, displays a progress bar of the download to stderr\n    '
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)
func_dict = {'resnet18': (resnet18, 512), 'resnet50': (resnet50, 2048)}