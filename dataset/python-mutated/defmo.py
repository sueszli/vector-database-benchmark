from typing import Callable, Dict, List, Optional, Type
import torch
from torch import nn
from kornia.core import Module, Tensor, concatenate, stack
from kornia.utils.helpers import map_location_to_cpu
urls: Dict[str, str] = {}
urls['defmo_encoder'] = 'http://ptak.felk.cvut.cz/personal/rozumden/defmo_saved_models/encoder_best.pt'
urls['defmo_rendering'] = 'http://ptak.felk.cvut.cz/personal/rozumden/defmo_saved_models/rendering_best.pt'

def conv1x1(in_planes: int, out_planes: int, stride: int=1) -> nn.Conv2d:
    if False:
        while True:
            i = 10
    '1x1 convolution.'
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes: int, out_planes: int, stride: int=1, groups: int=1, dilation: int=1) -> nn.Conv2d:
    if False:
        return 10
    '3x3 convolution with padding.'
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

class Bottleneck(Module):
    expansion: int = 4

    def __init__(self, inplanes: int, planes: int, stride: int=1, downsample: Optional[Module]=None, groups: int=1, base_width: int=64, dilation: int=1, norm_layer: Optional[Callable[..., Module]]=None) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
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
            print('Hello World!')
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

class ResNet(Module):

    def __init__(self, block: Type[Bottleneck], layers: List[int], num_classes: int=1000, zero_init_residual: bool=False, groups: int=1, width_per_group: int=64, replace_stride_with_dilation: Optional[List[bool]]=None, norm_layer: Optional[Callable[..., Module]]=None) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(f'replace_stride_with_dilation should be None or a 3-element tuple, got {replace_stride_with_dilation}')
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
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and isinstance(m.bn3.weight, Tensor):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block: Type[Bottleneck], planes: int, blocks: int, stride: int=1, dilate: bool=False) -> nn.Sequential:
        if False:
            print('Hello World!')
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
            print('Hello World!')
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        if False:
            i = 10
            return i + 15
        return self._forward_impl(x)

class EncoderDeFMO(Module):

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        model = ResNet(Bottleneck, [3, 4, 6, 3])
        modelc1 = nn.Sequential(*list(model.children())[:3])
        modelc2 = nn.Sequential(*list(model.children())[4:8])
        modelc1[0] = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.net = nn.Sequential(modelc1, modelc2)

    def forward(self, input_data: Tensor) -> Tensor:
        if False:
            return 10
        return self.net(input_data)

class RenderingDeFMO(Module):

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self.tsr_steps: int = 24
        model = nn.Sequential(nn.Conv2d(2049, 1024, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), nn.ReLU(inplace=True), Bottleneck(1024, 256), nn.PixelShuffle(2), Bottleneck(256, 64), nn.PixelShuffle(2), Bottleneck(64, 16), nn.PixelShuffle(2), nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False), nn.PixelShuffle(2), nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(inplace=True), nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, bias=True))
        self.net = model
        self.times = torch.linspace(0, 1, self.tsr_steps)

    def forward(self, latent: Tensor) -> Tensor:
        if False:
            for i in range(10):
                print('nop')
        times = self.times.to(latent.device).unsqueeze(0).repeat(latent.shape[0], 1)
        renders = []
        for ki in range(times.shape[1]):
            t_tensor = times[list(range(times.shape[0])), ki].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, latent.shape[2], latent.shape[3])
            latenti = concatenate((t_tensor, latent), 1)
            result = self.net(latenti)
            renders.append(result)
        renders_stacked = stack(renders, 1).contiguous()
        renders_stacked[:, :, :4] = torch.sigmoid(renders_stacked[:, :, :4])
        return renders_stacked

class DeFMO(Module):
    """Module that disentangle a fast-moving object from the background and performs deblurring.

    This is based on the original code from paper "DeFMO: Deblurring and Shape Recovery
        of Fast Moving Objects". See :cite:`DeFMO2021` for more details.

    Args:
        pretrained: Download and set pretrained weights to the model. Default: false.
    Returns:
        Temporal super-resolution without background.
    Shape:
        - Input: (B, 6, H, W)
        - Output: (B, S, 4, H, W)

    Examples:
        >>> import kornia
        >>> input = torch.rand(2, 6, 240, 320)
        >>> defmo = kornia.feature.DeFMO()
        >>> tsr_nobgr = defmo(input) # 2x24x4x240x320
    """

    def __init__(self, pretrained: bool=False) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self.encoder = EncoderDeFMO()
        self.rendering = RenderingDeFMO()
        if pretrained:
            pretrained_dict = torch.hub.load_state_dict_from_url(urls['defmo_encoder'], map_location=map_location_to_cpu)
            self.encoder.load_state_dict(pretrained_dict, strict=True)
            pretrained_dict_ren = torch.hub.load_state_dict_from_url(urls['defmo_rendering'], map_location=map_location_to_cpu)
            self.rendering.load_state_dict(pretrained_dict_ren, strict=True)
        self.eval()

    def forward(self, input_data: Tensor) -> Tensor:
        if False:
            for i in range(10):
                print('nop')
        latent = self.encoder(input_data)
        x_out = self.rendering(latent)
        return x_out