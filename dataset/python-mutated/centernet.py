from torchvision.models import mobilenet, resnet
import torch
import torch.nn as nn
import torch.nn.functional as F
_backbones = {'resnet18': (resnet.resnet18, 512), 'resnet34': (resnet.resnet34, 512), 'resnet50': (resnet.resnet50, 2048), 'resnet101': (resnet.resnet101, 2048), 'resnet152': (resnet.resnet152, 2048), 'mobilenet_v2': (mobilenet.mobilenet_v2, 1280)}

class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        if False:
            return 10
        x = self.conv(x)
        return x

class Interpolate(nn.Module):

    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None):
        if False:
            while True:
                i = 10
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor

    def forward(self, inputs):
        if False:
            while True:
                i = 10
        return F.interpolate(inputs, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners, recompute_scale_factor=self.recompute_scale_factor)

class UpDoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mode=None):
        if False:
            return 10
        super().__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.mode = mode
        if mode is None:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        else:
            align_corners = None if mode == 'nearest' else True
            self.up = Interpolate(scale_factor=2, mode=mode, align_corners=align_corners)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2=None):
        if False:
            print('Hello World!')
        x1 = self.up(x1)
        if x2 is not None:
            x = torch.cat([x2, x1], dim=1)
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        else:
            x = x1
        x = self.conv(x)
        return x

class CenterNet(nn.Module):

    def __init__(self, num_classes=1, backbone='resnet18', upsample_mode='nearest'):
        if False:
            while True:
                i = 10
        super().__init__()
        basemodel = _backbones[backbone][0](pretrained=True)
        if backbone == 'mobilenet_v2':
            layers = list(basemodel.children())[:-1]
        else:
            layers = list(basemodel.children())[:-2]
        basemodel = nn.Sequential(*layers)
        self.base_model = basemodel
        self.upsample_mode = upsample_mode
        num_ch = _backbones[backbone][1]
        self.up1 = UpDoubleConv(num_ch, 512, upsample_mode)
        self.up2 = UpDoubleConv(512, 256, upsample_mode)
        self.up3 = UpDoubleConv(256, 256, upsample_mode)
        self.out_classification = nn.Conv2d(256, num_classes, 1)
        self.out_residue = nn.Conv2d(256, 2, 1)

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        x = self.base_model(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        c = self.out_classification(x)
        r = self.out_residue(x)
        return (c, r)