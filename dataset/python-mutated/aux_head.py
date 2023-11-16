import torch.nn as nn

class DistillHeadCIFAR(nn.Module):

    def __init__(self, C, size, num_classes, bn_affine=False):
        if False:
            for i in range(10):
                print('nop')
        'assuming input size 8x8 or 16x16'
        super(DistillHeadCIFAR, self).__init__()
        self.features = nn.Sequential(nn.ReLU(), nn.AvgPool2d(size, stride=2, padding=0, count_include_pad=False), nn.Conv2d(C, 128, 1, bias=False), nn.BatchNorm2d(128, affine=bn_affine), nn.ReLU(), nn.Conv2d(128, 768, 2, bias=False), nn.BatchNorm2d(768, affine=bn_affine), nn.ReLU())
        self.classifier = nn.Linear(768, num_classes)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x

class DistillHeadImagenet(nn.Module):

    def __init__(self, C, size, num_classes, bn_affine=False):
        if False:
            for i in range(10):
                print('nop')
        'assuming input size 7x7 or 14x14'
        super(DistillHeadImagenet, self).__init__()
        self.features = nn.Sequential(nn.ReLU(), nn.AvgPool2d(size, stride=2, padding=0, count_include_pad=False), nn.Conv2d(C, 128, 1, bias=False), nn.BatchNorm2d(128, affine=bn_affine), nn.ReLU(), nn.Conv2d(128, 768, 2, bias=False), nn.BatchNorm2d(768, affine=bn_affine), nn.ReLU())
        self.classifier = nn.Linear(768, num_classes)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        if False:
            return 10
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x

class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, C, size=5, num_classes=10):
        if False:
            while True:
                i = 10
        'assuming input size 8x8'
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(nn.ReLU(inplace=True), nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), nn.Conv2d(C, 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, 768, 2, bias=False), nn.BatchNorm2d(768), nn.ReLU(inplace=True))
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        if False:
            return 10
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x

class AuxiliaryHeadImageNet(nn.Module):

    def __init__(self, C, size=5, num_classes=1000):
        if False:
            while True:
                i = 10
        'assuming input size 7x7'
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(nn.ReLU(inplace=True), nn.AvgPool2d(size, stride=2, padding=0, count_include_pad=False), nn.Conv2d(C, 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, 768, 2, bias=False), nn.ReLU(inplace=True))
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        if False:
            while True:
                i = 10
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x