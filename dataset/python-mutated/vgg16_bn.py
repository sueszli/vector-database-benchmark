import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
from torchvision import models
from packaging import version

def init_weights(modules):
    if False:
        return 10
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

class vgg16_bn(torch.nn.Module):

    def __init__(self, pretrained=True, freeze=True):
        if False:
            return 10
        super(vgg16_bn, self).__init__()
        if version.parse(torchvision.__version__) >= version.parse('0.13'):
            vgg_pretrained_features = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT if pretrained else None).features
        else:
            models.vgg.model_urls['vgg16_bn'] = models.vgg.model_urls['vgg16_bn'].replace('https://', 'http://')
            vgg_pretrained_features = models.vgg16_bn(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(12):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 19):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(19, 29):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(29, 39):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        self.slice5 = torch.nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1), nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6), nn.Conv2d(1024, 1024, kernel_size=1))
        if not pretrained:
            init_weights(self.slice1.modules())
            init_weights(self.slice2.modules())
            init_weights(self.slice3.modules())
            init_weights(self.slice4.modules())
        init_weights(self.slice5.modules())
        if freeze:
            for param in self.slice1.parameters():
                param.requires_grad = False

    def forward(self, X):
        if False:
            while True:
                i = 10
        h = self.slice1(X)
        h_relu2_2 = h
        h = self.slice2(h)
        h_relu3_2 = h
        h = self.slice3(h)
        h_relu4_3 = h
        h = self.slice4(h)
        h_relu5_3 = h
        h = self.slice5(h)
        h_fc7 = h
        return (h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)