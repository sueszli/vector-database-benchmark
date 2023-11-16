import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import torchvision
from torchvision import models
import pytorch_lightning as pl

class LeakySoftplus(nn.Module):

    def __init__(self, negative_slope: float=0.01):
        if False:
            print('Hello World!')
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input):
        if False:
            print('Hello World!')
        return F.softplus(input) + F.logsigmoid(input) * self.negative_slope
grelu = nn.LeakyReLU(0.2)

class Generator(pl.LightningModule):

    def __init__(self, norm_layer='batch_norm', use_bias=False, resnet_blocks=7, tanh=True, filters=[32, 64, 128, 128, 128, 64], input_channels=3, output_channels=3, append_smoothers=False):
        if False:
            i = 10
            return i + 15
        super().__init__()
        assert norm_layer in [None, 'batch_norm', 'instance_norm'], "norm_layer should be None, 'batch_norm' or 'instance_norm', not {}".format(norm_layer)
        self.norm_layer = None
        if norm_layer == 'batch_norm':
            self.norm_layer = nn.BatchNorm2d
        elif norm_layer == 'instance_norm':
            self.norm_layer = nn.InstanceNorm2d
        self.use_bias = use_bias
        self.resnet_blocks = resnet_blocks
        self.append_smoothers = append_smoothers
        stride1 = 2
        stride2 = 2
        self.conv0 = self.relu_layer(in_filters=input_channels, out_filters=filters[0], kernel_size=7, stride=1, padding=3, bias=self.use_bias, norm_layer=self.norm_layer, nonlinearity=grelu)
        self.conv1 = self.relu_layer(in_filters=filters[0], out_filters=filters[1], kernel_size=3, stride=stride1, padding=1, bias=self.use_bias, norm_layer=self.norm_layer, nonlinearity=grelu)
        self.conv2 = self.relu_layer(in_filters=filters[1], out_filters=filters[2], kernel_size=3, stride=stride2, padding=1, bias=self.use_bias, norm_layer=self.norm_layer, nonlinearity=grelu)
        self.resnets = nn.ModuleList()
        for i in range(self.resnet_blocks):
            self.resnets.append(self.resnet_block(in_filters=filters[2], out_filters=filters[2], kernel_size=3, stride=1, padding=1, bias=self.use_bias, norm_layer=self.norm_layer, nonlinearity=grelu))
        self.upconv2 = self.upconv_layer_upsample_and_conv(in_filters=filters[3] + filters[2], out_filters=filters[4], scale_factor=stride2, kernel_size=3, stride=1, padding=1, bias=self.use_bias, norm_layer=self.norm_layer, nonlinearity=grelu)
        self.upconv1 = self.upconv_layer_upsample_and_conv(in_filters=filters[4] + filters[1], out_filters=filters[4], scale_factor=stride1, kernel_size=3, stride=1, padding=1, bias=self.use_bias, norm_layer=self.norm_layer, nonlinearity=grelu)
        self.conv_11 = nn.Sequential(nn.Conv2d(in_channels=filters[0] + filters[4] + input_channels, out_channels=filters[5], kernel_size=7, stride=1, padding=3, bias=self.use_bias, padding_mode='zeros'), grelu)
        if self.append_smoothers:
            self.conv_11_a = nn.Sequential(nn.Conv2d(filters[5], filters[5], kernel_size=3, bias=self.use_bias, padding=1, padding_mode='zeros'), grelu, nn.BatchNorm2d(num_features=filters[5]), nn.Conv2d(filters[5], filters[5], kernel_size=3, bias=self.use_bias, padding=1, padding_mode='zeros'), grelu)
        if tanh:
            self.conv_12 = nn.Sequential(nn.Conv2d(filters[5], output_channels, kernel_size=1, stride=1, padding=0, bias=True, padding_mode='zeros'), nn.Sigmoid())
        else:
            self.conv_12 = nn.Conv2d(filters[5], output_channels, kernel_size=1, stride=1, padding=0, bias=True, padding_mode='zeros')

    def log_tensors(self, logger, tag, img_tensor):
        if False:
            print('Hello World!')
        logger.experiment.add_images(tag, img_tensor)

    def forward(self, input):
        if False:
            return 10
        output_d0 = self.conv0(input)
        output_d1 = self.conv1(output_d0)
        output_d2 = self.conv2(output_d1)
        output = output_d2
        for layer in self.resnets:
            output = layer(output) + output
        output_u2 = self.upconv2(torch.cat((output, output_d2), dim=1))
        output_u1 = self.upconv1(torch.cat((output_u2, output_d1), dim=1))
        output = torch.cat((output_u1, output_d0, input), dim=1)
        output_11 = self.conv_11(output)
        if self.append_smoothers:
            output_11_a = self.conv_11_a(output_11)
        else:
            output_11_a = output_11
        output_12 = self.conv_12(output_11_a)
        output = output_12
        return output

    def relu_layer(self, in_filters, out_filters, kernel_size, stride, padding, bias, norm_layer, nonlinearity):
        if False:
            while True:
                i = 10
        out = nn.Sequential()
        out.add_module('conv', nn.Conv2d(in_channels=in_filters, out_channels=out_filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, padding_mode='zeros'))
        if norm_layer:
            out.add_module('normalization', norm_layer(num_features=out_filters))
        if nonlinearity:
            out.add_module('nonlinearity', nonlinearity)
        return out

    def resnet_block(self, in_filters, out_filters, kernel_size, stride, padding, bias, norm_layer, nonlinearity):
        if False:
            print('Hello World!')
        out = nn.Sequential()
        if nonlinearity:
            out.add_module('nonlinearity_0', nonlinearity)
        out.add_module('conv_0', nn.Conv2d(in_channels=in_filters, out_channels=out_filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, padding_mode='zeros'))
        if norm_layer:
            out.add_module('normalization', norm_layer(num_features=out_filters))
        if nonlinearity:
            out.add_module('nonlinearity_1', nonlinearity)
        out.add_module('conv_1', nn.Conv2d(in_channels=in_filters, out_channels=out_filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, padding_mode='zeros'))
        return out

    def upconv_layer_upsample_and_conv(self, in_filters, out_filters, scale_factor, kernel_size, stride, padding, bias, norm_layer, nonlinearity):
        if False:
            for i in range(10):
                print('nop')
        parts = [nn.Upsample(scale_factor=scale_factor), nn.Conv2d(in_filters, out_filters, kernel_size, stride, padding=padding, bias=False, padding_mode='zeros')]
        if norm_layer:
            parts.append(norm_layer(num_features=out_filters))
        if nonlinearity:
            parts.append(nonlinearity)
        return nn.Sequential(*parts)
relu = grelu
relu = nn.LeakyReLU(0.2)

class Discriminator(nn.Module):

    def __init__(self, num_filters=12, input_channels=3, n_layers=2, norm_layer='instance_norm', use_bias=True):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.num_filters = num_filters
        self.input_channels = input_channels
        self.use_bias = use_bias
        if norm_layer == 'batch_norm':
            self.norm_layer = nn.BatchNorm2d
        else:
            self.norm_layer = nn.InstanceNorm2d
        self.net = self.make_net(n_layers, self.input_channels, 1, 4, 2, self.use_bias)

    def make_net(self, n, flt_in, flt_out=1, k=4, stride=2, bias=True):
        if False:
            print('Hello World!')
        padding = 1
        model = nn.Sequential()
        model.add_module('conv0', self.make_block(flt_in, self.num_filters, k, stride, padding, bias, None, relu))
        (flt_mult, flt_mult_prev) = (1, 1)
        for l in range(1, n):
            flt_mult_prev = flt_mult
            flt_mult = min(2 ** l, 8)
            model.add_module('conv_%d' % l, self.make_block(self.num_filters * flt_mult_prev, self.num_filters * flt_mult, k, stride, padding, bias, self.norm_layer, relu))
        flt_mult_prev = flt_mult
        flt_mult = min(2 ** n, 8)
        model.add_module('conv_%d' % n, self.make_block(self.num_filters * flt_mult_prev, self.num_filters * flt_mult, k, 1, padding, bias, self.norm_layer, relu))
        model.add_module('conv_out', self.make_block(self.num_filters * flt_mult, 1, k, 1, padding, bias, None, None))
        return model

    def make_block(self, flt_in, flt_out, k, stride, padding, bias, norm, relu):
        if False:
            for i in range(10):
                print('nop')
        m = nn.Sequential()
        m.add_module('conv', nn.Conv2d(flt_in, flt_out, k, stride=stride, padding=padding, bias=bias, padding_mode='zeros'))
        if norm is not None:
            m.add_module('norm', norm(flt_out))
        if relu is not None:
            m.add_module('relu', relu)
        return m

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        output = self.net(x)
        return output

class PerceptualVGG19(nn.Module):

    def __init__(self, feature_layers=[0, 3, 5], use_normalization=False):
        if False:
            print('Hello World!')
        super().__init__()
        model = models.squeezenet1_1(pretrained=True)
        model.float()
        model.eval()
        self.model = model
        self.feature_layers = feature_layers
        self.mean = torch.FloatTensor([0.485, 0.456, 0.406])
        self.mean_tensor = None
        self.std = torch.FloatTensor([0.229, 0.224, 0.225])
        self.std_tensor = None
        self.use_normalization = use_normalization
        for param in self.parameters():
            param.requires_grad = False

    def normalize(self, x):
        if False:
            print('Hello World!')
        if not self.use_normalization:
            return x
        if self.mean_tensor is None:
            self.mean_tensor = Variable(self.mean.view(1, 3, 1, 1).expand(x.shape), requires_grad=False)
            self.std_tensor = Variable(self.std.view(1, 3, 1, 1).expand(x.shape), requires_grad=False)
        x = (x + 1) / 2
        return (x - self.mean_tensor) / self.std_tensor

    def run(self, x):
        if False:
            return 10
        features = []
        h = x
        for f in range(max(self.feature_layers) + 1):
            h = self.model.features[f](h)
            if f in self.feature_layers:
                not_normed_features = h.clone().view(h.size(0), -1)
                features.append(not_normed_features)
        return torch.cat(features, dim=1)

    def forward(self, x):
        if False:
            print('Hello World!')
        h = self.normalize(x)
        return self.run(h)