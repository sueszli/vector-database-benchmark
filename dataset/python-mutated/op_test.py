import torch
import torch.nn as nn

class DummyNet(nn.Module):

    def __init__(self, num_classes=1000):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.features = nn.Sequential(nn.LeakyReLU(0.02), nn.BatchNorm2d(3), nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False))

    def forward(self, x):
        if False:
            print('Hello World!')
        output = self.features(x)
        return output.view(-1, 1).squeeze(1)

class ConcatNet(nn.Module):

    def forward(self, inputs):
        if False:
            i = 10
            return i + 15
        return torch.cat(inputs, 1)

class PermuteNet(nn.Module):

    def forward(self, input):
        if False:
            return 10
        return input.permute(2, 3, 0, 1)

class PReluNet(nn.Module):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.features = nn.Sequential(nn.PReLU(3))

    def forward(self, x):
        if False:
            return 10
        output = self.features(x)
        return output

class FakeQuantNet(nn.Module):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.fake_quant = torch.ao.quantization.FakeQuantize()
        self.fake_quant.disable_observer()

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        output = self.fake_quant(x)
        return output