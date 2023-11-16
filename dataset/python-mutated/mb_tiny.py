import torch.nn as nn
import torch.nn.functional as F

class Mb_Tiny(nn.Module):

    def __init__(self, num_classes=2):
        if False:
            return 10
        super(Mb_Tiny, self).__init__()
        self.base_channel = 8 * 2

        def conv_bn(inp, oup, stride):
            if False:
                while True:
                    i = 10
            return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True))

        def conv_dw(inp, oup, stride):
            if False:
                print('Hello World!')
            return nn.Sequential(nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False), nn.BatchNorm2d(inp), nn.ReLU(inplace=True), nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True))
        self.model = nn.Sequential(conv_bn(3, self.base_channel, 2), conv_dw(self.base_channel, self.base_channel * 2, 1), conv_dw(self.base_channel * 2, self.base_channel * 2, 2), conv_dw(self.base_channel * 2, self.base_channel * 2, 1), conv_dw(self.base_channel * 2, self.base_channel * 4, 2), conv_dw(self.base_channel * 4, self.base_channel * 4, 1), conv_dw(self.base_channel * 4, self.base_channel * 4, 1), conv_dw(self.base_channel * 4, self.base_channel * 4, 1), conv_dw(self.base_channel * 4, self.base_channel * 8, 2), conv_dw(self.base_channel * 8, self.base_channel * 8, 1), conv_dw(self.base_channel * 8, self.base_channel * 8, 1), conv_dw(self.base_channel * 8, self.base_channel * 16, 2), conv_dw(self.base_channel * 16, self.base_channel * 16, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        x = self.model(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x