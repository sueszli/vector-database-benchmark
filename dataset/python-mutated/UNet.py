import torch
import torch.nn as nn
import torch.nn.functional as F

class down(nn.Module):

    def __init__(self, inChannels, outChannels, filterSize):
        if False:
            i = 10
            return i + 15
        super(down, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2))
        self.conv2 = nn.Conv2d(outChannels, outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2))

    def forward(self, x):
        if False:
            while True:
                i = 10
        x = F.avg_pool2d(x, 2)
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        return x

class up(nn.Module):

    def __init__(self, inChannels, outChannels):
        if False:
            i = 10
            return i + 15
        super(up, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, outChannels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(2 * outChannels, outChannels, 3, stride=1, padding=1)

    def forward(self, x, skpCn):
        if False:
            i = 10
            return i + 15
        x = F.interpolate(x, size=[skpCn.size(2), skpCn.size(3)], mode='bilinear', align_corners=False)
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(torch.cat((x, skpCn), 1)), negative_slope=0.1)
        return x

class Small_UNet(nn.Module):

    def __init__(self, inChannels, outChannels):
        if False:
            return 10
        super(Small_UNet, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, 32, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.down1 = down(32, 64, 5)
        self.down2 = down(64, 128, 3)
        self.down3 = down(128, 128, 3)
        self.up1 = up(128, 128)
        self.up2 = up(128, 64)
        self.up3 = up(64, 32)
        self.conv3 = nn.Conv2d(32, outChannels, 3, stride=1, padding=1)

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        s1 = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        x = self.down3(s3)
        x = self.up1(x, s3)
        x = self.up2(x, s2)
        x1 = self.up3(x, s1)
        x = self.conv3(x1)
        return (x, x1)

class Small_UNet_Ds(nn.Module):

    def __init__(self, inChannels, outChannels):
        if False:
            i = 10
            return i + 15
        super(Small_UNet_Ds, self).__init__()
        self.conv1_1 = nn.Conv2d(inChannels, 32, 5, stride=1, padding=2)
        self.conv1_2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.down1 = down(32, 64, 5)
        self.down2 = down(64, 128, 3)
        self.down3 = down(128, 128, 3)
        self.up1 = up(128, 128)
        self.up2 = up(128, 64)
        self.up3 = up(64, 32)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, outChannels, 3, stride=1, padding=1)

    def forward(self, x):
        if False:
            print('Hello World!')
        x0 = F.leaky_relu(self.conv1_1(x), negative_slope=0.1)
        x0 = F.leaky_relu(self.conv1_2(x0), negative_slope=0.1)
        x = F.interpolate(x0, size=[x0.size(2) // 2, x0.size(3) // 2], mode='bilinear', align_corners=False)
        x = F.leaky_relu(self.conv2_1(x), negative_slope=0.1)
        s1 = F.leaky_relu(self.conv2_2(x), negative_slope=0.1)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        x = self.down3(s3)
        x = self.up1(x, s3)
        x = self.up2(x, s2)
        x1 = self.up3(x, s1)
        x1 = F.interpolate(x1, size=[x0.size(2), x0.size(3)], mode='bilinear', align_corners=False)
        x1 = F.leaky_relu(self.conv3(x1), negative_slope=0.1)
        x = self.conv4(x1)
        return (x, x1)