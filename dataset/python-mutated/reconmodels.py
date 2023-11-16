import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    if False:
        print('Hello World!')
    '3x3 convolution with padding'
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    if False:
        return 10
    '1x1 convolution'
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ChannelAttention(nn.Module):

    def __init__(self, in_planes, ratio=16):
        if False:
            while True:
                i = 10
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if False:
            while True:
                i = 10
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        if False:
            for i in range(10):
                print('nop')
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if False:
            while True:
                i = 10
        avg_out = torch.mean(x, dim=1, keepdim=True)
        (max_out, _) = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        if False:
            return 10
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

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
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

class BottleneckWithCBAM(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        if False:
            return 10
        super(BottleneckWithCBAM, self).__init__()
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
        self.channelatt = ChannelAttention(planes * self.expansion)
        self.spaciaoatt = SpatialAttention()

    def forward(self, x):
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
        out = self.channelatt(out) * out
        out = self.spaciaoatt(out) * out
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=2048, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        if False:
            while True:
                i = 10
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
                if isinstance(m, BottleneckWithCBAM):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        if False:
            i = 10
            return i + 15
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

    def _forward(self, x):
        if False:
            print('Hello World!')
        if x.is_cuda:
            x = x.type(torch.FloatTensor).cuda()
        else:
            x = x.type(torch.FloatTensor)
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
    forward = _forward

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    if False:
        while True:
            i = 10
    model = ResNet(block, layers, **kwargs)
    return model

def resnet50withcbam(pretrained=False, progress=True, **kwargs):
    if False:
        i = 10
        return i + 15
    'ResNet-50 model from\n    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_\n\n    Args:\n        pretrained (bool): If True, returns a model pre-trained on ImageNet\n        progress (bool): If True, displays a progress bar of the download to stderr\n    '
    return _resnet('resnet50', BottleneckWithCBAM, [3, 4, 6, 3], pretrained, progress, **kwargs)

class RecurrentModel(nn.Module):

    def __init__(self, combine_model, batch_size=4, seq_length=15, hidden_units=256, number_of_outputs=8, input_size=3328, number_of_layers=2):
        if False:
            while True:
                i = 10
        super(RecurrentModel, self).__init__()
        self.combine_model = combine_model
        self.input_size = input_size
        self.number_of_outputs = number_of_outputs
        self.hidden_units = hidden_units
        self.number_of_layers = number_of_layers
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.lstm = nn.LSTM(self.input_size, self.hidden_units, self.number_of_layers)
        self.linear = nn.Linear(self.hidden_units * self.seq_length, self.number_of_outputs)
        self.optimizer = torch.optim.Adam([{'params': self.lstm.parameters()}, {'params': self.linear.parameters()}, {'params': self.combine_model.audio_model.parameters(), 'lr': 0.0001}, {'params': self.combine_model.video_model.parameters(), 'lr': 0.0001}], lr=1e-05)

    def forward(self, audio_input, frame_input):
        if False:
            i = 10
            return i + 15
        if torch.cuda.is_available():
            features = torch.tensor([]).cuda()
        else:
            features = torch.tensor([])
        count = 0
        for i in range(audio_input.shape[0]):
            count += 1
            temp = self.combine_model.forward(audio_input[i], frame_input[i])
            features = torch.cat((features, temp), 0)
        features = features.view(self.seq_length, -1, self.input_size)
        (features, _) = self.lstm.forward(features)
        features = features.permute(1, 0, 2).reshape(-1, self.seq_length * self.hidden_units)
        return self.linear(features)

class AudioModel(nn.Module):

    def __init__(self, batch_size=4, seq_length=16000 // 30, num_features=1280, conv_filters=15):
        if False:
            for i in range(10):
                print('nop')
        super(AudioModel, self).__init__()
        self.conv_filters = conv_filters
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_features = num_features
        self.conv1 = nn.Conv2d(1, conv_filters, (1, 80))
        self.max_pool = nn.MaxPool2d((1, 2), stride=(1, 2))
        self.max_pool2 = nn.MaxPool2d((1, 1, 10, 1), stride=(1, 1, 10, 1))
        self.conv2 = nn.Conv2d(40, conv_filters, (1, 40))
        self.linear_out = nn.Linear(3958, self.num_features)

    def forward(self, audio_input):
        if False:
            print('Hello World!')
        if audio_input.is_cuda:
            audio_input = audio_input.type(torch.FloatTensor).cuda()
        else:
            audio_input = audio_input.type(torch.FloatTensor)
        audio_input = audio_input.view(1, 1, 1, -1)
        audio_input = F.dropout(audio_input)
        audio_input = self.conv1(audio_input)
        audio_input = self.max_pool(audio_input)
        return torch.squeeze(self.linear_out(audio_input))

class CombinedModel(nn.Module):

    def __init__(self, audio_model, video_model):
        if False:
            while True:
                i = 10
        super(CombinedModel, self).__init__()
        self.audio_model = audio_model
        self.video_model = video_model

    def forward(self, audio_frames, video_frames):
        if False:
            return 10
        video_features = self.video_model.forward(video_frames)
        audio_features = self.audio_model.forward(audio_frames)
        return torch.cat((audio_features, video_features), 1)