import torch
import torch.nn as nn
import torch.nn.functional as F

class cSE(nn.Module):
    """
    The channel-wise SE (Squeeze and Excitation) block from the
    `Squeeze-and-Excitation Networks`__ paper.

    Adapted from
    https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65939
    and
    https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178

    Shape:

    - Input: (batch, channels, height, width)
    - Output: (batch, channels, height, width) (same shape as input)

    __ https://arxiv.org/abs/1709.01507
    """

    def __init__(self, in_channels: int, r: int=16):
        if False:
            return 10
        '\n        Args:\n            in_channels: The number of channels\n                in the feature map of the input.\n            r: The reduction ratio of the intermediate channels.\n                Default: 16.\n        '
        super().__init__()
        self.linear1 = nn.Linear(in_channels, in_channels // r)
        self.linear2 = nn.Linear(in_channels // r, in_channels)

    def forward(self, x: torch.Tensor):
        if False:
            return 10
        'Forward call.'
        input_x = x
        x = x.view(*x.shape[:-2], -1).mean(-1)
        x = F.relu(self.linear1(x), inplace=True)
        x = self.linear2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.sigmoid(x)
        x = torch.mul(input_x, x)
        return x

class sSE(nn.Module):
    """
    The sSE (Channel Squeeze and Spatial Excitation) block from the
    `Concurrent Spatial and Channel ‘Squeeze & Excitation’
    in Fully Convolutional Networks`__ paper.

    Adapted from
    https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178

    Shape:

    - Input: (batch, channels, height, width)
    - Output: (batch, channels, height, width) (same shape as input)

    __ https://arxiv.org/abs/1803.02579
    """

    def __init__(self, in_channels: int):
        if False:
            while True:
                i = 10
        '\n        Args:\n            in_channels: The number of channels\n                in the feature map of the input.\n        '
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor):
        if False:
            while True:
                i = 10
        'Forward call.'
        input_x = x
        x = self.conv(x)
        x = torch.sigmoid(x)
        x = torch.mul(input_x, x)
        return x

class scSE(nn.Module):
    """
    The scSE (Concurrent Spatial and Channel Squeeze and Channel Excitation)
    block from the `Concurrent Spatial and Channel ‘Squeeze & Excitation’
    in Fully Convolutional Networks`__ paper.

    Adapted from
    https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178

    Shape:

    - Input: (batch, channels, height, width)
    - Output: (batch, channels, height, width) (same shape as input)

    __ https://arxiv.org/abs/1803.02579
    """

    def __init__(self, in_channels: int, r: int=16):
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            in_channels: The number of channels\n                in the feature map of the input.\n            r: The reduction ratio of the intermediate channels.\n                Default: 16.\n        '
        super().__init__()
        self.cse_block = cSE(in_channels, r)
        self.sse_block = sSE(in_channels)

    def forward(self, x: torch.Tensor):
        if False:
            while True:
                i = 10
        'Forward call.'
        cse = self.cse_block(x)
        sse = self.sse_block(x)
        x = torch.add(cse, sse)
        return x
__all__ = ['sSE', 'scSE', 'cSE']