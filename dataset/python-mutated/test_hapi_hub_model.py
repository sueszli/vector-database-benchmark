import paddle.nn.functional as F
from paddle import nn

class MM(nn.Layer):

    def __init__(self, out_channels):
        if False:
            while True:
                i = 10
        super().__init__()
        self.conv = nn.Conv2D(3, out_channels, 3, 2, 1)

    def forward(self, x):
        if False:
            while True:
                i = 10
        out = self.conv(x)
        out = F.relu(out)
        return out