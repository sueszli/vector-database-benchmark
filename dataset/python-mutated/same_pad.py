from torch import nn

class SamePad(nn.Module):

    def __init__(self, kernel_size, causal=False):
        if False:
            while True:
                i = 10
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if False:
            print('Hello World!')
        if self.remove > 0:
            x = x[:, :, :-self.remove]
        return x

class SamePad2d(nn.Module):

    def __init__(self, kernel_size):
        if False:
            while True:
                i = 10
        super().__init__()
        self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if False:
            while True:
                i = 10
        assert len(x.size()) == 4
        if self.remove > 0:
            x = x[:, :, :-self.remove, :-self.remove]
        return x