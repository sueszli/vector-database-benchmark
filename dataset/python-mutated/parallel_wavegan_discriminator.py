import math
import torch
from torch import nn
from torch.nn.utils.parametrize import remove_parametrizations
from TTS.vocoder.layers.parallel_wavegan import ResidualBlock

class ParallelWaveganDiscriminator(nn.Module):
    """PWGAN discriminator as in https://arxiv.org/abs/1910.11480.
    It classifies each audio window real/fake and returns a sequence
    of predictions.
        It is a stack of convolutional blocks with dilation.
    """

    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, num_layers=10, conv_channels=64, dilation_factor=1, nonlinear_activation='LeakyReLU', nonlinear_activation_params={'negative_slope': 0.2}, bias=True):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, ' [!] does not support even number kernel size.'
        assert dilation_factor > 0, ' [!] dilation factor must be > 0.'
        self.conv_layers = nn.ModuleList()
        conv_in_channels = in_channels
        for i in range(num_layers - 1):
            if i == 0:
                dilation = 1
            else:
                dilation = i if dilation_factor == 1 else dilation_factor ** i
                conv_in_channels = conv_channels
            padding = (kernel_size - 1) // 2 * dilation
            conv_layer = [nn.Conv1d(conv_in_channels, conv_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=bias), getattr(nn, nonlinear_activation)(inplace=True, **nonlinear_activation_params)]
            self.conv_layers += conv_layer
        padding = (kernel_size - 1) // 2
        last_conv_layer = nn.Conv1d(conv_in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv_layers += [last_conv_layer]
        self.apply_weight_norm()

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        '\n            x : (B, 1, T).\n        Returns:\n            Tensor: (B, 1, T)\n        '
        for f in self.conv_layers:
            x = f(x)
        return x

    def apply_weight_norm(self):
        if False:
            i = 10
            return i + 15

        def _apply_weight_norm(m):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d)):
                torch.nn.utils.parametrizations.weight_norm(m)
        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        if False:
            return 10

        def _remove_weight_norm(m):
            if False:
                for i in range(10):
                    print('nop')
            try:
                remove_parametrizations(m, 'weight')
            except ValueError:
                return
        self.apply(_remove_weight_norm)

class ResidualParallelWaveganDiscriminator(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, num_layers=30, stacks=3, res_channels=64, gate_channels=128, skip_channels=64, dropout=0.0, bias=True, nonlinear_activation='LeakyReLU', nonlinear_activation_params={'negative_slope': 0.2}):
        if False:
            return 10
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, 'Not support even number kernel size.'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.stacks = stacks
        self.kernel_size = kernel_size
        self.res_factor = math.sqrt(1.0 / num_layers)
        assert num_layers % stacks == 0
        layers_per_stack = num_layers // stacks
        self.first_conv = nn.Sequential(nn.Conv1d(in_channels, res_channels, kernel_size=1, padding=0, dilation=1, bias=True), getattr(nn, nonlinear_activation)(inplace=True, **nonlinear_activation_params))
        self.conv_layers = nn.ModuleList()
        for layer in range(num_layers):
            dilation = 2 ** (layer % layers_per_stack)
            conv = ResidualBlock(kernel_size=kernel_size, res_channels=res_channels, gate_channels=gate_channels, skip_channels=skip_channels, aux_channels=-1, dilation=dilation, dropout=dropout, bias=bias, use_causal_conv=False)
            self.conv_layers += [conv]
        self.last_conv_layers = nn.ModuleList([getattr(nn, nonlinear_activation)(inplace=True, **nonlinear_activation_params), nn.Conv1d(skip_channels, skip_channels, kernel_size=1, padding=0, dilation=1, bias=True), getattr(nn, nonlinear_activation)(inplace=True, **nonlinear_activation_params), nn.Conv1d(skip_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=True)])
        self.apply_weight_norm()

    def forward(self, x):
        if False:
            print('Hello World!')
        '\n        x: (B, 1, T).\n        '
        x = self.first_conv(x)
        skips = 0
        for f in self.conv_layers:
            (x, h) = f(x, None)
            skips += h
        skips *= self.res_factor
        x = skips
        for f in self.last_conv_layers:
            x = f(x)
        return x

    def apply_weight_norm(self):
        if False:
            print('Hello World!')

        def _apply_weight_norm(m):
            if False:
                print('Hello World!')
            if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d)):
                torch.nn.utils.parametrizations.weight_norm(m)
        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        if False:
            while True:
                i = 10

        def _remove_weight_norm(m):
            if False:
                i = 10
                return i + 15
            try:
                print(f'Weight norm is removed from {m}.')
                remove_parametrizations(m, 'weight')
            except ValueError:
                return
        self.apply(_remove_weight_norm)