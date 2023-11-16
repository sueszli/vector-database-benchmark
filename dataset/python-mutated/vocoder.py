from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
MAX_WAV_VALUE = 32768.0

class KernelPredictor(torch.nn.Module):
    """Kernel predictor for the location-variable convolutions"""

    def __init__(self, cond_channels, conv_in_channels, conv_out_channels, conv_layers, conv_kernel_size=3, kpnet_hidden_channels=64, kpnet_conv_size=3, kpnet_dropout=0.0, kpnet_nonlinear_activation='LeakyReLU', kpnet_nonlinear_activation_params={'negative_slope': 0.1}):
        if False:
            while True:
                i = 10
        '\n        Args:\n            cond_channels (int): number of channel for the conditioning sequence,\n            conv_in_channels (int): number of channel for the input sequence,\n            conv_out_channels (int): number of channel for the output sequence,\n            conv_layers (int): number of layers\n        '
        super().__init__()
        self.conv_in_channels = conv_in_channels
        self.conv_out_channels = conv_out_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_layers = conv_layers
        kpnet_kernel_channels = conv_in_channels * conv_out_channels * conv_kernel_size * conv_layers
        kpnet_bias_channels = conv_out_channels * conv_layers
        self.input_conv = nn.Sequential(nn.utils.parametrizations.weight_norm(nn.Conv1d(cond_channels, kpnet_hidden_channels, 5, padding=2, bias=True)), getattr(nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params))
        self.residual_convs = nn.ModuleList()
        padding = (kpnet_conv_size - 1) // 2
        for _ in range(3):
            self.residual_convs.append(nn.Sequential(nn.Dropout(kpnet_dropout), nn.utils.parametrizations.weight_norm(nn.Conv1d(kpnet_hidden_channels, kpnet_hidden_channels, kpnet_conv_size, padding=padding, bias=True)), getattr(nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params), nn.utils.parametrizations.weight_norm(nn.Conv1d(kpnet_hidden_channels, kpnet_hidden_channels, kpnet_conv_size, padding=padding, bias=True)), getattr(nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params)))
        self.kernel_conv = nn.utils.parametrizations.weight_norm(nn.Conv1d(kpnet_hidden_channels, kpnet_kernel_channels, kpnet_conv_size, padding=padding, bias=True))
        self.bias_conv = nn.utils.parametrizations.weight_norm(nn.Conv1d(kpnet_hidden_channels, kpnet_bias_channels, kpnet_conv_size, padding=padding, bias=True))

    def forward(self, c):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            c (Tensor): the conditioning sequence (batch, cond_channels, cond_length)\n        '
        (batch, _, cond_length) = c.shape
        c = self.input_conv(c)
        for residual_conv in self.residual_convs:
            residual_conv.to(c.device)
            c = c + residual_conv(c)
        k = self.kernel_conv(c)
        b = self.bias_conv(c)
        kernels = k.contiguous().view(batch, self.conv_layers, self.conv_in_channels, self.conv_out_channels, self.conv_kernel_size, cond_length)
        bias = b.contiguous().view(batch, self.conv_layers, self.conv_out_channels, cond_length)
        return (kernels, bias)

    def remove_weight_norm(self):
        if False:
            while True:
                i = 10
        parametrize.remove_parametrizations(self.input_conv[0], 'weight')
        parametrize.remove_parametrizations(self.kernel_conv, 'weight')
        parametrize.remove_parametrizations(self.bias_conv)
        for block in self.residual_convs:
            parametrize.remove_parametrizations(block[1], 'weight')
            parametrize.remove_parametrizations(block[3], 'weight')

class LVCBlock(torch.nn.Module):
    """the location-variable convolutions"""

    def __init__(self, in_channels, cond_channels, stride, dilations=[1, 3, 9, 27], lReLU_slope=0.2, conv_kernel_size=3, cond_hop_length=256, kpnet_hidden_channels=64, kpnet_conv_size=3, kpnet_dropout=0.0):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.cond_hop_length = cond_hop_length
        self.conv_layers = len(dilations)
        self.conv_kernel_size = conv_kernel_size
        self.kernel_predictor = KernelPredictor(cond_channels=cond_channels, conv_in_channels=in_channels, conv_out_channels=2 * in_channels, conv_layers=len(dilations), conv_kernel_size=conv_kernel_size, kpnet_hidden_channels=kpnet_hidden_channels, kpnet_conv_size=kpnet_conv_size, kpnet_dropout=kpnet_dropout, kpnet_nonlinear_activation_params={'negative_slope': lReLU_slope})
        self.convt_pre = nn.Sequential(nn.LeakyReLU(lReLU_slope), nn.utils.parametrizations.weight_norm(nn.ConvTranspose1d(in_channels, in_channels, 2 * stride, stride=stride, padding=stride // 2 + stride % 2, output_padding=stride % 2)))
        self.conv_blocks = nn.ModuleList()
        for dilation in dilations:
            self.conv_blocks.append(nn.Sequential(nn.LeakyReLU(lReLU_slope), nn.utils.parametrizations.weight_norm(nn.Conv1d(in_channels, in_channels, conv_kernel_size, padding=dilation * (conv_kernel_size - 1) // 2, dilation=dilation)), nn.LeakyReLU(lReLU_slope)))

    def forward(self, x, c):
        if False:
            i = 10
            return i + 15
        'forward propagation of the location-variable convolutions.\n        Args:\n            x (Tensor): the input sequence (batch, in_channels, in_length)\n            c (Tensor): the conditioning sequence (batch, cond_channels, cond_length)\n\n        Returns:\n            Tensor: the output sequence (batch, in_channels, in_length)\n        '
        (_, in_channels, _) = x.shape
        x = self.convt_pre(x)
        (kernels, bias) = self.kernel_predictor(c)
        for (i, conv) in enumerate(self.conv_blocks):
            output = conv(x)
            k = kernels[:, i, :, :, :, :]
            b = bias[:, i, :, :]
            output = self.location_variable_convolution(output, k, b, hop_size=self.cond_hop_length)
            x = x + torch.sigmoid(output[:, :in_channels, :]) * torch.tanh(output[:, in_channels:, :])
        return x

    def location_variable_convolution(self, x, kernel, bias, dilation=1, hop_size=256):
        if False:
            for i in range(10):
                print('nop')
        'perform location-variable convolution operation on the input sequence (x) using the local convolution kernl.\n        Time: 414 μs ± 309 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each), test on NVIDIA V100.\n        Args:\n            x (Tensor): the input sequence (batch, in_channels, in_length).\n            kernel (Tensor): the local convolution kernel (batch, in_channel, out_channels, kernel_size, kernel_length)\n            bias (Tensor): the bias for the local convolution (batch, out_channels, kernel_length)\n            dilation (int): the dilation of convolution.\n            hop_size (int): the hop_size of the conditioning sequence.\n        Returns:\n            (Tensor): the output sequence after performing local convolution. (batch, out_channels, in_length).\n        '
        (batch, _, in_length) = x.shape
        (batch, _, out_channels, kernel_size, kernel_length) = kernel.shape
        assert in_length == kernel_length * hop_size, 'length of (x, kernel) is not matched'
        padding = dilation * int((kernel_size - 1) / 2)
        x = F.pad(x, (padding, padding), 'constant', 0)
        x = x.unfold(2, hop_size + 2 * padding, hop_size)
        if hop_size < dilation:
            x = F.pad(x, (0, dilation), 'constant', 0)
        x = x.unfold(3, dilation, dilation)
        x = x[:, :, :, :, :hop_size]
        x = x.transpose(3, 4)
        x = x.unfold(4, kernel_size, 1)
        o = torch.einsum('bildsk,biokl->bolsd', x, kernel)
        o = o.to(memory_format=torch.channels_last_3d)
        bias = bias.unsqueeze(-1).unsqueeze(-1).to(memory_format=torch.channels_last_3d)
        o = o + bias
        o = o.contiguous().view(batch, out_channels, -1)
        return o

    def remove_weight_norm(self):
        if False:
            return 10
        self.kernel_predictor.remove_weight_norm()
        parametrize.remove_parametrizations(self.convt_pre[1], 'weight')
        for block in self.conv_blocks:
            parametrize.remove_parametrizations(block[1], 'weight')

class UnivNetGenerator(nn.Module):
    """
    UnivNet Generator

    Originally from https://github.com/mindslab-ai/univnet/blob/master/model/generator.py.
    """

    def __init__(self, noise_dim=64, channel_size=32, dilations=[1, 3, 9, 27], strides=[8, 8, 4], lReLU_slope=0.2, kpnet_conv_size=3, hop_length=256, n_mel_channels=100):
        if False:
            i = 10
            return i + 15
        super(UnivNetGenerator, self).__init__()
        self.mel_channel = n_mel_channels
        self.noise_dim = noise_dim
        self.hop_length = hop_length
        channel_size = channel_size
        kpnet_conv_size = kpnet_conv_size
        self.res_stack = nn.ModuleList()
        hop_length = 1
        for stride in strides:
            hop_length = stride * hop_length
            self.res_stack.append(LVCBlock(channel_size, n_mel_channels, stride=stride, dilations=dilations, lReLU_slope=lReLU_slope, cond_hop_length=hop_length, kpnet_conv_size=kpnet_conv_size))
        self.conv_pre = nn.utils.parametrizations.weight_norm(nn.Conv1d(noise_dim, channel_size, 7, padding=3, padding_mode='reflect'))
        self.conv_post = nn.Sequential(nn.LeakyReLU(lReLU_slope), nn.utils.parametrizations.weight_norm(nn.Conv1d(channel_size, 1, 7, padding=3, padding_mode='reflect')), nn.Tanh())

    def forward(self, c, z):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            c (Tensor): the conditioning sequence of mel-spectrogram (batch, mel_channels, in_length)\n            z (Tensor): the noise sequence (batch, noise_dim, in_length)\n\n        '
        z = self.conv_pre(z)
        for res_block in self.res_stack:
            res_block.to(z.device)
            z = res_block(z, c)
        z = self.conv_post(z)
        return z

    def eval(self, inference=False):
        if False:
            for i in range(10):
                print('nop')
        super(UnivNetGenerator, self).eval()
        if inference:
            self.remove_weight_norm()

    def remove_weight_norm(self):
        if False:
            i = 10
            return i + 15
        parametrize.remove_parametrizations(self.conv_pre, 'weight')
        for layer in self.conv_post:
            if len(layer.state_dict()) != 0:
                parametrize.remove_parametrizations(layer, 'weight')
        for res_block in self.res_stack:
            res_block.remove_weight_norm()

    def inference(self, c, z=None):
        if False:
            return 10
        zero = torch.full((c.shape[0], self.mel_channel, 10), -11.5129).to(c.device)
        mel = torch.cat((c, zero), dim=2)
        if z is None:
            z = torch.randn(c.shape[0], self.noise_dim, mel.size(2)).to(mel.device)
        audio = self.forward(mel, z)
        audio = audio[:, :, :-(self.hop_length * 10)]
        audio = audio.clamp(min=-1, max=1)
        return audio

@dataclass
class VocType:
    constructor: Callable[[], nn.Module]
    model_path: str
    subkey: Optional[str] = None

    def optionally_index(self, model_dict):
        if False:
            while True:
                i = 10
        if self.subkey is not None:
            return model_dict[self.subkey]
        return model_dict

class VocConf(Enum):
    Univnet = VocType(UnivNetGenerator, 'vocoder.pth', 'model_g')
if __name__ == '__main__':
    model = UnivNetGenerator()
    c = torch.randn(3, 100, 10)
    z = torch.randn(3, 64, 10)
    print(c.shape)
    y = model(c, z)
    print(y.shape)
    assert y.shape == torch.Size([3, 1, 2560])
    pytorch_total_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
    print(pytorch_total_params)