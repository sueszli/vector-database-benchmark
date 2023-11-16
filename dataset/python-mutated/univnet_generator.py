from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import parametrize
from TTS.vocoder.layers.lvc_block import LVCBlock
LRELU_SLOPE = 0.1

class UnivnetGenerator(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, cond_channels: int, upsample_factors: List[int], lvc_layers_each_block: int, lvc_kernel_size: int, kpnet_hidden_channels: int, kpnet_conv_size: int, dropout: float, use_weight_norm=True):
        if False:
            i = 10
            return i + 15
        'Univnet Generator network.\n\n        Paper: https://arxiv.org/pdf/2106.07889.pdf\n\n        Args:\n            in_channels (int): Number of input tensor channels.\n            out_channels (int): Number of channels of the output tensor.\n            hidden_channels (int): Number of hidden network channels.\n            cond_channels (int): Number of channels of the conditioning tensors.\n            upsample_factors (List[int]): List of uplsample factors for the upsampling layers.\n            lvc_layers_each_block (int): Number of LVC layers in each block.\n            lvc_kernel_size (int): Kernel size of the LVC layers.\n            kpnet_hidden_channels (int): Number of hidden channels in the key-point network.\n            kpnet_conv_size (int): Number of convolution channels in the key-point network.\n            dropout (float): Dropout rate.\n            use_weight_norm (bool, optional): Enable/disable weight norm. Defaults to True.\n        '
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_channels = cond_channels
        self.upsample_scale = np.prod(upsample_factors)
        self.lvc_block_nums = len(upsample_factors)
        self.first_conv = torch.nn.Conv1d(in_channels, hidden_channels, kernel_size=7, padding=(7 - 1) // 2, dilation=1, bias=True)
        self.lvc_blocks = torch.nn.ModuleList()
        cond_hop_length = 1
        for n in range(self.lvc_block_nums):
            cond_hop_length = cond_hop_length * upsample_factors[n]
            lvcb = LVCBlock(in_channels=hidden_channels, cond_channels=cond_channels, upsample_ratio=upsample_factors[n], conv_layers=lvc_layers_each_block, conv_kernel_size=lvc_kernel_size, cond_hop_length=cond_hop_length, kpnet_hidden_channels=kpnet_hidden_channels, kpnet_conv_size=kpnet_conv_size, kpnet_dropout=dropout)
            self.lvc_blocks += [lvcb]
        self.last_conv_layers = torch.nn.ModuleList([torch.nn.Conv1d(hidden_channels, out_channels, kernel_size=7, padding=(7 - 1) // 2, dilation=1, bias=True)])
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, c):
        if False:
            while True:
                i = 10
        "Calculate forward propagation.\n        Args:\n            c (Tensor): Local conditioning auxiliary features (B, C ,T').\n        Returns:\n            Tensor: Output tensor (B, out_channels, T)\n        "
        x = torch.randn([c.shape[0], self.in_channels, c.shape[2]])
        x = x.to(self.first_conv.bias.device)
        x = self.first_conv(x)
        for n in range(self.lvc_block_nums):
            x = self.lvc_blocks[n](x, c)
        for f in self.last_conv_layers:
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = f(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        if False:
            while True:
                i = 10
        'Remove weight normalization module from all of the layers.'

        def _remove_weight_norm(m):
            if False:
                return 10
            try:
                parametrize.remove_parametrizations(m, 'weight')
            except ValueError:
                return
        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        if False:
            for i in range(10):
                print('nop')
        'Apply weight normalization module from all of the layers.'

        def _apply_weight_norm(m):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d)):
                torch.nn.utils.parametrizations.weight_norm(m)
        self.apply(_apply_weight_norm)

    @staticmethod
    def _get_receptive_field_size(layers, stacks, kernel_size, dilation=lambda x: 2 ** x):
        if False:
            return 10
        assert layers % stacks == 0
        layers_per_cycle = layers // stacks
        dilations = [dilation(i % layers_per_cycle) for i in range(layers)]
        return (kernel_size - 1) * sum(dilations) + 1

    @property
    def receptive_field_size(self):
        if False:
            print('Hello World!')
        'Return receptive field size.'
        return self._get_receptive_field_size(self.layers, self.stacks, self.kernel_size)

    @torch.no_grad()
    def inference(self, c):
        if False:
            i = 10
            return i + 15
        'Perform inference.\n        Args:\n            c (Tensor): Local conditioning auxiliary features :math:`(B, C, T)`.\n        Returns:\n            Tensor: Output tensor (T, out_channels)\n        '
        x = torch.randn([c.shape[0], self.in_channels, c.shape[2]])
        x = x.to(self.first_conv.bias.device)
        c = c.to(next(self.parameters()))
        return self.forward(c)