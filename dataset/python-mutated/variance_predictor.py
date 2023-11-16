import torch
import torch.nn as nn
from TTS.tts.layers.delightful_tts.conv_layers import ConvTransposed

class VariancePredictor(nn.Module):
    """
    Network is 2-layer 1D convolutions with leaky relu activation and then
    followed by layer normalization then a dropout layer and finally an
    extra linear layer to project the hidden states into the output sequence.

    Args:
        channels_in (int): Number of in channels for conv layers.
        channels_out (int): Number of out channels for the last linear layer.
        kernel_size (int): Size the kernel for the conv layers.
        p_dropout (float): Probability of dropout.
        lrelu_slope (float): Slope for the leaky relu.

    Inputs: inputs, mask
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **mask** (batch, time): Tensor containing indices to be masked
    Returns:
        - **outputs** (batch, time): Tensor produced by last linear layer.
    """

    def __init__(self, channels_in: int, channels: int, channels_out: int, kernel_size: int, p_dropout: float, lrelu_slope: float):
        if False:
            print('Hello World!')
        super().__init__()
        self.layers = nn.ModuleList([ConvTransposed(channels_in, channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2), nn.LeakyReLU(lrelu_slope), nn.LayerNorm(channels), nn.Dropout(p_dropout), ConvTransposed(channels, channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2), nn.LeakyReLU(lrelu_slope), nn.LayerNorm(channels), nn.Dropout(p_dropout)])
        self.linear_layer = nn.Linear(channels, channels_out)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        '\n        Shapes:\n            x: :math: `[B, T_src, C]`\n            mask: :math: `[B, T_src]`\n        '
        for layer in self.layers:
            x = layer(x)
        x = self.linear_layer(x)
        x = x.squeeze(-1)
        x = x.masked_fill(mask, 0.0)
        return x