import torch
from torch.nn import functional as F

class ResidualBlock(torch.nn.Module):
    """Residual block module in WaveNet."""

    def __init__(self, kernel_size=3, res_channels=64, gate_channels=128, skip_channels=64, aux_channels=80, dropout=0.0, dilation=1, bias=True, use_causal_conv=False):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.dropout = dropout
        if use_causal_conv:
            padding = (kernel_size - 1) * dilation
        else:
            assert (kernel_size - 1) % 2 == 0, 'Not support even number kernel size.'
            padding = (kernel_size - 1) // 2 * dilation
        self.use_causal_conv = use_causal_conv
        self.conv = torch.nn.Conv1d(res_channels, gate_channels, kernel_size, padding=padding, dilation=dilation, bias=bias)
        if aux_channels > 0:
            self.conv1x1_aux = torch.nn.Conv1d(aux_channels, gate_channels, 1, bias=False)
        else:
            self.conv1x1_aux = None
        gate_out_channels = gate_channels // 2
        self.conv1x1_out = torch.nn.Conv1d(gate_out_channels, res_channels, 1, bias=bias)
        self.conv1x1_skip = torch.nn.Conv1d(gate_out_channels, skip_channels, 1, bias=bias)

    def forward(self, x, c):
        if False:
            while True:
                i = 10
        '\n        x: B x D_res x T\n        c: B x D_aux x T\n        '
        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv(x)
        x = x[:, :, :residual.size(-1)] if self.use_causal_conv else x
        splitdim = 1
        (xa, xb) = x.split(x.size(splitdim) // 2, dim=splitdim)
        if c is not None:
            assert self.conv1x1_aux is not None
            c = self.conv1x1_aux(c)
            (ca, cb) = c.split(c.size(splitdim) // 2, dim=splitdim)
            (xa, xb) = (xa + ca, xb + cb)
        x = torch.tanh(xa) * torch.sigmoid(xb)
        s = self.conv1x1_skip(x)
        x = (self.conv1x1_out(x) + residual) * 0.5 ** 2
        return (x, s)