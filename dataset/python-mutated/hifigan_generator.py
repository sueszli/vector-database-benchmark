import torch
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations
from TTS.utils.io import load_fsspec
LRELU_SLOPE = 0.1

def get_padding(k, d):
    if False:
        return 10
    return int((k * d - d) / 2)

class ResBlock1(torch.nn.Module):
    """Residual Block Type 1. It has 3 convolutional layers in each convolutional block.

    Network::

        x -> lrelu -> conv1_1 -> conv1_2 -> conv1_3 -> z -> lrelu -> conv2_1 -> conv2_2 -> conv2_3 -> o -> + -> o
        |--------------------------------------------------------------------------------------------------|


    Args:
        channels (int): number of hidden channels for the convolutional layers.
        kernel_size (int): size of the convolution filter in each layer.
        dilations (list): list of dilation value for each conv layer in a block.
    """

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        if False:
            return 10
        super().__init__()
        self.convs1 = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], padding=get_padding(kernel_size, dilation[2])))])
        self.convs2 = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))])

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            x (Tensor): input tensor.\n        Returns:\n            Tensor: output tensor.\n        Shapes:\n            x: [B, C, T]\n        '
        for (c1, c2) in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        if False:
            print('Hello World!')
        for l in self.convs1:
            remove_parametrizations(l, 'weight')
        for l in self.convs2:
            remove_parametrizations(l, 'weight')

class ResBlock2(torch.nn.Module):
    """Residual Block Type 2. It has 1 convolutional layers in each convolutional block.

    Network::

        x -> lrelu -> conv1-> -> z -> lrelu -> conv2-> o -> + -> o
        |---------------------------------------------------|


    Args:
        channels (int): number of hidden channels for the convolutional layers.
        kernel_size (int): size of the convolution filter in each layer.
        dilations (list): list of dilation value for each conv layer in a block.
    """

    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.convs = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1])))])

    def forward(self, x):
        if False:
            return 10
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        if False:
            i = 10
            return i + 15
        for l in self.convs:
            remove_parametrizations(l, 'weight')

class HifiganGenerator(torch.nn.Module):

    def __init__(self, in_channels, out_channels, resblock_type, resblock_dilation_sizes, resblock_kernel_sizes, upsample_kernel_sizes, upsample_initial_channel, upsample_factors, inference_padding=5, cond_channels=0, conv_pre_weight_norm=True, conv_post_weight_norm=True, conv_post_bias=True):
        if False:
            print('Hello World!')
        "HiFiGAN Generator with Multi-Receptive Field Fusion (MRF)\n\n        Network:\n            x -> lrelu -> upsampling_layer -> resblock1_k1x1 -> z1 -> + -> z_sum / #resblocks -> lrelu -> conv_post_7x1 -> tanh -> o\n                                                 ..          -> zI ---|\n                                              resblockN_kNx1 -> zN ---'\n\n        Args:\n            in_channels (int): number of input tensor channels.\n            out_channels (int): number of output tensor channels.\n            resblock_type (str): type of the `ResBlock`. '1' or '2'.\n            resblock_dilation_sizes (List[List[int]]): list of dilation values in each layer of a `ResBlock`.\n            resblock_kernel_sizes (List[int]): list of kernel sizes for each `ResBlock`.\n            upsample_kernel_sizes (List[int]): list of kernel sizes for each transposed convolution.\n            upsample_initial_channel (int): number of channels for the first upsampling layer. This is divided by 2\n                for each consecutive upsampling layer.\n            upsample_factors (List[int]): upsampling factors (stride) for each upsampling layer.\n            inference_padding (int): constant padding applied to the input at inference time. Defaults to 5.\n        "
        super().__init__()
        self.inference_padding = inference_padding
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_factors)
        self.conv_pre = weight_norm(Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if resblock_type == '1' else ResBlock2
        self.ups = nn.ModuleList()
        for (i, (u, k)) in enumerate(zip(upsample_factors, upsample_kernel_sizes)):
            self.ups.append(weight_norm(ConvTranspose1d(upsample_initial_channel // 2 ** i, upsample_initial_channel // 2 ** (i + 1), k, u, padding=(k - u) // 2)))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // 2 ** (i + 1)
            for (_, (k, d)) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))
        self.conv_post = weight_norm(Conv1d(ch, out_channels, 7, 1, padding=3, bias=conv_post_bias))
        if cond_channels > 0:
            self.cond_layer = nn.Conv1d(cond_channels, upsample_initial_channel, 1)
        if not conv_pre_weight_norm:
            remove_parametrizations(self.conv_pre, 'weight')
        if not conv_post_weight_norm:
            remove_parametrizations(self.conv_post, 'weight')

    def forward(self, x, g=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            x (Tensor): feature input tensor.\n            g (Tensor): global conditioning input tensor.\n\n        Returns:\n            Tensor: output waveform.\n\n        Shapes:\n            x: [B, C, T]\n            Tensor: [B, 1, T]\n        '
        o = self.conv_pre(x)
        if hasattr(self, 'cond_layer'):
            o = o + self.cond_layer(g)
        for i in range(self.num_upsamples):
            o = F.leaky_relu(o, LRELU_SLOPE)
            o = self.ups[i](o)
            z_sum = None
            for j in range(self.num_kernels):
                if z_sum is None:
                    z_sum = self.resblocks[i * self.num_kernels + j](o)
                else:
                    z_sum += self.resblocks[i * self.num_kernels + j](o)
            o = z_sum / self.num_kernels
        o = F.leaky_relu(o)
        o = self.conv_post(o)
        o = torch.tanh(o)
        return o

    @torch.no_grad()
    def inference(self, c):
        if False:
            return 10
        '\n        Args:\n            x (Tensor): conditioning input tensor.\n\n        Returns:\n            Tensor: output waveform.\n\n        Shapes:\n            x: [B, C, T]\n            Tensor: [B, 1, T]\n        '
        c = c.to(self.conv_pre.weight.device)
        c = torch.nn.functional.pad(c, (self.inference_padding, self.inference_padding), 'replicate')
        return self.forward(c)

    def remove_weight_norm(self):
        if False:
            print('Hello World!')
        print('Removing weight norm...')
        for l in self.ups:
            remove_parametrizations(l, 'weight')
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_parametrizations(self.conv_pre, 'weight')
        remove_parametrizations(self.conv_post, 'weight')

    def load_checkpoint(self, config, checkpoint_path, eval=False, cache=False):
        if False:
            print('Hello World!')
        state = load_fsspec(checkpoint_path, map_location=torch.device('cpu'), cache=cache)
        self.load_state_dict(state['model'])
        if eval:
            self.eval()
            assert not self.training
            self.remove_weight_norm()