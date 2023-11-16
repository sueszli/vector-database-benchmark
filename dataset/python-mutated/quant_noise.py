import torch
import torch.nn as nn

def quant_noise(module, p, block_size):
    if False:
        for i in range(10):
            print('nop')
    '\n    Wraps modules and applies quantization noise to the weights for\n    subsequent quantization with Iterative Product Quantization as\n    described in "Training with Quantization Noise for Extreme Model Compression"\n\n    Args:\n        - module: nn.Module\n        - p: amount of Quantization Noise\n        - block_size: size of the blocks for subsequent quantization with iPQ\n\n    Remarks:\n        - Module weights must have the right sizes wrt the block size\n        - Only Linear, Embedding and Conv2d modules are supported for the moment\n        - For more detail on how to quantize by blocks with convolutional weights,\n          see "And the Bit Goes Down: Revisiting the Quantization of Neural Networks"\n        - We implement the simplest form of noise here as stated in the paper\n          which consists in randomly dropping blocks\n    '
    if p <= 0:
        return module
    assert isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d))
    is_conv = module.weight.ndim == 4
    if not is_conv:
        assert module.weight.size(1) % block_size == 0, 'Input features must be a multiple of block sizes'
    elif module.kernel_size == (1, 1):
        assert module.in_channels % block_size == 0, 'Input channels must be a multiple of block sizes'
    else:
        k = module.kernel_size[0] * module.kernel_size[1]
        assert k % block_size == 0, 'Kernel size must be a multiple of block size'

    def _forward_pre_hook(mod, input):
        if False:
            print('Hello World!')
        if mod.training:
            if not is_conv:
                weight = mod.weight
                in_features = weight.size(1)
                out_features = weight.size(0)
                mask = torch.zeros(in_features // block_size * out_features, device=weight.device)
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)
            else:
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels
                if mod.kernel_size == (1, 1):
                    mask = torch.zeros(int(in_channels // block_size * out_channels), device=weight.device)
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(-1, in_channels)
                else:
                    mask = torch.zeros(weight.size(0), weight.size(1), device=weight.device)
                    mask.bernoulli_(p)
                    mask = mask.unsqueeze(2).unsqueeze(3).repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])
            mask = mask.to(torch.bool)
            s = 1 / (1 - p)
            mod.weight.data = s * weight.masked_fill(mask, 0)
    module.register_forward_pre_hook(_forward_pre_hook)
    return module