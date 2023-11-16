import torch
from torch import Tensor
from torch.ao.quantization.experimental.quantizer import quantize_APoT, dequantize_APoT

class fake_quantize_function(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor, alpha: Tensor, gamma: Tensor, quantization_levels: Tensor, level_indices: Tensor) -> Tensor:
        if False:
            i = 10
            return i + 15
        quantized_result = quantize_APoT(x, alpha, gamma, quantization_levels, level_indices)
        mask = x.detach().apply_(lambda x: x <= alpha and x >= -alpha)
        result = dequantize_APoT(quantized_result)
        ctx.save_for_backward(mask)
        return result

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        if False:
            print('Hello World!')
        mask = ctx.saved_tensors
        return grad_output * mask