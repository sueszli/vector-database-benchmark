import torch
from torch import Tensor
import numpy as np
from torch.ao.quantization.experimental.apot_utils import float_to_apot, apot_to_float, quant_dequant_util

class APoTQuantizer:
    alpha: torch.Tensor
    gamma: torch.Tensor
    quantization_levels: torch.Tensor
    level_indices: torch.Tensor

    def __init__(self, alpha: torch.Tensor, gamma: torch.Tensor, quantization_levels: torch.Tensor, level_indices: torch.Tensor) -> None:
        if False:
            while True:
                i = 10
        self.alpha = alpha
        self.gamma = gamma
        self.quantization_levels = quantization_levels
        self.level_indices = level_indices
    ' Quantizes fp Tensor to integer APoT representation.\n    Conversion is based on the qparams from a specified APoT non-uniform observer.\n    The approach follows the method outlined in the APoT paper: https://arxiv.org/pdf/1909.13144.pdf.\n    Args:\n        tensor2quantize: fp Tensor\n    Returns:\n        result: APoT Tensor representation of tensor2quantize\n    '

    def quantize(self, tensor2quantize: Tensor):
        if False:
            print('Hello World!')
        result = torch.tensor([])
        tensor2quantize = tensor2quantize.detach().apply_(lambda x: float_to_apot(x, self.quantization_levels, self.level_indices, self.alpha))
        tensor2quantize = tensor2quantize.int()
        from torch.ao.quantization.experimental.APoT_tensor import TensorAPoT
        result = TensorAPoT(self, tensor2quantize)
        return result
    ' Dequantizes integer Tensor to floating point (fp) representation\n    based on the calculated quantization levels from a specified APoT non-uniform observer.\n    The approach follows the method outlined in the APoT paper: https://arxiv.org/pdf/1909.13144.pdf.\n    Args:\n        tensor2quantize: fp Tensor\n    Returns:\n        result: fp reduced precision representation of input Tensor\n    '

    def dequantize(self, apot_tensor) -> Tensor:
        if False:
            i = 10
            return i + 15
        orig_size = apot_tensor.data.size()
        apot_tensor_data = apot_tensor.data.flatten()
        print(apot_tensor_data)
        result_temp = np.empty(shape=apot_tensor_data.size())
        for i in range(len(apot_tensor_data)):
            new_ele = apot_to_float(apot_tensor_data[i], self.quantization_levels, self.level_indices)
            result_temp[i] = new_ele
        result = torch.from_numpy(result_temp).reshape(orig_size)
        return result
    ' Returns result of quantize -> dequantize on a fp Tensor (reduced precision)\n    based on the calculated quantization levels from a specified APoT non-uniform observer.\n    The approach follows the method outlined in the APoT paper: https://arxiv.org/pdf/1909.13144.pdf.\n    Args:\n        apot_tensor: quantized APoT Tensor to dequantize\n    Returns:\n        result: fp representation of input Tensor\n    '

    def quant_dequant(self, tensor2quantize: Tensor) -> Tensor:
        if False:
            i = 10
            return i + 15
        levels_lst = list(self.quantization_levels)
        result = tensor2quantize.apply_(lambda x: quant_dequant_util(x, levels_lst))
        return result

    def q_apot_alpha(self) -> float:
        if False:
            print('Hello World!')
        raise NotImplementedError
' Global method to create quantizer and call quantizer quantize_APoT\n    Args:\n        tensor2quantize: fp Tensor to quantize\n        alpha: Tensor qparam alpha (clipping level)\n        gamma: Tensor qparam gamma (scale factor for quantization levels)\n        quantization levels: Tensor with fp quantization levels\n        level indices: Tensor with integer quantization level indices\n    Returns:\n        result: ApoT Tensor representation of tensor2quantize\n'

def quantize_APoT(tensor2quantize: Tensor, alpha: Tensor, gamma: Tensor, quantization_levels: Tensor, level_indices: Tensor):
    if False:
        for i in range(10):
            print('nop')
    quantizer = APoTQuantizer(alpha=alpha, gamma=gamma, quantization_levels=quantization_levels, level_indices=level_indices)
    result = quantizer.quantize(tensor2quantize)
    return result
' Global method to create quantizer and call quantizer dequantize_APoT\n    Args:\n        apot_tensor: APoT Tensor to dequantize\n    Returns:\n        result: fp Tensor dequantized from apot_tensor\n'

def dequantize_APoT(apot_tensor) -> Tensor:
    if False:
        for i in range(10):
            print('nop')
    quantizer = apot_tensor.quantizer
    result = quantizer.dequantize(apot_tensor)
    return result
' Global method to create quantizer and call quantizer quant_dequant\n    Args:\n        tensor2quantize: fp Tensor to quantize\n        alpha: Tensor qparam alpha (clipping level)\n        gamma: Tensor qparam gamma (scale factor for quantization levels)\n        quantization levels: Tensor with fp quantization levels\n        level indices: Tensor with integer quantization level indices\n    Returns:\n        result: fp reduced precision Tensor from tensor2quantize\n'

def quant_dequant_APoT(tensor2quantize: Tensor, alpha: Tensor, gamma: Tensor, quantization_levels: Tensor, level_indices: Tensor) -> Tensor:
    if False:
        print('Hello World!')
    quantizer = APoTQuantizer(alpha=alpha, gamma=gamma, quantization_levels=quantization_levels, level_indices=level_indices)
    result = quantizer.quant_dequant(tensor2quantize)
    return result