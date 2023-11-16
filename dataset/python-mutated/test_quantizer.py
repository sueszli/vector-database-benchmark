import torch
from torch import quantize_per_tensor
from torch.ao.quantization.observer import MinMaxObserver
from torch.ao.quantization.experimental.observer import APoTObserver
from torch.ao.quantization.experimental.quantizer import APoTQuantizer, quantize_APoT, dequantize_APoT
import unittest
import random

class TestQuantizer(unittest.TestCase):
    """ Tests quantize_APoT result on random 1-dim tensor
        and hardcoded values for b, k by comparing to uniform quantization
        (non-uniform quantization reduces to uniform for k = 1)
        quantized tensor (https://pytorch.org/docs/stable/generated/torch.quantize_per_tensor.html)
        * tensor2quantize: Tensor
        * b: 8
        * k: 1
    """

    def test_quantize_APoT_rand_k1(self):
        if False:
            while True:
                i = 10
        size = random.randint(1, 20)
        tensor2quantize = 1000 * torch.rand(size, dtype=torch.float)
        apot_observer = APoTObserver(b=8, k=1)
        apot_observer(tensor2quantize)
        (alpha, gamma, quantization_levels, level_indices) = apot_observer.calculate_qparams(signed=False)
        qtensor = quantize_APoT(tensor2quantize=tensor2quantize, alpha=alpha, gamma=gamma, quantization_levels=quantization_levels, level_indices=level_indices)
        uniform_observer = MinMaxObserver()
        uniform_observer(tensor2quantize)
        (scale, zero_point) = uniform_observer.calculate_qparams()
        uniform_quantized = quantize_per_tensor(input=tensor2quantize, scale=scale, zero_point=zero_point, dtype=torch.quint8).int_repr()
        qtensor_data = qtensor.data.int()
        uniform_quantized_tensor = uniform_quantized.data.int()
        self.assertTrue(torch.equal(qtensor_data, uniform_quantized_tensor))
    ' Tests quantize_APoT for k != 1.\n        Tests quantize_APoT result on random 1-dim tensor and hardcoded values for\n        b=4, k=2 by comparing results to hand-calculated results from APoT paper\n        https://arxiv.org/pdf/1909.13144.pdf\n        * tensor2quantize: Tensor\n        * b: 4\n        * k: 2\n    '

    def test_quantize_APoT_k2(self):
        if False:
            while True:
                i = 10
        '\n        given b = 4, k = 2, alpha = 1.0, we know:\n        (from APoT paper example: https://arxiv.org/pdf/1909.13144.pdf)\n\n        quantization_levels = tensor([0.0000, 0.0208, 0.0417, 0.0625, 0.0833, 0.1250, 0.1667,\n        0.1875, 0.2500, 0.3333, 0.3750, 0.5000, 0.6667, 0.6875, 0.7500, 1.0000])\n\n        level_indices = tensor([ 0, 3, 12, 15,  2, 14,  8, 11, 10, 1, 13,  9,  4,  7,  6,  5]))\n        '
        tensor2quantize = torch.tensor([0, 0.0215, 0.1692, 0.385, 1, 0.0391])
        observer = APoTObserver(b=4, k=2)
        observer.forward(tensor2quantize)
        (alpha, gamma, quantization_levels, level_indices) = observer.calculate_qparams(signed=False)
        qtensor = quantize_APoT(tensor2quantize=tensor2quantize, alpha=alpha, gamma=gamma, quantization_levels=quantization_levels, level_indices=level_indices)
        qtensor_data = qtensor.data.int()
        expected_qtensor = torch.tensor([0, 3, 8, 13, 5, 12], dtype=torch.int32)
        self.assertTrue(torch.equal(qtensor_data, expected_qtensor))
    ' Tests dequantize_apot result on random 1-dim tensor\n        and hardcoded values for b, k.\n        Dequant -> quant an input tensor and verify that\n        result is equivalent to input\n        * tensor2quantize: Tensor\n        * b: 4\n        * k: 2\n    '

    def test_dequantize_quantize_rand_b4(self):
        if False:
            return 10
        observer = APoTObserver(4, 2)
        size = random.randint(1, 20)
        tensor2quantize = 1000 * torch.rand(size, dtype=torch.float)
        observer.forward(tensor2quantize)
        (alpha, gamma, quantization_levels, level_indices) = observer.calculate_qparams(signed=False)
        original_apot = quantize_APoT(tensor2quantize=tensor2quantize, alpha=alpha, gamma=gamma, quantization_levels=quantization_levels, level_indices=level_indices)
        original_input = torch.clone(original_apot.data).int()
        dequantize_result = dequantize_APoT(apot_tensor=original_apot)
        final_apot = quantize_APoT(tensor2quantize=dequantize_result, alpha=alpha, gamma=gamma, quantization_levels=quantization_levels, level_indices=level_indices)
        result = final_apot.data.int()
        self.assertTrue(torch.equal(original_input, result))
    ' Tests dequantize_apot result on random 1-dim tensor\n        and hardcoded values for b, k.\n        Dequant -> quant an input tensor and verify that\n        result is equivalent to input\n        * tensor2quantize: Tensor\n        * b: 12\n        * k: 4\n    '

    def test_dequantize_quantize_rand_b6(self):
        if False:
            for i in range(10):
                print('nop')
        observer = APoTObserver(12, 4)
        size = random.randint(1, 20)
        tensor2quantize = 1000 * torch.rand(size, dtype=torch.float)
        observer.forward(tensor2quantize)
        (alpha, gamma, quantization_levels, level_indices) = observer.calculate_qparams(signed=False)
        original_apot = quantize_APoT(tensor2quantize=tensor2quantize, alpha=alpha, gamma=gamma, quantization_levels=quantization_levels, level_indices=level_indices)
        original_input = torch.clone(original_apot.data).int()
        dequantize_result = dequantize_APoT(apot_tensor=original_apot)
        final_apot = quantize_APoT(tensor2quantize=dequantize_result, alpha=alpha, gamma=gamma, quantization_levels=quantization_levels, level_indices=level_indices)
        result = final_apot.data.int()
        self.assertTrue(torch.equal(original_input, result))
    ' Tests for correct dimensions in dequantize_apot result\n         on random 3-dim tensor with random dimension sizes\n         and hardcoded values for b, k.\n         Dequant an input tensor and verify that\n         dimensions are same as input.\n         * tensor2quantize: Tensor\n         * b: 4\n         * k: 2\n    '

    def test_dequantize_dim(self):
        if False:
            print('Hello World!')
        observer = APoTObserver(4, 2)
        size1 = random.randint(1, 20)
        size2 = random.randint(1, 20)
        size3 = random.randint(1, 20)
        tensor2quantize = 1000 * torch.rand(size1, size2, size3, dtype=torch.float)
        observer.forward(tensor2quantize)
        (alpha, gamma, quantization_levels, level_indices) = observer.calculate_qparams(signed=False)
        original_apot = quantize_APoT(tensor2quantize=tensor2quantize, alpha=alpha, gamma=gamma, quantization_levels=quantization_levels, level_indices=level_indices)
        dequantize_result = dequantize_APoT(apot_tensor=original_apot)
        self.assertEqual(original_apot.data.size(), dequantize_result.size())

    def test_q_apot_alpha(self):
        if False:
            print('Hello World!')
        with self.assertRaises(NotImplementedError):
            APoTQuantizer.q_apot_alpha(self)
if __name__ == '__main__':
    unittest.main()