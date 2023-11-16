import torch
import unittest
from torch.ao.quantization.experimental.observer import APoTObserver
from torch.ao.quantization.experimental.quantizer import quantize_APoT, dequantize_APoT
from torch.ao.quantization.experimental.fake_quantize import APoTFakeQuantize
from torch.ao.quantization.experimental.fake_quantize_function import fake_quantize_function
forward_helper = fake_quantize_function.forward
backward = fake_quantize_function.backward
from torch.autograd import gradcheck

class TestFakeQuantize(unittest.TestCase):
    """ Tests fake quantize calculate_qparams() method
         by comparing with result from observer calculate_qparams.
         Uses hard-coded values: alpha=1.0, b=4, k=2.
    """

    def test_fake_calc_qparams(self):
        if False:
            print('Hello World!')
        apot_fake = APoTFakeQuantize(b=4, k=2)
        apot_fake.activation_post_process.min_val = torch.tensor([0.0])
        apot_fake.activation_post_process.max_val = torch.tensor([1.0])
        (alpha, gamma, quantization_levels, level_indices) = apot_fake.calculate_qparams(signed=False)
        observer = APoTObserver(b=4, k=2)
        observer.min_val = torch.tensor([0.0])
        observer.max_val = torch.tensor([1.0])
        qparams_expected = observer.calculate_qparams(signed=False)
        self.assertEqual(alpha, qparams_expected[0])
        self.assertTrue(torch.equal(gamma, qparams_expected[1]))
        self.assertTrue(torch.equal(quantization_levels, qparams_expected[2]))
        self.assertTrue(torch.equal(level_indices, qparams_expected[3]))
    ' Tests fake quantize forward() method\n         by comparing result with expected\n         quant_dequant_APoT mapping of input tensor.\n         Uses input tensor with random values from 0 -> 1000\n         and APoT observer with hard-coded values b=4, k=2\n    '

    def test_forward(self):
        if False:
            while True:
                i = 10
        X = 1000 * torch.rand(20)
        observer = APoTObserver(b=4, k=2)
        observer.forward(X)
        (alpha, gamma, quantization_levels, level_indices) = observer.calculate_qparams(signed=False)
        apot_fake = APoTFakeQuantize(b=4, k=2)
        apot_fake.enable_observer()
        apot_fake.enable_fake_quant()
        X_reduced_precision_fp = apot_fake.forward(torch.clone(X), False)
        X_to_apot = quantize_APoT(X, alpha, gamma, quantization_levels, level_indices)
        X_expected = dequantize_APoT(X_to_apot)
        self.assertTrue(torch.equal(X_reduced_precision_fp, X_expected))
    ' Tests fake quantize forward() method\n         throws error when qparams are None\n    '

    def test_forward_exception(self):
        if False:
            for i in range(10):
                print('nop')
        X = 1000 * torch.rand(20)
        apot_fake = APoTFakeQuantize(b=4, k=2)
        apot_fake.disable_observer()
        apot_fake.enable_fake_quant()
        with self.assertRaises(Exception):
            apot_fake.forward(torch.clone(X), False)
    ' Tests fake quantize helper backward() method\n         using torch.autograd.gradcheck function.\n    '

    def test_backward(self):
        if False:
            i = 10
            return i + 15
        input = torch.randn(20, dtype=torch.double, requires_grad=True)
        observer = APoTObserver(b=4, k=2)
        observer(input)
        (alpha, gamma, quantization_levels, level_indices) = observer.calculate_qparams(signed=False)
        test = gradcheck(fake_quantize_function.apply, (input, alpha, gamma, quantization_levels, level_indices), atol=0.0001)
if __name__ == '__main__':
    unittest.main()