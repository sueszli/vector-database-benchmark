import unittest
import torch
import torch.nn as nn
from fairseq.modules import ConvTBC

class TestConvTBC(unittest.TestCase):

    def test_convtbc(self):
        if False:
            for i in range(10):
                print('nop')
        conv_tbc = ConvTBC(4, 5, kernel_size=3, padding=1)
        conv1d = nn.Conv1d(4, 5, kernel_size=3, padding=1)
        conv_tbc.weight.data.copy_(conv1d.weight.data.transpose(0, 2))
        conv_tbc.bias.data.copy_(conv1d.bias.data)
        input_tbc = torch.randn(7, 2, 4, requires_grad=True)
        input1d = input_tbc.data.transpose(0, 1).transpose(1, 2)
        input1d.requires_grad = True
        output_tbc = conv_tbc(input_tbc)
        output1d = conv1d(input1d)
        self.assertAlmostEqual(output_tbc.data.transpose(0, 1).transpose(1, 2), output1d.data)
        grad_tbc = torch.randn(output_tbc.size())
        grad1d = grad_tbc.transpose(0, 1).transpose(1, 2).contiguous()
        output_tbc.backward(grad_tbc)
        output1d.backward(grad1d)
        self.assertAlmostEqual(conv_tbc.weight.grad.data.transpose(0, 2), conv1d.weight.grad.data)
        self.assertAlmostEqual(conv_tbc.bias.grad.data, conv1d.bias.grad.data)
        self.assertAlmostEqual(input_tbc.grad.data.transpose(0, 1).transpose(1, 2), input1d.grad.data)

    def assertAlmostEqual(self, t1, t2):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(t1.size(), t2.size(), 'size mismatch')
        self.assertLess((t1 - t2).abs().max(), 0.0001)
if __name__ == '__main__':
    unittest.main()