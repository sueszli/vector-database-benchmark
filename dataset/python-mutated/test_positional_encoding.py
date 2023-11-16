import unittest
import torch
from fairseq.modules import RelPositionalEncoding
import numpy as np

class TestRelPositionalEncoding(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        self.T = 3
        self.B = 1
        self.C = 2
        torch.manual_seed(0)
        self.sample = torch.randn(self.T, self.B, self.C)
        self.rel_pos_enc = RelPositionalEncoding(max_len=4, d_model=self.C)

    def test_extend_pe(self):
        if False:
            while True:
                i = 10
        inp = self.sample.transpose(0, 1)
        self.rel_pos_enc.extend_pe(inp)
        expected_pe = torch.tensor([[[0.1411, -0.99], [0.9093, -0.4161], [0.8415, 0.5403], [0.0, 1.0], [-0.8415, 0.5403], [-0.9093, -0.4161], [-0.1411, -0.99]]])
        self.assertTrue(np.allclose(expected_pe.cpu().detach().numpy(), self.rel_pos_enc.pe.cpu().detach().numpy(), atol=0.0001))

    def test_forward(self):
        if False:
            for i in range(10):
                print('nop')
        pos_enc = self.rel_pos_enc(self.sample)
        expected_pos_enc = torch.tensor([[[0.9093, -0.4161]], [[0.8415, 0.5403]], [[0.0, 1.0]], [[-0.8415, 0.5403]], [[-0.9093, -0.4161]]])
        self.assertTrue(np.allclose(pos_enc.cpu().detach().numpy(), expected_pos_enc.cpu().detach().numpy(), atol=0.0001))
if __name__ == '__main__':
    unittest.main()