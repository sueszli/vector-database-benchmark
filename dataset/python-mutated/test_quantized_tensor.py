import torch
import unittest
from torch.ao.quantization.experimental.observer import APoTObserver
from torch.ao.quantization.experimental.quantizer import quantize_APoT

class TestQuantizedTensor(unittest.TestCase):
    """ Tests int_repr on APoTQuantizer with random tensor2quantize
    and hard-coded values
    """

    def test_int_repr(self):
        if False:
            for i in range(10):
                print('nop')
        tensor2quantize = tensor2quantize = torch.tensor([0, 0.0215, 0.1692, 0.385, 1, 0.0391])
        observer = APoTObserver(b=4, k=2)
        observer.forward(tensor2quantize)
        qparams = observer.calculate_qparams(signed=False)
        qtensor = quantize_APoT(tensor2quantize=tensor2quantize, alpha=qparams[0], gamma=qparams[1], quantization_levels=qparams[2], level_indices=qparams[3])
        qtensor_data = qtensor.int_repr().int()
        expected_qtensor_data = torch.tensor([0, 3, 8, 13, 5, 12], dtype=torch.int32)
        self.assertTrue(torch.equal(qtensor_data, expected_qtensor_data))
if __name__ == '__main__':
    unittest.main()