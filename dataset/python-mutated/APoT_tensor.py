import torch
from torch.ao.quantization.experimental.quantizer import APoTQuantizer

class TensorAPoT:
    quantizer: APoTQuantizer
    data: torch.Tensor

    def __init__(self, quantizer: APoTQuantizer, apot_data: torch.Tensor):
        if False:
            while True:
                i = 10
        self.quantizer = quantizer
        self.data = apot_data

    def int_repr(self):
        if False:
            while True:
                i = 10
        return self.data