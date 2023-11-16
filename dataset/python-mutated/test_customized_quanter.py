import unittest
from typing import Iterable, Union
import numpy as np
import paddle
from paddle.nn import Linear
from paddle.quantization.base_quanter import BaseQuanter
from paddle.quantization.factory import quanter
linear_quant_axis = 1

@quanter('CustomizedQuanter')
class CustomizedQuanterLayer(BaseQuanter):

    def __init__(self, layer, bit_length=8, kwargs1=None):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._layer = layer
        self._bit_length = bit_length
        self._kwargs1 = kwargs1

    def scales(self) -> Union[paddle.Tensor, np.ndarray]:
        if False:
            for i in range(10):
                print('nop')
        return None

    def bit_length(self):
        if False:
            while True:
                i = 10
        return self._bit_length

    def quant_axis(self) -> Union[int, Iterable]:
        if False:
            return 10
        return linear_quant_axis if isinstance(self._layer, Linear) else None

    def zero_points(self) -> Union[paddle.Tensor, np.ndarray]:
        if False:
            return 10
        return None

    def forward(self, input):
        if False:
            return 10
        return input

class TestCustomizedQuanter(unittest.TestCase):

    def test_details(self):
        if False:
            return 10
        layer = Linear(5, 5)
        bit_length = 4
        quanter = CustomizedQuanter(bit_length=bit_length, kwargs1='test')
        quanter = quanter._instance(layer)
        self.assertEqual(quanter.bit_length(), bit_length)
        self.assertEqual(quanter.quant_axis(), linear_quant_axis)
        self.assertEqual(quanter._kwargs1, 'test')
if __name__ == '__main__':
    unittest.main()