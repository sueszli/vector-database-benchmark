import abc
from collections.abc import Iterable
from typing import Union
import numpy as np
import paddle
from paddle.nn import Layer

class BaseQuanter(Layer, metaclass=abc.ABCMeta):
    """
    Built-in quanters and customized quanters should extend this base quanter
    and implement abstract methods.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()

    @abc.abstractmethod
    def forward(self, input):
        if False:
            print('Hello World!')
        pass

    @abc.abstractmethod
    def scales(self) -> Union[paddle.Tensor, np.ndarray]:
        if False:
            print('Hello World!')
        "\n        Get the scales used for quantization.\n        It can be none which meams the quanter didn't hold scales for quantization.\n        "
        pass

    @abc.abstractmethod
    def zero_points(self) -> Union[paddle.Tensor, np.ndarray]:
        if False:
            i = 10
            return i + 15
        "\n        Get the zero points used for quantization.\n        It can be none which meams the quanter didn't hold zero points for quantization.\n        "
        pass

    @abc.abstractmethod
    def quant_axis(self) -> Union[int, Iterable]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the axis of quantization. None means tensor-wise quantization.\n        '
        pass

    @abc.abstractmethod
    def bit_length(self):
        if False:
            i = 10
            return i + 15
        '\n        Get the bit length of quantization.\n        '
        pass