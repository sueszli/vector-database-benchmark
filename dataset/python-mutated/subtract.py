import numpy as np
from prml.nn.function import Function

class Subtract(Function):
    enable_auto_broadcast = True

    @staticmethod
    def _forward(x, y):
        if False:
            i = 10
            return i + 15
        return x - y

    @staticmethod
    def _backward(delta, x, y):
        if False:
            print('Hello World!')
        return (delta, -delta)

def subtract(x, y):
    if False:
        return 10
    return Subtract().forward(x, y)

def rsubtract(x, y):
    if False:
        while True:
            i = 10
    return Subtract().forward(y, x)