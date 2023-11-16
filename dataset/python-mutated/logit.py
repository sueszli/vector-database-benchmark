import numpy as np
from prml.nn.function import Function

class Logit(Function):

    @staticmethod
    def _forward(x):
        if False:
            i = 10
            return i + 15
        return np.arctanh(2 * x - 1) * 2

    @staticmethod
    def _backward(delta, x):
        if False:
            for i in range(10):
                print('nop')
        return delta / x / (1 - x)

def logit(x):
    if False:
        while True:
            i = 10
    return Logit().forward(x)