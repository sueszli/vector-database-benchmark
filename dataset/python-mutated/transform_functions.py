"""Nonlinear Transformation classes


Created on Sat Apr 16 16:06:11 2011

Author: Josef Perktold
License : BSD
"""
import numpy as np

class TransformFunction:

    def __call__(self, x):
        if False:
            i = 10
            return i + 15
        self.func(x)

class SquareFunc(TransformFunction):
    """class to hold quadratic function with inverse function and derivative

    using instance methods instead of class methods, if we want extension
    to parametrized function
    """

    def func(self, x):
        if False:
            for i in range(10):
                print('nop')
        return np.power(x, 2.0)

    def inverseplus(self, x):
        if False:
            return 10
        return np.sqrt(x)

    def inverseminus(self, x):
        if False:
            i = 10
            return i + 15
        return 0.0 - np.sqrt(x)

    def derivplus(self, x):
        if False:
            print('Hello World!')
        return 0.5 / np.sqrt(x)

    def derivminus(self, x):
        if False:
            return 10
        return 0.0 - 0.5 / np.sqrt(x)

class NegSquareFunc(TransformFunction):
    """negative quadratic function

    """

    def func(self, x):
        if False:
            return 10
        return -np.power(x, 2)

    def inverseplus(self, x):
        if False:
            return 10
        return np.sqrt(-x)

    def inverseminus(self, x):
        if False:
            i = 10
            return i + 15
        return 0.0 - np.sqrt(-x)

    def derivplus(self, x):
        if False:
            while True:
                i = 10
        return 0.0 - 0.5 / np.sqrt(-x)

    def derivminus(self, x):
        if False:
            return 10
        return 0.5 / np.sqrt(-x)

class AbsFunc(TransformFunction):
    """class for absolute value transformation
    """

    def func(self, x):
        if False:
            return 10
        return np.abs(x)

    def inverseplus(self, x):
        if False:
            for i in range(10):
                print('nop')
        return x

    def inverseminus(self, x):
        if False:
            while True:
                i = 10
        return 0.0 - x

    def derivplus(self, x):
        if False:
            i = 10
            return i + 15
        return 1.0

    def derivminus(self, x):
        if False:
            for i in range(10):
                print('nop')
        return 0.0 - 1.0

class LogFunc(TransformFunction):

    def func(self, x):
        if False:
            while True:
                i = 10
        return np.log(x)

    def inverse(self, y):
        if False:
            i = 10
            return i + 15
        return np.exp(y)

    def deriv(self, x):
        if False:
            while True:
                i = 10
        return 1.0 / x

class ExpFunc(TransformFunction):

    def func(self, x):
        if False:
            for i in range(10):
                print('nop')
        return np.exp(x)

    def inverse(self, y):
        if False:
            for i in range(10):
                print('nop')
        return np.log(y)

    def deriv(self, x):
        if False:
            print('Hello World!')
        return np.exp(x)

class BoxCoxNonzeroFunc(TransformFunction):

    def __init__(self, lamda):
        if False:
            return 10
        self.lamda = lamda

    def func(self, x):
        if False:
            print('Hello World!')
        return (np.power(x, self.lamda) - 1) / self.lamda

    def inverse(self, y):
        if False:
            for i in range(10):
                print('nop')
        return (self.lamda * y + 1) / self.lamda

    def deriv(self, x):
        if False:
            i = 10
            return i + 15
        return np.power(x, self.lamda - 1)

class AffineFunc(TransformFunction):

    def __init__(self, constant, slope):
        if False:
            for i in range(10):
                print('nop')
        self.constant = constant
        self.slope = slope

    def func(self, x):
        if False:
            return 10
        return self.constant + self.slope * x

    def inverse(self, y):
        if False:
            return 10
        return (y - self.constant) / self.slope

    def deriv(self, x):
        if False:
            while True:
                i = 10
        return self.slope

class ChainFunc(TransformFunction):

    def __init__(self, finn, fout):
        if False:
            return 10
        self.finn = finn
        self.fout = fout

    def func(self, x):
        if False:
            return 10
        return self.fout.func(self.finn.func(x))

    def inverse(self, y):
        if False:
            for i in range(10):
                print('nop')
        return self.f1.inverse(self.fout.inverse(y))

    def deriv(self, x):
        if False:
            i = 10
            return i + 15
        z = self.finn.func(x)
        return self.fout.deriv(z) * self.finn.deriv(x)
if __name__ == '__main__':
    absf = AbsFunc()
    absf.func(5) == 5
    absf.func(-5) == 5
    absf.inverseplus(5) == 5
    absf.inverseminus(5) == -5
    chainf = ChainFunc(AffineFunc(1, 2), BoxCoxNonzeroFunc(2))
    print(chainf.func(3.0))
    chainf2 = ChainFunc(BoxCoxNonzeroFunc(2), AffineFunc(1, 2))
    print(chainf.func(3.0))