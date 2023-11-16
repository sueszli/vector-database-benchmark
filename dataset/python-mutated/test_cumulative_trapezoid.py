import unittest
import numpy as np
from scipy.integrate import cumulative_trapezoid
from test_trapezoid import Testfp16Trapezoid, TestTrapezoidAPI, TestTrapezoidError
import paddle

class TestCumulativeTrapezoidAPI(TestTrapezoidAPI):

    def set_api(self):
        if False:
            for i in range(10):
                print('nop')
        self.ref_api = cumulative_trapezoid
        self.paddle_api = paddle.cumulative_trapezoid

class TestCumulativeTrapezoidWithX(TestCumulativeTrapezoidAPI):

    def set_args(self):
        if False:
            while True:
                i = 10
        self.y = np.array([[2, 4, 8], [3, 5, 9]]).astype('float32')
        self.x = np.array([[1, 2, 3], [3, 4, 5]]).astype('float32')
        self.dx = None
        self.axis = -1

class TestCumulativeTrapezoidAxis(TestCumulativeTrapezoidAPI):

    def set_args(self):
        if False:
            return 10
        self.y = np.array([[2, 4, 8], [3, 5, 9]]).astype('float32')
        self.x = None
        self.dx = 1.0
        self.axis = 0

class TestCumulativeTrapezoidWithDx(TestCumulativeTrapezoidAPI):

    def set_args(self):
        if False:
            i = 10
            return i + 15
        self.y = np.array([[2, 4, 8], [3, 5, 9]]).astype('float32')
        self.x = None
        self.dx = 3.0
        self.axis = -1

class TestCumulativeTrapezoidfloat64(TestCumulativeTrapezoidAPI):

    def set_args(self):
        if False:
            while True:
                i = 10
        self.y = np.array([[2, 4, 8], [3, 5, 9]]).astype('float64')
        self.x = np.array([[1, 2, 3], [3, 4, 5]]).astype('float64')
        self.dx = None
        self.axis = -1

class TestCumulativeTrapezoidWithOutDxX(TestCumulativeTrapezoidAPI):

    def set_args(self):
        if False:
            i = 10
            return i + 15
        self.y = np.array([[2, 4, 8], [3, 5, 9]]).astype('float64')
        self.x = None
        self.dx = None
        self.axis = -1

class TestCumulativeTrapezoidBroadcast(TestCumulativeTrapezoidAPI):

    def set_args(self):
        if False:
            while True:
                i = 10
        self.y = np.random.random((3, 3, 4)).astype('float32')
        self.x = np.random.random(3).astype('float32')
        self.dx = None
        self.axis = 1

class TestCumulativeTrapezoidAxis1(TestCumulativeTrapezoidAPI):

    def set_args(self):
        if False:
            for i in range(10):
                print('nop')
        self.y = np.random.random((3, 3, 4)).astype('float32')
        self.x = None
        self.dx = 1
        self.axis = 1

class TestCumulativeTrapezoidError(TestTrapezoidError):

    def set_api(self):
        if False:
            while True:
                i = 10
        self.paddle_api = paddle.cumulative_trapezoid

class Testfp16CumulativeTrapezoid(Testfp16Trapezoid):

    def set_api(self):
        if False:
            while True:
                i = 10
        self.paddle_api = paddle.cumulative_trapezoid
        self.ref_api = cumulative_trapezoid
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()