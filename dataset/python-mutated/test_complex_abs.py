import unittest
import numpy as np
from op_test import OpTest
import paddle
import paddle.base.dygraph as dg

class TestComplexAbsOp(OpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        paddle.enable_static()
        self.python_api = paddle.abs
        self.op_type = 'abs'
        self.dtype = np.float64
        self.shape = (2, 3, 4, 5)
        self.init_input_output()
        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(self.x)}
        self.outputs = {'Out': self.out}

    def init_input_output(self):
        if False:
            return 10
        self.x = np.random.random(self.shape).astype(self.dtype) + 1j * np.random.random(self.shape).astype(self.dtype)
        self.out = np.abs(self.x)

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output()

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        self.check_grad(['X'], 'Out')

class TestComplexAbsOpZeroValues(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        paddle.enable_static()
        self.op_type = 'abs'
        self.python_api = paddle.abs
        self.dtype = np.float64
        self.shape = (2, 3, 4, 5)
        self.init_input_output()
        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(self.x)}
        self.outputs = {'Out': self.out}

    def init_input_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.x = np.zeros(self.shape).astype(self.dtype) + 1j * np.zeros(self.shape).astype(self.dtype)
        self.out = np.abs(self.x)

    def test_check_output(self):
        if False:
            return 10
        self.check_output()

    def test_check_grad(self):
        if False:
            print('Hello World!')
        self.check_grad(['X'], 'Out')

class TestAbs(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self._dtypes = ['float32', 'float64']
        self._places = [paddle.CPUPlace()]
        if paddle.is_compiled_with_cuda():
            self._places.append(paddle.CUDAPlace(0))

    def test_all_positive(self):
        if False:
            for i in range(10):
                print('nop')
        for dtype in self._dtypes:
            x = 1 + 10 * np.random.random([13, 3, 3]).astype(dtype)
            for place in self._places:
                with dg.guard(place):
                    y = paddle.abs(paddle.to_tensor(x))
                    np.testing.assert_allclose(np.abs(x), y.numpy(), rtol=1e-05)

class TestRealAbsOp(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        paddle.enable_static()
        self.python_api = paddle.abs
        self.op_type = 'abs'
        self.dtype = np.float64
        self.shape = (2, 3, 4, 5)
        self.init_input_output()
        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(self.x)}
        self.outputs = {'Out': self.out}

    def init_input_output(self):
        if False:
            return 10
        self.x = 1 + np.random.random(self.shape).astype(self.dtype)
        self.out = np.abs(self.x)

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output()

    def test_check_grad(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_grad(['X'], 'Out')
if __name__ == '__main__':
    unittest.main()