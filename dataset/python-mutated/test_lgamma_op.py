import math
import unittest
import numpy as np
from op_test import OpTest, convert_float_to_uint16
from scipy import special
import paddle
from paddle.base import core
paddle.enable_static()

class TestLgammaOp(OpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'lgamma'
        self.python_api = paddle.lgamma
        self.init_dtype_type()
        shape = (5, 20)
        data = np.random.random(shape).astype(self.dtype) + 1
        self.inputs = {'X': data}
        result = np.ones(shape).astype(self.dtype)
        for i in range(shape[0]):
            for j in range(shape[1]):
                result[i][j] = math.lgamma(data[i][j])
        self.outputs = {'Out': result}

    def init_dtype_type(self):
        if False:
            for i in range(10):
                print('nop')
        self.dtype = np.float64

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        if False:
            while True:
                i = 10
        self.check_grad(['X'], 'Out', numeric_grad_delta=1e-07, check_pir=True)

class TestLgammaOpFp32(TestLgammaOp):

    def init_dtype_type(self):
        if False:
            while True:
                i = 10
        self.dtype = np.float32

    def test_check_grad_normal(self):
        if False:
            return 10
        self.check_grad(['X'], 'Out', numeric_grad_delta=0.005, check_pir=True)

class TestLgammaFP16Op(TestLgammaOp):

    def init_dtype_type(self):
        if False:
            print('Hello World!')
        self.dtype = np.float16

    def test_check_grad_normal(self):
        if False:
            return 10
        self.check_grad(['X'], 'Out', check_pir=True)

@unittest.skipIf(not core.is_compiled_with_cuda() or not core.is_bfloat16_supported(core.CUDAPlace(0)), 'core is not compiled with CUDA or not support bfloat16')
class TestLgammaBF16Op(OpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'lgamma'
        self.python_api = paddle.lgamma
        self.dtype = np.uint16
        shape = (5, 20)
        data = np.random.random(shape).astype('float32') + 1
        self.inputs = {'X': convert_float_to_uint16(data)}
        result = np.ones(shape).astype('float32')
        for i in range(shape[0]):
            for j in range(shape[1]):
                result[i][j] = math.lgamma(data[i][j])
        self.outputs = {'Out': convert_float_to_uint16(result)}

    def test_check_output(self):
        if False:
            print('Hello World!')
        self.check_output_with_place(core.CUDAPlace(0), check_pir=True)

    def test_check_grad_normal(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_grad_with_place(core.CUDAPlace(0), ['X'], 'Out', check_pir=True)

class TestLgammaOpApi(unittest.TestCase):

    def test_lgamma(self):
        if False:
            while True:
                i = 10
        paddle.disable_static()
        self.dtype = 'float32'
        shape = (1, 4)
        data = np.random.random(shape).astype(self.dtype) + 1
        data_ = paddle.to_tensor(data)
        out = paddle.lgamma(data_)
        result = special.gammaln(data)
        np.testing.assert_allclose(result, out.numpy(), rtol=1e-05)
        paddle.enable_static()
if __name__ == '__main__':
    unittest.main()