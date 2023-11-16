import unittest
import numpy as np
from op_test import OpTest, convert_float_to_uint16, skip_check_grad_ci
import paddle
from paddle import base
from paddle.base import core

def l2_norm(x, axis, epsilon):
    if False:
        i = 10
        return i + 15
    x2 = x ** 2
    s = np.sum(x2, axis=axis, keepdims=True)
    r = np.sqrt(s + epsilon)
    y = x / np.broadcast_to(r, x.shape)
    return (y, r)

def norm_wrapper(x, axis=1, epsilon=1e-12, is_test=False):
    if False:
        for i in range(10):
            print('nop')
    return paddle.nn.functional.normalize(x, axis=axis, epsilon=epsilon)

class TestNormOp(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'norm'
        self.python_api = norm_wrapper
        self.init_test_case()
        self.init_dtype()
        x = np.random.random(self.shape).astype(self.dtype)
        (y, norm) = l2_norm(x, self.axis, self.epsilon)
        self.inputs = {'X': x}
        self.attrs = {'epsilon': self.epsilon, 'axis': self.axis}
        self.outputs = {'Out': y, 'Norm': norm}
        self.python_out_sig = ['Out']

    def test_check_output(self):
        if False:
            print('Hello World!')
        self.check_output(check_cinn=True)

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        self.check_grad(['X'], 'Out', check_cinn=True)

    def init_test_case(self):
        if False:
            print('Hello World!')
        self.shape = [2, 3, 4, 5]
        self.axis = 1
        self.epsilon = 1e-08

    def init_dtype(self):
        if False:
            return 10
        self.dtype = 'float64'

class TestNormOp2(TestNormOp):

    def init_test_case(self):
        if False:
            for i in range(10):
                print('nop')
        self.shape = [5, 3, 9, 7]
        self.axis = 0
        self.epsilon = 1e-08

class TestNormOp3(TestNormOp):

    def init_test_case(self):
        if False:
            print('Hello World!')
        self.shape = [5, 3, 2, 7]
        self.axis = -1
        self.epsilon = 1e-08

@skip_check_grad_ci(reason="'check_grad' on large inputs is too slow, " + 'however it is desirable to cover the forward pass')
class TestNormOp4(TestNormOp):

    def init_test_case(self):
        if False:
            i = 10
            return i + 15
        self.shape = [128, 1024, 14, 14]
        self.axis = 2
        self.epsilon = 1e-08

    def test_check_grad(self):
        if False:
            return 10
        pass

@skip_check_grad_ci(reason="'check_grad' on large inputs is too slow, " + 'however it is desirable to cover the forward pass')
class TestNormOp5(TestNormOp):

    def init_test_case(self):
        if False:
            while True:
                i = 10
        self.shape = [2048, 2048]
        self.axis = 1
        self.epsilon = 1e-08

    def test_check_grad(self):
        if False:
            while True:
                i = 10
        pass

class TestNormOp6(TestNormOp):

    def init_dtype(self):
        if False:
            return 10
        self.dtype = 'float32'

    def test_check_grad(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_grad(['X'], 'Out', max_relative_error=0.008, check_cinn=True)

@unittest.skipIf(not base.core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestNormOp7(TestNormOp):

    def init_dtype(self):
        if False:
            while True:
                i = 10
        self.dtype = 'float16'

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output_with_place(base.core.CUDAPlace(0), atol=0.05, check_cinn=True)

    def test_check_grad(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_grad_with_place(base.core.CUDAPlace(0), ['X'], 'Out', max_relative_error=0.05, check_cinn=True)

@skip_check_grad_ci(reason='skip check grad for test mode.')
class TestNormTestOp(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'norm'
        self.python_api = norm_wrapper
        self.init_test_case()
        x = np.random.random(self.shape).astype('float64')
        (y, norm) = l2_norm(x, self.axis, self.epsilon)
        self.inputs = {'X': x}
        self.attrs = {'epsilon': self.epsilon, 'axis': int(self.axis), 'is_test': True}
        self.outputs = {'Out': y}
        self.python_out_sig = ['out']

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(check_dygraph=True, check_cinn=True)

    def test_check_grad(self):
        if False:
            return 10
        pass

    def init_test_case(self):
        if False:
            while True:
                i = 10
        self.shape = [2, 3, 4, 5]
        self.axis = 1
        self.epsilon = 1e-08

@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA and not support the bfloat16')
class TestNormBF16Op(OpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'norm'
        self.python_api = norm_wrapper
        self.init_test_case()
        self.dtype = 'float32'
        x = np.random.random(self.shape).astype(self.dtype)
        (y, norm) = l2_norm(x, self.axis, self.epsilon)
        self.inputs = {'X': convert_float_to_uint16(x)}
        self.attrs = {'epsilon': self.epsilon, 'axis': self.axis}
        self.outputs = {'Out': convert_float_to_uint16(y), 'Norm': norm}
        self.python_out_sig = ['Out']

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output_with_place(core.CUDAPlace(0), atol=0.1, check_cinn=True)

    def test_check_grad(self):
        if False:
            while True:
                i = 10
        self.check_grad_with_place(core.CUDAPlace(0), ['X'], 'Out', max_relative_error=0.01, check_cinn=True)

    def init_test_case(self):
        if False:
            i = 10
            return i + 15
        self.shape = [2, 3, 4, 5]
        self.axis = 1
        self.epsilon = 1e-08

class API_NormTest(unittest.TestCase):

    def test_errors(self):
        if False:
            return 10
        with base.program_guard(base.Program()):

            def test_norm_x_type():
                if False:
                    print('Hello World!')
                data = paddle.static.data(name='x', shape=[3, 3], dtype='int64')
                out = paddle.nn.functional.normalize(data)
            self.assertRaises(TypeError, test_norm_x_type)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()