import unittest
import numpy as np
from op_test import OpTest, convert_float_to_uint16
import paddle
from paddle.base import core
np.random.seed(10)

def logit(x, eps):
    if False:
        return 10
    x_min = np.minimum(x, 1.0 - eps)
    x_max = np.maximum(x_min, eps)
    return np.log(x_max / (1.0 - x_max))

def logit_grad(x, eps=1e-08):
    if False:
        return 10
    tmp_x = np.select([x < eps, x > 1.0 - eps], [x * 0.0, x * 0.0], default=-1.0)
    x_1 = 1.0 - x
    _x = np.select([tmp_x == -1.0], [np.reciprocal(x * x_1)], default=0.0)
    dout = np.full_like(x, fill_value=1.0 / _x.size)
    dx = dout * _x
    return dx

class TestLogitOp(OpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'logit'
        self.python_api = paddle.logit
        self.set_attrs()
        x = np.random.uniform(-1.0, 1.0, self.shape).astype(self.dtype)
        out = logit(x, self.eps)
        self.x_grad = logit_grad(x, self.eps)
        self.inputs = {'X': x}
        self.outputs = {'Out': out}
        self.attrs = {'eps': self.eps}

    def set_attrs(self):
        if False:
            i = 10
            return i + 15
        self.dtype = np.float64
        self.shape = [120]
        self.eps = 1e-08

    def test_check_output(self):
        if False:
            print('Hello World!')
        self.check_output(check_pir=True)

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        self.check_grad(['X'], ['Out'], user_defined_grads=[self.x_grad], check_pir=True)

class TestLogitOpFp32(TestLogitOp):

    def set_attrs(self):
        if False:
            while True:
                i = 10
        self.dtype = np.float32
        self.shape = [120]
        self.eps = 1e-08

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output(check_pir=True)

    def test_check_grad(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_grad(['X'], ['Out'], user_defined_grads=[self.x_grad], check_pir=True)

class TestLogitOpFp16(TestLogitOp):

    def set_attrs(self):
        if False:
            i = 10
            return i + 15
        self.dtype = np.float16
        self.shape = [120]
        self.eps = 1e-08

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(check_pir=True)

    def test_check_grad(self):
        if False:
            while True:
                i = 10
        self.check_grad(['X'], ['Out'], user_defined_grads=[self.x_grad], check_pir=True)

@unittest.skipIf(not core.is_compiled_with_cuda() or not core.is_bfloat16_supported(core.CUDAPlace(0)), 'core is not compiled with CUDA and not support the bfloat16')
class TestLogitOpBf16(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'logit'
        self.python_api = paddle.logit
        self.set_attrs()
        x = np.random.uniform(-0.5, 0.5, self.shape).astype(np.float32)
        out = logit(x, self.eps)
        self.x_grad = logit_grad(x, self.eps)
        self.inputs = {'X': convert_float_to_uint16(x)}
        self.outputs = {'Out': convert_float_to_uint16(out)}
        self.attrs = {'eps': self.eps}

    def set_attrs(self):
        if False:
            while True:
                i = 10
        self.dtype = np.uint16
        self.shape = [120]
        self.eps = 1e-08

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, check_pir=True)

    def test_check_grad(self):
        if False:
            print('Hello World!')
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_grad_with_place(place, ['X'], ['Out'], user_defined_grads=[self.x_grad], check_pir=True)

class TestLogitShape(TestLogitOp):

    def set_attrs(self):
        if False:
            print('Hello World!')
        self.dtype = np.float64
        self.shape = [2, 60]
        self.eps = 1e-08

class TestLogitEps(TestLogitOp):

    def set_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        self.dtype = np.float32
        self.shape = [120]
        self.eps = 1e-08

class TestLogitAPI(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.x_shape = [120]
        self.x = np.random.uniform(0.0, 1.0, self.x_shape).astype(np.float32)
        self.place = paddle.CUDAPlace(0) if paddle.base.core.is_compiled_with_cuda() else paddle.CPUPlace()

    def check_api(self, eps=1e-08):
        if False:
            i = 10
            return i + 15
        ref_out = logit(self.x, eps)
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name='x', shape=self.x_shape)
            y = paddle.logit(x, eps)
            exe = paddle.static.Executor(self.place)
            out = exe.run(feed={'x': self.x}, fetch_list=[y])
        np.testing.assert_allclose(out[0], ref_out, rtol=1e-05)
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        y = paddle.logit(x, 1e-08)
        np.testing.assert_allclose(y.numpy(), ref_out, rtol=1e-05)
        paddle.enable_static()

    def test_check_api(self):
        if False:
            print('Hello World!')
        paddle.enable_static()
        for eps in [1e-06, 0.0]:
            self.check_api(eps)

    def test_errors(self):
        if False:
            while True:
                i = 10
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name='X1', shape=[100], dtype='int32')
            self.assertRaises(TypeError, paddle.logit, x)
            x = paddle.static.data(name='X2', shape=[100], dtype='float32')
            self.assertRaises(TypeError, paddle.logit, x, dtype='int32')
if __name__ == '__main__':
    unittest.main()