import unittest
import numpy as np
from prim.composite_ops.utils import TOLERANCE
np.random.seed(2013)
import paddle
import paddle.nn.functional as F
from paddle.base import core
from paddle.incubate.autograd import primapi

def generate_data(shape, dtype='float32'):
    if False:
        i = 10
        return i + 15
    np_data = np.random.random(shape).astype(dtype)
    return np_data

class Attr:

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        self.dtype = 'float32'
        self.shape = None
        self.approximate = False

    def set_dtype(self, dtype) -> None:
        if False:
            i = 10
            return i + 15
        self.dtype = dtype

    def set_shape(self, shape) -> None:
        if False:
            i = 10
            return i + 15
        self.shape = shape

    def set_approximate(self, approximate) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.approximate = approximate

    def get_rtol(self, flag):
        if False:
            print('Hello World!')
        rtol = TOLERANCE[self.dtype][flag].get('rtol')
        return rtol

    def get_atol(self, flag):
        if False:
            while True:
                i = 10
        atol = TOLERANCE[self.dtype][flag].get('atol')
        return atol
attrs = Attr()

def fn(x):
    if False:
        print('Hello World!')
    return F.gelu(x, approximate=attrs.approximate)

def expect_grad(inputs):
    if False:
        return 10
    paddle.disable_static()
    inputs.stop_gradient = False
    res = fn(inputs)
    gradients = paddle.grad(res, inputs)
    return gradients

class TestCompositeGelu(unittest.TestCase):
    """test composite gelu: prim forward"""

    def setUp(self):
        if False:
            while True:
                i = 10
        self.dtypes = ['float16', 'float32', 'float64']
        self.shapes = [[16, 16, 64, 64], [2, 3, 4], [2, 3]]
        self.approximates = [True, False]

    def cal_composite_grad(self, inputs):
        if False:
            print('Hello World!')
        paddle.enable_static()
        core._set_prim_forward_enabled(True)
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data('x', shape=inputs.shape, dtype=str(inputs.dtype))
            x.stop_gradient = False
            y = fn(x)
            blocks = main_program.blocks
            fwd_ops = [op.type for op in blocks[0].ops]
            self.assertTrue('gelu' in fwd_ops)
            primapi.to_prim(blocks)
            fwd_ops_new = [op.type for op in blocks[0].ops]
            self.assertTrue('gelu' not in fwd_ops_new)
            z = paddle.static.gradients([y], x)
            fwd_ops_grad = [op.type for op in blocks[0].ops]
            self.assertTrue('gelu_grad' not in fwd_ops_grad)
        exe = paddle.static.Executor()
        exe.run(startup_program)
        res = exe.run(main_program, feed={'x': inputs}, fetch_list=[z])
        paddle.disable_static()
        core._set_prim_forward_enabled(False)
        return res

    def compare_backward(self):
        if False:
            return 10
        np_data = generate_data(attrs.shape, attrs.dtype)
        tensor_data = paddle.to_tensor(np_data)
        expect = expect_grad(tensor_data)[0].numpy()
        actual = self.cal_composite_grad(np_data)[0]
        assert expect.dtype == actual.dtype
        np.testing.assert_allclose(expect, actual, rtol=attrs.get_rtol('backward'), atol=attrs.get_atol('backward'))

    def test_backward(self):
        if False:
            for i in range(10):
                print('nop')
        for i in self.approximates:
            for j in self.dtypes:
                for t in self.shapes:
                    if paddle.device.get_device() == 'cpu' and j == 'float16':
                        print('need pass this case')
                        continue
                    attrs.set_approximate(i)
                    attrs.set_dtype(j)
                    attrs.set_shape(t)
                    self.compare_backward()

class TestCompositeGeluPrimBackward(unittest.TestCase):
    """test composite gelu: prim forward and backward"""

    def setUp(self):
        if False:
            while True:
                i = 10
        self.dtypes = ['float16', 'float32', 'float64']
        self.shapes = [[16, 16, 64, 64], [2, 3, 4], [2, 3]]
        self.approximates = [True, False]

    def cal_composite_grad(self, inputs):
        if False:
            while True:
                i = 10
        paddle.enable_static()
        core._set_prim_all_enabled(True)
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data('x', shape=inputs.shape, dtype=str(inputs.dtype))
            x.stop_gradient = False
            y = fn(x)
            blocks = main_program.blocks
            primapi.to_prim(blocks)
            z = paddle.static.gradients([y], x)
        exe = paddle.static.Executor()
        exe.run(startup_program)
        res = exe.run(main_program, feed={'x': inputs}, fetch_list=[z])
        paddle.disable_static()
        core._set_prim_all_enabled(False)
        return res

    def compare_backward(self):
        if False:
            return 10
        np_data = generate_data(attrs.shape, attrs.dtype)
        tensor_data = paddle.to_tensor(np_data)
        expect = expect_grad(tensor_data)[0].numpy()
        actual = self.cal_composite_grad(np_data)[0]
        assert expect.dtype == actual.dtype
        np.testing.assert_allclose(expect, actual, rtol=attrs.get_rtol('prim_backward'), atol=attrs.get_rtol('prim_backward'))

    def test_prim_backward(self):
        if False:
            print('Hello World!')
        for i in self.approximates:
            for j in self.dtypes:
                for t in self.shapes:
                    if paddle.device.get_device() == 'cpu' and j == 'float16':
                        print('need pass this case')
                        continue
                    attrs.set_approximate(i)
                    attrs.set_dtype(j)
                    attrs.set_shape(t)
                    self.compare_backward()
if __name__ == '__main__':
    unittest.main()