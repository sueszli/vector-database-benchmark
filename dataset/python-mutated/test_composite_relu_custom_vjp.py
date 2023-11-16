import unittest
import numpy as np
from prim.composite_ops.utils import TOLERANCE
import paddle
import paddle.nn.functional as F
from paddle.base import core

def generate_data(shape, dtype='float32'):
    if False:
        while True:
            i = 10
    np_data = np.random.random(shape).astype(dtype)
    return np_data

class Attr:

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self.dtype = None
        self.shape = None

    def set_dtype(self, dtype) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.dtype = dtype

    def set_shape(self, shape) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.shape = shape

    def get_rtol(self, flag):
        if False:
            return 10
        rtol = TOLERANCE[self.dtype][flag].get('rtol')
        return rtol

    def get_atol(self, flag):
        if False:
            for i in range(10):
                print('nop')
        atol = TOLERANCE[self.dtype][flag].get('atol')
        return atol
attrs = Attr()

def fn(x):
    if False:
        print('Hello World!')
    return F.relu(x)

def expect_grad(inputs):
    if False:
        for i in range(10):
            print('nop')
    paddle.disable_static()
    inputs.stop_gradient = False
    res = fn(inputs)
    gradients = paddle.grad(res, inputs)
    return gradients

class TestCompositeReluPrimBackward(unittest.TestCase):
    """test composite relu and prim backward"""

    def setUp(self):
        if False:
            print('Hello World!')
        core._set_prim_backward_enabled(True)
        self.dtypes = ['float16', 'float32', 'float64']
        self.shapes = [[2, 3, 4], [2, 3]]

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
            z = paddle.static.gradients([y], x)
            paddle.incubate.autograd.primapi.to_prim(blocks)
        exe = paddle.static.Executor()
        exe.run(startup_program)
        res = exe.run(main_program, feed={'x': inputs}, fetch_list=[z])
        paddle.disable_static()
        core._set_prim_all_enabled(False)
        return res

    def compare_backward(self):
        if False:
            for i in range(10):
                print('nop')
        np_data = generate_data(attrs.shape)
        tensor_data = paddle.to_tensor(np_data)
        expect = expect_grad(tensor_data)[0].numpy()
        actual = self.cal_composite_grad(np_data)[0]
        assert expect.dtype == actual.dtype
        np.testing.assert_allclose(expect, actual, rtol=attrs.get_rtol('prim_backward'), atol=attrs.get_rtol('prim_backward'))

    def test_prim_backward(self):
        if False:
            for i in range(10):
                print('nop')
        for j in self.dtypes:
            for t in self.shapes:
                attrs.set_dtype(j)
                attrs.set_shape(t)
                self.compare_backward()
if __name__ == '__main__':
    unittest.main()