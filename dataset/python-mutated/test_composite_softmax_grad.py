import unittest
import numpy as np
from prim.composite_ops.utils import TOLERANCE
import paddle
import paddle.nn.functional as F
from paddle.base import core
from paddle.incubate.autograd import primapi

def generate_data(shape, dtype='float32'):
    if False:
        return 10
    np_data = np.random.random(shape).astype(dtype)
    return np_data

class Attr:

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self.dtype = None
        self.axis = -1
        self.shape = None

    def set_dtype(self, dtype) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.dtype = dtype

    def set_axis(self, axis) -> None:
        if False:
            while True:
                i = 10
        self.axis = axis

    def set_shape(self, shape) -> None:
        if False:
            while True:
                i = 10
        self.shape = shape

    def get_rtol(self, flag):
        if False:
            print('Hello World!')
        rtol = TOLERANCE[self.dtype][flag].get('rtol')
        return rtol

    def get_atol(self, flag):
        if False:
            print('Hello World!')
        atol = TOLERANCE[self.dtype][flag].get('atol')
        return atol
attrs = Attr()

def fn(x):
    if False:
        for i in range(10):
            print('nop')
    return F.softmax(x, axis=attrs.axis, dtype=attrs.dtype)

def expect_grad(inputs):
    if False:
        for i in range(10):
            print('nop')
    paddle.disable_static()
    inputs.stop_gradient = False
    res = fn(inputs)
    gradients = paddle.grad(res, inputs)
    return gradients

class TestCompositeSoftmax(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.dtypes = ['float32', 'float64']
        self.shapes = [[2, 3, 4], [2, 3]]
        self.axes = [-1, 0, 1]

    def cal_composite_grad(self, inputs):
        if False:
            return 10
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
            self.assertTrue('softmax' in fwd_ops)
            primapi.to_prim(blocks)
            fwd_ops_new = [op.type for op in blocks[0].ops]
            self.assertTrue('softmax' not in fwd_ops_new)
            z = paddle.static.gradients([y], x)
            fwd_ops_grad = [op.type for op in blocks[0].ops]
            self.assertTrue('softmax_grad' not in fwd_ops_grad)
        exe = paddle.static.Executor()
        exe.run(startup_program)
        res = exe.run(main_program, feed={'x': inputs}, fetch_list=[z])
        paddle.disable_static()
        core._set_prim_forward_enabled(False)
        return res

    def compare_backward(self):
        if False:
            i = 10
            return i + 15
        np_data = generate_data(attrs.shape)
        tensor_data = paddle.to_tensor(np_data)
        expect = expect_grad(tensor_data)[0].numpy()
        actual = self.cal_composite_grad(np_data)[0]
        assert expect.dtype == actual.dtype
        np.testing.assert_allclose(expect, actual, rtol=attrs.get_rtol('backward'), atol=attrs.get_atol('backward'))

    def test_backward(self):
        if False:
            print('Hello World!')
        for i in self.axes:
            for j in self.dtypes:
                for t in self.shapes:
                    attrs.set_axis(i)
                    attrs.set_dtype(j)
                    attrs.set_shape(t)
                    self.compare_backward()

class TestCompositeSoftmaxPrimBackward(unittest.TestCase):
    """test composite softmax and prim backward"""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        core._set_prim_backward_enabled(True)
        self.dtypes = ['float32', 'float64']
        self.shapes = [[], [2, 3, 4], [2, 3]]
        self.axes = [-1, 0, 1]

    def cal_composite_grad(self, inputs):
        if False:
            return 10
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
        if not attrs.shape and attrs.axis not in [-1, 0]:
            return
        np_data = generate_data(attrs.shape)
        tensor_data = paddle.to_tensor(np_data)
        expect = expect_grad(tensor_data)[0].numpy()
        actual = self.cal_composite_grad(np_data)[0]
        assert expect.dtype == actual.dtype
        np.testing.assert_allclose(expect, actual, rtol=attrs.get_rtol('prim_backward'), atol=attrs.get_rtol('prim_backward'))

    def test_prim_backward(self):
        if False:
            while True:
                i = 10
        for i in self.axes:
            for j in self.dtypes:
                for t in self.shapes:
                    attrs.set_axis(i)
                    attrs.set_dtype(j)
                    attrs.set_shape(t)
                    self.compare_backward()
if __name__ == '__main__':
    unittest.main()