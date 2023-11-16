import unittest
import numpy as np
from prim.composite_ops.utils import TOLERANCE
import paddle
from paddle import tensor
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
        self.keepdim = False
        self.axis = None
        self.shape = None

    def set_dtype(self, dtype) -> None:
        if False:
            i = 10
            return i + 15
        self.dtype = dtype

    def set_keepdim(self, keepdim) -> None:
        if False:
            print('Hello World!')
        self.keepdim = keepdim

    def set_axis(self, axis) -> None:
        if False:
            i = 10
            return i + 15
        self.axis = axis

    def set_shape(self, shape) -> None:
        if False:
            while True:
                i = 10
        self.shape = shape

    def get_rtol(self, flag):
        if False:
            return 10
        rtol = TOLERANCE[self.dtype][flag].get('rtol')
        return rtol

    def get_atol(self, flag):
        if False:
            i = 10
            return i + 15
        atol = TOLERANCE[self.dtype][flag].get('atol')
        return atol
attrs = Attr()

def fn(x):
    if False:
        print('Hello World!')
    return tensor.mean(x, axis=attrs.axis, keepdim=attrs.keepdim)

def expect_forward(inputs):
    if False:
        while True:
            i = 10
    return fn(inputs)

class TestCompositeMean(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.dtypes = ['float16', 'float32', 'float64']
        self.keepdim = [False, True]
        self.shapes = [[16, 16, 64, 64], [2, 3, 4], [2, 3]]
        self.axes = [-1, 0, 1]

    def cal_composite(self, inputs):
        if False:
            while True:
                i = 10
        paddle.enable_static()
        core._set_prim_forward_enabled(True)
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data('x', shape=inputs.shape, dtype=str(inputs.dtype))
            y = fn(x)
            blocks = main_program.blocks
            fwd_ops = [op.type for op in blocks[0].ops]
            self.assertTrue('reduce_mean' in fwd_ops)
            primapi.to_prim(blocks)
            fwd_ops_new = [op.type for op in blocks[0].ops]
            self.assertTrue('reduce_mean' not in fwd_ops_new)
        exe = paddle.static.Executor()
        exe.run(startup_program)
        res = exe.run(main_program, feed={'x': inputs}, fetch_list=[y])
        paddle.disable_static()
        core._set_prim_forward_enabled(False)
        return res

    def compare_forward(self):
        if False:
            print('Hello World!')
        np_data = generate_data(attrs.shape, attrs.dtype)
        tensor_data = paddle.to_tensor(np_data)
        expect = expect_forward(tensor_data).numpy()
        actual = self.cal_composite(np_data)[0]
        assert expect.dtype == actual.dtype
        np.testing.assert_allclose(expect, actual, rtol=attrs.get_rtol('forward'), atol=attrs.get_atol('forward'))

    def test_forward(self):
        if False:
            print('Hello World!')
        for i in self.axes:
            for j in self.dtypes:
                for t in self.shapes:
                    for k in self.keepdim:
                        if paddle.device.get_device() == 'cpu' and j == 'float16':
                            print('need pass this case')
                            continue
                        attrs.set_axis(i)
                        attrs.set_dtype(j)
                        attrs.set_shape(t)
                        attrs.set_keepdim(k)
                        self.compare_forward()
if __name__ == '__main__':
    unittest.main()