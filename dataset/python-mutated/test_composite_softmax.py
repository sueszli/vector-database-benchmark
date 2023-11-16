import unittest
import numpy as np
from prim.composite_ops.utils import TOLERANCE
import paddle
import paddle.nn.functional as F
from paddle.base import core, framework
from paddle.incubate.autograd import primapi

def generate_data(shape, dtype='float32'):
    if False:
        return 10
    np_data = np.random.random(shape).astype(dtype)
    return np_data

class Attr:

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self.dtype = None
        self.axis = -1
        self.shape = None

    def set_dtype(self, dtype) -> None:
        if False:
            print('Hello World!')
        self.dtype = dtype

    def set_axis(self, axis) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.axis = axis

    def set_shape(self, shape) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.shape = shape

    def get_rtol(self, flag):
        if False:
            print('Hello World!')
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
        while True:
            i = 10
    return F.softmax(x, axis=attrs.axis, dtype=attrs.dtype)

def expect_forward(inputs):
    if False:
        return 10
    return fn(inputs)

class TestCompositeSoftmax(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.dtypes = ['float32', 'float64']
        self.shapes = [[], [2, 3, 4], [2, 3]]
        self.axes = [-1, 0, 1]

    def cal_composite(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()
        core._set_prim_forward_enabled(True)
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data('x', shape=inputs.shape, dtype=str(inputs.dtype))
            y = fn(x)
            blocks = main_program.blocks
            fwd_ops = [op.type for op in blocks[0].ops]
            self.assertTrue('softmax' in fwd_ops)
            primapi.to_prim(blocks)
            fwd_ops_new = [op.type for op in blocks[0].ops]
            self.assertTrue('softmax' not in fwd_ops_new)
        exe = paddle.static.Executor()
        exe.run(startup_program)
        res = exe.run(main_program, feed={'x': inputs}, fetch_list=[y])
        paddle.disable_static()
        core._set_prim_forward_enabled(False)
        return res

    def compare_forward(self):
        if False:
            while True:
                i = 10
        if not attrs.shape and attrs.axis not in [-1, 0]:
            return
        np_data = generate_data(attrs.shape)
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
                    attrs.set_axis(i)
                    attrs.set_dtype(j)
                    attrs.set_shape(t)
                    self.compare_forward()

def apply_to_static(net, use_cinn):
    if False:
        i = 10
        return i + 15
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(net, build_strategy=False)

class PrimeNet(paddle.nn.Layer):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.sf = F.softmax

    def forward(self, x, current_axis):
        if False:
            return 10
        out = self.sf(x, axis=current_axis)
        return out

class TestPrimForwardAndBackward(unittest.TestCase):
    """
    Test PrimeNet with @to_static + prim forward + prim backward + cinn v.s Dygraph
    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        paddle.seed(2022)
        self.shapes = [[], [2, 3, 4], [2, 3]]
        self.axes = [-1, 0, 1]

    def train(self, use_prim):
        if False:
            print('Hello World!')
        self.x = paddle.randn(attrs.shape, dtype='float32')
        self.x.stop_gradient = False
        core._set_prim_all_enabled(use_prim)
        paddle.seed(2022)
        net = PrimeNet()
        sgd = paddle.optimizer.SGD(learning_rate=0.1, parameters=net.parameters())
        net = paddle.amp.decorate(models=net, level='O2')
        net = apply_to_static(net, False)
        with paddle.amp.auto_cast(level='O2'):
            out = net(self.x, attrs.axis)
            loss = paddle.mean(out)
            grad = paddle.grad(loss, self.x)
            return (loss, grad)

    def compare_forward(self):
        if False:
            i = 10
            return i + 15
        if not attrs.shape and attrs.axis not in [-1, 0]:
            return
        if not isinstance(framework._current_expected_place(), core.CPUPlace):
            expected = self.train(False)
            actual = self.train(True)
            np.testing.assert_allclose(expected[0], actual[0], rtol=0.001, atol=0.001)
            np.testing.assert_allclose(expected[1], actual[1], rtol=0.001, atol=0.001)

    def test_forward(self):
        if False:
            for i in range(10):
                print('nop')
        for i in self.axes:
            for t in self.shapes:
                attrs.set_axis(i)
                attrs.set_shape(t)
                self.compare_forward()
if __name__ == '__main__':
    unittest.main()