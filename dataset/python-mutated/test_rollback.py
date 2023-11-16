import unittest
import numpy as np
from dygraph_to_static_utils_new import Dy2StTestBase, test_ast_only, test_legacy_and_pir
import paddle
from paddle.jit.dy2static.program_translator import StaticFunction
from paddle.jit.dy2static.utils import func_to_source_code

class Net(paddle.nn.Layer):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.sub = SubNet()

    def forward(self, x):
        if False:
            print('Hello World!')
        x = self.sub(x)
        x = foo(x)
        out = self.sub.bar(x)
        return out

    def infer(self, x):
        if False:
            return 10
        x = self.sub.bar(x)
        out = foo(x)
        return out

class SubNet(paddle.nn.Layer):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()

    def forward(self, x, flag=True):
        if False:
            while True:
                i = 10
        if flag:
            out = x + 1
        else:
            out = x - 1
        return out

    def bar(self, x, flag=True):
        if False:
            while True:
                i = 10
        if flag:
            out = x + 2
        else:
            out = x - 2
        return out

def foo(x, flag=False):
    if False:
        print('Hello World!')
    if flag:
        out = x * 2.0
    else:
        out = x / 2.0
    return out

class TestRollBackPlainFunction(Dy2StTestBase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        paddle.set_device('cpu')

    @test_legacy_and_pir
    def test_plain_func(self):
        if False:
            while True:
                i = 10
        st_foo = paddle.jit.to_static(foo)
        x = paddle.randn([3, 4])
        st_out = st_foo(x)
        self.assertTrue(isinstance(st_foo, StaticFunction))
        st_foo = st_foo.rollback()
        dy_out = st_foo(x)
        self.assertTrue(func_to_source_code(foo) == func_to_source_code(st_foo))
        np.testing.assert_array_equal(st_out.numpy(), dy_out.numpy())

class TestRollBackNet(Dy2StTestBase):

    def setUp(self):
        if False:
            return 10
        paddle.set_device('cpu')

    @test_ast_only
    @test_legacy_and_pir
    def test_net(self):
        if False:
            i = 10
            return i + 15
        net = paddle.jit.to_static(Net())
        x = paddle.randn([3, 4])
        st_fwd_out = net(x)
        self.assertTrue(isinstance(net.forward, StaticFunction))
        self.assertTrue('true_fn' in func_to_source_code(net.sub.forward))
        self.assertFalse('true_fn' in func_to_source_code(net.sub.bar))
        net.infer = paddle.jit.to_static(net.infer)
        st_infer_out = net.infer(x)
        self.assertTrue(isinstance(net.infer, StaticFunction))
        self.assertFalse('true_fn' in func_to_source_code(net.sub.bar))
        net.forward = net.forward.rollback()
        self.assertFalse(isinstance(net.forward, StaticFunction))
        self.assertFalse('true_fn' in func_to_source_code(net.sub.forward))
        dy_fwd_out = net(x)
        np.testing.assert_array_equal(st_fwd_out.numpy(), dy_fwd_out.numpy())
        net.infer.rollback()
        self.assertFalse(isinstance(net.infer, StaticFunction))
        self.assertFalse('true_fn' in func_to_source_code(net.sub.forward))
        dy_infer_out = net.infer(x)
        np.testing.assert_array_equal(st_infer_out.numpy(), dy_infer_out.numpy())

class FuncRollback(paddle.nn.Layer):

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()

    def forward(self, x):
        if False:
            while True:
                i = 10
        return x + 1

    @paddle.jit.to_static
    def func(self, x):
        if False:
            for i in range(10):
                print('nop')
        return x + 2

class TestRollBackNotForward(Dy2StTestBase):

    @test_ast_only
    @test_legacy_and_pir
    def test_rollback(self):
        if False:
            for i in range(10):
                print('nop')
        x = paddle.zeros([2, 2])
        net = FuncRollback()
        out = net.func(x)
        net.func.rollback()
        self.assertTrue(not isinstance(net.func, StaticFunction))
if __name__ == '__main__':
    unittest.main()