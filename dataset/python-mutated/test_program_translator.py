import inspect
import textwrap
import unittest
import astor
import numpy as np
from dygraph_to_static_utils_new import Dy2StTestBase, test_ast_only
from ifelse_simple_func import dyfunc_with_if_else_early_return1, dyfunc_with_if_else_early_return2
import paddle
import paddle.jit.dy2static as _jst
from paddle import base
from paddle.jit.api import to_static
from paddle.jit.dy2static.utils import func_to_source_code
from paddle.utils import gast
np.random.seed(0)

def simple_func(x, weight_numpy):
    if False:
        i = 10
        return i + 15
    x = base.dygraph.to_variable(x)
    w = base.dygraph.to_variable(weight_numpy)
    y = paddle.matmul(x, w)
    z = paddle.mean(y)
    return z

@to_static
def decorated_simple_func(x, weight_numpy):
    if False:
        i = 10
        return i + 15
    x = base.dygraph.to_variable(x)
    w = base.dygraph.to_variable(weight_numpy)
    y = paddle.matmul(x, w)
    z = paddle.mean(y)
    return z

def get_source_code(func):
    if False:
        return 10
    raw_code = inspect.getsource(func)
    code = textwrap.dedent(raw_code)
    root = gast.parse(code)
    source_code = astor.to_source(gast.gast_to_ast(root))
    return source_code

class StaticCode1:

    def dyfunc_with_if_else(x_v, label=None):
        if False:
            for i in range(10):
                print('nop')
        loss = _jst.UndefinedVar('loss')
        __return_1 = _jst.UndefinedVar('__return_1')
        __return_0 = _jst.UndefinedVar('__return_0')
        __return_value_0 = None

        def get_args_0():
            if False:
                i = 10
                return i + 15
            nonlocal x_v
            return (x_v,)

        def set_args_0(__args):
            if False:
                i = 10
                return i + 15
            nonlocal x_v
            (x_v,) = __args

        def true_fn_0():
            if False:
                return 10
            nonlocal x_v
            x_v = x_v - 1
            return

        def false_fn_0():
            if False:
                for i in range(10):
                    print('nop')
            nonlocal x_v
            x_v = x_v + 1
            return
        _jst.IfElse(paddle.mean(x_v)[0] > 5, true_fn_0, false_fn_0, get_args_0, set_args_0, ('x_v',), push_pop_names=None)

        def get_args_1():
            if False:
                print('Hello World!')
            nonlocal __return_0, __return_1, __return_value_0, loss
            return (__return_0, __return_1, __return_value_0, loss)

        def set_args_1(__args):
            if False:
                while True:
                    i = 10
            nonlocal __return_0, __return_1, __return_value_0, loss
            (__return_0, __return_1, __return_value_0, loss) = __args

        def true_fn_1():
            if False:
                for i in range(10):
                    print('nop')
            nonlocal __return_0, __return_1, __return_value_0, loss
            loss = paddle.nn.functional.cross_entropy(x_v, label, reduction='none', use_softmax=False)
            __return_0 = _jst.create_bool_as_type(label is not None, True)
            __return_value_0 = loss
            return

        def false_fn_1():
            if False:
                return 10
            nonlocal __return_0, __return_1, __return_value_0, loss
            __return_1 = _jst.create_bool_as_type(label is not None, True)
            __return_value_0 = x_v
            return
        _jst.IfElse(label is not None, true_fn_1, false_fn_1, get_args_1, set_args_1, ('__return_0', '__return_1', '__return_value_0', 'loss'), push_pop_names=None)
        return __return_value_0

class StaticCode2:

    def dyfunc_with_if_else(x_v, label=None):
        if False:
            i = 10
            return i + 15
        loss = _jst.UndefinedVar('loss')
        __return_3 = _jst.UndefinedVar('__return_3')
        __return_2 = _jst.UndefinedVar('__return_2')
        __return_value_1 = None

        def get_args_2():
            if False:
                print('Hello World!')
            nonlocal x_v
            return (x_v,)

        def set_args_2(__args):
            if False:
                return 10
            nonlocal x_v
            (x_v,) = __args

        def true_fn_2():
            if False:
                while True:
                    i = 10
            nonlocal x_v
            x_v = x_v - 1
            return

        def false_fn_2():
            if False:
                i = 10
                return i + 15
            nonlocal x_v
            x_v = x_v + 1
            return
        _jst.IfElse(paddle.mean(x_v)[0] > 5, true_fn_2, false_fn_2, get_args_2, set_args_2, ('x_v',), push_pop_names=None)

        def get_args_3():
            if False:
                for i in range(10):
                    print('nop')
            nonlocal __return_2, __return_3, __return_value_1, loss
            return (__return_2, __return_3, __return_value_1, loss)

        def set_args_3(__args):
            if False:
                i = 10
                return i + 15
            nonlocal __return_2, __return_3, __return_value_1, loss
            (__return_2, __return_3, __return_value_1, loss) = __args

        def true_fn_3():
            if False:
                while True:
                    i = 10
            nonlocal __return_2, __return_3, __return_value_1, loss
            loss = paddle.nn.functional.cross_entropy(x_v, label, reduction='none', use_softmax=False)
            __return_2 = _jst.create_bool_as_type(label is not None, True)
            __return_value_1 = loss
            return

        def false_fn_3():
            if False:
                i = 10
                return i + 15
            nonlocal __return_2, __return_3, __return_value_1, loss
            __return_3 = _jst.create_bool_as_type(label is not None, True)
            __return_value_1 = x_v
            return
        _jst.IfElse(label is not None, true_fn_3, false_fn_3, get_args_3, set_args_3, ('__return_2', '__return_3', '__return_value_1', 'loss'), push_pop_names=None)
        return __return_value_1

class NetWithError(paddle.nn.Layer):

    @to_static(full_graph=True)
    def forward(self, x):
        if False:
            print('Hello World!')
        linear = paddle.nn.Linear(32, 64)
        y = linear(x)
        return y

class TestEnableDeclarative(Dy2StTestBase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.x = np.random.randn(30, 10, 32).astype('float32')
        self.weight = np.random.randn(32, 64).astype('float32')

    @test_ast_only
    def test_raise_error(self):
        if False:
            print('Hello World!')
        with base.dygraph.guard():
            paddle.jit.enable_to_static(True)
            net = NetWithError()
            with self.assertRaises(ValueError):
                net(base.dygraph.to_variable(self.x))

    def test_enable_disable_declarative(self):
        if False:
            print('Hello World!')
        paddle.jit.enable_to_static(True)
        with base.dygraph.guard():
            static_output = decorated_simple_func(self.x, self.weight)
        paddle.jit.enable_to_static(False)
        with base.dygraph.guard():
            dygraph_output = decorated_simple_func(self.x, self.weight)
            np.testing.assert_allclose(static_output.numpy(), dygraph_output.numpy(), rtol=1e-05, atol=0.0001)

class Net(paddle.nn.Layer):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        return x + 1

class SwitchModeNet(paddle.nn.Layer):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()

    @paddle.jit.to_static
    def forward(self, x):
        if False:
            while True:
                i = 10
        return x + 1

    @paddle.jit.to_static
    def foo(self):
        if False:
            print('Hello World!')
        return True

@paddle.jit.to_static(full_graph=True)
def switch_mode_function():
    if False:
        while True:
            i = 10
    return True

class TestFunctionTrainEvalMode(Dy2StTestBase):

    @test_ast_only
    def test_switch_mode(self):
        if False:
            print('Hello World!')
        paddle.disable_static()
        switch_mode_function.eval()
        switch_mode_function()
        self.assertEqual(switch_mode_function._training, False)
        (_, partial_layer) = switch_mode_function.program_cache.last()[-1]
        self.assertEqual(partial_layer.training, False)
        switch_mode_function.train()
        switch_mode_function()
        self.assertEqual(switch_mode_function._training, True)
        (_, partial_layer) = switch_mode_function.program_cache.last()[-1]
        self.assertEqual(partial_layer.training, True)

    def test_raise_error(self):
        if False:
            i = 10
            return i + 15
        paddle.disable_static()
        net = SwitchModeNet()
        self.assertEqual(net.training, True)
        with self.assertRaises(RuntimeError):
            net.forward.eval()
        net.eval()
        self.assertEqual(net.training, False)
        with self.assertRaises(RuntimeError):
            net.foo.train()

class TestIfElseEarlyReturn(Dy2StTestBase):

    def test_ifelse_early_return1(self):
        if False:
            return 10
        answer = np.zeros([2, 2]) + 1
        static_func = paddle.jit.to_static(dyfunc_with_if_else_early_return1)
        out = static_func()
        np.testing.assert_allclose(answer, out[0].numpy(), rtol=1e-05)

    def test_ifelse_early_return2(self):
        if False:
            i = 10
            return i + 15
        answer = np.zeros([2, 2]) + 3
        static_func = paddle.jit.to_static(dyfunc_with_if_else_early_return2)
        out = static_func()
        np.testing.assert_allclose(answer, out[0].numpy(), rtol=1e-05)

class TestRemoveCommentInDy2St(Dy2StTestBase):

    def func_with_comment(self):
        if False:
            print('Hello World!')
        x = paddle.to_tensor([1, 2, 3])
        y = paddle.to_tensor([4, 5, 6])

    def test_remove_comment(self):
        if False:
            i = 10
            return i + 15
        code_string = func_to_source_code(self.func_with_comment)
        self.assertEqual('#' not in code_string, True)

class Obj:

    def __init__(self):
        if False:
            print('Hello World!')
        pass

    def func(self, x):
        if False:
            i = 10
            return i + 15
        return x + 1
obj = Obj()

class Net2:

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.layer1 = paddle.nn.Linear(10, 10)

    def forward(self, data):
        if False:
            i = 10
            return i + 15

        @paddle.jit.to_static
        def func(ins, x, loss_fn):
            if False:
                return 10
            x = ins.layer1(x)
            return loss_fn(x)

        def func1(x):
            if False:
                while True:
                    i = 10
            return func(self, x, obj.func)
        return func1(data)

class TestParameterRecorder(Dy2StTestBase):

    def test_recorder(self):
        if False:
            return 10
        'function calls nn.Layer case.'
        net = Net()
        x = paddle.randn([5, 10])
        out = net.forward(x)
if __name__ == '__main__':
    unittest.main()