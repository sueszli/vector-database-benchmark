import os
import tempfile
import unittest
import numpy as np
from dygraph_to_static_utils_new import Dy2StTestBase, test_ast_only, test_legacy_and_pir
from test_basic_api_transformation import dyfunc_to_variable
import paddle
from paddle import base
from paddle.base.dygraph import to_variable
from paddle.jit.api import to_static
from paddle.jit.dy2static.program_translator import ConcreteProgram, StaticFunction
from paddle.nn import Layer
from paddle.static import InputSpec

class SimpleNet(Layer):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.linear = paddle.nn.Linear(10, 3)

    @to_static(input_spec=[InputSpec(shape=[None, 10], dtype='float32')], full_graph=True)
    def forward(self, x, a=1, b=2):
        if False:
            for i in range(10):
                print('nop')
        y = self.inner_function(x)
        return y

    @to_static(full_graph=True)
    def inner_function(self, x):
        if False:
            return 10
        y = self.linear(x)
        return y

    def add_func(self, x, y):
        if False:
            print('Hello World!')
        z = x + y
        return z

    @to_static(input_spec=[[InputSpec([None, 10]), InputSpec([None, 10])]], full_graph=True)
    def func_with_list(self, l, int_val=1):
        if False:
            i = 10
            return i + 15
        (x, y) = l
        z = x + y
        z = z + int_val
        return z

    @to_static(input_spec=[{'x': InputSpec([None, 10]), 'y': InputSpec([None, 10])}], full_graph=True)
    def func_with_dict(self, d):
        if False:
            print('Hello World!')
        x = d['x']
        y = d['y']
        z = x + y
        return z

    @to_static(input_spec=[[InputSpec([None]), {'x': InputSpec([None, 10]), 'y': InputSpec([None, 10])}]], full_graph=True)
    def func_with_list_dict(self, dl):
        if False:
            while True:
                i = 10
        bias = dl[0]
        x = dl[1]['x']
        y = dl[1]['y']
        z = x + y
        z = z + bias
        return z

class TestStaticFunctionInstance(Dy2StTestBase):

    def test_instance_same_class(self):
        if False:
            return 10
        with base.dygraph.guard(base.CPUPlace()):
            net_1 = SimpleNet()
            net_2 = SimpleNet()
            self.assertTrue(isinstance(net_1.forward, StaticFunction))
            self.assertTrue(isinstance(net_2.forward, StaticFunction))
            self.assertNotEqual(net_1.forward, net_2.forward)
            net_1.forward.concrete_program
            self.assertTrue(len(net_1.forward.program_cache) == 1)
            self.assertTrue(len(net_2.forward.program_cache) == 0)

class TestInputSpec(Dy2StTestBase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.temp_dir.name, 'simple_net')

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.temp_dir.cleanup()

    @test_legacy_and_pir
    @test_ast_only
    def test_with_input_spec(self):
        if False:
            print('Hello World!')
        with base.dygraph.guard(base.CPUPlace()):
            x = to_variable(np.ones([4, 10]).astype('float32'))
            y = to_variable(np.ones([4, 10]).astype('float32') * 2)
            int_val = 4.0
            net = SimpleNet()
            out = net(x)
            self.assertTrue(len(net.forward.program_cache) == 1)
            net.inner_function(x)
            paddle.jit.save(net, self.model_path)
            infer_net = paddle.jit.load(self.model_path)
            pred = infer_net(x)
            np.testing.assert_allclose(out.numpy(), pred.numpy(), rtol=1e-05)
            x_2 = to_variable(np.ones([4, 20]).astype('float32'))
            net.add_func = to_static(net.add_func)
            out = net.add_func(x_2, np.ones([20]).astype('float32'))
            self.assertTrue(len(net.add_func.program_cache) == 1)
            out = net.func_with_list([x, y], int_val)
            out = net.func_with_dict({'x': x, 'y': y})
            int_np = np.ones([1]).astype('float32')
            out = net.func_with_list_dict([int_np, {'x': x, 'y': y}])

    def test_with_error(self):
        if False:
            while True:
                i = 10
        with base.dygraph.guard(base.CPUPlace()):
            x = to_variable(np.ones([4, 10]).astype('float32'))
            y = to_variable(np.ones([4, 10]).astype('float32') * 2)
            int_val = 4.0
            net = SimpleNet()
            with self.assertRaises(ValueError):
                net(x, a=1, other_kwarg=2)
            with self.assertRaises(ValueError):
                net.add_func = to_static(net.add_func, input_spec=[InputSpec([-1, 10]), InputSpec([-1, 10]), InputSpec([10])])
                net.add_func(x, y)

    @test_ast_only
    def test_concrete_program(self):
        if False:
            for i in range(10):
                print('nop')
        with base.dygraph.guard(base.CPUPlace()):
            x = to_variable(np.ones([4, 10]).astype('float32'))
            y = to_variable(np.ones([4, 10]).astype('float32') * 2)
            int_val = 4.0
            net = SimpleNet()
            net.add_func = to_static(net.add_func, input_spec=[InputSpec([-1, 10]), InputSpec([-1, 10], name='y')])
            cp1 = net.add_func.concrete_program
            self.assertTrue(cp1.inputs[-1].shape == (-1, 10))
            self.assertTrue(cp1.inputs[-1].name == 'y')
            net.add_func = to_static(net.add_func, input_spec=[InputSpec([10]), InputSpec([10], name='label')])
            cp2 = net.add_func.concrete_program
            self.assertTrue(cp2.inputs[-1].shape == (10,))
            self.assertTrue(cp2.inputs[-1].name == 'label')
            self.assertTrue(len(net.add_func.program_cache) == 1)
            self.assertTrue(cp1 != cp2)

def foo_func(a, b, c=1, d=2):
    if False:
        for i in range(10):
            print('nop')
    z = a + b
    return z

class TestDifferentInputSpecCacheProgram(Dy2StTestBase):

    def setUp(self):
        if False:
            print('Hello World!')
        paddle.jit.enable_to_static(True)

    @test_legacy_and_pir
    @test_ast_only
    def test_with_different_input(self):
        if False:
            i = 10
            return i + 15
        with base.dygraph.guard(base.CPUPlace()):
            x_data = np.ones([16, 10]).astype('float32')
            y_data = np.ones([10]).astype('float32') * 2
            z_data = np.ones([10]).astype('float32') * 2.2
            foo = to_static(foo_func)
            out_1 = foo(to_variable(x_data), to_variable(y_data))
            np.testing.assert_allclose(x_data + y_data, out_1.numpy(), rtol=1e-05)
            self.assertTrue(len(foo.program_cache) == 1)
            self.assertTrue(len(foo.program_cache.concrete_programs()) == 1)
            first_program = foo.program_cache.last()
            out_2 = foo(to_variable(x_data), y_data)
            np.testing.assert_allclose(x_data + y_data, out_2.numpy(), rtol=1e-05)
            self.assertTrue(len(foo.program_cache) == 1)
            out_3 = foo(to_variable(x_data), z_data)
            np.testing.assert_allclose(x_data + z_data, out_3.numpy(), rtol=1e-05)
            self.assertTrue(len(foo.program_cache) == 1)
            out_4 = foo(to_variable(x_data), z_data, 3)
            np.testing.assert_allclose(x_data + z_data, out_4.numpy(), rtol=1e-05)
            self.assertTrue(len(foo.program_cache) == 2)
            foo(to_variable(x_data), y_data)
            recent_program = foo.program_cache.last()
            self.assertTrue(first_program == recent_program)

    @test_ast_only
    def test_get_concrete_program(self):
        if False:
            while True:
                i = 10
        foo = to_static(foo_func)
        concrete_program_1 = foo.get_concrete_program(InputSpec([None, 10]), InputSpec([10]))
        self.assertTrue(len(foo.program_cache) == 1)
        concrete_program_2 = foo.get_concrete_program(InputSpec([None, 10]), InputSpec([10]), 1, 2)
        self.assertTrue(concrete_program_2 == concrete_program_1)
        self.assertTrue(len(foo.program_cache) == 1)
        concrete_program_3 = foo.get_concrete_program(InputSpec([None, 10]), InputSpec([10]), c=2)
        self.assertTrue(concrete_program_3 != concrete_program_1)
        self.assertTrue(len(foo.program_cache) == 2)
        concrete_program_4 = foo.get_concrete_program(InputSpec([10]), InputSpec([10]))
        self.assertTrue(concrete_program_4 != concrete_program_1)
        self.assertTrue(len(foo.program_cache) == 3)
        with self.assertRaises(ValueError):
            concrete_program_5 = foo.get_concrete_program(InputSpec([10]))
        with self.assertRaises(TypeError):
            concrete_program_5 = foo.get_concrete_program(InputSpec([10]), InputSpec([10]), e=4)

    @test_legacy_and_pir
    @test_ast_only
    def test_concrete_program(self):
        if False:
            return 10
        with base.dygraph.guard(base.CPUPlace()):
            foo_1 = paddle.jit.to_static(foo_func, input_spec=[InputSpec([10], name='x'), InputSpec([10], name='y')])
            self.assertTrue(isinstance(foo_1.concrete_program, ConcreteProgram))
            foo_2 = paddle.jit.to_static(foo_func)
            out = foo_2(paddle.rand([10]), paddle.rand([10]))
            self.assertTrue(isinstance(foo_2.concrete_program, ConcreteProgram))
            foo_3 = paddle.jit.to_static(foo_func)
            with self.assertRaises(ValueError):
                foo_3.concrete_program

class TestInputDefaultName(Dy2StTestBase):

    def setUp(self):
        if False:
            print('Hello World!')
        paddle.disable_static()
        self.net = SimpleNet()

    def assert_default_name(self, func_name, input_names):
        if False:
            print('Hello World!')
        decorated_func = getattr(self.net, func_name)
        spec_names = [x.name for x in decorated_func.inputs]
        self.assertListEqual(spec_names, input_names)

    def test_common_input(self):
        if False:
            return 10
        self.assert_default_name('forward', ['x'])

    def test_list_input(self):
        if False:
            i = 10
            return i + 15
        self.assert_default_name('func_with_list', ['l_0', 'l_1'])

    def test_dict_input(self):
        if False:
            print('Hello World!')
        self.assert_default_name('func_with_dict', ['x', 'y'])

    def test_nest_input(self):
        if False:
            i = 10
            return i + 15
        self.assert_default_name('func_with_list_dict', ['dl_0', 'x', 'y'])

class TestDeclarativeAPI(Dy2StTestBase):

    @test_ast_only
    def test_error(self):
        if False:
            print('Hello World!')
        func = to_static(dyfunc_to_variable)
        paddle.enable_static()
        with self.assertRaises(RuntimeError):
            func(np.ones(5).astype('int32'))
        paddle.jit.enable_to_static(False)
        with self.assertRaises(AssertionError):
            func(np.ones(5).astype('int32'))

class TestDecorateModelDirectly(Dy2StTestBase):

    def setUp(self):
        if False:
            return 10
        paddle.disable_static()
        paddle.jit.enable_to_static(True)
        self.x = to_variable(np.ones([4, 10]).astype('float32'))

    @test_legacy_and_pir
    @test_ast_only
    def test_fake_input(self):
        if False:
            for i in range(10):
                print('nop')
        net = SimpleNet()
        net = to_static(net)
        y = net(self.x)
        self.assertTrue(len(net.forward.program_cache) == 1)

    @test_ast_only
    def test_input_spec(self):
        if False:
            while True:
                i = 10
        net = SimpleNet()
        net = to_static(net, input_spec=[InputSpec([None, 8, 10])])
        self.assertTrue(len(net.forward.inputs) == 1)
        self.assertTrue(len(net.forward.program_cache) == 1)
        input_shape = net.forward.inputs[0].shape
        self.assertListEqual(list(input_shape), [-1, 8, 10])
        net = to_static(net, input_spec=[InputSpec([None, 16, 10])])
        input_shape = net.forward.inputs[0].shape
        self.assertListEqual(list(input_shape), [-1, 16, 10])

class TestErrorWithInitFromStaticMode(Dy2StTestBase):

    def test_raise_error(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()
        net = SimpleNet()
        with self.assertRaisesRegex(RuntimeError, 'only available in dynamic mode'):
            net.forward.concrete_program
        with self.assertRaisesRegex(RuntimeError, 'only available in dynamic mode'):
            net.forward.inputs
        with self.assertRaisesRegex(RuntimeError, 'only available in dynamic mode'):
            net.forward.outputs

class CallNonForwardFuncNet(paddle.nn.Layer):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.sub = CallNonForwardFuncSubNet()

    @paddle.jit.to_static(full_graph=True)
    def forward(self):
        if False:
            i = 10
            return i + 15
        return self.sub.func()

class CallNonForwardFuncSubNet(paddle.nn.Layer):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.a = paddle.to_tensor([1, 2])

    def func(self):
        if False:
            for i in range(10):
                print('nop')
        x = self.a * 2
        return x

class TestCallNonForwardFunc(Dy2StTestBase):

    @test_legacy_and_pir
    def test_call_non_forward(self):
        if False:
            return 10
        paddle.disable_static()
        net = CallNonForwardFuncNet()
        out = net()
        self.assertEqual(out.numpy().tolist(), [2, 4])
        paddle.enable_static()

class SetBuffersNet1(paddle.nn.Layer):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.a = paddle.to_tensor([1])

    @paddle.jit.to_static(full_graph=True)
    def forward(self):
        if False:
            return 10
        self.a = self.a + 1
        return self.a

class SetBuffersNet2(paddle.nn.Layer):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.b = paddle.to_tensor([2])

    @paddle.jit.to_static(full_graph=True)
    def forward(self):
        if False:
            print('Hello World!')
        self.b = None
        self.b = paddle.to_tensor([3])
        return self.b

class TestSetBuffers(Dy2StTestBase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.temp_dir.name, 'SetBuffersNet1')

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.temp_dir.cleanup()

    @test_legacy_and_pir
    def test_set_buffers1(self):
        if False:
            return 10
        paddle.disable_static()
        net = SetBuffersNet1()
        out = net()
        self.assertEqual(out.numpy().tolist(), [2])
        paddle.jit.save(net, self.model_path)
        paddle.enable_static()

    @test_ast_only
    def test_set_buffers2(self):
        if False:
            i = 10
            return i + 15
        paddle.disable_static()
        net = SetBuffersNet2()
        with self.assertRaises(RuntimeError):
            out = net()
        paddle.enable_static()

class ClassNoInheritLayer:

    def func(self, x):
        if False:
            i = 10
            return i + 15
        return x + 1

class TestClassNoInheritLayer(Dy2StTestBase):

    def test_to_static(self):
        if False:
            while True:
                i = 10
        paddle.disable_static()
        net = ClassNoInheritLayer()
        input_spec = [paddle.static.InputSpec(name='x', shape=[1])]
        with self.assertRaises(TypeError):
            static_func = paddle.jit.to_static(net.func, input_spec=input_spec)
if __name__ == '__main__':
    unittest.main()