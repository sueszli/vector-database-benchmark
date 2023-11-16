import logging
import unittest
import numpy as np
from dygraph_to_static_utils_new import Dy2StTestBase, test_ast_only, test_legacy_and_pir, test_legacy_and_pir_api
import paddle
import paddle.jit.dy2static as _jst
from paddle import base
from paddle.jit.dy2static.convert_call_func import CONVERSION_OPTIONS
from paddle.jit.dy2static.utils import func_to_source_code
SEED = 2020
np.random.seed(SEED)

@paddle.jit.to_static(full_graph=True)
def dyfunc_with_if(x_v):
    if False:
        for i in range(10):
            print('nop')
    if paddle.mean(x_v).numpy() > 5:
        x_v = x_v - 1
    else:
        x_v = x_v + 1
    return x_v

@paddle.jit.to_static(full_graph=True)
def nested_func(x_v):
    if False:
        while True:
            i = 10
    x_v = base.dygraph.to_variable(x_v)

    def fn1():
        if False:
            return 10
        return x_v
    res = fn1()
    return res

@paddle.jit.to_static(full_graph=True)
def dyfunc_with_third_library_logging(x_v):
    if False:
        print('Hello World!')
    logging.info('test dyfunc_with_third_library_logging')
    if paddle.mean(x_v).numpy() > 5:
        x_v = x_v - 1
    else:
        x_v = x_v + 1
    return x_v

class A:

    @staticmethod
    def add(a, b):
        if False:
            return 10
        '\n        dygraph mode, return a numpy object.\n        static graph mode, return a variable object.\n        '
        return paddle.to_tensor(a.numpy() + b.numpy())

@paddle.jit.to_static(full_graph=True)
def dyfunc_with_staticmethod(x_v):
    if False:
        return 10
    a = A()
    return a.add(x_v, x_v)

class TestRecursiveCall1(Dy2StTestBase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.input = np.random.random([10, 16]).astype('float32')

    def init_test_func(self):
        if False:
            while True:
                i = 10
        self.dyfunc = nested_func

    def get_dygraph_output(self):
        if False:
            return 10
        paddle.jit.enable_to_static(False)
        res = self.dyfunc(self.input).numpy()
        return res

    def get_static_output(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.jit.enable_to_static(True)
        res = self.dyfunc(self.input).numpy()
        return res

    @test_legacy_and_pir_api
    def test_transformed_static_result(self):
        if False:
            for i in range(10):
                print('nop')
        self.init_test_func()
        static_res = self.get_static_output()
        dygraph_res = self.get_dygraph_output()
        np.testing.assert_allclose(dygraph_res, static_res, rtol=1e-05, err_msg=f'dygraph res is {dygraph_res}\nstatic_res is {static_res}')
lambda_fun = lambda x: x

class MyConvLayer(paddle.nn.Layer):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self._conv = paddle.nn.Conv2D(in_channels=3, out_channels=2, kernel_size=3, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.99)), bias_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.5)))

    @paddle.jit.to_static(full_graph=True)
    def forward(self, inputs):
        if False:
            return 10
        y = dyfunc_with_if(inputs)
        y = lambda_fun(y)
        y = self.dymethod(y)
        return y

    @paddle.jit.to_static(full_graph=True)
    def dymethod(self, x_v):
        if False:
            print('Hello World!')
        x_v = paddle.assign(x_v)
        return x_v

class MyLayer(paddle.nn.Layer):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.conv = MyConvLayer()
        self.fc = paddle.nn.Linear(in_features=5, out_features=1, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.99)), bias_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.5)))
        self.act = paddle.nn.ReLU()

    @paddle.jit.to_static(full_graph=True)
    def forward(self, inputs):
        if False:
            i = 10
            return i + 15
        h = self.conv(inputs)
        out = self.fc(h)
        return self.act(out)

class TestRecursiveCall2(Dy2StTestBase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.input = np.random.random((1, 3, 3, 5)).astype('float32')

    def set_func(self):
        if False:
            print('Hello World!')
        self.dygraph_func = MyLayer()

    def _run(self):
        if False:
            return 10
        data = base.dygraph.to_variable(self.input)
        res = self.dygraph_func(data)
        return res.numpy()

    def get_dygraph_output(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.jit.enable_to_static(False)
        return self._run()

    def get_static_output(self):
        if False:
            i = 10
            return i + 15
        paddle.jit.enable_to_static(True)
        return self._run()

    @test_legacy_and_pir
    def test_transformed_static_result(self):
        if False:
            print('Hello World!')
        self.set_func()
        dygraph_res = self.get_dygraph_output()
        static_res = self.get_static_output()
        np.testing.assert_allclose(dygraph_res, static_res, rtol=1e-05)

class TestThirdPartyLibrary(TestRecursiveCall2):

    def set_func(self):
        if False:
            print('Hello World!')
        self.dygraph_func = dyfunc_with_third_library_logging

class TestStaticMethod(TestRecursiveCall2):

    def set_func(self):
        if False:
            i = 10
            return i + 15
        self.dygraph_func = dyfunc_with_staticmethod

class NotToStaticHelper(paddle.nn.Layer):

    def __init__(self):
        if False:
            return 10
        super().__init__()

    def sum(self, x):
        if False:
            i = 10
            return i + 15
        if x.shape[0] > 1:
            res = x + 1
        res = paddle.sum(x)
        return res

    def outer(self, x):
        if False:
            for i in range(10):
                print('nop')
        res = self.sum(x)
        return res

    def inner(self, x):
        if False:
            while True:
                i = 10
        return self.outer(x)

class TestNotToConvert(TestRecursiveCall2):

    def set_func(self):
        if False:
            i = 10
            return i + 15
        self.net = NotToStaticHelper()
        paddle.jit.not_to_static(self.net.sum)
        self.dygraph_func = paddle.jit.to_static(self.net.outer)

    @test_legacy_and_pir_api
    def test_conversion_options(self):
        if False:
            return 10
        self.set_func()
        options = getattr(self.net.sum, CONVERSION_OPTIONS, None)
        self.assertIsNotNone(options)
        self.assertTrue(options.not_convert)

    @test_legacy_and_pir_api
    def test_code(self):
        if False:
            i = 10
            return i + 15
        self.set_func()
        self.assertIn('if x.shape[0] > 1', func_to_source_code(_jst.Call(self.net.sum)))

class TestNotToConvert2(TestRecursiveCall2):

    def set_func(self):
        if False:
            while True:
                i = 10
        self.net = NotToStaticHelper()
        paddle.jit.not_to_static(self.net.sum)
        self.dygraph_func = paddle.jit.to_static(self.net.sum)

    @test_legacy_and_pir_api
    def test_conversion_options(self):
        if False:
            while True:
                i = 10
        self.set_func()
        options = getattr(self.net.sum, CONVERSION_OPTIONS, None)
        self.assertIsNotNone(options)
        self.assertTrue(options.not_convert)

    @test_ast_only
    @test_legacy_and_pir_api
    def test_code(self):
        if False:
            print('Hello World!')
        self.set_func()
        self.dygraph_func = paddle.jit.to_static(self.net.sum)
        self.assertIn('if x.shape[0] > 1', self.dygraph_func.code)

@paddle.jit.not_to_static
def forward(self, x):
    if False:
        for i in range(10):
            print('nop')
    if x.shape[0] > 1:
        x = x + 1
    return x

class TestConvertPaddleAPI(Dy2StTestBase):

    @test_ast_only
    @test_legacy_and_pir_api
    def test_functional_api(self):
        if False:
            for i in range(10):
                print('nop')
        func = paddle.nn.functional.relu
        func = paddle.jit.to_static(func)
        self.assertNotIn('_jst.IfElse', func.code)
        self.assertIn('if in_dynamic_or_pir_mode()', func.code)

    @test_ast_only
    @test_legacy_and_pir_api
    def test_class_api(self):
        if False:
            for i in range(10):
                print('nop')
        bn = paddle.nn.SyncBatchNorm(2)
        paddle.jit.to_static(bn)
        self.assertNotIn('_jst.IfElse', bn.forward.code)
        self.assertIn('if in_dynamic_mode()', bn.forward.code)

    @test_ast_only
    @test_legacy_and_pir_api
    def test_class_patch_api(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.nn.SyncBatchNorm.forward = forward
        bn = paddle.nn.SyncBatchNorm(2)
        paddle.jit.to_static(bn)
        self.assertNotIn('_jst.IfElse', bn.forward.code)
        self.assertIn('if x.shape[0] > 1', bn.forward.code)
if __name__ == '__main__':
    unittest.main()