import unittest
from unittest import TestCase
import numpy as np
import paddle
from paddle import base
from paddle.base.wrapped_decorator import wrap_decorator

def _dygraph_guard_(func):
    if False:
        while True:
            i = 10

    def __impl__(*args, **kwargs):
        if False:
            print('Hello World!')
        if base.in_dygraph_mode():
            return func(*args, **kwargs)
        else:
            with base.dygraph.guard():
                return func(*args, **kwargs)
    return __impl__
dygraph_guard = wrap_decorator(_dygraph_guard_)

def random_var(size, low=-1, high=1, dtype='float32'):
    if False:
        for i in range(10):
            print('nop')
    np.random.seed(2021)
    x_np = np.random.uniform(low=low, high=high, size=size).astype(dtype)
    return base.dygraph.to_variable(x_np)

class TestDygraphTripleGradMatmul(TestCase):

    def test_matmul_triple_grad(self):
        if False:
            i = 10
            return i + 15
        input_numpy = np.ones([3, 3]) * 2
        x = paddle.to_tensor(input_numpy, stop_gradient=False, dtype='float32')
        y = paddle.to_tensor(input_numpy, stop_gradient=False, dtype='float32')
        out = paddle.matmul(x, y, False, False)
        new_out_g = paddle.to_tensor(np.ones([3, 3]), stop_gradient=False, dtype='float32')
        (new_x_g, new_y_g) = paddle.grad([out], [x, y], [new_out_g], retain_graph=True, create_graph=True)
        new_x_g_g = paddle.to_tensor(np.ones([3, 3]), stop_gradient=False, dtype='float32')
        new_y_g_g = paddle.to_tensor(np.ones([3, 3]), stop_gradient=False, dtype='float32')
        (new_a, new_b, new_c) = paddle.grad([new_x_g, new_y_g], [x, y, new_out_g], [new_x_g_g, new_y_g_g], retain_graph=True, create_graph=True)
        new_a.backward()
        out_ref = np.ones([3, 3]) * 12.0
        np.testing.assert_array_equal(out.numpy(), out_ref)
        new_x_g_ref = np.ones([3, 3]) * 6.0
        new_y_g_ref = np.ones([3, 3]) * 6.0
        np.testing.assert_array_equal(new_x_g.numpy(), new_x_g_ref)
        np.testing.assert_array_equal(new_y_g.numpy(), new_y_g_ref)
        new_a_ref = np.ones([3, 3]) * 3.0
        new_b_ref = np.ones([3, 3]) * 3.0
        new_c_ref = np.ones([3, 3]) * 12.0
        np.testing.assert_array_equal(new_a.numpy(), new_a_ref)
        np.testing.assert_array_equal(new_b.numpy(), new_b_ref)
        np.testing.assert_array_equal(new_c.numpy(), new_c_ref)
        x_grad_ref = np.ones([3, 3]) * 0.0
        assert x.grad is None
        y_grad_ref = np.ones([3, 3]) * 0.0
        assert y.grad is None
        new_out_g_ref = np.ones([3, 3]) * 3.0
        np.testing.assert_array_equal(new_out_g.grad.numpy(), new_out_g_ref)
        new_x_g_g_ref = np.ones([3, 3]) * 0.0
        new_y_g_g_ref = np.ones([3, 3]) * 3.0
        assert new_x_g_g.grad is None
        np.testing.assert_array_equal(new_y_g_g.grad.numpy(), new_y_g_g_ref)

class TestDygraphTripleGrad(TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.sort_sum_gradient = False
        self.shape = [5, 5]

    def grad(self, outputs, inputs, grad_outputs=None, no_grad_vars=None, retain_graph=None, create_graph=False, allow_unused=False):
        if False:
            for i in range(10):
                print('nop')
        base.set_flags({'FLAGS_sort_sum_gradient': self.sort_sum_gradient})
        return base.dygraph.grad(outputs=outputs, inputs=inputs, grad_outputs=grad_outputs, no_grad_vars=no_grad_vars, retain_graph=retain_graph, create_graph=create_graph, allow_unused=allow_unused)

    @dygraph_guard
    def func_exception(self):
        if False:
            print('Hello World!')
        with self.assertRaises(AssertionError):
            self.grad(None, None)
        shape = self.shape
        with self.assertRaises(AssertionError):
            self.grad(1, random_var(shape))
        with self.assertRaises(AssertionError):
            self.grad(random_var(shape), 1)
        with self.assertRaises(AssertionError):
            self.grad([1], [random_var(shape)])
        with self.assertRaises(AssertionError):
            self.grad([random_var(shape)], [1])
        with self.assertRaises(AssertionError):
            self.grad([random_var(shape), random_var(shape)], [random_var(shape)], [random_var(shape)])
        with self.assertRaises(AssertionError):
            self.grad([random_var(shape)], [random_var(shape)], no_grad_vars=[1])
        with self.assertRaises(AssertionError):
            self.grad([random_var(shape)], [random_var(shape)], no_grad_vars=1)

    @dygraph_guard
    def func_example_with_gradient_and_create_graph(self):
        if False:
            while True:
                i = 10
        x = random_var(self.shape)
        x.retain_grads()
        x_np = x.numpy()
        x.stop_gradient = False
        y = random_var(self.shape)
        y_np = y.numpy()
        y.stop_gradient = False
        z = random_var(self.shape)
        z_np = z.numpy()
        numel = z_np.size
        z.stop_gradient = False
        out = paddle.nn.functional.sigmoid(paddle.matmul(x, y) + z)
        out_np = out.numpy()
        (dx_actual,) = self.grad([out], [x], create_graph=True)
        dout = np.ones(self.shape).astype('float32')
        dx_expected = np.matmul(dout * out_np * (1 - out_np), np.transpose(y_np))
        np.testing.assert_allclose(dx_actual.numpy(), dx_expected, rtol=1e-05)
        (ddx_actual,) = self.grad([dx_actual], [x], create_graph=True)
        DDY = np.zeros(self.shape).astype('float32')
        DDX = np.ones(self.shape).astype('float32')
        double_grad_tmp1 = np.matmul(dout * out_np * (1 - out_np), np.transpose(DDY))
        double_grad_tmp2 = np.matmul(DDX, y_np) + np.matmul(x_np, DDY)
        double_grad_tmp3 = (1 - 2 * out_np) * dout * double_grad_tmp2 * out_np * (1 - out_np)
        ddx_expected = double_grad_tmp1 + np.matmul(double_grad_tmp3, np.transpose(y_np))
        np.testing.assert_allclose(ddx_actual.numpy(), ddx_expected, rtol=1e-05)
        d_ddout = np.zeros(self.shape).astype('float32')
        tmp0 = np.matmul(DDX, y_np) + np.matmul(x_np, DDY)
        tmp1 = (1 - 2 * out_np) * ((1 - 2 * out_np) * dout * tmp0 * tmp0)
        tmp2 = tmp0 * (1 - 2 * out_np) * d_ddout - 2 * dout * (1 - out_np) * out_np * tmp0 * tmp0
        dddx_expected = np.matmul((tmp1 + tmp2) * out_np * (1 - out_np), np.transpose(y_np))
        ddx_actual.backward()
        dddx_grad_actual = x.gradient()
        np.testing.assert_allclose(dddx_grad_actual, dddx_expected, rtol=1e-05)

    def test_all_cases(self):
        if False:
            return 10
        self.func_exception()
        self.func_example_with_gradient_and_create_graph()

class TestDygraphTripleGradBradcastCase(TestCase):

    def setUp(self):
        if False:
            return 10
        self.sort_sum_gradient = False
        self.x_shape = [3, 2, 2]
        self.y_shape = [1, 2, 2]
        self.z_shape = [2, 2]

    def grad(self, outputs, inputs, grad_outputs=None, no_grad_vars=None, retain_graph=None, create_graph=False, allow_unused=False):
        if False:
            while True:
                i = 10
        base.set_flags({'FLAGS_sort_sum_gradient': self.sort_sum_gradient})
        return base.dygraph.grad(outputs=outputs, inputs=inputs, grad_outputs=grad_outputs, no_grad_vars=no_grad_vars, retain_graph=retain_graph, create_graph=create_graph, allow_unused=allow_unused)

    @dygraph_guard
    def func_example_with_gradient_and_create_graph(self):
        if False:
            i = 10
            return i + 15
        x = random_var(self.x_shape)
        x.retain_grads()
        x_np = x.numpy()
        x.stop_gradient = False
        y = random_var(self.y_shape)
        y_np = y.numpy()
        y.stop_gradient = False
        z = random_var(self.z_shape)
        z_np = z.numpy()
        numel = z_np.size
        z.stop_gradient = False
        out = paddle.nn.functional.sigmoid(paddle.matmul(x, y) + z)
        out_np = out.numpy()
        (dx_actual,) = self.grad([out], [x], create_graph=True)
        dout = np.ones(self.x_shape).astype('float32')
        dx_expected = np.matmul(dout * out_np * (1 - out_np), np.transpose(y_np, axes=(0, 2, 1)))
        np.testing.assert_allclose(dx_actual.numpy(), dx_expected, rtol=1e-05)
        (ddx_actual,) = self.grad([dx_actual], [x], create_graph=True)
        DDY = np.zeros(self.y_shape).astype('float32')
        DDX = np.ones(self.x_shape).astype('float32')
        double_grad_tmp1 = np.matmul(dout * out_np * (1 - out_np), np.transpose(DDY, axes=(0, 2, 1)))
        double_grad_tmp2 = np.matmul(DDX, y_np) + np.matmul(x_np, DDY)
        double_grad_tmp3 = (1 - 2 * out_np) * dout * double_grad_tmp2 * out_np * (1 - out_np)
        ddx_expected = double_grad_tmp1 + np.matmul(double_grad_tmp3, np.transpose(y_np, axes=(0, 2, 1)))
        np.testing.assert_allclose(ddx_actual.numpy(), ddx_expected, rtol=1e-05)
        d_ddout = np.zeros(self.x_shape).astype('float32')
        tmp0 = np.matmul(DDX, y_np) + np.matmul(x_np, DDY)
        tmp1 = (1 - 2 * out_np) * ((1 - 2 * out_np) * dout * tmp0 * tmp0)
        tmp2 = tmp0 * (1 - 2 * out_np) * d_ddout - 2 * dout * (1 - out_np) * out_np * tmp0 * tmp0
        dddx_expected = np.matmul((tmp1 + tmp2) * out_np * (1 - out_np), np.transpose(y_np, axes=(0, 2, 1)))
        ddx_actual.backward()
        dddx_grad_actual = x.gradient()
        np.testing.assert_allclose(dddx_grad_actual, dddx_expected, rtol=1e-05)

    def test_all_cases(self):
        if False:
            print('Hello World!')
        self.func_example_with_gradient_and_create_graph()

class TestDygraphTripleGradMatmulcase1(TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.input_numpy_x = None
        self.input_numpy_y = None
        self.input_numpy_dout = None
        self.input_numpy_ddx = None
        self.input_numpy_ddy = None
        self.places = ['cpu']
        if paddle.is_compiled_with_cuda():
            self.places.append('gpu')

    def actual(self):
        if False:
            while True:
                i = 10
        x = paddle.to_tensor(self.input_numpy_x, stop_gradient=False, dtype='float32')
        y = paddle.to_tensor(self.input_numpy_y, stop_gradient=False, dtype='float32')
        out = paddle.matmul(x, y, False, False)
        dout = paddle.to_tensor(self.input_numpy_dout, stop_gradient=False, dtype='float32')
        (dx, dy) = paddle.grad([out], [x, y], [dout], retain_graph=True, create_graph=True)
        ddx = paddle.to_tensor(self.input_numpy_ddx, stop_gradient=False, dtype='float32')
        ddy = paddle.to_tensor(self.input_numpy_ddy, stop_gradient=False, dtype='float32')
        (dx_double_grad, dy_double_grad) = paddle.grad([dx, dy], [x, y], [ddx, ddy], retain_graph=True, create_graph=True)
        (d_dout, d_ddx, d_ddy) = paddle.grad([dx_double_grad, dy_double_grad], [dout, ddx, ddy], retain_graph=False, create_graph=False)
        return (d_dout, d_ddx, d_ddy)

    def test_matmul_triple_grad_case1(self):
        if False:
            print('Hello World!')

        def init_data():
            if False:
                return 10
            self.input_numpy_x = np.random.random([3, 3]).astype('float32')
            self.input_numpy_y = np.random.random([3, 3]).astype('float32')
            self.input_numpy_dout = np.ones([3, 3], dtype='float32')
            self.input_numpy_ddx = np.ones([3, 3], dtype='float32')
            self.input_numpy_ddy = np.ones([3, 3], dtype='float32')
        init_data()
        d_dout_expected = np.ones([3, 3], dtype='float32') * 6
        d_ddx_expected = np.ones([3, 3], dtype='float32') * 3
        d_ddy_expected = np.ones([3, 3], dtype='float32') * 3
        expected_results = (d_dout_expected, d_ddx_expected, d_ddy_expected)
        for place in self.places:
            paddle.device.set_device(place)
            actual_results = self.actual()
            for (expected_result, actual_result) in zip(expected_results, actual_results):
                np.testing.assert_allclose(expected_result, actual_result, rtol=1e-06)

    def test_matmul_triple_grad_case2(self):
        if False:
            return 10

        def init_data():
            if False:
                for i in range(10):
                    print('nop')
            self.input_numpy_x = np.random.random([3]).astype('float32')
            self.input_numpy_y = np.random.random([3]).astype('float32')
            self.input_numpy_dout = np.ones([1], dtype='float32')
            self.input_numpy_ddx = np.ones([3], dtype='float32')
            self.input_numpy_ddy = np.ones([3], dtype='float32')
        init_data()
        d_dout_expected = np.ones([1], dtype='float32') * 6
        d_ddx_expected = np.ones([3], dtype='float32')
        d_ddy_expected = np.ones([3], dtype='float32')
        expected_results = (d_dout_expected, d_ddx_expected, d_ddy_expected)
        for place in self.places:
            paddle.device.set_device(place)
            actual_results = self.actual()
            for (expected_result, actual_result) in zip(expected_results, actual_results):
                np.testing.assert_allclose(expected_result, actual_result, rtol=1e-06)

    def test_matmul_triple_grad_case3(self):
        if False:
            return 10

        def init_data():
            if False:
                i = 10
                return i + 15
            self.input_numpy_x = np.random.random([3, 1]).astype('float32')
            self.input_numpy_y = np.random.random([1]).astype('float32')
            self.input_numpy_dout = np.ones([3], dtype='float32')
            self.input_numpy_ddx = np.ones([3, 1], dtype='float32')
            self.input_numpy_ddy = np.ones([1], dtype='float32')
        init_data()
        d_dout_expected = np.ones([3], dtype='float32') * 2
        d_ddx_expected = np.ones([3, 1], dtype='float32')
        d_ddy_expected = np.ones([1], dtype='float32') * 3
        expected_results = (d_dout_expected, d_ddx_expected, d_ddy_expected)
        for place in self.places:
            paddle.device.set_device(place)
            actual_results = self.actual()
            for (expected_result, actual_result) in zip(expected_results, actual_results):
                np.testing.assert_allclose(expected_result, actual_result, rtol=1e-06)
'\n# d_ddout is none, dtype is complex64\nclass TestDygraphTripleGradMatmulcase2(TestCase):\n    def setUp(self):\n        self.input_numpy_x = None\n        self.input_numpy_y = None\n        self.input_numpy_dout = None\n        self.input_numpy_ddx = None\n        self.input_numpy_ddy = None\n        self.input_numpy_ddx_conj = None\n        self.input_numpy_ddy_conj = None\n        self.input_numpy_dout_conj = None\n        self.places = ["cpu"]\n        if paddle.is_compiled_with_cuda():\n            self.places.append("gpu")\n\n    def actual(self):\n        x = paddle.to_tensor(\n            self.input_numpy_x, stop_gradient=False, dtype=\'complex64\'\n        )\n        y = paddle.to_tensor(\n            self.input_numpy_y, stop_gradient=False, dtype=\'complex64\'\n        )\n        out = paddle.matmul(x, y, False, False)\n\n        dout = paddle.to_tensor(\n            self.input_numpy_dout, stop_gradient=False, dtype=\'complex64\'\n        )\n        (dx, dy) = paddle.grad(\n            [out], [x, y], [dout], retain_graph=True, create_graph=True\n        )\n        ddx = paddle.to_tensor(\n            self.input_numpy_ddx, stop_gradient=False, dtype=\'complex64\'\n        )\n        ddy = paddle.to_tensor(\n            self.input_numpy_ddy, stop_gradient=False, dtype=\'complex64\'\n        )\n        dx_double_grad, dy_double_grad = paddle.grad(\n            [dx, dy],\n            [x, y],\n            [ddx, ddy],\n            retain_graph=True,\n            create_graph=True,\n        )\n        d_x, d_y, d_dout, d_ddx, d_ddy = paddle.grad(\n            [dx_double_grad, dy_double_grad],\n            [x, y, dout, ddx, ddy],\n            retain_graph=False,\n            create_graph=False,\n        )\n        return d_x, d_y, d_dout, d_ddx, d_ddy\n\n    # case1: no d_ddout, dims = 1, dtype is complex64\n    def test_matmul_triple_grad_case1(self):\n        def init_data():\n            self.input_numpy_x = np.random.random([3]).astype(\n                \'float32\'\n            ) + 1j * np.random.random(\n                [\n                    3,\n                ]\n            ).astype(\n                \'float32\'\n            )\n            self.input_numpy_y = np.random.random([3]).astype(\n                \'float32\'\n            ) + 1j * np.random.random(\n                [\n                    3,\n                ]\n            ).astype(\n                \'float32\'\n            )\n            self.input_numpy_dout = np.ones(\n                [\n                    1,\n                ],\n                dtype="float32",\n            )\n            self.input_numpy_ddx = np.ones(\n                [\n                    3,\n                ],\n                dtype="float32",\n            )\n            self.input_numpy_ddy = np.ones(\n                [\n                    3,\n                ],\n                dtype="float32",\n            )\n            self.input_numpy_ddx_conj = np.conjugate(self.input_numpy_ddx)\n            self.input_numpy_ddy_conj = np.conjugate(self.input_numpy_ddy)\n            self.input_numpy_dout_conj = np.conjugate(self.input_numpy_dout)\n\n        init_data()\n        d_x_expected = np.zeros(\n            [\n                3,\n            ],\n            dtype="float32",\n        )\n        d_y_expected = np.zeros(\n            [\n                3,\n            ],\n            dtype="float32",\n        )\n        d_dout_expected = np.matmul(\n            self.input_numpy_ddy_conj,\n            np.ones(\n                [\n                    3,\n                ],\n                dtype="float32",\n            ),\n        ) + np.matmul(\n            self.input_numpy_ddx_conj,\n            np.ones(\n                [\n                    3,\n                ],\n                dtype="float32",\n            ),\n        )\n        d_ddx_expected = (\n            np.ones(\n                [\n                    3,\n                ],\n                dtype="float32",\n            )\n            * self.input_numpy_dout_conj[0]\n        )\n        d_ddy_expected = (\n            np.ones(\n                [\n                    3,\n                ],\n                dtype="float32",\n            )\n            * self.input_numpy_dout_conj[0]\n        )\n        expected_results = (\n            d_x_expected,\n            d_y_expected,\n            d_dout_expected,\n            d_ddx_expected,\n            d_ddy_expected,\n        )\n\n        for place in self.places:\n            paddle.device.set_device(place)\n            actual_results = self.actual()\n            for expected_result, actual_result in zip(\n                expected_results, actual_results\n            ):\n                np.testing.assert_allclose(\n                    expected_result, actual_result, rtol=1e-6\n                )\n'

class TestDygraphTripleGradMatmulcase3(TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.input_numpy_x = None
        self.input_numpy_y = None
        self.input_numpy_dout = None
        self.input_numpy_ddx = None
        self.input_numpy_ddy = None
        self.places = ['cpu']
        if paddle.is_compiled_with_cuda():
            self.places.append('gpu')

    def actual(self):
        if False:
            i = 10
            return i + 15
        x = paddle.to_tensor(self.input_numpy_x, stop_gradient=False, dtype='float32')
        y = paddle.to_tensor(self.input_numpy_y, stop_gradient=False, dtype='float32')
        out = paddle.matmul(x, y, False, False)
        dout = paddle.to_tensor(self.input_numpy_dout, stop_gradient=False, dtype='float32')
        (dx, dy) = paddle.grad([out], [x, y], [dout], retain_graph=True, create_graph=True)
        ddx = paddle.to_tensor(self.input_numpy_ddx, stop_gradient=False, dtype='float32')
        ddy = paddle.to_tensor(self.input_numpy_ddy, stop_gradient=False, dtype='float32')
        (dy_double_grad,) = paddle.grad([dx, dy], [y], [ddx, ddy], retain_graph=True, create_graph=True)
        (d_dout, d_ddx) = paddle.grad([dy_double_grad], [dout, ddx], retain_graph=False, create_graph=False)
        return (d_dout, d_ddx)

    def test_matmul_triple_grad_case1(self):
        if False:
            return 10

        def init_data():
            if False:
                for i in range(10):
                    print('nop')
            self.input_numpy_x = np.random.random([3, 3]).astype('float32')
            self.input_numpy_y = np.random.random([3, 3]).astype('float32')
            self.input_numpy_dout = np.ones([3, 3], dtype='float32')
            self.input_numpy_ddx = np.ones([3, 3], dtype='float32')
            self.input_numpy_ddy = np.ones([3, 3], dtype='float32')
        init_data()
        d_dout_expected = np.ones([3, 3], dtype='float32') * 3
        d_ddx_expected = np.ones([3, 3], dtype='float32') * 3
        expected_results = (d_dout_expected, d_ddx_expected)
        for place in self.places:
            paddle.device.set_device(place)
            actual_results = self.actual()
            for (expected_result, actual_result) in zip(expected_results, actual_results):
                np.testing.assert_allclose(expected_result, actual_result, rtol=1e-06)

    def test_matmul_triple_grad_case2(self):
        if False:
            return 10

        def init_data():
            if False:
                print('Hello World!')
            self.input_numpy_x = np.random.random([3]).astype('float32')
            self.input_numpy_y = np.random.random([3]).astype('float32')
            self.input_numpy_dout = np.ones([1], dtype='float32')
            self.input_numpy_ddx = np.ones([3], dtype='float32')
            self.input_numpy_ddy = np.ones([3], dtype='float32')
        init_data()
        d_dout_expected = np.ones([1], dtype='float32') * 3
        d_ddx_expected = np.ones([3], dtype='float32')
        expected_results = (d_dout_expected, d_ddx_expected)
        for place in self.places:
            paddle.device.set_device(place)
            actual_results = self.actual()
            for (expected_result, actual_result) in zip(expected_results, actual_results):
                np.testing.assert_allclose(expected_result, actual_result, rtol=1e-06)

    def test_matmul_triple_grad_case3(self):
        if False:
            while True:
                i = 10

        def init_data():
            if False:
                for i in range(10):
                    print('nop')
            self.input_numpy_x = np.random.random([3, 1]).astype('float32')
            self.input_numpy_y = np.random.random([1]).astype('float32')
            self.input_numpy_dout = np.ones([3], dtype='float32')
            self.input_numpy_ddx = np.ones([3, 1], dtype='float32')
            self.input_numpy_ddy = np.ones([1], dtype='float32')
        init_data()
        d_dout_expected = np.ones([3], dtype='float32')
        d_ddx_expected = np.ones([3, 1], dtype='float32')
        expected_results = (d_dout_expected, d_ddx_expected)
        for place in self.places:
            paddle.device.set_device(place)
            actual_results = self.actual()
            for (expected_result, actual_result) in zip(expected_results, actual_results):
                np.testing.assert_allclose(expected_result, actual_result, rtol=1e-06)
'\n# d_ddout is none, d_dx is none, dtype is complex64\nclass TestDygraphTripleGradMatmulcase4(TestCase):\n    def setUp(self):\n        self.input_numpy_x = None\n        self.input_numpy_y = None\n        self.input_numpy_dout = None\n        self.input_numpy_ddx = None\n        self.input_numpy_ddy = None\n        self.input_numpy_ddx_conj = None\n        self.input_numpy_dout_conj = None\n        self.places = ["cpu"]\n        if paddle.is_compiled_with_cuda():\n            self.places.append("gpu")\n\n    def actual(self):\n        x = paddle.to_tensor(\n            self.input_numpy_x, stop_gradient=False, dtype=\'complex64\'\n        )\n        y = paddle.to_tensor(\n            self.input_numpy_y, stop_gradient=False, dtype=\'complex64\'\n        )\n        out = paddle.matmul(x, y, False, False)\n\n        dout = paddle.to_tensor(\n            self.input_numpy_dout, stop_gradient=False, dtype=\'complex64\'\n        )\n        (dx, dy) = paddle.grad(\n            [out], [x, y], [dout], retain_graph=True, create_graph=True\n        )\n        ddx = paddle.to_tensor(\n            self.input_numpy_ddx, stop_gradient=False, dtype=\'complex64\'\n        )\n        ddy = paddle.to_tensor(\n            self.input_numpy_ddy, stop_gradient=False, dtype=\'complex64\'\n        )\n        (dy_double_grad,) = paddle.grad(\n            [dx, dy],\n            [y],\n            [ddx, ddy],\n            retain_graph=True,\n            create_graph=True,\n        )\n        d_x, d_y, d_dout, d_ddx, d_ddy = paddle.grad(\n            [dy_double_grad],\n            [x, y, dout, ddx, ddy],\n            retain_graph=False,\n            create_graph=False,\n        )\n        return d_x, d_y, d_dout, d_ddx, d_ddy\n\n    # case1: no d_ddout,no d_dx, dims = 1\n    def test_matmul_triple_grad_case1(self):\n        def init_data():\n            self.input_numpy_x = np.random.random([3]).astype(\n                \'float32\'\n            ) + 1j * np.random.random(\n                [\n                    3,\n                ]\n            ).astype(\n                \'float32\'\n            )\n            self.input_numpy_y = np.random.random([3]).astype(\n                \'float32\'\n            ) + 1j * np.random.random(\n                [\n                    3,\n                ]\n            ).astype(\n                \'float32\'\n            )\n            self.input_numpy_dout = np.ones(\n                [\n                    1,\n                ],\n                dtype="float32",\n            )\n            self.input_numpy_ddx = np.ones(\n                [\n                    3,\n                ],\n                dtype="float32",\n            )\n            self.input_numpy_ddy = np.ones(\n                [\n                    3,\n                ],\n                dtype="float32",\n            )\n            self.input_numpy_ddx_conj = np.conjugate(self.input_numpy_ddx)\n            self.input_numpy_dout_conj = np.conjugate(self.input_numpy_dout)\n\n        init_data()\n        d_x_expected = np.zeros(\n            [\n                3,\n            ],\n            dtype="float32",\n        )\n        d_y_expected = np.zeros(\n            [\n                3,\n            ],\n            dtype="float32",\n        )\n        d_dout_expected = np.matmul(\n            self.input_numpy_ddx_conj,\n            np.ones(\n                [\n                    3,\n                ],\n                dtype="float32",\n            ),\n        )\n        d_ddx_expected = (\n            np.ones(\n                [\n                    3,\n                ],\n                dtype="float32",\n            )\n            * self.input_numpy_dout_conj[0]\n        )\n        d_ddy_expected = np.zeros(\n            [\n                3,\n            ],\n            dtype="float32",\n        )\n        expected_results = (\n            d_x_expected,\n            d_y_expected,\n            d_dout_expected,\n            d_ddx_expected,\n            d_ddy_expected,\n        )\n\n        for place in self.places:\n            paddle.device.set_device(place)\n            actual_results = self.actual()\n            for expected_result, actual_result in zip(\n                expected_results, actual_results\n            ):\n                np.testing.assert_allclose(\n                    expected_result, actual_result, rtol=1e-6\n                )\n'

class TestDygraphTripleGradMatmulcase5(TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.input_numpy_x = None
        self.input_numpy_y = None
        self.input_numpy_dout = None
        self.input_numpy_ddx = None
        self.input_numpy_ddy = None
        self.places = ['cpu']
        if paddle.is_compiled_with_cuda():
            self.places.append('gpu')

    def actual(self):
        if False:
            print('Hello World!')
        x = paddle.to_tensor(self.input_numpy_x, stop_gradient=False, dtype='float32')
        y = paddle.to_tensor(self.input_numpy_y, stop_gradient=False, dtype='float32')
        out = paddle.matmul(x, y, False, False)
        dout = paddle.to_tensor(self.input_numpy_dout, stop_gradient=False, dtype='float32')
        (dx, dy) = paddle.grad([out], [x, y], [dout], retain_graph=True, create_graph=True)
        ddx = paddle.to_tensor(self.input_numpy_ddx, stop_gradient=False, dtype='float32')
        ddy = paddle.to_tensor(self.input_numpy_ddy, stop_gradient=False, dtype='float32')
        (dx_double_grad,) = paddle.grad([dx, dy], [x], [ddx, ddy], retain_graph=True, create_graph=True)
        (d_dout, d_ddy) = paddle.grad([dx_double_grad], [dout, ddy], retain_graph=False, create_graph=False)
        return (d_dout, d_ddy)

    def test_matmul_triple_grad_case1(self):
        if False:
            return 10

        def init_data():
            if False:
                print('Hello World!')
            self.input_numpy_x = np.random.random([3, 3]).astype('float32')
            self.input_numpy_y = np.random.random([3, 3]).astype('float32')
            self.input_numpy_dout = np.ones([3, 3], dtype='float32')
            self.input_numpy_ddx = np.ones([3, 3], dtype='float32')
            self.input_numpy_ddy = np.ones([3, 3], dtype='float32')
        init_data()
        d_dout_expected = np.ones([3, 3], dtype='float32') * 3
        d_ddy_expected = np.ones([3, 3], dtype='float32') * 3
        expected_results = (d_dout_expected, d_ddy_expected)
        for place in self.places:
            paddle.device.set_device(place)
            actual_results = self.actual()
            for (expected_result, actual_result) in zip(expected_results, actual_results):
                np.testing.assert_allclose(expected_result, actual_result, rtol=1e-06)

    def test_matmul_triple_grad_case2(self):
        if False:
            print('Hello World!')

        def init_data():
            if False:
                for i in range(10):
                    print('nop')
            self.input_numpy_x = np.random.random([3]).astype('float32')
            self.input_numpy_y = np.random.random([3]).astype('float32')
            self.input_numpy_dout = np.ones([1], dtype='float32')
            self.input_numpy_ddx = np.ones([3], dtype='float32')
            self.input_numpy_ddy = np.ones([3], dtype='float32')
        init_data()
        d_dout_expected = np.ones([1], dtype='float32') * 3
        d_ddy_expected = np.ones([3], dtype='float32')
        expected_results = (d_dout_expected, d_ddy_expected)
        for place in self.places:
            paddle.device.set_device(place)
            actual_results = self.actual()
            for (expected_result, actual_result) in zip(expected_results, actual_results):
                np.testing.assert_allclose(expected_result, actual_result, rtol=1e-06)

    def test_matmul_triple_grad_case3(self):
        if False:
            for i in range(10):
                print('nop')

        def init_data():
            if False:
                print('Hello World!')
            self.input_numpy_x = np.random.random([3, 1]).astype('float32')
            self.input_numpy_y = np.random.random([1]).astype('float32')
            self.input_numpy_dout = np.ones([3], dtype='float32')
            self.input_numpy_ddx = np.ones([3, 1], dtype='float32')
            self.input_numpy_ddy = np.ones([1], dtype='float32')
        init_data()
        d_dout_expected = np.ones([3], dtype='float32')
        d_ddy_expected = np.ones([1], dtype='float32') * 3
        expected_results = (d_dout_expected, d_ddy_expected)
        for place in self.places:
            paddle.device.set_device(place)
            actual_results = self.actual()
            for (expected_result, actual_result) in zip(expected_results, actual_results):
                np.testing.assert_allclose(expected_result, actual_result, rtol=1e-06)
'\nTODO(Ruting) test complex dtype when composite api support\n# d_ddout is none, d_dy is none, dtype is complex64\nclass TestDygraphTripleGradMatmulcase6(TestCase):\n    def setUp(self):\n        self.input_numpy_x = None\n        self.input_numpy_y = None\n        self.input_numpy_dout = None\n        self.input_numpy_ddx = None\n        self.input_numpy_ddy = None\n        self.input_numpy_ddy_conj = None\n        self.input_numpy_dout_conj = None\n        self.places = ["cpu"]\n        if paddle.is_compiled_with_cuda():\n            self.places.append("gpu")\n\n    def actual(self):\n        x = paddle.to_tensor(\n            self.input_numpy_x, stop_gradient=False, dtype=\'complex64\'\n        )\n        y = paddle.to_tensor(\n            self.input_numpy_y, stop_gradient=False, dtype=\'complex64\'\n        )\n        out = paddle.matmul(x, y, False, False)\n\n        dout = paddle.to_tensor(\n            self.input_numpy_dout, stop_gradient=False, dtype=\'complex64\'\n        )\n        (dx, dy) = paddle.grad(\n            [out], [x, y], [dout], retain_graph=True, create_graph=True\n        )\n        ddx = paddle.to_tensor(\n            self.input_numpy_ddx, stop_gradient=False, dtype=\'complex64\'\n        )\n        ddy = paddle.to_tensor(\n            self.input_numpy_ddy, stop_gradient=False, dtype=\'complex64\'\n        )\n        (dx_double_grad,) = paddle.grad(\n            [dx, dy],\n            [x],\n            [ddx, ddy],\n            retain_graph=True,\n            create_graph=True,\n        )\n        d_x, d_y, d_dout, d_ddx, d_ddy = paddle.grad(\n            [dx_double_grad],\n            [x, y, dout, ddx, ddy],\n            retain_graph=False,\n            create_graph=False,\n        )\n        return d_x, d_y, d_dout, d_ddx, d_ddy\n\n    # case1: no d_ddout,no d_dy, dims = 1\n    def test_matmul_triple_grad_case1(self):\n        def init_data():\n            self.input_numpy_x = np.random.random([3]).astype(\n                \'float32\'\n            ) + 1j * np.random.random(\n                [\n                    3,\n                ]\n            ).astype(\n                \'float32\'\n            )\n            self.input_numpy_y = np.random.random([3]).astype(\n                \'float32\'\n            ) + 1j * np.random.random(\n                [\n                    3,\n                ]\n            ).astype(\n                \'float32\'\n            )\n            self.input_numpy_dout = np.ones(\n                [\n                    1,\n                ],\n                dtype="float32",\n            )\n            self.input_numpy_ddx = np.ones(\n                [\n                    3,\n                ],\n                dtype="float32",\n            )\n            self.input_numpy_ddy = np.ones(\n                [\n                    3,\n                ],\n                dtype="float32",\n            )\n            self.input_numpy_ddy_conj = np.conjugate(self.input_numpy_ddy)\n            self.input_numpy_dout_conj = np.conjugate(self.input_numpy_dout)\n\n        init_data()\n        d_x_expected = np.zeros(\n            [\n                3,\n            ],\n            dtype="float32",\n        )\n        d_y_expected = np.zeros(\n            [\n                3,\n            ],\n            dtype="float32",\n        )\n        d_dout_expected = np.matmul(\n            self.input_numpy_ddy_conj,\n            np.ones(\n                [\n                    3,\n                ],\n                dtype="float32",\n            ),\n        )\n        d_ddx_expected = np.zeros(\n            [\n                3,\n            ],\n            dtype="float32",\n        )\n        d_ddy_expected = (\n            np.ones(\n                [\n                    3,\n                ],\n                dtype="float32",\n            )\n            * self.input_numpy_dout_conj[0]\n        )\n        expected_results = (\n            d_x_expected,\n            d_y_expected,\n            d_dout_expected,\n            d_ddx_expected,\n            d_ddy_expected,\n        )\n\n        for place in self.places:\n            paddle.device.set_device(place)\n            actual_results = self.actual()\n            for expected_result, actual_result in zip(\n                expected_results, actual_results\n            ):\n                np.testing.assert_allclose(\n                    expected_result, actual_result, rtol=1e-6\n                )\n'
if __name__ == '__main__':
    unittest.main()