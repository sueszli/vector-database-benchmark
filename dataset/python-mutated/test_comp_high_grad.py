import sys
import unittest
sys.path.append('../../../legacy_test')
import gradient_checker
import numpy as np
import parameterized as param
from decorator_helper import prog_scope
import paddle
from paddle import base
from paddle.base import core

@param.parameterized_class(('shape1', 'shape2'), [([2, 3, 4], [2, 3, 4]), ([2, 3, 3, 4], [3, 1, 4]), ([2, 3, 3, 4], [3, 1, 1]), ([2, 3, 3, 4], [2, 3, 1, 4]), ([2, 3, 3, 4], [2, 3, 1, 1])])
class TestAddHighGradCheck(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        cls.shape1 = cls.shape1
        cls.shape2 = cls.shape2

    def add_wrapper(self, x):
        if False:
            i = 10
            return i + 15
        return paddle.add(x[0], x[1])

    @prog_scope()
    def func_double(self, place):
        if False:
            while True:
                i = 10
        shape1 = self.shape1
        shape2 = self.shape2
        eps = 0.0005
        dtype = np.float64
        x = paddle.static.data('x', shape1, dtype=dtype)
        y = paddle.static.data('y', shape2, dtype=dtype)
        x.persistable = True
        y.persistable = True
        out = paddle.add(x, y)
        x_arr = np.random.uniform(-1, 1, shape1).astype(dtype)
        y_arr = np.random.uniform(-2, 2, shape2).astype(dtype)
        x_arr[np.abs(x_arr) < 0.005] = 0.002
        y_arr[np.abs(y_arr) < 0.005] = 0.002
        from paddle.base import core
        core._set_prim_backward_enabled(True)
        gradient_checker.double_grad_check([x, y], y=out, x_init=[x_arr, y_arr], place=place, eps=eps)
        gradient_checker.double_grad_check_for_dygraph(self.add_wrapper, [x, y], y=out, x_init=[x_arr, y_arr], place=place)
        core._set_prim_backward_enabled(False)

    @prog_scope()
    def func_triple(self, place):
        if False:
            i = 10
            return i + 15
        shape1 = self.shape1
        shape2 = self.shape2
        eps = 0.0005
        dtype = np.float64
        x = paddle.static.data('x', shape1, dtype=dtype)
        y = paddle.static.data('y', shape2, dtype=dtype)
        x.persistable = True
        y.persistable = True
        out = paddle.add(x, y)
        x_arr = np.random.uniform(-1, 1, shape1).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape2).astype(dtype)
        x_arr[np.abs(x_arr) < 0.005] = 0.002
        y_arr[np.abs(y_arr) < 0.005] = 0.002
        from paddle.base import core
        core._set_prim_backward_enabled(True)
        gradient_checker.triple_grad_check([x, y], y=out, x_init=[x_arr, y_arr], place=place, eps=eps)
        gradient_checker.triple_grad_check_for_dygraph(self.add_wrapper, [x, y], y=out, x_init=[x_arr, y_arr], place=place)
        core._set_prim_backward_enabled(False)

    def test_high_grad(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func_double(p)
            self.func_triple(p)

@param.parameterized_class(('shape1', 'shape2'), [([2, 3, 4], [2, 3, 4]), ([2, 3, 3, 4], [3, 1, 4]), ([2, 3, 3, 4], [3, 1, 1]), ([2, 3, 3, 4], [2, 3, 1, 4]), ([2, 3, 3, 4], [2, 3, 1, 1])])
class TestSubtractHighGradCheck(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        cls.shape1 = cls.shape1
        cls.shape2 = cls.shape2

    def subtract_wrapper(self, x):
        if False:
            print('Hello World!')
        return paddle.subtract(x[0], x[1])

    @prog_scope()
    def func_double(self, place):
        if False:
            print('Hello World!')
        shape1 = self.shape1
        shape2 = self.shape2
        eps = 0.0005
        dtype = np.float64
        x = paddle.static.data('x', shape1, dtype=dtype)
        y = paddle.static.data('y', shape2, dtype=dtype)
        x.persistable = True
        y.persistable = True
        out = paddle.subtract(x, y)
        x_arr = np.random.uniform(-1, 1, shape1).astype(dtype)
        y_arr = np.random.uniform(-2, 2, shape2).astype(dtype)
        x_arr[np.abs(x_arr) < 0.005] = 0.002
        y_arr[np.abs(y_arr) < 0.005] = 0.002
        from paddle.base import core
        core._set_prim_backward_enabled(True)
        gradient_checker.double_grad_check([x, y], y=out, x_init=[x_arr, y_arr], place=place, eps=eps)
        gradient_checker.double_grad_check_for_dygraph(self.subtract_wrapper, [x, y], y=out, x_init=[x_arr, y_arr], place=place)
        core._set_prim_backward_enabled(False)

    @prog_scope()
    def func_triple(self, place):
        if False:
            print('Hello World!')
        shape1 = self.shape1
        shape2 = self.shape2
        eps = 0.0005
        dtype = np.float64
        x = paddle.static.data('x', shape1, dtype=dtype)
        y = paddle.static.data('y', shape2, dtype=dtype)
        x.persistable = True
        y.persistable = True
        out = paddle.subtract(x, y)
        x_arr = np.random.uniform(-1, 1, shape1).astype(dtype)
        y_arr = np.random.uniform(-2, 2, shape2).astype(dtype)
        x_arr[np.abs(x_arr) < 0.005] = 0.002
        y_arr[np.abs(y_arr) < 0.005] = 0.002
        from paddle.base import core
        core._set_prim_backward_enabled(True)
        gradient_checker.triple_grad_check([x, y], y=out, x_init=[x_arr, y_arr], place=place, eps=eps)
        gradient_checker.triple_grad_check_for_dygraph(self.subtract_wrapper, [x, y], y=out, x_init=[x_arr, y_arr], place=place)
        core._set_prim_backward_enabled(False)

    def test_high_grad(self):
        if False:
            while True:
                i = 10
        paddle.enable_static()
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func_double(p)
            self.func_triple(p)

@param.parameterized_class(('shape1', 'shape2'), [([2, 3, 4], [2, 3, 4]), ([2, 3, 3, 4], [3, 1, 4]), ([2, 3, 3, 4], [3, 1, 1]), ([2, 3, 3, 4], [2, 3, 1, 4]), ([2, 3, 3, 4], [2, 3, 1, 1])])
class TestMultiplyHighGradCheck(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        cls.shape1 = cls.shape1
        cls.shape2 = cls.shape2

    def multiply_wrapper(self, x):
        if False:
            while True:
                i = 10
        return paddle.multiply(x[0], x[1])

    @prog_scope()
    def func_double(self, place):
        if False:
            while True:
                i = 10
        shape1 = self.shape1
        shape2 = self.shape2
        eps = 0.0005
        dtype = np.float64
        x = paddle.static.data('x', shape1, dtype=dtype)
        y = paddle.static.data('y', shape2, dtype=dtype)
        x.persistable = True
        y.persistable = True
        out = paddle.multiply(x, y)
        x_arr = np.random.uniform(-1, 1, shape1).astype(dtype)
        y_arr = np.random.uniform(-2, 2, shape2).astype(dtype)
        x_arr[np.abs(x_arr) < 0.005] = 0.002
        y_arr[np.abs(y_arr) < 0.005] = 0.002
        from paddle.base import core
        core._set_prim_backward_enabled(True)
        gradient_checker.double_grad_check([x, y], y=out, x_init=[x_arr, y_arr], place=place, eps=eps)
        gradient_checker.double_grad_check_for_dygraph(self.multiply_wrapper, [x, y], y=out, x_init=[x_arr, y_arr], place=place)
        core._set_prim_backward_enabled(False)

    @prog_scope()
    def func_triple(self, place):
        if False:
            for i in range(10):
                print('nop')
        shape1 = self.shape1
        shape2 = self.shape2
        eps = 0.0005
        dtype = np.float64
        x = paddle.static.data('x', shape1, dtype=dtype)
        y = paddle.static.data('y', shape2, dtype=dtype)
        x.persistable = True
        y.persistable = True
        out = paddle.multiply(x, y)
        x_arr = np.random.uniform(-1, 1, shape1).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape2).astype(dtype)
        x_arr[np.abs(x_arr) < 0.005] = 0.002
        y_arr[np.abs(y_arr) < 0.005] = 0.002
        from paddle.base import core
        core._set_prim_backward_enabled(True)
        gradient_checker.triple_grad_check([x, y], y=out, x_init=[x_arr, y_arr], place=place, eps=eps)
        gradient_checker.triple_grad_check_for_dygraph(self.multiply_wrapper, [x, y], y=out, x_init=[x_arr, y_arr], place=place)
        core._set_prim_backward_enabled(False)

    def test_high_grad(self):
        if False:
            print('Hello World!')
        paddle.enable_static()
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func_double(p)
            self.func_triple(p)

@param.parameterized_class('shape1', [([2],), ([2, 3],), ([2, 3, 4],), ([2, 3, 3, 4],)])
class TestSiluHighGradCheck(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        cls.shape1 = cls.shape1

    def silu_wrapper(self, x):
        if False:
            print('Hello World!')
        return paddle.nn.functional.silu(x[0])

    @prog_scope()
    def func_double(self, place):
        if False:
            while True:
                i = 10
        shape1 = self.shape1
        eps = 0.0005
        dtype = np.float64
        x = paddle.static.data('x', shape1, dtype=dtype)
        x.stop_gradient = False
        x.persistable = True
        out = paddle.nn.functional.silu(x)
        x_arr = np.random.uniform(-1, 1, shape1).astype(dtype)
        x_arr[np.abs(x_arr) < 0.005] = 0.002
        from paddle.base import core
        core._set_prim_backward_enabled(True)
        gradient_checker.double_grad_check([x], y=out, x_init=[x_arr], place=place, eps=eps)
        gradient_checker.double_grad_check_for_dygraph(self.silu_wrapper, [x], y=out, x_init=[x_arr], place=place)
        core._set_prim_backward_enabled(False)

    @prog_scope()
    def func_triple(self, place):
        if False:
            for i in range(10):
                print('nop')
        shape1 = self.shape1
        eps = 0.0005
        dtype = np.float64
        x = paddle.static.data('x', shape1, dtype=dtype)
        x.stop_gradient = False
        x.persistable = True
        out = paddle.nn.functional.silu(x)
        x_arr = np.random.uniform(-1, 1, shape1).astype(dtype)
        x_arr[np.abs(x_arr) < 0.005] = 0.002
        from paddle.base import core
        core._set_prim_backward_enabled(True)
        gradient_checker.triple_grad_check([x], y=out, x_init=[x_arr], place=place, eps=eps)
        gradient_checker.triple_grad_check_for_dygraph(self.silu_wrapper, [x], y=out, x_init=[x_arr], place=place)
        core._set_prim_backward_enabled(False)

    def test_high_grad(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func_double(p)
            self.func_triple(p)
if __name__ == '__main__':
    unittest.main()