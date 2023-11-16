import unittest
import gradient_checker
import numpy as np
from decorator_helper import prog_scope
import paddle
from paddle import base
from paddle.base import core

class TestElementwiseMulDoubleGradCheck(unittest.TestCase):

    @prog_scope()
    def func(self, place):
        if False:
            while True:
                i = 10
        shape = [2, 3, 4, 5]
        eps = 0.005
        dtype = np.float64
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.data('y', shape, dtype)
        x.persistable = True
        y.persistable = True
        out = paddle.multiply(x, y)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        gradient_checker.double_grad_check([x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps)

    def test_grad(self):
        if False:
            return 10
        paddle.enable_static()
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)

class TestElementwiseMulBroadcastDoubleGradCheck(unittest.TestCase):

    @prog_scope()
    def func(self, place):
        if False:
            i = 10
            return i + 15
        shape = [2, 3, 4, 5]
        eps = 0.005
        dtype = np.float64
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.data('y', shape[:-1], dtype)
        x.persistable = True
        y.persistable = True
        out = paddle.tensor.math._multiply_with_axis(x, y, axis=0)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape[:-1]).astype(dtype)
        gradient_checker.double_grad_check([x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps)

    def test_grad(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)

class TestElementwiseAddDoubleGradCheck(unittest.TestCase):

    @prog_scope()
    def func(self, place):
        if False:
            return 10
        shape = [2, 3, 4, 5]
        eps = 0.005
        dtype = np.float64
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.data('y', shape, dtype)
        x.persistable = True
        y.persistable = True
        out = paddle.add(x, y)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        gradient_checker.double_grad_check([x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps)

    def test_grad(self):
        if False:
            return 10
        paddle.enable_static()
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)

class TestElementwiseAddBroadcastDoubleGradCheck(unittest.TestCase):

    @prog_scope()
    def func(self, place):
        if False:
            print('Hello World!')
        shape = [2, 3, 4, 5]
        eps = 0.005
        dtype = np.float64
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.data('y', shape[:-1], dtype)
        x.persistable = True
        y.persistable = True
        out = paddle.tensor.math._add_with_axis(x, y, axis=0)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape[:-1]).astype(dtype)
        gradient_checker.double_grad_check([x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps)

    def test_grad(self):
        if False:
            return 10
        paddle.enable_static()
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)

class TestElementwiseSubDoubleGradCheck(unittest.TestCase):

    def subtract_wrapper(self, x):
        if False:
            return 10
        return paddle.subtract(x[0], x[1])

    @prog_scope()
    def func(self, place):
        if False:
            while True:
                i = 10
        shape = [2, 3, 4, 5]
        eps = 0.005
        dtype = np.float64
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.data('y', shape, dtype)
        x.persistable = True
        y.persistable = True
        out = paddle.subtract(x, y)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        gradient_checker.double_grad_check([x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps)
        gradient_checker.double_grad_check_for_dygraph(self.subtract_wrapper, [x, y], out, x_init=[x_arr, y_arr], place=place)

    def test_grad(self):
        if False:
            while True:
                i = 10
        paddle.enable_static()
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)

class TestElementwiseSubBroadcastDoubleGradCheck(unittest.TestCase):

    @prog_scope()
    def func(self, place):
        if False:
            print('Hello World!')
        shape = [2, 3, 4, 5]
        eps = 0.005
        dtype = np.float64
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.data('y', shape[:-1], dtype)
        x.persistable = True
        y.persistable = True
        out = paddle.tensor.math._subtract_with_axis(x, y, axis=0)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape[:-1]).astype(dtype)
        gradient_checker.double_grad_check([x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps)

    def test_grad(self):
        if False:
            return 10
        paddle.enable_static()
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)

class TestElementwiseDivDoubleGradCheck(unittest.TestCase):

    def divide_wrapper(self, x):
        if False:
            while True:
                i = 10
        return paddle.divide(x[0], x[1])

    @prog_scope()
    def func(self, place):
        if False:
            while True:
                i = 10
        shape = [2, 3, 4, 5]
        eps = 0.0001
        dtype = np.float64
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.data('y', shape, dtype)
        x.persistable = True
        y.persistable = True
        out = paddle.tensor.math.divide(x, y)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr[np.abs(y_arr) < 0.005] = 0.02
        gradient_checker.double_grad_check([x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps, atol=0.001)
        gradient_checker.double_grad_check_for_dygraph(self.divide_wrapper, [x, y], out, x_init=[x_arr, y_arr], place=place, atol=0.001)

    def test_grad(self):
        if False:
            print('Hello World!')
        paddle.enable_static()
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)

class TestElementwiseDivBroadcastDoubleGradCheck(unittest.TestCase):

    @prog_scope()
    def func(self, place):
        if False:
            i = 10
            return i + 15
        shape = [2, 3, 4, 5]
        eps = 0.0001
        dtype = np.float64
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.data('y', shape[1:-1], dtype)
        x.persistable = True
        y.persistable = True
        out = paddle.tensor.math._divide_with_axis(x, y, axis=1)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape[1:-1]).astype(dtype)
        y_arr[np.abs(y_arr) < 0.005] = 0.02
        gradient_checker.double_grad_check([x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps, atol=0.001)

    def test_grad(self):
        if False:
            return 10
        paddle.enable_static()
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)

class TestElementwiseAddTripleGradCheck(unittest.TestCase):

    @prog_scope()
    def func(self, place):
        if False:
            print('Hello World!')
        shape = [2, 3, 4, 5]
        eps = 0.005
        dtype = np.float64
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.data('y', shape, dtype)
        x.persistable = True
        y.persistable = True
        out = paddle.add(x, y)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        gradient_checker.triple_grad_check([x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps)

    def test_grad(self):
        if False:
            print('Hello World!')
        paddle.enable_static()
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)

class TestElementwiseAddBroadcastTripleGradCheck(unittest.TestCase):

    @prog_scope()
    def func(self, place):
        if False:
            print('Hello World!')
        shape = [2, 3, 4, 5]
        eps = 0.005
        dtype = np.float64
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.data('y', shape[:-1], dtype)
        x.persistable = True
        y.persistable = True
        out = paddle.tensor.math._add_with_axis(x, y, axis=0)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape[:-1]).astype(dtype)
        gradient_checker.triple_grad_check([x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps)

    def test_grad(self):
        if False:
            print('Hello World!')
        paddle.enable_static()
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)

class TestElementwiseMulTripleGradCheck(unittest.TestCase):

    def multiply_wrapper(self, x):
        if False:
            for i in range(10):
                print('nop')
        return paddle.multiply(x[0], x[1])

    @prog_scope()
    def func(self, place):
        if False:
            print('Hello World!')
        shape = [2, 3, 4, 5]
        eps = 0.005
        dtype = np.float64
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.data('y', shape, dtype)
        x.persistable = True
        y.persistable = True
        out = paddle.multiply(x, y)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        gradient_checker.triple_grad_check([x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps)
        gradient_checker.triple_grad_check_for_dygraph(self.multiply_wrapper, [x, y], out, x_init=[x_arr, y_arr], place=place)

    def test_grad(self):
        if False:
            while True:
                i = 10
        paddle.enable_static()
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)

class TestElementwiseMulBroadcastTripleGradCheck(unittest.TestCase):

    @prog_scope()
    def func(self, place):
        if False:
            while True:
                i = 10
        shape = [2, 3, 4, 5]
        eps = 0.005
        dtype = np.float64
        x = paddle.static.data('x', shape, dtype)
        y = paddle.static.data('y', shape[:-1], dtype)
        x.persistable = True
        y.persistable = True
        out = paddle.tensor.math._add_with_axis(x, y, axis=0)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape[:-1]).astype(dtype)
        gradient_checker.triple_grad_check([x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps)

    def test_grad(self):
        if False:
            i = 10
            return i + 15
        paddle.enable_static()
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)
if __name__ == '__main__':
    unittest.main()