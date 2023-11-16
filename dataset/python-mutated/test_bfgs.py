import unittest
import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.incubate.optimizer.functional.bfgs import minimize_bfgs
np.random.seed(123)

def test_static_graph(func, x0, line_search_fn='strong_wolfe', dtype='float32'):
    if False:
        for i in range(10):
            print('nop')
    dimension = x0.shape[0]
    paddle.enable_static()
    main = paddle.static.Program()
    startup = paddle.static.Program()
    with paddle.static.program_guard(main, startup):
        X = paddle.static.data(name='x', shape=[dimension], dtype=dtype)
        Y = minimize_bfgs(func, X, line_search_fn=line_search_fn, dtype=dtype)
    exe = paddle.static.Executor()
    exe.run(startup)
    return exe.run(main, feed={'x': x0}, fetch_list=[Y])

def test_static_graph_H0(func, x0, H0, dtype='float32'):
    if False:
        for i in range(10):
            print('nop')
    paddle.enable_static()
    main = paddle.static.Program()
    startup = paddle.static.Program()
    with paddle.static.program_guard(main, startup):
        X = paddle.static.data(name='x', shape=[x0.shape[0]], dtype=dtype)
        H = paddle.static.data(name='h', shape=[H0.shape[0], H0.shape[1]], dtype=dtype)
        Y = minimize_bfgs(func, X, initial_inverse_hessian_estimate=H, dtype=dtype)
    exe = paddle.static.Executor()
    exe.run(startup)
    return exe.run(main, feed={'x': x0, 'h': H0}, fetch_list=[Y])

def test_dynamic_graph(func, x0, H0=None, line_search_fn='strong_wolfe', dtype='float32'):
    if False:
        while True:
            i = 10
    paddle.disable_static()
    x0 = paddle.to_tensor(x0)
    if H0 is not None:
        H0 = paddle.to_tensor(H0)
    return minimize_bfgs(func, x0, initial_inverse_hessian_estimate=H0, line_search_fn=line_search_fn, dtype=dtype)

class TestBfgs(unittest.TestCase):

    def test_quadratic_nd(self):
        if False:
            for i in range(10):
                print('nop')
        for dimension in [1, 10]:
            minimum = np.random.random(size=[dimension]).astype('float32')
            scale = np.exp(np.random.random(size=[dimension]).astype('float32'))

            def func(x):
                if False:
                    while True:
                        i = 10
                minimum_ = paddle.assign(minimum)
                scale_ = paddle.assign(scale)
                return paddle.sum(paddle.multiply(scale_, F.square_error_cost(x, minimum_)))
            x0 = np.random.random(size=[dimension]).astype('float32')
            results = test_static_graph(func=func, x0=x0)
            np.testing.assert_allclose(minimum, results[2], rtol=1e-05, atol=1e-08)
            results = test_dynamic_graph(func=func, x0=x0)
            np.testing.assert_allclose(minimum, results[2].numpy(), rtol=1e-05, atol=1e-08)

    def test_inf_minima(self):
        if False:
            i = 10
            return i + 15
        extream_point = np.array([-1, 2]).astype('float32')

        def func(x):
            if False:
                while True:
                    i = 10
            return x * x * x / 3.0 - (extream_point[0] + extream_point[1]) * x * x / 2 + extream_point[0] * extream_point[1] * x
        x0 = np.array([-1.7]).astype('float32')
        results = test_static_graph(func, x0)
        self.assertFalse(results[0][0])

    def test_multi_minima(self):
        if False:
            while True:
                i = 10

        def func(x):
            if False:
                while True:
                    i = 10
            return 3 * x ** 4 + 0.4 * x ** 3 - 5.64 * x ** 2 + 2.112 * x
        x0 = np.array([0.82], dtype='float64')
        results = test_static_graph(func, x0, dtype='float64')
        np.testing.assert_allclose(0.8, results[2], rtol=1e-05, atol=1e-08)

    def test_rosenbrock(self):
        if False:
            print('Hello World!')
        a = np.random.random(size=[1]).astype('float32')
        minimum = [a.item(), (a ** 2).item()]
        b = np.random.random(size=[1]).astype('float32')

        def func(position):
            if False:
                i = 10
                return i + 15
            (x, y) = (position[0], position[1])
            c = (a - x) ** 2 + b * (y - x ** 2) ** 2
            return c[0]
        x0 = np.random.random(size=[2]).astype('float32')
        results = test_dynamic_graph(func, x0)
        np.testing.assert_allclose(minimum, results[2], rtol=1e-05, atol=1e-08)

    def test_exception(self):
        if False:
            for i in range(10):
                print('nop')

        def func(x):
            if False:
                while True:
                    i = 10
            return paddle.dot(x, x)
        x0 = np.random.random(size=[2]).astype('float32')
        H0 = np.array([[2.0, 0.0], [0.0, 0.9]]).astype('float32')
        results = test_static_graph_H0(func, x0, H0, dtype='float32')
        np.testing.assert_allclose([0.0, 0.0], results[2], rtol=1e-05, atol=1e-08)
        self.assertTrue(results[0][0])
        H1 = np.array([[1.0, 2.0], [2.0, 1.0]]).astype('float32')
        self.assertRaises(ValueError, test_dynamic_graph, func, x0, H0=H1)
        self.assertRaises(NotImplementedError, test_static_graph, func, x0, line_search_fn='other')
if __name__ == '__main__':
    unittest.main()