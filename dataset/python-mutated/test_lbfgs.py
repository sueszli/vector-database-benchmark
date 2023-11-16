import unittest
import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.incubate.optimizer.functional.lbfgs import minimize_lbfgs
np.random.seed(123)

def test_static_graph(func, x0, line_search_fn='strong_wolfe', dtype='float32'):
    if False:
        return 10
    dimension = x0.shape[0]
    paddle.enable_static()
    main = paddle.static.Program()
    startup = paddle.static.Program()
    with paddle.static.program_guard(main, startup):
        X = paddle.static.data(name='x', shape=[dimension], dtype=dtype)
        Y = minimize_lbfgs(func, X, line_search_fn=line_search_fn, dtype=dtype)
    exe = paddle.static.Executor()
    exe.run(startup)
    return exe.run(main, feed={'x': x0}, fetch_list=[Y])

def test_static_graph_H0(func, x0, H0, dtype='float32'):
    if False:
        while True:
            i = 10
    paddle.enable_static()
    main = paddle.static.Program()
    startup = paddle.static.Program()
    with paddle.static.program_guard(main, startup):
        X = paddle.static.data(name='x', shape=[x0.shape[0]], dtype=dtype)
        H = paddle.static.data(name='h', shape=[H0.shape[0], H0.shape[1]], dtype=dtype)
        Y = minimize_lbfgs(func, X, initial_inverse_hessian_estimate=H, dtype=dtype)
    exe = paddle.static.Executor()
    exe.run(startup)
    return exe.run(main, feed={'x': x0, 'h': H0}, fetch_list=[Y])

def test_dynamic_graph(func, x0, H0=None, line_search_fn='strong_wolfe', dtype='float32'):
    if False:
        return 10
    paddle.disable_static()
    x0 = paddle.to_tensor(x0)
    if H0 is not None:
        H0 = paddle.to_tensor(H0)
    return minimize_lbfgs(func, x0, initial_inverse_hessian_estimate=H0, line_search_fn=line_search_fn, dtype=dtype)

class TestLbfgs(unittest.TestCase):

    def test_quadratic_nd(self):
        if False:
            for i in range(10):
                print('nop')
        for dimension in [1, 10]:
            minimum = np.random.random(size=[dimension]).astype('float32')
            scale = np.exp(np.random.random(size=[dimension]).astype('float32'))

            def func(x):
                if False:
                    return 10
                minimum_ = paddle.assign(minimum)
                scale_ = paddle.assign(scale)
                return paddle.sum(paddle.multiply(scale_, F.square_error_cost(x, minimum_)))
            x0 = np.random.random(size=[dimension]).astype('float32')
            results = test_static_graph(func, x0)
            np.testing.assert_allclose(minimum, results[2], rtol=1e-05)
            results = test_dynamic_graph(func, x0)
            np.testing.assert_allclose(minimum, results[2].numpy(), rtol=1e-05)

    def test_inf_minima(self):
        if False:
            for i in range(10):
                print('nop')
        extream_point = np.array([-1, 2]).astype('float32')

        def func(x):
            if False:
                for i in range(10):
                    print('nop')
            return x * x * x / 3.0 - (extream_point[0] + extream_point[1]) * x * x / 2 + extream_point[0] * extream_point[1] * x
        x0 = np.array([-1.7]).astype('float32')
        results = test_static_graph(func, x0)
        self.assertFalse(results[0][0])

    def test_multi_minima(self):
        if False:
            return 10

        def func(x):
            if False:
                while True:
                    i = 10
            return 3 * x ** 4 + 0.4 * x ** 3 - 5.64 * x ** 2 + 2.112 * x
        x0 = np.array([0.82], dtype='float64')
        results = test_static_graph(func, x0, dtype='float64')
        np.testing.assert_allclose(0.8, results[2], rtol=1e-05)

    def test_rosenbrock(self):
        if False:
            print('Hello World!')
        a = np.random.random(size=[1]).astype('float32')
        minimum = [a.item(), (a ** 2).item()]
        b = np.random.random(size=[1]).astype('float32')

        def func(position):
            if False:
                print('Hello World!')
            (x, y) = (position[0], position[1])
            c = (a - x) ** 2 + b * (y - x ** 2) ** 2
            return c[0]
        x0 = np.random.random(size=[2]).astype('float32')
        results = test_dynamic_graph(func, x0)
        np.testing.assert_allclose(minimum, results[2], rtol=1e-05)

    def test_exception(self):
        if False:
            print('Hello World!')

        def func(x):
            if False:
                print('Hello World!')
            return paddle.dot(x, x)
        x0 = np.random.random(size=[2]).astype('float32')
        H0 = np.array([[2.0, 0.0], [0.0, 0.9]]).astype('float32')
        x1 = np.random.random(size=[2]).astype('int32')
        self.assertRaises(ValueError, test_static_graph, func, x1, dtype='int32')
        results = test_static_graph_H0(func, x0, H0, dtype='float32')
        np.testing.assert_allclose([0.0, 0.0], results[2], rtol=1e-05)
        self.assertTrue(results[0][0])
        x2 = np.random.random(size=[2]).astype('float64')
        H1 = np.array([[1.0, 2.0], [3.0, 1.0]]).astype('float64')
        self.assertRaises(ValueError, test_static_graph_H0, func, x2, H0=H1, dtype='float64')
if __name__ == '__main__':
    unittest.main()