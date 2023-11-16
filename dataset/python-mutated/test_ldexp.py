import unittest
import numpy as np
import paddle
from paddle.base import core
from paddle.static import Program, program_guard
DYNAMIC = 1
STATIC = 2

def _run_ldexp(mode, x, y, device='cpu'):
    if False:
        print('Hello World!')
    if mode == DYNAMIC:
        paddle.disable_static()
        paddle.set_device(device)
        x_ = paddle.to_tensor(x)
        if isinstance(y, int):
            y_ = y
        else:
            y_ = paddle.to_tensor(y)
        res = paddle.ldexp(x_, y_)
        return res.numpy()
    elif mode == STATIC:
        paddle.enable_static()
        if isinstance(y, int):
            with program_guard(Program(), Program()):
                x_ = paddle.static.data(name='x', shape=x.shape, dtype=x.dtype)
                y_ = y
                res = paddle.ldexp(x_, y_)
                place = paddle.CPUPlace() if device == 'cpu' else paddle.CUDAPlace(0)
                exe = paddle.static.Executor(place)
                outs = exe.run(feed={'x': x, 'y': y}, fetch_list=[res])
                return outs[0]
        else:
            with program_guard(Program(), Program()):
                x_ = paddle.static.data(name='x', shape=x.shape, dtype=x.dtype)
                y_ = paddle.static.data(name='y', shape=y.shape, dtype=y.dtype)
                res = paddle.ldexp(x_, y_)
                place = paddle.CPUPlace() if device == 'cpu' else paddle.CUDAPlace(0)
                exe = paddle.static.Executor(place)
                outs = exe.run(feed={'x': x, 'y': y}, fetch_list=[res])
                return outs[0]

def check_dtype(input, desired_dtype):
    if False:
        print('Hello World!')
    if input.dtype != desired_dtype:
        raise ValueError('The expected data type to be obtained is {}, but got {}'.format(desired_dtype, input.dtype))

class TestLdexpAPI(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.places = ['cpu']
        if core.is_compiled_with_cuda():
            self.places.append('gpu')

    def test_ldexp(self):
        if False:
            return 10
        np.random.seed(7)
        for place in self.places:
            dims = (np.random.randint(200, 300),)
            x = (np.random.rand(*dims) * 10).astype(np.float64)
            y = np.random.randint(-10, 10, dims).astype(np.int32)
            res = _run_ldexp(DYNAMIC, x, y, place)
            check_dtype(res, np.float64)
            np.testing.assert_allclose(res, np.ldexp(x, y))
            res = _run_ldexp(STATIC, x, y, place)
            check_dtype(res, np.float64)
            np.testing.assert_allclose(res, np.ldexp(x, y))
            dims = (np.random.randint(200, 300),)
            x = (np.random.rand(*dims) * 10).astype(np.float32)
            y = np.random.randint(-10, 10, dims).astype(np.int32)
            res = _run_ldexp(DYNAMIC, x, y, place)
            check_dtype(res, np.float32)
            np.testing.assert_allclose(res, np.ldexp(x, y))
            res = _run_ldexp(STATIC, x, y, place)
            check_dtype(res, np.float32)
            np.testing.assert_allclose(res, np.ldexp(x, y))
            dims = (np.random.randint(200, 300),)
            x = np.random.randint(-10, 10, dims).astype(np.int64)
            y = np.random.randint(-10, 10, dims).astype(np.int32)
            res = _run_ldexp(DYNAMIC, x, y, place)
            check_dtype(res, np.float32)
            np.testing.assert_allclose(res, np.ldexp(x, y))
            res = _run_ldexp(STATIC, x, y, place)
            check_dtype(res, np.float32)
            np.testing.assert_allclose(res, np.ldexp(x, y))
            dims = (np.random.randint(200, 300),)
            x = np.random.randint(-10, 10, dims).astype(np.int32)
            y = np.random.randint(-10, 10, dims).astype(np.int32)
            res = _run_ldexp(DYNAMIC, x, y, place)
            check_dtype(res, np.float32)
            np.testing.assert_allclose(res, np.ldexp(x, y))
            res = _run_ldexp(STATIC, x, y, place)
            check_dtype(res, np.float32)
            np.testing.assert_allclose(res, np.ldexp(x, y))
            dims = (np.random.randint(1, 10), np.random.randint(5, 10), np.random.randint(5, 10))
            x = (np.random.rand(*dims) * 10).astype(np.float64)
            y = np.random.randint(-10, 10, dims[-1]).astype(np.int32)
            res = _run_ldexp(DYNAMIC, x, y)
            check_dtype(res, np.float64)
            np.testing.assert_allclose(res, np.ldexp(x, y))
            res = _run_ldexp(STATIC, x, y)
            check_dtype(res, np.float64)
            np.testing.assert_allclose(res, np.ldexp(x, y))

class TestLdexpError(unittest.TestCase):
    """TestLdexpError."""

    def test_errors(self):
        if False:
            print('Hello World!')
        'test_errors.'
        np.random.seed(7)
        dims = (np.random.randint(200, 300),)
        x = (np.random.rand(*dims) * 10).astype(np.float64)
        y = np.random.randint(-10, 10, dims).astype(np.int32)
        self.assertRaises(TypeError, paddle.ldexp, x, paddle.to_tensor(y))
        dims = (np.random.randint(200, 300),)
        x = (np.random.rand(*dims) * 10).astype(np.float64)
        y = np.random.randint(-10, 10, dims).astype(np.int32)
        self.assertRaises(TypeError, paddle.ldexp, paddle.to_tensor(x), y)
if __name__ == '__main__':
    unittest.main()