import unittest
import numpy as np
import paddle
from paddle.base import core
from paddle.static import Program, program_guard
DYNAMIC = 1
STATIC = 2

def _run_power(mode, x, y, device='cpu'):
    if False:
        for i in range(10):
            print('nop')
    if mode == DYNAMIC:
        paddle.disable_static()
        paddle.set_device(device)
        if isinstance(y, (int, float)):
            x_ = paddle.to_tensor(x)
            y_ = y
            res = paddle.pow(x_, y_)
            return res.numpy()
        else:
            x_ = paddle.to_tensor(x)
            y_ = paddle.to_tensor(y)
            res = paddle.pow(x_, y_)
            return res.numpy()
    elif mode == STATIC:
        paddle.enable_static()
        if isinstance(y, (int, float)):
            with program_guard(Program(), Program()):
                x_ = paddle.static.data(name='x', shape=x.shape, dtype=x.dtype)
                y_ = y
                res = paddle.pow(x_, y_)
                place = paddle.CPUPlace() if device == 'cpu' else paddle.CUDAPlace(0)
                exe = paddle.static.Executor(place)
                outs = exe.run(feed={'x': x}, fetch_list=[res])
                return outs[0]
        else:
            with program_guard(Program(), Program()):
                x_ = paddle.static.data(name='x', shape=x.shape, dtype=x.dtype)
                y_ = paddle.static.data(name='y', shape=y.shape, dtype=y.dtype)
                res = paddle.pow(x_, y_)
                place = paddle.CPUPlace() if device == 'cpu' else paddle.CUDAPlace(0)
                exe = paddle.static.Executor(place)
                outs = exe.run(feed={'x': x, 'y': y}, fetch_list=[res])
                return outs[0]

class TestPowerAPI(unittest.TestCase):
    """TestPowerAPI."""

    def setUp(self):
        if False:
            while True:
                i = 10
        self.places = ['cpu']
        if core.is_compiled_with_cuda():
            self.places.append('gpu')

    def test_power(self):
        if False:
            for i in range(10):
                print('nop')
        'test_power.'
        np.random.seed(7)
        for place in self.places:
            dims = (np.random.randint(200, 300),)
            x = (np.random.rand(*dims) * 10).astype(np.float64)
            y = np.random.rand() * 10
            res = _run_power(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
            res = _run_power(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
            dims = (np.random.randint(200, 300),)
            x = (np.random.rand(*dims) * 10).astype(np.float64)
            y = int(np.random.rand() * 10)
            res = _run_power(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
            res = _run_power(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
            x = (np.random.rand(*dims) * 10).astype(np.int64)
            y = int(np.random.rand() * 10)
            res = _run_power(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
            res = _run_power(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
            dims = (np.random.randint(200, 300),)
            x = (np.random.rand(*dims) * 10).astype(np.float64)
            y = (np.random.rand(*dims) * 10).astype(np.float64)
            res = _run_power(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
            res = _run_power(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
            dims = (np.random.randint(200, 300),)
            x = (np.random.rand(*dims) * 10).astype(np.int64)
            y = (np.random.rand(*dims) * 10).astype(np.int64)
            res = _run_power(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
            res = _run_power(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
            dims = (np.random.randint(200, 300),)
            x = (np.random.rand(*dims) * 10).astype(np.int32)
            y = (np.random.rand(*dims) * 10).astype(np.int32)
            res = _run_power(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
            res = _run_power(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
            dims = (np.random.randint(200, 300),)
            x = (np.random.rand(*dims) * 10).astype(np.float32)
            y = (np.random.rand(*dims) * 10).astype(np.float32)
            res = _run_power(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
            res = _run_power(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
            dims = (np.random.randint(2, 10), np.random.randint(5, 10))
            x = np.random.rand() * 10
            y = (np.random.rand(*dims) * 10).astype(np.float32)
            res = _run_power(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
            dims = (np.random.randint(2, 10), np.random.randint(5, 10))
            x = (np.random.rand(*dims) * 10).astype(np.float32)
            y = np.random.rand() * 10
            res = _run_power(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
            res = _run_power(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
            dims = (np.random.randint(1, 10), np.random.randint(5, 10), np.random.randint(5, 10))
            x = (np.random.rand(*dims) * 10).astype(np.float64)
            y = (np.random.rand(dims[-1]) * 10).astype(np.float64)
            res = _run_power(DYNAMIC, x, y)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
            res = _run_power(STATIC, x, y)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)

class TestPowerError(unittest.TestCase):
    """TestPowerError."""

    def test_errors(self):
        if False:
            i = 10
            return i + 15
        'test_errors.'
        np.random.seed(7)
        dims = (np.random.randint(1, 10), np.random.randint(5, 10), np.random.randint(5, 10))
        x = (np.random.rand(*dims) * 10).astype(np.float64)
        y = (np.random.rand(dims[-1] + 1) * 10).astype(np.float64)
        self.assertRaises(ValueError, _run_power, DYNAMIC, x, y)
        self.assertRaises(ValueError, _run_power, STATIC, x, y)
        dims = (np.random.randint(1, 10), np.random.randint(5, 10), np.random.randint(5, 10))
        x = (np.random.rand(*dims) * 10).astype(np.float64)
        y = (np.random.rand(dims[-1] + 1) * 10).astype(np.int8)
        self.assertRaises(TypeError, paddle.pow, x, y)
        dims = (np.random.randint(200, 300),)
        x = (np.random.rand(*dims) * 10).astype(np.float64)
        y = int(np.random.rand() * 10)
        self.assertRaises(TypeError, paddle.pow, x, str(y))
if __name__ == '__main__':
    unittest.main()