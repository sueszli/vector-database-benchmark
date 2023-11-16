import unittest
from typing import Optional
import numpy as np
import paddle
from paddle.base import core

def np_nan_to_num(x: np.ndarray, nan: float=0.0, posinf: Optional[float]=None, neginf: Optional[float]=None) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    return np.nan_to_num(x, True, nan=nan, posinf=posinf, neginf=neginf)

def np_nan_to_num_op(x: np.ndarray, nan: float, replace_posinf_with_max: bool, posinf: float, replace_neginf_with_min: bool, neginf: float) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    if replace_posinf_with_max:
        posinf = None
    if replace_neginf_with_min:
        neginf = None
    return np.nan_to_num(x, True, nan=nan, posinf=posinf, neginf=neginf)

def np_nan_to_num_grad(x: np.ndarray, dout: np.ndarray) -> np.ndarray:
    if False:
        while True:
            i = 10
    dx = np.copy(dout)
    dx[np.isnan(x) | (x == np.inf) | (x == -np.inf)] = 0
    return dx

class TestNanToNum(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() else paddle.CPUPlace()

    def test_static(self):
        if False:
            while True:
                i = 10
        x_np = np.array([[1, np.nan, -2], [np.inf, 0, -np.inf]]).astype(np.float32)
        out1_np = np_nan_to_num(x_np)
        out2_np = np_nan_to_num(x_np, 1.0)
        out3_np = np_nan_to_num(x_np, 1.0, 9.0)
        out4_np = np_nan_to_num(x_np, 1.0, 9.0, -12.0)
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', x_np.shape)
            out1 = paddle.nan_to_num(x)
            out2 = paddle.nan_to_num(x, 1.0)
            out3 = paddle.nan_to_num(x, 1.0, 9.0)
            out4 = paddle.nan_to_num(x, 1.0, 9.0, -12.0)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': x_np}, fetch_list=[out1, out2, out3, out4])
        np.testing.assert_allclose(out1_np, res[0])
        np.testing.assert_allclose(out2_np, res[1])
        np.testing.assert_allclose(out3_np, res[2])
        np.testing.assert_allclose(out4_np, res[3])

    def test_dygraph(self):
        if False:
            while True:
                i = 10
        paddle.disable_static(place=self.place)
        with paddle.base.dygraph.guard():
            x_np = np.array([[1, np.nan, -2], [np.inf, 0, -np.inf]]).astype(np.float32)
            x_tensor = paddle.to_tensor(x_np, stop_gradient=False)
            out_tensor = paddle.nan_to_num(x_tensor)
            out_np = np_nan_to_num(x_np)
            np.testing.assert_allclose(out_tensor.numpy(), out_np)
            out_tensor = paddle.nan_to_num(x_tensor, 1.0, None, None)
            out_np = np_nan_to_num(x_np, 1, None, None)
            np.testing.assert_allclose(out_tensor.numpy(), out_np)
            out_tensor = paddle.nan_to_num(x_tensor, 1.0, 2.0, None)
            out_np = np_nan_to_num(x_np, 1, 2, None)
            np.testing.assert_allclose(out_tensor.numpy(), out_np)
            out_tensor = paddle.nan_to_num(x_tensor, 1.0, None, -10.0)
            out_np = np_nan_to_num(x_np, 1, None, -10)
            np.testing.assert_allclose(out_tensor.numpy(), out_np)
            out_tensor = paddle.nan_to_num(x_tensor, 1.0, 100.0, -10.0)
            out_np = np_nan_to_num(x_np, 1, 100, -10)
            np.testing.assert_allclose(out_tensor.numpy(), out_np)
        paddle.enable_static()

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        paddle.disable_static(place=self.place)
        x_np = np.array([[1, np.nan, -2], [np.inf, 0, -np.inf]]).astype(np.float32)
        x_tensor = paddle.to_tensor(x_np, stop_gradient=False)
        y = paddle.nan_to_num(x_tensor)
        dx = paddle.grad(y, x_tensor)[0].numpy()
        np_grad = np_nan_to_num_grad(x_np, np.ones_like(x_np))
        np.testing.assert_allclose(np_grad, dx)
        paddle.enable_static()
if __name__ == '__main__':
    unittest.main()