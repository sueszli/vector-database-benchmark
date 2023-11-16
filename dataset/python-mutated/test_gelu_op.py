import unittest
import numpy as np
from scipy.special import erf
import paddle
import paddle.base.dygraph as dg
import paddle.nn.functional as F
from paddle import base

def gelu(x, approximate):
    if False:
        return 10
    if approximate:
        y_ref = 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
    else:
        y_ref = 0.5 * x * (1 + erf(x / np.sqrt(2)))
    return y_ref.astype(x.dtype)

class TestGeluOp(unittest.TestCase):

    def _test_case1_cpu(self, approximate):
        if False:
            while True:
                i = 10
        x = np.random.uniform(-1, 1, size=(11, 17)).astype(np.float32)
        y_ref = gelu(x, approximate)
        place = base.CPUPlace()
        with dg.guard(place) as g:
            x_var = dg.to_variable(x)
            y_var = F.gelu(x_var, approximate)
            y_test = y_var.numpy()
        np.testing.assert_allclose(y_ref, y_test, rtol=1e-05, atol=1e-08)

    def _test_case1_gpu(self, approximate):
        if False:
            for i in range(10):
                print('nop')
        x = np.random.uniform(-1, 1, size=(11, 17)).astype(np.float32)
        y_ref = gelu(x, approximate)
        place = base.CUDAPlace(0)
        with dg.guard(place) as g:
            x_var = dg.to_variable(x)
            y_var = F.gelu(x_var, approximate)
            y_test = y_var.numpy()
        np.testing.assert_allclose(y_ref, y_test, rtol=1e-05, atol=1e-08)

    def test_cases(self):
        if False:
            return 10
        for approximate in [True, False]:
            self._test_case1_cpu(approximate)
            if base.is_compiled_with_cuda():
                self._test_case1_gpu(approximate)

    def test_fast_math(self):
        if False:
            for i in range(10):
                print('nop')
        if not paddle.is_compiled_with_cuda():
            return

        def use_fast_math(enabled):
            if False:
                return 10
            paddle.set_flags({'FLAGS_use_fast_math': enabled})
        shape = [11, 17, 8]
        x_np = np.random.uniform(-1, 1, size=shape).astype(np.float16)
        y_g_np = np.random.uniform(-1, 1, size=shape).astype(np.float16)

        def run_gelu_op(approximate):
            if False:
                for i in range(10):
                    print('nop')
            with dg.guard():
                x = paddle.to_tensor(x_np)
                x.stop_gradient = False
                y = F.gelu(x, approximate=approximate)
                x_grad = paddle.grad([y], [x], [paddle.to_tensor(y_g_np)])[0]
                return (y.numpy(), x_grad.numpy())
        use_fast_math(True)
        (y_fast_math, x_g_fast_math) = run_gelu_op(True)
        use_fast_math(False)
        (y_ref, x_g_ref) = run_gelu_op(True)
        np.testing.assert_allclose(y_ref, y_fast_math, rtol=1e-05, atol=0.0005)
        np.testing.assert_allclose(x_g_ref, x_g_fast_math, rtol=1e-05, atol=0.0005)
if __name__ == '__main__':
    unittest.main()