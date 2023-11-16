import unittest
import numpy as np
import paddle
from paddle.nn.utils.clip_grad_norm_ import clip_grad_norm_

class TestClipGradNorm(unittest.TestCase):

    def test_basic(self):
        if False:
            print('Hello World!')
        run_test_equal(self, shape=[16, 16], dtype=np.float32, max_norm=5, norm_type=2)
        run_test_equal(self, shape=(100,), dtype=np.float32, max_norm=1e+20, norm_type=2)
        run_test_equal(self, shape=[4, 8, 16], dtype=np.float32, max_norm=1.0, norm_type=float('inf'))

    def test_errors(self):
        if False:
            while True:
                i = 10

        def TestValueError():
            if False:
                print('Hello World!')
            input_pd = paddle.to_tensor(np.random.random([1, 2]).astype(np.float32))
            input_pd.grad = paddle.to_tensor(np.random.random([1, 2]).astype(np.float32))
            clip_grad_norm_(input_pd, max_norm=2, norm_type=float('-inf'))
        self.assertRaises(ValueError, TestValueError)

        def TestRuntimeError():
            if False:
                print('Hello World!')
            input_pd = paddle.to_tensor(np.random.random([1, 2]).astype(np.float32))
            input_pd.grad = paddle.full([1, 2], float('inf'))
            clip_grad_norm_(input_pd, max_norm=2, norm_type=2, error_if_nonfinite=True)
        self.assertRaises(RuntimeError, TestRuntimeError)

        def TestRuntimeErrorStaticMode():
            if False:
                return 10
            paddle.enable_static()
            input_pd = paddle.to_tensor(np.random.random([1, 2]).astype(np.float32))
            input_pd.grad = paddle.to_tensor(np.random.random([1, 2]).astype(np.float32))
            clip_grad_norm_(input_pd, max_norm=2, norm_type=float('inf'))
            paddle.disable_static()
        self.assertRaises(RuntimeError, TestRuntimeErrorStaticMode)

def run_test_equal(self, shape, dtype, max_norm, norm_type: float=2.0, error_if_nonfinite: bool=False):
    if False:
        while True:
            i = 10
    input = np.random.random(shape).astype(dtype)
    grad = np.random.random(shape).astype(dtype)
    input_pd = paddle.to_tensor(input)
    input_pd.grad = paddle.to_tensor(grad)
    if norm_type == 2:
        grad = grad.reshape(1, grad.size)
        output = np.linalg.norm(grad, 'fro')
    elif norm_type == np.inf:
        output = np.amax(np.abs(grad))
    else:
        output = np.linalg.norm(grad, norm_type)
    clip_grad_norm_result = clip_grad_norm_(input_pd, max_norm=max_norm, norm_type=norm_type, error_if_nonfinite=error_if_nonfinite)
    np.testing.assert_allclose(clip_grad_norm_result.numpy(), output, rtol=1e-05, atol=1e-05, equal_nan=False)
if __name__ == '__main__':
    unittest.main()