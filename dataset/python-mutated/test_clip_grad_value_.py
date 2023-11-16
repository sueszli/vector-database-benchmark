import unittest
import numpy as np
import paddle
from paddle.nn.utils.clip_grad_value_ import clip_grad_value_

class TestClipGradValue(unittest.TestCase):

    def test_basic(self):
        if False:
            while True:
                i = 10
        run_test_equal_np(self, shape=[16, 16], dtype=np.float32, clip_value=1)
        run_test_equal_np(self, shape=(100,), dtype=np.float32, clip_value=0.1)
        run_test_equal_np(self, shape=[4, 8, 16], dtype=np.float32, clip_value=0)
        run_test_equal_ClipGradByValue(self, shape=[16, 16], dtype=np.float32, clip_value=1)
        run_test_equal_ClipGradByValue(self, shape=(100,), dtype=np.float32, clip_value=0.1)
        run_test_equal_ClipGradByValue(self, shape=[4, 8, 16], dtype=np.float32, clip_value=0)

    def test_errors(self):
        if False:
            print('Hello World!')

        def TestValueError():
            if False:
                i = 10
                return i + 15
            input_pd = paddle.to_tensor(np.random.random([1, 2]).astype(np.float32))
            input_pd.grad = paddle.to_tensor(np.random.random([1, 2]).astype(np.float32))
            clip_grad_value_(input_pd, clip_value=-1)
        self.assertRaises(ValueError, TestValueError)

        def TestRuntimeErrorStaticMode():
            if False:
                print('Hello World!')
            paddle.enable_static()
            input_pd = paddle.to_tensor(np.random.random([1, 2]).astype(np.float32))
            input_pd.grad = paddle.to_tensor(np.random.random([1, 2]).astype(np.float32))
            clip_grad_value_(input_pd, clip_value=1)
            paddle.disable_static()
        self.assertRaises(RuntimeError, TestRuntimeErrorStaticMode)

def run_test_equal_np(self, shape, dtype, clip_value):
    if False:
        print('Hello World!')
    input = np.random.random(shape).astype(dtype)
    grad = np.random.random(shape).astype(dtype)
    input_pd = paddle.to_tensor(input)
    input_pd.grad = paddle.to_tensor(grad)
    output = np.clip(grad, a_min=-clip_value, a_max=clip_value)
    clip_grad_value_(input_pd, clip_value=clip_value)
    np.testing.assert_allclose(input_pd.grad.numpy(), output, rtol=1e-05, atol=1e-05, equal_nan=False)

def run_test_equal_ClipGradByValue(self, shape, dtype, clip_value):
    if False:
        while True:
            i = 10
    input = np.random.random(shape).astype(dtype)
    grad = np.random.random(shape).astype(dtype)
    input_pd = paddle.to_tensor(input)
    input_pd.grad = paddle.to_tensor(grad)
    clip = paddle.nn.ClipGradByValue(max=clip_value, min=-clip_value)
    output = clip([(input_pd, input_pd.grad)])[0][1]
    clip_grad_value_(input_pd, clip_value=clip_value)
    np.testing.assert_allclose(input_pd.grad, output, rtol=1e-05, atol=1e-05, equal_nan=False)
if __name__ == '__main__':
    unittest.main()