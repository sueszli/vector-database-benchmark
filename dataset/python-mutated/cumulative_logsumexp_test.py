"""Functional tests for cumulative_logsumexp op."""
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

class CumulativeLogsumexpTest(test.TestCase):
    valid_dtypes = [dtypes.float32, dtypes.float64, dtypes.float16, dtypes.bfloat16]

    def _computeLogSumExp(self, x, **kwargs):
        if False:
            i = 10
            return i + 15
        result_naive = math_ops.cumsum(math_ops.exp(x), **kwargs)
        result_fused = math_ops.exp(math_ops.cumulative_logsumexp(x, **kwargs))
        return (result_naive, result_fused)

    def _testLogSumExp(self, x, dtype=dtypes.float32, use_gpu=False, **kwargs):
        if False:
            i = 10
            return i + 15
        with self.cached_session(use_gpu=use_gpu):
            x = ops.convert_to_tensor(x, dtype=dtype)
            (result_naive, result_fused) = self.evaluate(self._computeLogSumExp(x, **kwargs))
        tol = 0.02 if dtype in [dtypes.float16, dtypes.bfloat16] else 1e-06
        self.assertAllClose(result_naive, result_fused, rtol=tol, atol=tol)

    def _testLogSumExpAllArgs(self, x, axis=0, use_gpu=False):
        if False:
            return 10
        for dtype in self.valid_dtypes:
            for reverse in (True, False):
                for exclusive in (True, False):
                    self._testLogSumExp(x, dtype=dtype, use_gpu=use_gpu, reverse=reverse, exclusive=exclusive, axis=axis)

    def testMinusInfinity(self):
        if False:
            i = 10
            return i + 15
        x = np.log([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
        self._testLogSumExpAllArgs(x, use_gpu=False)
        self._testLogSumExpAllArgs(x, use_gpu=True)

    def test1D(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.arange(10) / 10.0 - 0.5
        self._testLogSumExpAllArgs(x, use_gpu=False)
        self._testLogSumExpAllArgs(x, use_gpu=True)

    def test2D(self):
        if False:
            i = 10
            return i + 15
        x = np.reshape(np.arange(20) / 20.0 - 0.5, (2, 10))
        for axis in (-2, -1, 0, 1):
            self._testLogSumExpAllArgs(x, axis=axis, use_gpu=False)
            self._testLogSumExpAllArgs(x, axis=axis, use_gpu=True)

    def _testGradient(self, x, use_gpu=False, **kwargs):
        if False:
            while True:
                i = 10
        with self.cached_session(use_gpu=use_gpu):
            x = ops.convert_to_tensor(x, dtype=dtypes.float64)
            (grad_naive_theoretical, _) = gradient_checker_v2.compute_gradient(lambda y: math_ops.cumsum(math_ops.exp(y), **kwargs), [x])
            (grad_fused_theoretical, _) = gradient_checker_v2.compute_gradient(lambda y: math_ops.exp(math_ops.cumulative_logsumexp(y, **kwargs)), [x])
            self.assertAllClose(grad_fused_theoretical, grad_naive_theoretical)

    def testGradient(self):
        if False:
            i = 10
            return i + 15
        for reverse in (True, False):
            for exclusive in (True, False):
                x = np.arange(10) / 10.0 - 0.5
                self._testGradient(x, use_gpu=False, reverse=reverse, exclusive=exclusive)
                self._testGradient(x, use_gpu=True, reverse=reverse, exclusive=exclusive)

    def _logSumExpMap(self, x):
        if False:
            while True:
                i = 10
        return map_fn.map_fn(lambda i: math_ops.reduce_logsumexp(x[:i + 1]), math_ops.range(array_ops.shape(x)[0]), dtype=x.dtype)

    def test1DLarge(self):
        if False:
            print('Hello World!')
        x_np = np.arange(20) * 20.0
        for use_gpu in (True, False):
            with self.cached_session(use_gpu=use_gpu):
                x_tf = ops.convert_to_tensor(x_np, dtype=dtypes.float32)
                result_fused = self.evaluate(math_ops.cumulative_logsumexp(x_tf))
                result_map = self.evaluate(self._logSumExpMap(x_tf))
            self.assertAllClose(result_fused, result_map)
if __name__ == '__main__':
    test.main()