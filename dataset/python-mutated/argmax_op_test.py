"""Tests for tensorflow.ops.argmax_op."""
import functools
from absl.testing import parameterized
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

class ArgMaxTest(test.TestCase, parameterized.TestCase):

    def _testArg(self, method, x, axis, expected_values, use_gpu=False, expected_err_re=None):
        if False:
            while True:
                i = 10
        with self.session(use_gpu=use_gpu):
            ans = method(x, axis=axis)
            if expected_err_re is None:
                tf_ans = self.evaluate(ans)
                self.assertEqual(np.int64, tf_ans.dtype)
                self.assertAllEqual(tf_ans, expected_values)
                self.assertShapeEqual(expected_values, ans)
            else:
                with self.assertRaisesOpError(expected_err_re):
                    self.evaluate(ans)

    def _testBothArg(self, method, x, axis, expected_values, expected_err_re=None):
        if False:
            while True:
                i = 10
        self._testArg(method, x, axis, expected_values, True, expected_err_re)
        if not test_util.is_xla_enabled():
            self._testArg(method, x, axis, expected_values, False, expected_err_re)

    def _testBasic(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        x = np.arange(200, dtype=np.float32).astype(dtype)
        np.random.shuffle(x)
        self._testBothArg(math_ops.argmax, x, 0, x.argmax())
        self._testBothArg(math_ops.argmin, x, 0, x.argmin())

    def _testTieBreaking(self, dtype):
        if False:
            print('Hello World!')
        x = np.zeros(200, dtype=dtype)
        self._testBothArg(math_ops.argmax, x, 0, x.argmax())
        self._testBothArg(math_ops.argmin, x, 0, x.argmin())
        x = np.array([[0, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 1]], dtype=dtype)
        self._testBothArg(math_ops.argmax, x, 1, x.argmax(axis=1))
        self._testBothArg(math_ops.argmin, x, 1, x.argmin(axis=1))

    def _testDim(self, dtype):
        if False:
            return 10
        shape = (3, 2, 4, 5, 6, 3, 7)
        x = np.arange(functools.reduce(lambda x, y: x * y, shape), dtype=np.float32).astype(dtype)
        np.random.shuffle(x)
        x = x.reshape(shape)
        for axis in range(-7, 7):
            self._testBothArg(math_ops.argmax, x, axis, x.argmax(axis))
            self._testBothArg(math_ops.argmin, x, axis, x.argmin(axis))

    @parameterized.parameters(np.float16, np.float32, np.float64, np.int16, np.int32, np.int64, np.bool_, dtypes.bfloat16.as_numpy_dtype)
    def testTypes(self, dtype):
        if False:
            print('Hello World!')
        self._testBasic(dtype)
        self._testTieBreaking(dtype)
        self._testDim(dtype)

    def testFloatInt32Output(self):
        if False:
            print('Hello World!')
        x = np.asarray(100 * np.random.randn(200), dtype=np.float32)
        expected_values = x.argmax()
        with self.session():
            ans = math_ops.argmax(x, axis=0, output_type=dtypes.int32)
            tf_ans = self.evaluate(ans)
            self.assertEqual(np.int32, tf_ans.dtype)
            self.assertAllEqual(tf_ans, expected_values)
        expected_values = x.argmin()
        with self.session():
            ans = math_ops.argmin(x, axis=0, output_type=dtypes.int32)
            tf_ans = self.evaluate(ans)
            self.assertEqual(np.int32, tf_ans.dtype)
            self.assertAllEqual(tf_ans, expected_values)

    def testEmpty(self):
        if False:
            return 10
        with self.cached_session():
            for op in (math_ops.argmin, math_ops.argmax):
                with self.assertRaisesOpError('Reduction axis 0 is empty in shape \\[0\\]'):
                    op([], 0).eval()

    @test_util.run_deprecated_v1
    def testDefaultAxis(self):
        if False:
            while True:
                i = 10
        with self.cached_session():
            for op in (math_ops.argmin, math_ops.argmax):
                ans = op([1]).eval()
                self.assertAllEqual(ans, 0)

    @test_util.run_deprecated_v1
    def testOutputEmpty(self):
        if False:
            return 10
        with self.cached_session():
            for op in (math_ops.argmin, math_ops.argmax):
                ret = op(array_ops.zeros(shape=[1, 0, 2]), axis=-1).eval()
                self.assertEqual(ret.shape, (1, 0))
if __name__ == '__main__':
    test.main()