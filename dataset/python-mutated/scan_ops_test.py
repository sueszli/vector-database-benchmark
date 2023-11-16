"""Functional tests for scan ops."""
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

def numpy_reverse(x, axis):
    if False:
        i = 10
        return i + 15
    length = len(x.shape)
    if axis < 0:
        axis = length + axis
    ix = tuple((slice(None, None, -1) if i == axis else slice(None) for i in range(length)))
    return x[ix]

def handle_options(func, init_fn, x, axis, exclusive, reverse):
    if False:
        while True:
            i = 10
    'Adds tf options to numpy scan ops.'
    length = len(x.shape)
    if axis < 0:
        axis = length + axis
    if reverse:
        x = numpy_reverse(x, axis)
    if exclusive:
        ix_head = tuple((slice(0, 1) if i == axis else slice(None) for i in range(length)))
        ix_init = tuple((slice(0, -1) if i == axis else slice(None) for i in range(length)))
        init = init_fn(x[ix_head])
        x = np.concatenate([init, func(x[ix_init], axis=axis)], axis=axis)
    else:
        x = func(x, axis=axis)
    if reverse:
        x = numpy_reverse(x, axis)
    return x

class CumsumTest(xla_test.XLATestCase):
    valid_dtypes = [np.float32, np.int32, np.int64]

    def axis_dtypes(self):
        if False:
            for i in range(10):
                print('nop')
        return set(self.int_types).intersection([np.int32, np.int64])

    def _compare(self, x, axis, exclusive, reverse):
        if False:
            for i in range(10):
                print('nop')
        np_out = handle_options(np.cumsum, np.zeros_like, x, axis, exclusive, reverse)
        with self.session(), self.test_scope():
            p = array_ops.placeholder(x.dtype)
            tf_out = math_ops.cumsum(p, axis, exclusive, reverse).eval(feed_dict={p: x})
        self.assertAllClose(np_out, tf_out)

    def _compareAll(self, x, axis):
        if False:
            return 10
        for exclusive in [True, False]:
            for reverse in [True, False]:
                self._compare(x, axis, exclusive, reverse)

    def testEmpty(self):
        if False:
            i = 10
            return i + 15
        for dtype in self.valid_dtypes:
            x = np.zeros([0]).astype(dtype)
            for axis in (-1, 0):
                self._compareAll(x, axis)

    def testAxisType(self):
        if False:
            i = 10
            return i + 15
        for dtype in self.valid_dtypes:
            x = np.arange(1, 6).reshape([5]).astype(dtype)
            for axis_dtype in self.axis_dtypes():
                with self.session(), self.test_scope():
                    p = array_ops.placeholder(x.dtype)
                    axis = constant_op.constant(0, axis_dtype)
                    math_ops.cumsum(p, axis).eval(feed_dict={p: x})

    def test1D(self):
        if False:
            print('Hello World!')
        for dtype in self.valid_dtypes:
            x = np.arange(1, 6).reshape([5]).astype(dtype)
            for axis in (-1, 0):
                self._compareAll(x, axis)

    def test2D(self):
        if False:
            while True:
                i = 10
        for dtype in self.valid_dtypes:
            x = np.arange(0, 10).reshape([2, 5]).astype(dtype)
            for axis in (-2, -1, 0, 1):
                self._compareAll(x, axis)

    def test3D(self):
        if False:
            for i in range(10):
                print('nop')
        for dtype in self.valid_dtypes:
            x = np.arange(0, 20).reshape([2, 2, 5]).astype(dtype)
            for axis in (-3, -2, -1, 0, 1, 2):
                self._compareAll(x, axis)

    def test6D(self):
        if False:
            while True:
                i = 10
        for dtype in self.valid_dtypes:
            x = np.arange(1, 145).reshape([2, 2, 3, 3, 2, 2]).astype(dtype)
            for axis in range(-6, 6, 3):
                self._compareAll(x, axis)

    def testMixedPrecision(self):
        if False:
            while True:
                i = 10
        with self.session(), self.test_scope():
            y = math_ops.cumsum(constant_op.constant([1.0, 2.0, 3.0, 4.0], dtypes.bfloat16), -1, exclusive=True).eval()
        self.assertAllEqual(y, [0.0, 1.0, 3.0, 6.0])

    @test_util.disable_mlir_bridge('Error handling')
    def testInvalidAxis(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.arange(0, 10).reshape([2, 5]).astype(np.float32)
        with self.session(), self.test_scope():
            input_tensor = ops.convert_to_tensor(x)
            with self.assertRaisesWithPredicateMatch(errors_impl.InvalidArgumentError, lambda e: 'Expected scan axis in the range [-2, 2)' in str(e)):
                math_ops.cumsum(input_tensor, -3).eval()
            with self.assertRaisesWithPredicateMatch(errors_impl.InvalidArgumentError, lambda e: 'Expected scan axis in the range [-2, 2)' in str(e)):
                math_ops.cumsum(input_tensor, 2).eval()
            with self.assertRaisesWithPredicateMatch(errors_impl.InvalidArgumentError, lambda e: 'axis must be a scalar' in str(e)):
                math_ops.cumsum(input_tensor, [0]).eval()

class CumulativeLogSumExpTest(xla_test.XLATestCase):
    valid_dtypes = [np.float32, np.float64]

    def axis_dtypes(self):
        if False:
            for i in range(10):
                print('nop')
        return set(self.int_types).intersection([np.int32, np.int64])

    def _compare(self, x, axis, exclusive, reverse):
        if False:
            return 10

        def neginf_like(x):
            if False:
                return 10
            return -np.inf * np.ones_like(x)
        np_out = handle_options(np.logaddexp.accumulate, neginf_like, x, axis, exclusive, reverse)
        with self.session(), self.test_scope():
            p = array_ops.placeholder(x.dtype)
            tf_out = math_ops.cumulative_logsumexp(p, axis, exclusive, reverse).eval(feed_dict={p: x})
        self.assertAllClose(np_out, tf_out, rtol=4e-05)

    def _compareAll(self, x, axis):
        if False:
            for i in range(10):
                print('nop')
        for exclusive in [True, False]:
            for reverse in [True, False]:
                self._compare(x, axis, exclusive, reverse)

    def testEmpty(self):
        if False:
            i = 10
            return i + 15
        for dtype in self.valid_dtypes:
            x = np.zeros([0]).astype(dtype)
            for axis in (-1, 0):
                self._compareAll(x, axis)

    def testAxisType(self):
        if False:
            while True:
                i = 10
        for dtype in self.valid_dtypes:
            x = np.arange(1, 6).reshape([5]).astype(dtype)
            for axis_dtype in self.axis_dtypes():
                with self.session(), self.test_scope():
                    p = array_ops.placeholder(x.dtype)
                    axis = constant_op.constant(0, axis_dtype)
                    math_ops.cumulative_logsumexp(p, axis).eval(feed_dict={p: x})

    def test1D(self):
        if False:
            while True:
                i = 10
        for dtype in self.valid_dtypes:
            x = np.arange(1, 6).reshape([5]).astype(dtype)
            for axis in (-1, 0):
                self._compareAll(x, axis)

    def test2D(self):
        if False:
            while True:
                i = 10
        for dtype in self.valid_dtypes:
            x = np.arange(0, 10).reshape([2, 5]).astype(dtype)
            for axis in (-2, -1, 0, 1):
                self._compareAll(x, axis)

    def test3D(self):
        if False:
            for i in range(10):
                print('nop')
        for dtype in self.valid_dtypes:
            x = np.arange(0, 20).reshape([2, 2, 5]).astype(dtype)
            for axis in (-3, -2, -1, 0, 1, 2):
                self._compareAll(x, axis)

    def test6D(self):
        if False:
            while True:
                i = 10
        for dtype in self.valid_dtypes:
            x = np.arange(1, 145).reshape([2, 2, 3, 3, 2, 2]).astype(dtype)
            for axis in range(-6, 6, 3):
                self._compareAll(x, axis)

    @test_util.disable_mlir_bridge('Error handling')
    def testInvalidAxis(self):
        if False:
            return 10
        x = np.arange(0, 10).reshape([2, 5]).astype(np.float32)
        with self.session(), self.test_scope():
            input_tensor = ops.convert_to_tensor(x)
            with self.assertRaisesWithPredicateMatch(errors_impl.InvalidArgumentError, lambda e: 'Expected scan axis in the range [-2, 2)' in str(e)):
                math_ops.cumulative_logsumexp(input_tensor, -3).eval()
            with self.assertRaisesWithPredicateMatch(errors_impl.InvalidArgumentError, lambda e: 'Expected scan axis in the range [-2, 2)' in str(e)):
                math_ops.cumulative_logsumexp(input_tensor, 2).eval()
            with self.assertRaisesWithPredicateMatch(errors_impl.InvalidArgumentError, lambda e: 'axis must be a scalar' in str(e)):
                math_ops.cumulative_logsumexp(input_tensor, [0]).eval()

class CumprodTest(xla_test.XLATestCase):
    valid_dtypes = [np.float32, np.int32]

    def axis_dtypes(self):
        if False:
            return 10
        return set(self.int_types).intersection([np.int32, np.int64])

    def _compare(self, x, axis, exclusive, reverse):
        if False:
            while True:
                i = 10
        np_out = handle_options(np.cumprod, np.ones_like, x, axis, exclusive, reverse)
        with self.session(), self.test_scope():
            p = array_ops.placeholder(x.dtype)
            prod = math_ops.cumprod(p, axis, exclusive, reverse)
            tf_out = prod.eval(feed_dict={p: x})
        self.assertAllClose(np_out, tf_out)

    def _compareAll(self, x, axis):
        if False:
            return 10
        for exclusive in [True, False]:
            for reverse in [True, False]:
                self._compare(x, axis, exclusive, reverse)

    def testEmpty(self):
        if False:
            return 10
        for dtype in self.valid_dtypes:
            x = np.zeros([0]).astype(dtype)
            for axis in (-1, 0):
                self._compareAll(x, axis)

    def testAxisType(self):
        if False:
            i = 10
            return i + 15
        for dtype in self.valid_dtypes:
            x = np.arange(1, 6).reshape([5]).astype(dtype)
            for axis_dtype in self.axis_dtypes():
                with self.session(), self.test_scope():
                    p = array_ops.placeholder(x.dtype)
                    axis = constant_op.constant(0, axis_dtype)
                    math_ops.cumprod(x, axis).eval(feed_dict={p: x})

    def test1D(self):
        if False:
            while True:
                i = 10
        for dtype in self.valid_dtypes:
            x = np.arange(1, 6).reshape([5]).astype(dtype)
            for axis in (-1, 0):
                self._compareAll(x, axis)

    def test2D(self):
        if False:
            for i in range(10):
                print('nop')
        for dtype in self.valid_dtypes:
            x = np.arange(1, 11).reshape([2, 5]).astype(dtype)
            for axis in (-2, -1, 0, 1):
                self._compareAll(x, axis)

    def test3D(self):
        if False:
            return 10
        for dtype in self.valid_dtypes:
            x = np.arange(1, 21).reshape([2, 2, 5]).astype(dtype)
            for axis in (-3, -2, -1, 0, 1, 2):
                self._compareAll(x, axis)

    def test6D(self):
        if False:
            while True:
                i = 10
        for dtype in self.valid_dtypes:
            x = np.arange(1, 145).reshape([2, 2, 3, 3, 2, 2]).astype(dtype)
            for axis in range(-6, 6, 3):
                self._compareAll(x, axis)

    @test_util.disable_mlir_bridge('Error handling')
    def testInvalidAxis(self):
        if False:
            print('Hello World!')
        x = np.arange(0, 10).reshape([2, 5]).astype(np.float32)
        with self.session(), self.test_scope():
            input_tensor = ops.convert_to_tensor(x)
            with self.assertRaisesWithPredicateMatch(errors_impl.InvalidArgumentError, lambda e: 'Expected scan axis in the range [-2, 2)' in str(e)):
                math_ops.cumprod(input_tensor, -3).eval()
            with self.assertRaisesWithPredicateMatch(errors_impl.InvalidArgumentError, lambda e: 'Expected scan axis in the range [-2, 2)' in str(e)):
                math_ops.cumprod(input_tensor, 2).eval()
            with self.assertRaisesWithPredicateMatch(errors_impl.InvalidArgumentError, lambda e: 'axis must be a scalar' in str(e)):
                math_ops.cumprod(input_tensor, [0]).eval()
if __name__ == '__main__':
    test.main()