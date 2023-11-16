"""Tests for tf numpy mathematical methods."""
import itertools
from absl.testing import parameterized
import numpy as np
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_math_ops
from tensorflow.python.platform import test

class MathTest(test.TestCase, parameterized.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super(MathTest, self).setUp()
        self.array_transforms = [lambda x: x, ops.convert_to_tensor, np.array, lambda x: np.array(x, dtype=np.float32), lambda x: np.array(x, dtype=np.float64), np_array_ops.array, lambda x: np_array_ops.array(x, dtype=np.float32), lambda x: np_array_ops.array(x, dtype=np.float64)]
        self.types = [np.int32, np.int64, np.float32, np.float64]

    def _testBinaryOp(self, math_fun, np_fun, name, operands=None, extra_operands=None, check_promotion=True, check_promotion_result_type=True):
        if False:
            while True:
                i = 10

        def run_test(a, b):
            if False:
                i = 10
                return i + 15
            for fn in self.array_transforms:
                arg1 = fn(a)
                arg2 = fn(b)
                self.match(math_fun(arg1, arg2), np_fun(arg1, arg2), msg='{}({}, {})'.format(name, arg1, arg2))
            for type_a in self.types:
                for type_b in self.types:
                    if not check_promotion and type_a != type_b:
                        continue
                    arg1 = np_array_ops.array(a, dtype=type_a)
                    arg2 = np_array_ops.array(b, dtype=type_b)
                    self.match(math_fun(arg1, arg2), np_fun(arg1, arg2), msg='{}({}, {})'.format(name, arg1, arg2), check_dtype=check_promotion_result_type)
        if operands is None:
            operands = [(5, 2), (5, [2, 3]), (5, [[2, 3], [6, 7]]), ([1, 2, 3], 7), ([1, 2, 3], [5, 6, 7])]
        for (operand1, operand2) in operands:
            run_test(operand1, operand2)
        if extra_operands is not None:
            for (operand1, operand2) in extra_operands:
                run_test(operand1, operand2)

    def testDot(self):
        if False:
            print('Hello World!')
        extra_operands = [([1, 2], [[5, 6, 7], [8, 9, 10]]), (np.arange(2 * 3 * 5).reshape([2, 3, 5]).tolist(), np.arange(5 * 7 * 11).reshape([7, 5, 11]).tolist())]
        return self._testBinaryOp(np_math_ops.dot, np.dot, 'dot', extra_operands=extra_operands)

    def testMinimum(self):
        if False:
            for i in range(10):
                print('nop')
        return self._testBinaryOp(np_math_ops.minimum, np.minimum, 'minimum', check_promotion_result_type=False)

    def testMaximum(self):
        if False:
            while True:
                i = 10
        return self._testBinaryOp(np_math_ops.maximum, np.maximum, 'maximum', check_promotion_result_type=False)

    def testMatmul(self):
        if False:
            i = 10
            return i + 15
        operands = [([[1, 2]], [[3, 4, 5], [6, 7, 8]])]
        return self._testBinaryOp(np_math_ops.matmul, np.matmul, 'matmul', operands=operands)

    def testMatmulError(self):
        if False:
            return 10
        with self.assertRaisesRegex(ValueError, ''):
            np_math_ops.matmul(np_array_ops.ones([], np.int32), np_array_ops.ones([2, 3], np.int32))
        with self.assertRaisesRegex(ValueError, ''):
            np_math_ops.matmul(np_array_ops.ones([2, 3], np.int32), np_array_ops.ones([], np.int32))

    def testVDot(self):
        if False:
            return 10
        operands = [([[1, 2], [3, 4]], [[3, 4], [6, 7]]), ([[1, 2], [3, 4]], [3, 4, 6, 7])]
        return self._testBinaryOp(np_math_ops.vdot, np.vdot, 'vdot', operands=operands)

    def testLcm(self):
        if False:
            for i in range(10):
                print('nop')
        a = np_array_ops.array(6, dtype=np.int8)
        b = np_array_ops.array(22, dtype=np.int8)
        res_tf = np_math_ops.lcm(a, b)
        res_np = np.lcm(np.array(a), np.array(b))
        self.assertEqual(res_tf, res_np)

    def _testUnaryOp(self, math_fun, np_fun, name):
        if False:
            i = 10
            return i + 15

        def run_test(a):
            if False:
                i = 10
                return i + 15
            for fn in self.array_transforms:
                arg1 = fn(a)
                self.match(math_fun(arg1), np_fun(arg1), msg='{}({})'.format(name, arg1))
        run_test(5)
        run_test([2, 3])
        run_test([[2, -3], [-6, 7]])

    def testLog(self):
        if False:
            return 10
        self._testUnaryOp(np_math_ops.log, np.log, 'log')

    def testExp(self):
        if False:
            while True:
                i = 10
        self._testUnaryOp(np_math_ops.exp, np.exp, 'exp')

    def testTanh(self):
        if False:
            for i in range(10):
                print('nop')
        self._testUnaryOp(np_math_ops.tanh, np.tanh, 'tanh')

    def testSqrt(self):
        if False:
            return 10
        self._testUnaryOp(np_math_ops.sqrt, np.sqrt, 'sqrt')

    def match(self, actual, expected, msg='', check_dtype=True):
        if False:
            while True:
                i = 10
        self.assertIsInstance(actual, np_arrays.ndarray)
        if check_dtype:
            self.assertEqual(actual.dtype, expected.dtype, 'Dtype mismatch.\nActual: {}\nExpected: {}\n{}'.format(actual.dtype.as_numpy_dtype, expected.dtype, msg))
        self.assertEqual(actual.shape, expected.shape, 'Shape mismatch.\nActual: {}\nExpected: {}\n{}'.format(actual.shape, expected.shape, msg))
        np.testing.assert_allclose(actual.tolist(), expected.tolist(), rtol=1e-06)

    def testArgsort(self):
        if False:
            i = 10
            return i + 15
        self._testUnaryOp(np_math_ops.argsort, np.argsort, 'argsort')
        r = np.arange(100)
        a = np.zeros(100)
        np.testing.assert_equal(np_math_ops.argsort(a, kind='stable'), r)

    def testArgMaxArgMin(self):
        if False:
            print('Hello World!')
        data = [0, 5, [1], [1, 2, 3], [[1, 2, 3]], [[4, 6], [7, 8]], [[[4, 6], [9, 10]], [[7, 8], [12, 34]]]]
        for (fn, d) in itertools.product(self.array_transforms, data):
            arr = fn(d)
            self.match(np_math_ops.argmax(arr), np.argmax(arr))
            self.match(np_math_ops.argmin(arr), np.argmin(arr))
            if hasattr(arr, 'shape'):
                ndims = len(arr.shape)
            else:
                ndims = np_array_ops.array(arr, copy=False).ndim
            if ndims == 0:
                ndims = 1
            for axis in range(-ndims, ndims):
                self.match(np_math_ops.argmax(arr, axis=axis), np.argmax(arr, axis=axis))
                self.match(np_math_ops.argmin(arr, axis=axis), np.argmin(arr, axis=axis))

    @parameterized.parameters([False, True])
    def testIsCloseEqualNan(self, equal_nan):
        if False:
            while True:
                i = 10
        a = np.asarray([1, 1, np.nan, 1, np.nan], np.float32)
        b = np.asarray([1, 2, 1, np.nan, np.nan], np.float32)
        self.match(np_math_ops.isclose(a, b, equal_nan=equal_nan), np.isclose(a, b, equal_nan=equal_nan))

    def testAverageWrongShape(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesWithPredicateMatch(errors.InvalidArgumentError, ''):
            np_math_ops.average(np.ones([2, 3]), weights=np.ones([2, 4]))
        with self.assertRaisesWithPredicateMatch(errors.InvalidArgumentError, ''):
            np_math_ops.average(np.ones([2, 3]), axis=0, weights=np.ones([2, 4]))
        with self.assertRaisesWithPredicateMatch(errors.InvalidArgumentError, ''):
            np_math_ops.average(np.ones([2, 3]), axis=0, weights=np.ones([]))
        with self.assertRaisesWithPredicateMatch(errors.InvalidArgumentError, ''):
            np_math_ops.average(np.ones([2, 3]), axis=0, weights=np.ones([5]))

    def testClip(self):
        if False:
            i = 10
            return i + 15

        def run_test(arr, *args, **kwargs):
            if False:
                while True:
                    i = 10
            check_dtype = kwargs.pop('check_dtype', True)
            for fn in self.array_transforms:
                arr = fn(arr)
                self.match(np_math_ops.clip(arr, *args, **kwargs), np.clip(arr, *args, **kwargs), check_dtype=check_dtype)
        run_test(0, -1, 5, check_dtype=False)
        run_test(-1, -1, 5, check_dtype=False)
        run_test(5, -1, 5, check_dtype=False)
        run_test(-10, -1, 5, check_dtype=False)
        run_test(10, -1, 5, check_dtype=False)
        run_test(10, None, 5, check_dtype=False)
        run_test(10, -1, None, check_dtype=False)
        run_test([0, 20, -5, 4], -1, 5, check_dtype=False)
        run_test([0, 20, -5, 4], None, 5, check_dtype=False)
        run_test([0, 20, -5, 4], -1, None, check_dtype=False)
        run_test([0.5, 20.2, -5.7, 4.4], -1.5, 5.1, check_dtype=False)
        run_test([0, 20, -5, 4], [-5, 0, -5, 0], [0, 5, 0, 5], check_dtype=False)
        run_test([[1, 2, 3], [4, 5, 6]], [2, 0, 2], 5, check_dtype=False)
        run_test([[1, 2, 3], [4, 5, 6]], 0, [5, 3, 1], check_dtype=False)

    def testPtp(self):
        if False:
            for i in range(10):
                print('nop')

        def run_test(arr, *args, **kwargs):
            if False:
                return 10
            for fn in self.array_transforms:
                arg = fn(arr)
                self.match(np_math_ops.ptp(arg, *args, **kwargs), np.ptp(arg, *args, **kwargs))
        run_test([1, 2, 3])
        run_test([1.0, 2.0, 3.0])
        run_test([[1, 2], [3, 4]], axis=1)
        run_test([[1, 2], [3, 4]], axis=0)
        run_test([[1, 2], [3, 4]], axis=-1)
        run_test([[1, 2], [3, 4]], axis=-2)

    def testLinSpace(self):
        if False:
            for i in range(10):
                print('nop')
        array_transforms = [lambda x: x, ops.convert_to_tensor, np.array, lambda x: np.array(x, dtype=np.float32), lambda x: np.array(x, dtype=np.float64), np_array_ops.array, lambda x: np_array_ops.array(x, dtype=np.float32), lambda x: np_array_ops.array(x, dtype=np.float64)]

        def run_test(start, stop, **kwargs):
            if False:
                while True:
                    i = 10
            for fn1 in array_transforms:
                for fn2 in array_transforms:
                    arg1 = fn1(start)
                    arg2 = fn2(stop)
                    self.match(np_math_ops.linspace(arg1, arg2, **kwargs), np.linspace(arg1, arg2, **kwargs), msg='linspace({}, {})'.format(arg1, arg2))
        run_test(0, 1)
        run_test(0, 1, num=10)
        run_test(0, 1, endpoint=False)
        run_test(0, -1)
        run_test(0, -1, num=10)
        run_test(0, -1, endpoint=False)

    def testLogSpace(self):
        if False:
            return 10
        array_transforms = [lambda x: x, ops.convert_to_tensor, np.array, lambda x: np.array(x, dtype=np.float32), lambda x: np.array(x, dtype=np.float64), np_array_ops.array, lambda x: np_array_ops.array(x, dtype=np.float32), lambda x: np_array_ops.array(x, dtype=np.float64)]

        def run_test(start, stop, **kwargs):
            if False:
                print('Hello World!')
            for fn1 in array_transforms:
                for fn2 in array_transforms:
                    arg1 = fn1(start)
                    arg2 = fn2(stop)
                    self.match(np_math_ops.logspace(arg1, arg2, **kwargs), np.logspace(arg1, arg2, **kwargs), msg='logspace({}, {})'.format(arg1, arg2))
        run_test(0, 5)
        run_test(0, 5, num=10)
        run_test(0, 5, endpoint=False)
        run_test(0, 5, base=2.0)
        run_test(0, -5)
        run_test(0, -5, num=10)
        run_test(0, -5, endpoint=False)
        run_test(0, -5, base=2.0)

    def testGeomSpace(self):
        if False:
            i = 10
            return i + 15

        def run_test(start, stop, **kwargs):
            if False:
                while True:
                    i = 10
            arg1 = start
            arg2 = stop
            self.match(np_math_ops.geomspace(arg1, arg2, **kwargs), np.geomspace(arg1, arg2, **kwargs), msg='geomspace({}, {})'.format(arg1, arg2))
        run_test(1, 1000, num=5)
        run_test(1, 1000, num=5, endpoint=False)
        run_test(-1, -1000, num=5)
        run_test(-1, -1000, num=5, endpoint=False)

    @parameterized.parameters(['T', 'ndim', 'size', 'data', '__pos__', '__round__', 'tolist', 'flatten', 'transpose', 'reshape', 'ravel', 'clip', 'astype', 'max', 'mean', 'min'])
    def testNumpyMethodsOnTensor(self, np_method):
        if False:
            print('Hello World!')
        a = ops.convert_to_tensor([1, 2])
        self.assertTrue(hasattr(a, np_method))

    def testFlatten(self):
        if False:
            while True:
                i = 10
        a1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        a2 = ops.convert_to_tensor(a1)
        self.assertAllEqual(a1.flatten('C'), a2.flatten('C'))
        self.assertAllEqual(a1.flatten('F'), a2.flatten('F'))
        self.assertAllEqual(a1.flatten('C'), a2.flatten('A'))
        self.assertAllEqual(a1.flatten('C'), a2.flatten('K'))
        with self.assertRaises(ValueError):
            a2.flatten('invalid')

    def testIsInf(self):
        if False:
            for i in range(10):
                print('nop')
        x1 = ops.convert_to_tensor(-2147483648)
        x2 = ops.convert_to_tensor(2147483647)
        self.assertFalse(np_math_ops.isinf(x1))
        self.assertFalse(np_math_ops.isinf(x2))
        self.assertFalse(np_math_ops.isposinf(x1))
        self.assertFalse(np_math_ops.isposinf(x2))
        self.assertFalse(np_math_ops.isneginf(x1))
        self.assertFalse(np_math_ops.isneginf(x2))
if __name__ == '__main__':
    tensor.enable_tensor_equality()
    ops.enable_eager_execution()
    ops.set_dtype_conversion_mode('legacy')
    np_math_ops.enable_numpy_methods_on_tensor()
    test.main()