"""Functional tests for coefficient-wise operations."""
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_grad
from tensorflow.python.platform import test
_ADD = lambda x, y: x + y
_SUB = lambda x, y: x - y
_MUL = lambda x, y: x * y
_POW = lambda x, y: x ** y
_TRUEDIV = lambda x, y: x / y
_FLOORDIV = lambda x, y: x // y
_MOD = lambda x, y: x % y
_LT = lambda x, y: x < y
_LE = lambda x, y: x <= y
_GT = lambda x, y: x > y
_GE = lambda x, y: x >= y
_AND = lambda x, y: x & y
_OR = lambda x, y: x | y
_XOR = lambda x, y: x ^ y
_INV = lambda x: ~x

def _sparsify(x, thresh=0.5, index_dtype=np.int64):
    if False:
        print('Hello World!')
    x[x < thresh] = 0
    non_zero = np.where(x)
    x_indices = np.vstack(non_zero).astype(index_dtype).T
    x_values = x[non_zero]
    x_shape = x.shape
    return (sparse_tensor.SparseTensor(indices=x_indices, values=x_values, dense_shape=x_shape), x_values)

def _default_tolerance(dtype):
    if False:
        i = 10
        return i + 15
    'Returns a sensible default tolerance for comparing results of a given type.\n\n  Args:\n    dtype: A datatype.\n  '
    if dtype == np.float16:
        return 0.005
    elif dtype in (np.float32, np.complex64):
        return 0.001
    elif dtype in (np.float64, np.complex128):
        return 1e-05
    else:
        return None

class ComparisonOpTest(test.TestCase):

    def _compareScalar(self, func, x, y, dtype):
        if False:
            while True:
                i = 10
        with test_util.use_gpu():
            out = func(ops.convert_to_tensor(np.array([x]).astype(dtype)), ops.convert_to_tensor(np.array([y]).astype(dtype)))
            ret = self.evaluate(out)
        return ret[0]

    def testScalarCompareScalar(self):
        if False:
            i = 10
            return i + 15
        dtypes = [np.float16, np.float32, np.float64, np.int32, np.int64]
        data = [-1, 0, 1]
        for t in dtypes:
            for x in data:
                for y in data:
                    with self.subTest(t=t, x=x, y=y):
                        self.assertEqual(self._compareScalar(math_ops.less, x, y, t), x < y)
                        self.assertEqual(self._compareScalar(math_ops.less_equal, x, y, t), x <= y)
                        self.assertEqual(self._compareScalar(math_ops.greater, x, y, t), x > y)
                        self.assertEqual(self._compareScalar(math_ops.greater_equal, x, y, t), x >= y)
                        self.assertEqual(self._compareScalar(math_ops.equal, x, y, t), x == y)
                        self.assertEqual(self._compareScalar(math_ops.not_equal, x, y, t), x != y)
        data = [-1, 0, 1, -1j, 1j, 1 + 1j, 1 - 1j]
        for t in [np.complex64, np.complex128]:
            for x in data:
                for y in data:
                    with self.subTest(t=t, x=x, y=y):
                        self.assertEqual(self._compareScalar(math_ops.equal, x, y, t), x == y)
                        self.assertEqual(self._compareScalar(math_ops.not_equal, x, y, t), x != y)

    def _compare(self, x, y, np_func, tf_func):
        if False:
            return 10
        np_ans = np_func(x, y)
        with test_util.use_gpu():
            out = tf_func(ops.convert_to_tensor(x), ops.convert_to_tensor(y))
            tf_ans = self.evaluate(out)
        self.assertAllEqual(np_ans, tf_ans)

    def testTensorCompareTensor(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.linspace(-15, 15, 6).reshape(1, 3, 2)
        y = np.linspace(20, -10, 6).reshape(1, 3, 2)
        for t in [np.float16, np.float32, np.float64, np.int32, np.int64]:
            with self.subTest(t=t):
                xt = x.astype(t)
                yt = y.astype(t)
                self._compare(xt, yt, np.less, math_ops.less)
                self._compare(xt, yt, np.less_equal, math_ops.less_equal)
                self._compare(xt, yt, np.greater, math_ops.greater)
                self._compare(xt, yt, np.greater_equal, math_ops.greater_equal)
                self._compare(xt, yt, np.equal, math_ops.equal)
                self._compare(xt, yt, np.not_equal, math_ops.not_equal)
        for t in [np.complex64, np.complex128]:
            with self.subTest(t=t):
                xt = x.astype(t)
                xt -= 1j * xt
                yt = y.astype(t)
                yt -= 1j * yt
                self._compare(xt, yt, np.equal, math_ops.equal)
                self._compare(xt, yt, np.not_equal, math_ops.not_equal)

    def _compareBCast(self, xs, ys, dtype, np_func, tf_func):
        if False:
            for i in range(10):
                print('nop')
        x = np.linspace(-15, 15, np.prod(xs)).astype(dtype).reshape(xs)
        y = np.linspace(20, -10, np.prod(ys)).astype(dtype).reshape(ys)
        if dtype in (np.complex64, np.complex128):
            x -= 1j * x
            y -= 1j * y
        self._compare(x, y, np_func, tf_func)
        self._compare(y, x, np_func, tf_func)

    def _testBCastByFunc(self, np_func, tf_func, include_complex=False):
        if False:
            return 10
        shapes = [([1, 3, 2], [1]), ([1, 3, 2], [2]), ([1, 3, 2], [3, 2]), ([1, 3, 2], [3, 1]), ([1, 3, 2], [1, 3, 2]), ([1, 3, 2], [2, 3, 1]), ([1, 3, 2], [2, 1, 1]), ([1, 3, 2], [1, 3, 1]), ([2, 1, 5], [2, 3, 1]), ([2, 0, 5], [2, 0, 1]), ([2, 3, 0], [2, 3, 1])]
        dtypes = [np.float16, np.float32, np.float64, np.int32, np.int64]
        if include_complex:
            dtypes.extend([np.complex64, np.complex128])
        for (xs, ys) in shapes:
            for dtype in dtypes:
                with self.subTest(xs=xs, ys=ys, dtype=dtype):
                    self._compareBCast(xs, ys, dtype, np_func, tf_func)

    def testBCastLess(self):
        if False:
            for i in range(10):
                print('nop')
        self._testBCastByFunc(np.less, math_ops.less)

    def testBCastLessEqual(self):
        if False:
            while True:
                i = 10
        self._testBCastByFunc(np.less_equal, math_ops.less_equal)

    def testBCastGreater(self):
        if False:
            i = 10
            return i + 15
        self._testBCastByFunc(np.greater, math_ops.greater)

    def testBCastGreaterEqual(self):
        if False:
            print('Hello World!')
        self._testBCastByFunc(np.greater_equal, math_ops.greater_equal)

    def testBCastEqual(self):
        if False:
            print('Hello World!')
        self._testBCastByFunc(np.equal, math_ops.equal, include_complex=True)

    def testBCastNotEqual(self):
        if False:
            print('Hello World!')
        self._testBCastByFunc(np.not_equal, math_ops.not_equal, include_complex=True)

    def testShapeMismatch(self):
        if False:
            while True:
                i = 10
        dtypes = [np.float16, np.float32, np.float64, np.int32, np.int64]
        funcs = [math_ops.less, math_ops.less_equal, math_ops.greater, math_ops.greater_equal, math_ops.equal, math_ops.not_equal]
        x = np.arange(0, 10).reshape([2, 5])
        y = np.arange(0, 10).reshape([5, 2])
        for t in dtypes:
            for f in funcs:
                with self.subTest(t=t, f=f):
                    with self.assertRaisesIncompatibleShapesError((ValueError, errors.InvalidArgumentError)):
                        f(x.astype(t), y.astype(t))

class LogicalOpTest(test.TestCase):

    def _compareBinary(self, x, y, np_func, tf_func, use_gpu=False):
        if False:
            print('Hello World!')
        np_ans = np_func(x, y)
        with test_util.device(use_gpu=use_gpu):
            inx = ops.convert_to_tensor(x)
            iny = ops.convert_to_tensor(y)
            out = tf_func(inx, iny)
            tf_val = self.evaluate(out)
        self.assertEqual(out.dtype, dtypes_lib.bool)
        self.assertAllEqual(np_ans, tf_val)
        self.assertShapeEqual(np_ans, out)

    def _not(self, x, use_gpu=False):
        if False:
            return 10
        np_ans = np.logical_not(x)
        with test_util.device(use_gpu=use_gpu):
            out = math_ops.logical_not(ops.convert_to_tensor(x))
            tf_val = self.evaluate(out)
        self.assertEqual(out.dtype, dtypes_lib.bool)
        self.assertAllEqual(np_ans, tf_val)
        self.assertShapeEqual(np_ans, out)

    def testScalar(self):
        if False:
            print('Hello World!')
        data = [np.array([True]), np.array([False])]
        for use_gpu in [True, False]:
            for x in data:
                with self.subTest(use_gpu=use_gpu, x=x):
                    self._not(x, use_gpu)
            for x in data:
                for y in data:
                    with self.subTest(use_gpu=use_gpu, x=x, y=y):
                        self._compareBinary(x, y, np.logical_and, math_ops.logical_and, use_gpu)
                        self._compareBinary(x, y, np.logical_or, math_ops.logical_or, use_gpu)
                        self._compareBinary(x, y, np.logical_xor, math_ops.logical_xor, use_gpu)

    def testTensor(self):
        if False:
            while True:
                i = 10
        x = np.random.randint(0, 2, 6).astype(np.bool_).reshape(1, 3, 2)
        y = np.random.randint(0, 2, 6).astype(np.bool_).reshape(1, 3, 2)
        for use_gpu in [True, False]:
            with self.subTest(use_gpu=use_gpu):
                self._not(x, use_gpu)
                self._compareBinary(x, y, np.logical_and, math_ops.logical_and, use_gpu)
                self._compareBinary(x, y, np.logical_or, math_ops.logical_or, use_gpu)
                self._compareBinary(x, y, np.logical_xor, math_ops.logical_xor, use_gpu)

    def testBCast(self):
        if False:
            print('Hello World!')
        shapes = [([1, 3, 2], [1]), ([1, 3, 2], [2]), ([1, 3, 2], [3, 2]), ([1, 3, 2], [3, 1]), ([1, 3, 2], [1, 3, 2]), ([1, 3, 2], [2, 3, 1]), ([1, 3, 2], [2, 1, 1]), ([1, 3, 2], [1, 3, 1]), ([2, 1, 5], [2, 3, 1]), ([2, 0, 5], [2, 0, 1]), ([2, 3, 0], [2, 3, 1])]
        for (xs, ys) in shapes:
            x = np.random.randint(0, 2, np.prod(xs)).astype(np.bool_).reshape(xs)
            y = np.random.randint(0, 2, np.prod(ys)).astype(np.bool_).reshape(ys)
            for use_gpu in [True, False]:
                with self.subTest(xs=xs, ys=ys, use_gpu=use_gpu):
                    self._compareBinary(x, y, np.logical_and, math_ops.logical_and, use_gpu)
                    self._compareBinary(x, y, np.logical_or, math_ops.logical_or, use_gpu)
                    self._compareBinary(x, y, np.logical_xor, math_ops.logical_xor, use_gpu)

    @test_util.run_deprecated_v1
    def testShapeMismatch(self):
        if False:
            i = 10
            return i + 15
        x = np.random.randint(0, 2, 6).astype(np.bool_).reshape(1, 3, 2)
        y = np.random.randint(0, 2, 6).astype(np.bool_).reshape(3, 2, 1)
        for f in [math_ops.logical_and, math_ops.logical_or, math_ops.logical_xor]:
            with self.subTest(f=f):
                with self.assertRaisesWithPredicateMatch(ValueError, lambda e: 'Dimensions must' in str(e)):
                    f(x, y)

    @test_util.run_deprecated_v1
    def testUsingAsPythonValueFails(self):
        if False:
            i = 10
            return i + 15
        b = constant_op.constant(False)
        with self.assertRaises(TypeError):
            if b:
                pass
        x = constant_op.constant(3)
        y = constant_op.constant(4)
        with self.assertRaises(TypeError):
            if x > y:
                pass
        z = constant_op.constant(7)
        with self.assertRaises(TypeError):
            _ = x < y < z

class SelectOpTest(test.TestCase):

    def _compare(self, fn, c, x, y, use_gpu):
        if False:
            return 10
        np_ans = np.where(c, x, y)
        with test_util.device(use_gpu=use_gpu):
            out = fn(c, x, y)
            tf_ans = self.evaluate(out)
        self.assertAllEqual(np_ans, tf_ans)
        self.assertShapeEqual(np_ans, out)

    def _compareGradientX(self, fn, c, x, y, numeric_gradient_type=None, x_init_value=None):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            inx = ops.convert_to_tensor(x)
            iny = ops.convert_to_tensor(y)
            out = fn(c, inx, iny)
            s = list(np.shape(c))
            if x_init_value is None:
                x_init_value = x
            if x.shape != y.shape:
                x_init_value = np.broadcast_to(y, x.shape)
            (jacob_t, jacob_n) = gradient_checker.compute_gradient(inx, s, out, s, x_init_value=x_init_value)
            if numeric_gradient_type is not None:
                xf = x.astype(numeric_gradient_type)
                yf = y.astype(numeric_gradient_type)
                inxf = ops.convert_to_tensor(xf)
                inyf = ops.convert_to_tensor(yf)
                outf = fn(c, inxf, inyf)
                (_, jacob_n) = gradient_checker.compute_gradient(inxf, s, outf, s, x_init_value=xf)
                jacob_n = jacob_n.astype(x.dtype)
        if x.dtype == np.float16:
            self.assertAllClose(jacob_t, jacob_n, rtol=0.001, atol=0.001)
        elif x.dtype == np.float32:
            self.assertAllClose(jacob_t, jacob_n, rtol=0.001, atol=0.001)
        elif x.dtype == np.float64:
            self.assertAllClose(jacob_t, jacob_n, rtol=1e-05, atol=1e-05)

    def _compareGradientY(self, fn, c, x, y, numeric_gradient_type=None):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            inx = ops.convert_to_tensor(x)
            iny = ops.convert_to_tensor(y)
            out = fn(c, inx, iny)
            s = list(np.shape(c))
            (jacob_t, jacob_n) = gradient_checker.compute_gradient(iny, s, out, s, x_init_value=x, delta=1.0)
            if numeric_gradient_type is not None:
                xf = x.astype(numeric_gradient_type)
                yf = y.astype(numeric_gradient_type)
                inxf = ops.convert_to_tensor(xf)
                inyf = ops.convert_to_tensor(yf)
                outf = fn(c, inxf, inyf)
                (_, jacob_n) = gradient_checker.compute_gradient(inyf, s, outf, s, x_init_value=yf)
                jacob_n = jacob_n.astype(x.dtype)
        if x.dtype == np.float16:
            self.assertAllClose(jacob_t, jacob_n, rtol=0.001, atol=0.001)
        elif x.dtype == np.float32:
            self.assertAllClose(jacob_t, jacob_n, rtol=0.001, atol=0.001)
        elif x.dtype == np.float64:
            self.assertAllClose(jacob_t, jacob_n, rtol=1e-05, atol=1e-05)

    def _testScalar(self, fn):
        if False:
            for i in range(10):
                print('nop')
        c = True
        x = np.random.rand(1, 3, 2) * 100
        y = np.random.rand(1, 3, 2) * 100
        for t in [np.float16, np.float32, np.float64, np.int32, np.int64, np.complex64, np.complex128]:
            with self.subTest(t=t):
                xt = x.astype(t)
                yt = y.astype(t)
                self._compare(fn, c, xt, yt, use_gpu=False)
                if t in [np.float16, np.float32, np.float64]:
                    self._compare(fn, c, xt, yt, use_gpu=True)

    def testScalar(self):
        if False:
            print('Hello World!')
        self._testScalar(array_ops.where)
        self._testScalar(array_ops.where_v2)

    def _testScalarBroadcast(self, fn, c, x, y):
        if False:
            i = 10
            return i + 15
        for t in [np.float16, np.float32, np.float64, np.int32, np.int64, np.complex64, np.complex128]:
            with self.subTest(t=t):
                xt = x.astype(t)
                yt = y.astype(t)
                self._compare(fn, c, xt, yt, use_gpu=False)
                if t in [np.float16, np.float32, np.float64]:
                    self._compare(fn, c, xt, yt, use_gpu=True)

    def testScalarBroadcast(self):
        if False:
            while True:
                i = 10
        c = True
        x = np.random.rand(1, 3, 2) * 100
        y = np.random.rand(1, 1, 1) * 100
        self._testScalarBroadcast(array_ops.where_v2, c, x, y)
        self._testScalarBroadcast(array_ops.where_v2, c, y, x)
        x = np.random.rand(1, 3, 2) * 100
        y = np.random.rand(1, 3, 1) * 100
        self._testScalarBroadcast(array_ops.where_v2, c, x, y)
        self._testScalarBroadcast(array_ops.where_v2, c, y, x)
        x = np.random.rand(1, 3, 2) * 100
        y = np.random.rand(1, 1, 2) * 100
        self._testScalarBroadcast(array_ops.where_v2, c, x, y)
        self._testScalarBroadcast(array_ops.where_v2, c, y, x)
        x = np.random.rand(1, 3, 2) * 100
        y = np.random.rand(1, 1) * 100
        self._testScalarBroadcast(array_ops.where_v2, c, x, y)
        self._testScalarBroadcast(array_ops.where_v2, c, y, x)
        x = np.random.rand(1, 3, 2) * 100
        y = np.random.rand(1) * 100
        self._testScalarBroadcast(array_ops.where_v2, c, x, y)
        self._testScalarBroadcast(array_ops.where_v2, c, y, x)
        x = np.random.rand(1, 3, 2) * 100
        y = np.random.rand(1, 2) * 100
        self._testScalarBroadcast(array_ops.where_v2, c, x, y)
        self._testScalarBroadcast(array_ops.where_v2, c, y, x)
        x = np.random.rand(1, 3, 2) * 100
        y = np.random.rand(3, 2) * 100
        self._testScalarBroadcast(array_ops.where_v2, c, x, y)
        self._testScalarBroadcast(array_ops.where_v2, c, y, x)

    def _testBasic(self, fn):
        if False:
            return 10
        c = np.random.randint(0, 2, 6).astype(np.bool_).reshape(1, 3, 2)
        x = np.random.rand(1, 3, 2) * 100
        y = np.random.rand(1, 3, 2) * 100
        for t in [np.float16, np.float32, np.float64, np.int32, np.int64, np.complex64, np.complex128]:
            with self.subTest(t=t):
                xt = x.astype(t)
                yt = y.astype(t)
                self._compare(fn, c, xt, yt, use_gpu=False)
                if t in [np.float16, np.float32, np.float64]:
                    self._compare(fn, c, xt, yt, use_gpu=True)

    def testBasic(self):
        if False:
            return 10
        self._testBasic(array_ops.where)
        self._testBasic(array_ops.where_v2)

    def _testBasicBroadcast(self, fn, c, x, y):
        if False:
            return 10
        for t in [np.float16, np.float32, np.float64, np.int32, np.int64, np.complex64, np.complex128]:
            with self.subTest(t=t):
                xt = x.astype(t)
                yt = y.astype(t)
                self._compare(fn, c, xt, yt, use_gpu=False)
                if t in [np.float16, np.float32, np.float64]:
                    self._compare(fn, c, xt, yt, use_gpu=True)

    def testBasicBroadcast(self):
        if False:
            i = 10
            return i + 15
        c0 = np.random.randint(0, 2, 6).astype(np.bool_).reshape(1, 3, 2)
        c1 = np.random.randint(0, 2, 2).astype(np.bool_).reshape(1, 1, 2)
        c2 = np.random.randint(0, 2, 3).astype(np.bool_).reshape(1, 3, 1)
        c3 = np.random.randint(0, 2, 1).astype(np.bool_).reshape(1, 1, 1)
        for c in [c0, c1, c2, c3]:
            with self.subTest(c=c):
                x = np.random.rand(1, 3, 2) * 100
                y = np.random.rand(1, 1, 1) * 100
                self._testBasicBroadcast(array_ops.where_v2, c, x, y)
                self._testBasicBroadcast(array_ops.where_v2, c, y, x)
                x = np.random.rand(1, 3, 2) * 100
                y = np.random.rand(1, 3, 1) * 100
                self._testBasicBroadcast(array_ops.where_v2, c, x, y)
                self._testBasicBroadcast(array_ops.where_v2, c, y, x)
                x = np.random.rand(1, 3, 2) * 100
                y = np.random.rand(1, 1, 2) * 100
                self._testBasicBroadcast(array_ops.where_v2, c, x, y)
                self._testBasicBroadcast(array_ops.where_v2, c, y, x)
                x = np.random.rand(1, 3, 2) * 100
                y = np.random.rand(1, 1) * 100
                self._testBasicBroadcast(array_ops.where_v2, c, x, y)
                self._testBasicBroadcast(array_ops.where_v2, c, y, x)
                x = np.random.rand(1, 3, 2) * 100
                y = np.random.rand(1) * 100
                self._testBasicBroadcast(array_ops.where_v2, c, x, y)
                self._testBasicBroadcast(array_ops.where_v2, c, y, x)
                x = np.random.rand(1, 3, 2) * 100
                y = np.random.rand(1, 2) * 100
                self._testBasicBroadcast(array_ops.where_v2, c, x, y)
                self._testBasicBroadcast(array_ops.where_v2, c, y, x)
                x = np.random.rand(1, 3, 2) * 100
                y = np.random.rand(3, 2) * 100
                self._testBasicBroadcast(array_ops.where_v2, c, x, y)
                self._testBasicBroadcast(array_ops.where_v2, c, y, x)

    def _testGradients(self, fn):
        if False:
            for i in range(10):
                print('nop')
        c = np.random.randint(0, 2, 6).astype(np.bool_).reshape(1, 3, 2)
        x = np.random.rand(1, 3, 2) * 100
        y = np.random.rand(1, 3, 2) * 100
        for t in [np.float16, np.float32, np.float64]:
            with self.subTest(t=t):
                xt = x.astype(t)
                yt = y.astype(t)
                if t == np.float16:
                    self._compareGradientX(fn, c, xt, yt, np.float64)
                    self._compareGradientY(fn, c, xt, yt, np.float64)
                else:
                    self._compareGradientX(fn, c, xt, yt)
                    self._compareGradientY(fn, c, xt, yt)

    @test_util.run_deprecated_v1
    def testGradients(self):
        if False:
            for i in range(10):
                print('nop')
        self._testGradients(array_ops.where)
        self._testGradients(array_ops.where_v2)

    @test_util.run_deprecated_v1
    def testGradientsBroadcast(self):
        if False:
            while True:
                i = 10
        c = np.random.randint(0, 2, 6).astype(np.bool_).reshape(1, 3, 2)
        for t in [np.float32, np.float64]:
            with self.subTest(t=t):
                x = np.random.rand(1, 3, 2) * 100
                y = np.random.rand(1, 1, 1) * 100
                self._compareGradientX(array_ops.where_v2, c, x.astype(t), y.astype(t))
                x = np.random.rand(1, 3, 2) * 100
                y = np.random.rand(1, 3, 1) * 100
                self._compareGradientX(array_ops.where_v2, c, x.astype(t), y.astype(t))
                x = np.random.rand(1, 3, 2) * 100
                y = np.random.rand(1, 1, 2) * 100
                self._compareGradientX(array_ops.where_v2, c, x.astype(t), y.astype(t))
                x = np.random.rand(1, 3, 2) * 100
                y = np.random.rand(1, 1) * 100
                self._compareGradientX(array_ops.where_v2, c, x.astype(t), y.astype(t))
                x = np.random.rand(1, 3, 2) * 100
                y = np.random.rand(1) * 100
                self._compareGradientX(array_ops.where_v2, c, x.astype(t), y.astype(t))
                x = np.random.rand(1, 3, 2) * 100
                y = np.random.rand(1, 2) * 100
                self._compareGradientX(array_ops.where_v2, c, x.astype(t), y.astype(t))
                x = np.random.rand(1, 3, 2) * 100
                y = np.random.rand(3, 2) * 100
                self._compareGradientX(array_ops.where_v2, c, x.astype(t), y.astype(t))

    def _testShapeMismatch(self, fn):
        if False:
            print('Hello World!')
        c = np.random.randint(0, 2, 6).astype(np.bool_).reshape(1, 3, 2)
        x = np.random.rand(1, 3, 2) * 100
        y = np.random.rand(2, 5, 3) * 100
        for t in [np.float16, np.float32, np.float64, np.int32, np.int64, np.complex64, np.complex128]:
            with self.subTest(t=t):
                xt = x.astype(t)
                yt = y.astype(t)
                with self.assertRaises(ValueError):
                    fn(c, xt, yt)

    @test_util.run_deprecated_v1
    def testShapeMismatch(self):
        if False:
            return 10
        self._testShapeMismatch(array_ops.where)
        self._testShapeMismatch(array_ops.where_v2)

    def _testEmptyTensor(self, fn):
        if False:
            i = 10
            return i + 15
        c = np.random.randint(0, 3, 0).astype(np.bool_).reshape(1, 3, 0)
        x = np.random.rand(1, 3, 0) * 100
        y = np.random.rand(1, 3, 0) * 100
        z_expected = np.zeros((1, 3, 0), dtype=np.float32)
        with self.cached_session():
            xt = x.astype(np.float32)
            yt = y.astype(np.float32)
            z = fn(c, xt, yt).eval()
            self.assertAllEqual(z_expected, z)

    @test_util.run_deprecated_v1
    def testEmptyTensor(self):
        if False:
            print('Hello World!')
        self._testEmptyTensor(array_ops.where)
        self._testEmptyTensor(array_ops.where_v2)

    def _testNan(self, fn):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            for c in (False, True):
                for a in (7.0, np.nan):
                    for b in (5.0, np.nan):
                        with self.subTest(c=c, a=a, b=b):
                            x = fn(c, a, b).eval()
                            y = a if c else b
                            self.assertEqual(np.isnan(x), np.isnan(y))

    @test_util.run_deprecated_v1
    def testNan(self):
        if False:
            for i in range(10):
                print('nop')
        "Verify that nans don't propagate where they shouldn't."
        self._testNan(array_ops.where)
        self._testNan(array_ops.where_v2)

class BatchSelectOpTest(test.TestCase):
    """Test broadcasting of Select when 'c' is a vec and 't' &'e' are rank2+."""

    def _compare(self, c, x, y, use_gpu):
        if False:
            i = 10
            return i + 15
        np_ans = np.dstack([x_i if c_i else y_i for (c_i, x_i, y_i) in zip(c, x, y)]).transpose([2, 0, 1])
        with test_util.device(use_gpu=use_gpu):
            out = array_ops.where(c, x, y)
            tf_ans = self.evaluate(out)
        self.assertAllEqual(np_ans, tf_ans)
        self.assertShapeEqual(np_ans, out)

    def _compareGradientX(self, c, x, y, numeric_gradient_type=None):
        if False:
            print('Hello World!')
        with self.cached_session():
            inx = ops.convert_to_tensor(x)
            iny = ops.convert_to_tensor(y)
            out = array_ops.where(c, inx, iny)
            s = list(np.shape(x))
            (jacob_t, jacob_n) = gradient_checker.compute_gradient(inx, s, out, s, x_init_value=x)
            if numeric_gradient_type is not None:
                xf = x.astype(numeric_gradient_type)
                yf = y.astype(numeric_gradient_type)
                inxf = ops.convert_to_tensor(xf)
                inyf = ops.convert_to_tensor(yf)
                outf = array_ops.where(c, inxf, inyf)
                (_, jacob_n) = gradient_checker.compute_gradient(inxf, s, outf, s, x_init_value=xf)
                jacob_n = jacob_n.astype(x.dtype)
        if x.dtype == np.float16:
            self.assertAllClose(jacob_t, jacob_n, rtol=0.001, atol=0.001)
        elif x.dtype == np.float32:
            self.assertAllClose(jacob_t, jacob_n, rtol=0.001, atol=0.001)
        elif x.dtype == np.float64:
            self.assertAllClose(jacob_t, jacob_n, rtol=1e-05, atol=1e-05)

    def _compareGradientY(self, c, x, y, numeric_gradient_type=None):
        if False:
            while True:
                i = 10
        with self.cached_session():
            inx = ops.convert_to_tensor(x)
            iny = ops.convert_to_tensor(y)
            out = array_ops.where(c, inx, iny)
            s = list(np.shape(x))
            (jacob_t, jacob_n) = gradient_checker.compute_gradient(iny, s, out, s, x_init_value=y)
            if numeric_gradient_type is not None:
                xf = x.astype(numeric_gradient_type)
                yf = y.astype(numeric_gradient_type)
                inxf = ops.convert_to_tensor(xf)
                inyf = ops.convert_to_tensor(yf)
                outf = array_ops.where(c, inxf, inyf)
                (_, jacob_n) = gradient_checker.compute_gradient(inyf, s, outf, s, x_init_value=yf)
                jacob_n = jacob_n.astype(x.dtype)
        if x.dtype == np.float16:
            self.assertAllClose(jacob_t, jacob_n, rtol=0.001, atol=0.001)
        elif x.dtype == np.float32:
            self.assertAllClose(jacob_t, jacob_n, rtol=0.001, atol=0.001)
        elif x.dtype == np.float64:
            self.assertAllClose(jacob_t, jacob_n, rtol=1e-05, atol=1e-05)

    def testBasic(self):
        if False:
            return 10
        c = np.random.randint(0, 2, 16).astype(np.bool_)
        x = np.random.rand(16, 2, 8) * 100
        y = np.random.rand(16, 2, 8) * 100
        for t in [np.float16, np.float32, np.float64, np.int32, np.int64, np.complex64, np.complex128]:
            with self.subTest(t=t):
                xt = x.astype(t)
                yt = y.astype(t)
                self._compare(c, xt, yt, use_gpu=False)
                if t in [np.float16, np.float32, np.float64]:
                    self._compare(c, xt, yt, use_gpu=True)

    @test_util.run_deprecated_v1
    def testGradients(self):
        if False:
            print('Hello World!')
        c = np.random.randint(0, 2, 16).astype(np.bool_)
        x = np.random.rand(16, 2, 8) * 100
        y = np.random.rand(16, 2, 8) * 100
        for t in [np.float16, np.float32, np.float64]:
            with self.subTest(t=t):
                xt = x.astype(t)
                yt = y.astype(t)
                if t == np.float16:
                    self._compareGradientX(c, xt, yt, np.float64)
                    self._compareGradientY(c, xt, yt, np.float64)
                else:
                    self._compareGradientX(c, xt, yt)
                    self._compareGradientY(c, xt, yt)

    @test_util.run_deprecated_v1
    def testShapeMismatch(self):
        if False:
            while True:
                i = 10
        c = np.random.randint(0, 2, 8).astype(np.bool_)
        x = np.random.rand(16, 3, 2) * 100
        y = np.random.rand(16, 3, 2) * 100
        for t in [np.float16, np.float32, np.float64, np.int32, np.int64, np.complex64, np.complex128]:
            with self.subTest(t=t):
                xt = x.astype(t)
                yt = y.astype(t)
                with self.assertRaises(ValueError):
                    array_ops.where(c, xt, yt)

@test_util.with_eager_op_as_function
class MinMaxOpTest(test.TestCase):

    def _compare(self, x, y, use_gpu):
        if False:
            print('Hello World!')
        (np_min, np_max) = (np.minimum(x, y), np.maximum(x, y))
        with test_util.device(use_gpu=use_gpu):
            inx = ops.convert_to_tensor(x)
            iny = ops.convert_to_tensor(y)
            (omin, omax) = (math_ops.minimum(inx, iny), math_ops.maximum(inx, iny))
            (tf_min, tf_max) = self.evaluate([omin, omax])
        self.assertAllEqual(np_min, tf_min)
        self.assertAllEqual(np_max, tf_max)

    def testBasic(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.random.rand(1, 3, 2) * 100.0
        y = np.random.rand(1, 3, 2) * 100.0
        for t in [np.float16, np.float32, np.float64, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]:
            with self.subTest(t=t):
                self._compare(x.astype(t), y.astype(t), use_gpu=False)
                self._compare(x.astype(t), y.astype(t), use_gpu=True)

    @test_util.set_xla_env_flag(flag='--xla_cpu_enable_fast_min_max=false')
    @test_util.set_xla_env_flag(flag='--xla_gpu_enable_fast_min_max=false')
    def testNaNPropagation(self):
        if False:
            print('Hello World!')
        x = np.array([1.0, np.nan, 1.0, np.nan], dtype=np.float64)
        y = np.array([1.0, 1.0, np.nan, np.nan], dtype=np.float64)
        for t in [np.float16, np.float32, np.float64]:
            with self.subTest(t=t):
                self._compare(x.astype(t), y.astype(t), use_gpu=False)
                self._compare(x.astype(t), y.astype(t), use_gpu=True)

    def testDifferentShapes(self):
        if False:
            print('Hello World!')
        x = np.random.rand(1, 3, 2) * 100.0
        y = np.random.rand(2) * 100.0
        for t in [np.float16, np.float32, np.float64, np.int32, np.int64]:
            with self.subTest(t=t):
                self._compare(x.astype(t), y.astype(t), use_gpu=False)
                self._compare(x.astype(t), y.astype(t), use_gpu=True)

    def testScalar(self):
        if False:
            return 10
        x = np.random.rand(1, 3, 2) * 100.0
        y = np.random.rand(1).item() * 100.0
        for t in [np.float32, np.int32]:
            with self.subTest(t=t):
                self._compare(x.astype(t), t(y), use_gpu=False)
                self._compare(x.astype(t), t(y), use_gpu=True)

    def _compareGradientX(self, func, x, y):
        if False:
            i = 10
            return i + 15
        with self.cached_session():
            inx = ops.convert_to_tensor(x)
            iny = ops.convert_to_tensor(y)
            out = func(inx, iny)
            s = list(np.shape(x))
            (jacob_t, jacob_n) = gradient_checker.compute_gradient(inx, s, out, s, x_init_value=x)
        if x.dtype == np.float16:
            self.assertAllClose(jacob_t, jacob_n, rtol=0.001, atol=0.001)
        elif x.dtype == np.float32:
            self.assertAllClose(jacob_t, jacob_n, rtol=0.001, atol=0.001)
        elif x.dtype == np.float64:
            self.assertAllClose(jacob_t, jacob_n, rtol=1e-05, atol=1e-05)

    def _compareGradientY(self, func, x, y):
        if False:
            return 10
        with self.cached_session():
            inx = ops.convert_to_tensor(x)
            iny = ops.convert_to_tensor(y)
            out = func(inx, iny)
            s = list(np.shape(x))
            (jacob_t, jacob_n) = gradient_checker.compute_gradient(iny, s, out, s, x_init_value=y)
        if x.dtype == np.float16:
            self.assertAllClose(jacob_t, jacob_n, rtol=0.001, atol=0.001)
        elif x.dtype == np.float32:
            self.assertAllClose(jacob_t, jacob_n, rtol=0.001, atol=0.001)
        elif x.dtype == np.float64:
            self.assertAllClose(jacob_t, jacob_n, rtol=1e-05, atol=1e-05)

    @test_util.run_deprecated_v1
    def testGradients(self):
        if False:
            while True:
                i = 10
        x = np.random.rand(1, 3, 2) * 100.0
        y = x + (np.random.randint(2, size=x.shape) - 0.5) * 2
        self._compareGradientX(math_ops.maximum, x, y)
        self._compareGradientY(math_ops.maximum, x, y)
        self._compareGradientX(math_ops.minimum, x, y)
        self._compareGradientY(math_ops.minimum, x, y)

class MathOpsOverloadTest(test.TestCase):

    def _computeTensorAndLiteral(self, x, y, dtype, func):
        if False:
            i = 10
            return i + 15
        with test_util.force_cpu():
            inx = ops.convert_to_tensor(x, dtype=dtype)
            z = func(inx, y)
            return self.evaluate(z)

    def _computeLiteralAndTensor(self, x, y, dtype, func):
        if False:
            print('Hello World!')
        with test_util.force_cpu():
            iny = ops.convert_to_tensor(y, dtype=dtype)
            z = func(x, iny)
            return self.evaluate(z)

    def _compareBinary(self, x, y, dtype, np_func, tf_func):
        if False:
            i = 10
            return i + 15
        np_ans = np_func(x, y)
        if np_func != np.true_divide:
            np_ans = np_ans.astype(np.float32 if dtype == dtypes_lib.bfloat16 else dtype.as_numpy_dtype)
        rtol = 0.01 if dtype in (dtypes_lib.bfloat16, dtypes_lib.float16) else 1e-06
        self.assertAllClose(np_ans, self._computeTensorAndLiteral(x, y, dtype, tf_func), rtol=rtol)
        self.assertAllClose(np_ans, self._computeLiteralAndTensor(x, y, dtype, tf_func), rtol=rtol)

    def _compareUnary(self, x, dtype, np_func, tf_func):
        if False:
            i = 10
            return i + 15
        np_ans = np_func(x).astype(dtype.as_numpy_dtype)
        with test_util.force_cpu():
            self.assertAllClose(np_ans, self.evaluate(tf_func(ops.convert_to_tensor(x, dtype=dtype))))

    def testOverload(self):
        if False:
            while True:
                i = 10
        dtypes = [dtypes_lib.float16, dtypes_lib.float32, dtypes_lib.float64, dtypes_lib.bfloat16, dtypes_lib.uint8, dtypes_lib.uint16, dtypes_lib.uint32, dtypes_lib.uint64, dtypes_lib.int8, dtypes_lib.int16, dtypes_lib.int32, dtypes_lib.int64, dtypes_lib.complex64, dtypes_lib.complex128]
        funcs = [(np.add, _ADD), (np.subtract, _SUB), (np.multiply, _MUL), (np.power, _POW), (np.true_divide, _TRUEDIV), (np.floor_divide, _FLOORDIV), (np.mod, _MOD)]
        for dtype in dtypes:
            for (np_func, tf_func) in funcs:
                with self.subTest(dtype=dtype, np_func=np_func, tf_func=tf_func):
                    if dtype in (dtypes_lib.complex64, dtypes_lib.complex128) and tf_func in (_FLOORDIV, _MOD):
                        continue
                    if dtype in (dtypes_lib.uint8, dtypes_lib.uint16, dtypes_lib.uint32, dtypes_lib.uint64) and tf_func == _POW:
                        continue
                    self._compareBinary(10, 3, dtype, np_func, tf_func)

    def testOverloadComparisons(self):
        if False:
            while True:
                i = 10
        dtypes = [dtypes_lib.float16, dtypes_lib.float32, dtypes_lib.float64, dtypes_lib.uint8, dtypes_lib.uint16, dtypes_lib.uint32, dtypes_lib.uint64, dtypes_lib.int8, dtypes_lib.int16, dtypes_lib.int32, dtypes_lib.int64]
        funcs = [(np.less, _LT), (np.less_equal, _LE), (np.greater, _GT), (np.greater_equal, _GE)]
        for dtype in dtypes:
            for (np_func, tf_func) in funcs:
                with self.subTest(dtype=dtype, np_func=np_func, tf_func=tf_func):
                    self._compareBinary(10, 5, dtype, np_func, tf_func)
        logical_funcs = [(np.logical_and, _AND), (np.logical_or, _OR), (np.logical_xor, _XOR), (np.equal, math_ops.equal), (np.not_equal, math_ops.not_equal)]
        for (np_func, tf_func) in logical_funcs:
            with self.subTest(np_func=np_func, tf_func=tf_func):
                self._compareBinary(True, False, dtypes_lib.bool, np_func, tf_func)
                self._compareBinary(True, True, dtypes_lib.bool, np_func, tf_func)
                self._compareBinary(False, False, dtypes_lib.bool, np_func, tf_func)
                self._compareBinary(False, True, dtypes_lib.bool, np_func, tf_func)
                self._compareBinary([True, True, False, False], [True, False, True, False], dtypes_lib.bool, np_func, tf_func)
        self._compareUnary(True, dtypes_lib.bool, np.logical_not, _INV)
        self._compareUnary(False, dtypes_lib.bool, np.logical_not, _INV)
        self._compareUnary([True, False], dtypes_lib.bool, np.logical_not, _INV)

class IsFiniteInfNanTest(test.TestCase):

    def _compare(self, x, use_gpu):
        if False:
            for i in range(10):
                print('nop')
        with test_util.device(use_gpu=use_gpu):
            inx = ops.convert_to_tensor(x)
            (ofinite, oinf, onan) = (math_ops.is_finite(inx), math_ops.is_inf(inx), math_ops.is_nan(inx))
            (tf_finite, tf_inf, tf_nan) = self.evaluate([ofinite, oinf, onan])
        if x.dtype == dtypes_lib.bfloat16.as_numpy_dtype:
            x = x.astype(np.float32)
        (np_finite, np_inf, np_nan) = (np.isfinite(x), np.isinf(x), np.isnan(x))
        self.assertAllEqual(np_inf, tf_inf)
        self.assertAllEqual(np_nan, tf_nan)
        self.assertAllEqual(np_finite, tf_finite)
        self.assertShapeEqual(np_inf, oinf)
        self.assertShapeEqual(np_nan, onan)
        self.assertShapeEqual(np_finite, ofinite)

    def _testDtype(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        if dtype != dtypes_lib.bfloat16.as_numpy_dtype:
            fi = np.finfo(dtype)
            data = np.array([0, -1, 1, fi.resolution, -fi.resolution, fi.min, fi.max, -np.inf, np.inf, np.nan]).astype(dtype)
        else:
            data = np.array([0, -1, 1, 0.01, -0.01, -3.3895e+38, 3.3895e+38, -np.inf, np.inf, np.nan]).astype(dtype)
        self._compare(data, use_gpu=False)
        self._compare(data, use_gpu=True)

    def testHalf(self):
        if False:
            print('Hello World!')
        self._testDtype(np.float16)

    def testFloat(self):
        if False:
            while True:
                i = 10
        self._testDtype(np.float32)

    def testDouble(self):
        if False:
            print('Hello World!')
        self._testDtype(np.float64)

    def testBfloat16(self):
        if False:
            for i in range(10):
                print('nop')
        self._testDtype(dtypes_lib.bfloat16.as_numpy_dtype)

    def testSqrt(self):
        if False:
            return 10
        for dtype in [np.float16, np.float32, np.float64]:
            fi = np.finfo(dtype)
            for size in [1, 3, 4, 7, 8, 63, 64, 65]:
                for value in [fi.min, -2, -1, 0, fi.tiny, 1, 2, 1000, fi.max / 100]:
                    with self.subTest(dtype=dtype, size=size, value=value):
                        x = np.full((size,), value, dtype=dtype)
                        np_y = np.sqrt(x)
                        np_nan = np.isnan(np_y)
                        with test_util.use_gpu():
                            tf_y = math_ops.sqrt(x)
                            tf_nan = math_ops.is_nan(tf_y)
                            if value < 0:
                                self.assertAllEqual(np_nan, self.evaluate(tf_nan))
                            else:
                                self.assertAllCloseAccordingToType(np_y, self.evaluate(tf_y))

class RoundingTest(test.TestCase):

    def _compare_values(self, x, y=None):
        if False:
            return 10
        y = np.rint(x) if y is None else np.asarray(y)
        tf_rint = math_ops.rint(x)
        np_rint = self.evaluate(tf_rint)
        self.assertAllEqual(y, np_rint)
        self.assertShapeEqual(y, tf_rint)

    def _compare(self, x):
        if False:
            print('Hello World!')
        (np_floor, np_ceil) = (np.floor(x), np.ceil(x))
        inx = ops.convert_to_tensor(x)
        (ofloor, oceil) = (math_ops.floor(inx), math_ops.ceil(inx))
        (tf_floor, tf_ceil) = self.evaluate([ofloor, oceil])
        self.assertAllEqual(np_floor, tf_floor)
        self.assertAllEqual(np_ceil, tf_ceil)
        self.assertShapeEqual(np_floor, ofloor)
        self.assertShapeEqual(np_ceil, oceil)

    def _testDtype(self, dtype):
        if False:
            return 10
        data = (np.arange(-3, 3) / 4.0).reshape(1, 3, 2).astype(dtype)
        self._compare(data)
        if dtype in (np.float16, dtypes_lib.bfloat16.as_numpy_dtype):
            return
        self._compare_values(data)
        x = [0.5, 0.5000001]
        y = [0.0, 1.0]
        self._compare_values(x, y=y)
        x = [-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]
        y = [-2.0, -2.0, -0.0, 0.0, 2.0, 2.0, 2.0]
        self._compare_values(x, y=y)

    def testTypes(self):
        if False:
            while True:
                i = 10
        for dtype in [np.float16, np.float32, np.float64, dtypes_lib.bfloat16.as_numpy_dtype]:
            with self.subTest(dtype=dtype):
                self._testDtype(dtype)

class ComplexMakeRealImagTest(test.TestCase):

    def _compareMake(self, real, imag, use_gpu):
        if False:
            print('Hello World!')
        np_ans = real + 1j * imag
        with test_util.device(use_gpu=use_gpu):
            real = ops.convert_to_tensor(real)
            imag = ops.convert_to_tensor(imag)
            tf_ans = math_ops.complex(real, imag)
            out = self.evaluate(tf_ans)
        self.assertAllEqual(np_ans, out)
        self.assertShapeEqual(np_ans, tf_ans)

    def testMake(self):
        if False:
            while True:
                i = 10
        real = (np.arange(-3, 3) / 4.0).reshape([1, 3, 2]).astype(np.float32)
        imag = (np.arange(-3, 3) / 5.0).reshape([1, 3, 2]).astype(np.float32)
        for use_gpu in [False, True]:
            with self.subTest(use_gpu=use_gpu):
                self._compareMake(real, imag, use_gpu)
                self._compareMake(real, 12.0, use_gpu)
                self._compareMake(23.0, imag, use_gpu)

    def testRealImagNumericType(self):
        if False:
            i = 10
            return i + 15
        for use_gpu in [True, False]:
            for value in [1.0, 1j, 1.0 + 1j]:
                with self.subTest(use_gpu=use_gpu, value=value):
                    (np_real, np_imag) = (np.real(value), np.imag(value))
                    with test_util.device(use_gpu=use_gpu):
                        tf_real = math_ops.real(value)
                        tf_imag = math_ops.imag(value)
                        self.assertAllEqual(np_real, self.evaluate(tf_real))
                        self.assertAllEqual(np_imag, self.evaluate(tf_imag))

    def _compareRealImag(self, cplx, use_gpu):
        if False:
            while True:
                i = 10
        (np_real, np_imag) = (np.real(cplx), np.imag(cplx))
        np_zeros = np_real * 0
        with test_util.device(use_gpu=use_gpu):
            inx = ops.convert_to_tensor(cplx)
            tf_real = math_ops.real(inx)
            tf_imag = math_ops.imag(inx)
            tf_real_real = math_ops.real(tf_real)
            tf_imag_real = math_ops.imag(tf_real)
            self.assertAllEqual(np_real, self.evaluate(tf_real))
            self.assertAllEqual(np_imag, self.evaluate(tf_imag))
            self.assertAllEqual(np_real, self.evaluate(tf_real_real))
            self.assertAllEqual(np_zeros, self.evaluate(tf_imag_real))

    def testRealImag64(self):
        if False:
            return 10
        real = (np.arange(-3, 3) / 4.0).reshape([1, 3, 2]).astype(np.float32)
        imag = (np.arange(-3, 3) / 5.0).reshape([1, 3, 2]).astype(np.float32)
        cplx = real + 1j * imag
        self._compareRealImag(cplx, use_gpu=False)
        self._compareRealImag(cplx, use_gpu=True)

    def testRealImag128(self):
        if False:
            for i in range(10):
                print('nop')
        real = (np.arange(-3, 3) / 4.0).reshape([1, 3, 2]).astype(np.float64)
        imag = (np.arange(-3, 3) / 5.0).reshape([1, 3, 2]).astype(np.float64)
        cplx = real + 1j * imag
        self._compareRealImag(cplx, use_gpu=False)
        self._compareRealImag(cplx, use_gpu=True)

    def _compareAngle(self, cplx, use_gpu):
        if False:
            print('Hello World!')
        np_angle = np.angle(cplx)
        with test_util.device(use_gpu=use_gpu):
            inx = ops.convert_to_tensor(cplx)
            tf_angle = math_ops.angle(inx)
            tf_angle_val = self.evaluate(tf_angle)
        self.assertAllClose(np_angle, tf_angle_val)
        self.assertShapeEqual(np_angle, tf_angle)

    def testAngle(self):
        if False:
            for i in range(10):
                print('nop')
        mag = np.random.rand(10).astype(np.float32)
        angle = (2 * np.pi * np.arange(10) / 10.0).astype(np.float32)
        cplx = mag * np.exp(1j * angle)
        cplx = np.append(cplx, [1.0, 1j, -1.0, -1j])
        self._compareAngle(cplx, use_gpu=False)
        self._compareAngle(cplx, use_gpu=True)
        real = (np.arange(-2, 2) / 2.0).astype(np.float64)
        self._compareAngle(real, use_gpu=False)
        self._compareAngle(real, use_gpu=True)

    def testAngle64(self):
        if False:
            while True:
                i = 10
        mag = np.random.rand(10).astype(np.float64)
        angle = (2 * np.pi * np.arange(10) / 100.0).astype(np.float64)
        cplx = mag * np.exp(1j * angle)
        cplx = np.append(cplx, [1.0, 1j, -1.0, -1j])
        self._compareAngle(cplx, use_gpu=False)
        self._compareAngle(cplx, use_gpu=True)
        real = (np.arange(-2, 2) / 2.0).astype(np.float64)
        self._compareAngle(real, use_gpu=False)
        self._compareAngle(real, use_gpu=True)

    @test_util.run_deprecated_v1
    def testRealReal(self):
        if False:
            for i in range(10):
                print('nop')
        for dtype in (dtypes_lib.int32, dtypes_lib.int64, dtypes_lib.float32, dtypes_lib.float64):
            with self.subTest(dtype=dtype):
                x = array_ops.placeholder(dtype)
                y = math_ops.real(x)
                self.assertEqual(x, y)

    def _compareConj(self, cplx, use_gpu):
        if False:
            print('Hello World!')
        np_ans = np.conj(cplx)
        with test_util.device(use_gpu=use_gpu):
            inx = ops.convert_to_tensor(cplx)
            tf_conj = math_ops.conj(inx)
            tf_ans = self.evaluate(tf_conj)
        self.assertAllEqual(np_ans, tf_ans)
        self.assertShapeEqual(np_ans, tf_conj)

    def testConj64(self):
        if False:
            i = 10
            return i + 15
        real = (np.arange(-3, 3) / 4.0).reshape([1, 3, 2]).astype(np.float32)
        imag = (np.arange(-3, 3) / 5.0).reshape([1, 3, 2]).astype(np.float32)
        cplx = real + 1j * imag
        self._compareConj(cplx, use_gpu=False)
        self._compareConj(cplx, use_gpu=True)

    def testConj128(self):
        if False:
            return 10
        real = (np.arange(-3, 3) / 4.0).reshape([1, 3, 2]).astype(np.float64)
        imag = (np.arange(-3, 3) / 5.0).reshape([1, 3, 2]).astype(np.float64)
        cplx = real + 1j * imag
        self._compareConj(cplx, use_gpu=False)
        self._compareConj(cplx, use_gpu=True)

    @test_util.run_deprecated_v1
    def testConjReal(self):
        if False:
            i = 10
            return i + 15
        for dtype in (dtypes_lib.int32, dtypes_lib.int64, dtypes_lib.float16, dtypes_lib.float32, dtypes_lib.float64):
            with self.subTest(dtype=dtype):
                x = array_ops.placeholder(dtype)
                y = math_ops.conj(x)
                self.assertEqual(x, y)

    @test_util.run_deprecated_v1
    def testConjString(self):
        if False:
            print('Hello World!')
        x = array_ops.placeholder(dtypes_lib.string)
        with self.assertRaisesRegex(TypeError, 'Expected numeric or variant tensor'):
            math_ops.conj(x)

    def _compareGradient(self, x):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            inx = ops.convert_to_tensor(x)
            (real, imag) = array_ops.split(value=inx, num_or_size_splits=2, axis=1)
            (real, imag) = (array_ops.reshape(real, [-1]), array_ops.reshape(imag, [-1]))
            cplx = math_ops.complex(real, imag)
            cplx = math_ops.conj(cplx)
            loss = math_ops.reduce_sum(math_ops.square(math_ops.real(cplx))) + math_ops.reduce_sum(math_ops.square(math_ops.imag(cplx)))
            epsilon = 0.001
            (jacob_t, jacob_n) = gradient_checker.compute_gradient(inx, list(x.shape), loss, list(loss.shape), x_init_value=x, delta=epsilon)
        self.assertAllClose(jacob_t, jacob_n, rtol=epsilon, atol=epsilon)

    def _compareBroadcastGradient(self, x):
        if False:
            while True:
                i = 10
        x_ = ops.convert_to_tensor(x)
        epsilon = 0.001
        with self.cached_session():
            for args in [(x_, 0.0), (0.0, x_)]:
                with self.subTest(args=args):
                    z = math_ops.reduce_sum(math_ops.abs(math_ops.complex(*args)))
                    (jacob_t, jacob_n) = gradient_checker.compute_gradient(x_, list(x.shape), z, [1], x_init_value=x, delta=epsilon)
                    self.assertAllClose(jacob_t, jacob_n, rtol=epsilon, atol=epsilon)

    @test_util.run_deprecated_v1
    def testGradient(self):
        if False:
            return 10
        data = np.arange(1, 2, 0.1).reshape([5, 2]).astype(np.float32)
        self._compareGradient(data)
        self._compareBroadcastGradient(data)
        data = np.arange(1, 2, 0.1).reshape([5, 2]).astype(np.float64)
        self._compareGradient(data)

    def _compareMulGradient(self, data):
        if False:
            return 10
        with self.cached_session():
            inp = ops.convert_to_tensor(data)
            (xr, xi, yr, yi) = array_ops.split(value=inp, num_or_size_splits=4, axis=1)

            def vec(x):
                if False:
                    for i in range(10):
                        print('nop')
                return array_ops.reshape(x, [-1])
            (xr, xi, yr, yi) = (vec(xr), vec(xi), vec(yr), vec(yi))

            def cplx(r, i):
                if False:
                    return 10
                return math_ops.complex(r, i)
            (x, y) = (cplx(xr, xi), cplx(yr, yi))
            z = x * y
            loss = math_ops.reduce_sum(math_ops.real(z) + math_ops.imag(z))
            epsilon = 0.005
            (jacob_t, jacob_n) = gradient_checker.compute_gradient(inp, list(data.shape), loss, [1], x_init_value=data, delta=epsilon)
        self.assertAllClose(jacob_t, jacob_n, rtol=epsilon, atol=epsilon)

    @test_util.run_deprecated_v1
    def testMulGradient(self):
        if False:
            return 10
        data = np.arange(1, 2, 0.125).reshape([2, 4]).astype(np.float32)
        self._compareMulGradient(data)

class PolyvalTest(test.TestCase):

    def _runtest(self, dtype, degree):
        if False:
            i = 10
            return i + 15
        x = np.random.rand(2, 2).astype(dtype)
        coeffs = [np.random.rand(2, 2).astype(dtype) for _ in range(degree + 1)]
        np_val = np.polyval(coeffs, x)
        with self.cached_session():
            tf_val = math_ops.polyval(coeffs, x)
            self.assertAllClose(np_val, self.evaluate(tf_val))

    def testSimple(self):
        if False:
            for i in range(10):
                print('nop')
        for dtype in [np.int32, np.float32, np.float64, np.complex64, np.complex128]:
            for degree in range(5):
                with self.subTest(dtype=dtype, degree=degree):
                    self._runtest(dtype, degree)

    def testBroadcast(self):
        if False:
            for i in range(10):
                print('nop')
        dtype = np.float32
        degree = 3
        shapes = [(1,), (2, 1), (1, 2), (2, 2)]
        for x_shape in shapes:
            for coeff_shape in shapes:
                with self.subTest(x_shape=x_shape, coeff_shape=coeff_shape):
                    x = np.random.rand(*x_shape).astype(dtype)
                    coeffs = [np.random.rand(*coeff_shape).astype(dtype) for _ in range(degree + 1)]
                    np_val = np.polyval(coeffs, x)
                    with self.cached_session():
                        tf_val = math_ops.polyval(coeffs, x)
                        self.assertAllClose(np_val, self.evaluate(tf_val))

    def testEmpty(self):
        if False:
            i = 10
            return i + 15
        x = np.random.rand(2, 2).astype(np.float32)
        coeffs = []
        np_val = np.polyval(coeffs, x)
        with self.cached_session():
            tf_val = math_ops.polyval(coeffs, x)
            self.assertAllClose(np_val, self.evaluate(tf_val))

    def test_coeffs_raise(self):
        if False:
            i = 10
            return i + 15
        x = np.random.rand(2, 2).astype(np.float32)
        coeffs = {}
        with self.assertRaisesRegex(ValueError, 'Argument coeffs must be list'):
            math_ops.polyval(coeffs, x)

class RealTest(test.TestCase):

    def _run_test(self, input_values, expected_values):
        if False:
            i = 10
            return i + 15
        res = math_ops.real(input_values)
        self.assertAllEqual(res, expected_values)

    def test_real(self):
        if False:
            print('Hello World!')
        test_cases = [(np.complex64, [-2.25 + 4.75j, 3.25 + 5.75j], [-2.25, 3.25]), (np.complex128, [-2.25 + 4.75j, 3.25 + 5.75j], [-2.25, 3.25]), (np.float32, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), (np.float64, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])]
        for (dtype, input_values, expected_values) in test_cases:
            with self.subTest(dtype=dtype):
                self._run_test(input_values, expected_values)

    def test_real_raises_error_for_non_numeric_tensor(self):
        if False:
            return 10
        x = np.array(['Hello', 'World'])
        with self.assertRaisesRegex(TypeError, 'input must be a numeric tensor'):
            self._run_test(x, None)
if __name__ == '__main__':
    test.main()