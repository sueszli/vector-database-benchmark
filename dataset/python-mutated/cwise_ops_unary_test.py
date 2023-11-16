"""Functional tests for unary coefficient-wise operations."""
import math
import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_grad
from tensorflow.python.ops import special_math_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
_NEG = lambda x: -x
_ABS = abs

def _sparsify(x, thresh=0.5, index_dtype=np.int64):
    if False:
        while True:
            i = 10
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
    if dtype == dtypes_lib.bfloat16.as_numpy_dtype:
        return 0.005
    if dtype == np.float16:
        return 0.005
    elif dtype in (np.float32, np.complex64):
        return 0.001
    elif dtype in (np.float64, np.complex128):
        return 1e-05
    else:
        return None

class UnaryOpTest(test.TestCase):

    def _compareCpu(self, x, np_func, tf_func, grad_rtol=None, grad_atol=None):
        if False:
            print('Hello World!')
        if grad_rtol is None:
            grad_rtol = _default_tolerance(x.dtype)
        if grad_atol is None:
            grad_atol = _default_tolerance(x.dtype)
        np_ans = np_func(x)
        with self.cached_session(use_gpu=False):
            inx = ops.convert_to_tensor(x)
            y = tf_func(inx)
            tf_cpu = self.evaluate(y)
            self.assertShapeEqual(np_ans, y)
            if x.dtype == np.float16:
                self.assertAllClose(np_ans, tf_cpu, rtol=0.001, atol=0.001)
            elif x.dtype == dtypes_lib.bfloat16.as_numpy_dtype:
                self.assertAllClose(np_ans, tf_cpu, rtol=0.01, atol=0.01)
            else:
                self.assertAllClose(np_ans, tf_cpu)
            if x.dtype in (np.complex64, np.complex128) and tf_func == math_ops.sign:
                return
            if tf_func == math_ops.round:
                return
            if x.dtype in (np.float16, dtypes_lib.bfloat16.as_numpy_dtype):
                s = list(np.shape(x))
                (jacob_t, _) = gradient_checker.compute_gradient(inx, s, y, s, x_init_value=x)
                xf = x.astype(np.float64)
                inxf = ops.convert_to_tensor(xf)
                yf = tf_func(inxf)
                (_, jacob_n) = gradient_checker.compute_gradient(inxf, s, yf, s, x_init_value=xf, delta=0.01)
                jacob_n = jacob_n.astype(x.dtype)
                self.assertAllClose(jacob_t, jacob_n, rtol=grad_rtol, atol=grad_atol)
            elif x.dtype in (np.float32, np.complex64):
                s = list(np.shape(x))
                (jacob_t, jacob_n) = gradient_checker.compute_gradient(inx, s, y, s, x_init_value=x, delta=0.001)
                self.assertAllClose(jacob_t, jacob_n, rtol=grad_rtol, atol=grad_atol)
            elif x.dtype in (np.float64, np.complex128):
                s = list(np.shape(x))
                (jacob_t, jacob_n) = gradient_checker.compute_gradient(inx, s, y, s, x_init_value=x, delta=1e-05)
                self.assertAllClose(jacob_t, jacob_n, rtol=grad_rtol, atol=grad_atol)

    def _check(self, result_tensor, result_np, input_sp_t, tol):
        if False:
            return 10
        self.assertTrue(isinstance(result_tensor, sparse_tensor.SparseTensor))
        self.assertTrue(isinstance(input_sp_t, sparse_tensor.SparseTensor))
        self.assertAllEqual(input_sp_t.indices, result_tensor.indices)
        self.assertAllEqual(input_sp_t.dense_shape, result_tensor.dense_shape)
        if tol is None:
            self.assertAllClose(result_np, result_tensor.values)
        else:
            self.assertAllClose(result_np, result_tensor.values, rtol=tol, atol=tol)

    def _compareSparseCpu(self, x, np_func, tf_func, tol):
        if False:
            while True:
                i = 10
        (x_sp, x_sp_vals) = _sparsify(x)
        res_np = np_func(x_sp_vals)
        with test_util.force_cpu():
            self._check(tf_func(x_sp), res_np, x_sp, tol)

    def _compareGpu(self, x, np_func, tf_func):
        if False:
            print('Hello World!')
        np_ans = np_func(x)
        with test_util.use_gpu():
            result = tf_func(ops.convert_to_tensor(x))
            tf_gpu = self.evaluate(result)
            self.assertAllCloseAccordingToType(np_ans, tf_gpu, atol=2e-06)

    def _compareSparseGpu(self, x, np_func, tf_func, tol):
        if False:
            print('Hello World!')
        (x_sp, x_sp_vals) = _sparsify(x)
        res_np = np_func(x_sp_vals)
        with test_util.use_gpu():
            self._check(tf_func(x_sp), res_np, x_sp, tol)

    def _compareBoth(self, x, np_func, tf_func, grad_tol=None):
        if False:
            while True:
                i = 10
        self._compareCpu(x, np_func, tf_func, grad_rtol=grad_tol, grad_atol=grad_tol)
        self._compareGpu(x, np_func, tf_func)

    def _compareBothSparse(self, x, np_func, tf_func, tol=None):
        if False:
            while True:
                i = 10
        self._compareSparseCpu(x, np_func, tf_func, tol)
        self._compareSparseGpu(x, np_func, tf_func, tol)

    def _inv(self, x):
        if False:
            for i in range(10):
                print('nop')
        return 1.0 / x

    def _rsqrt(self, x):
        if False:
            print('Hello World!')
        return self._inv(np.sqrt(x))

    def _sigmoid(self, x):
        if False:
            i = 10
            return i + 15
        return 1.0 / (1.0 + np.exp(-x))

    def _log_sigmoid(self, x):
        if False:
            print('Hello World!')
        return np.log(self._sigmoid(x))

    def _replace_domain_error_with_inf(self, fn):
        if False:
            i = 10
            return i + 15

        def func(x):
            if False:
                while True:
                    i = 10
            try:
                return fn(x)
            except ValueError as e:
                if 'domain error' in str(e):
                    return np.inf * np.ones_like(x)
                else:
                    raise e
        return func

    @test_util.run_deprecated_v1
    def testFloatBasic(self):
        if False:
            return 10
        x = np.arange(-3, 3).reshape(1, 3, 2).astype(np.float32)
        w = x - x.min() + 1.02
        y = (x + 0.5).astype(np.float32)
        z = (x + 15.5).astype(np.float32)
        k = np.arange(-0.9, 0.9, 0.25).astype(np.float32)
        self._compareBoth(x, np.abs, math_ops.abs)
        self._compareBoth(x, np.abs, _ABS)
        self._compareBoth(x, np.negative, math_ops.negative)
        self._compareBoth(x, np.negative, _NEG)
        self._compareBoth(y, self._inv, math_ops.reciprocal)
        self._compareBoth(x, np.square, math_ops.square)
        self._compareBoth(z, np.sqrt, math_ops.sqrt)
        self._compareBoth(z, self._rsqrt, math_ops.rsqrt)
        self._compareBoth(x, np.exp, math_ops.exp)
        self._compareBoth(x, np.expm1, math_ops.expm1)
        self._compareBoth(z, np.log, math_ops.log)
        self._compareBoth(z, np.log1p, math_ops.log1p)
        self._compareBoth(x, np.sinh, math_ops.sinh)
        self._compareBoth(x, np.cosh, math_ops.cosh)
        self._compareBoth(x, np.tanh, math_ops.tanh)
        self._compareBoth(x, np.arcsinh, math_ops.asinh)
        self._compareBoth(w, np.arccosh, math_ops.acosh)
        self._compareBoth(k, np.arctanh, math_ops.atanh)
        self._compareBoth(x, self._sigmoid, math_ops.sigmoid)
        self._compareBoth(x, self._log_sigmoid, math_ops.log_sigmoid)
        self._compareBoth(y, np.sign, math_ops.sign)
        self._compareBoth(x, np.sin, math_ops.sin)
        self._compareBoth(x, np.cos, math_ops.cos)
        self._compareBoth(k, np.arcsin, math_ops.asin)
        self._compareBoth(k, np.arccos, math_ops.acos)
        self._compareBoth(x, np.arctan, math_ops.atan)
        self._compareBoth(x, np.tan, math_ops.tan)
        self._compareBoth(y, np.vectorize(self._replace_domain_error_with_inf(math.lgamma)), math_ops.lgamma)
        self._compareBoth(x, np.vectorize(math.erf), math_ops.erf)
        self._compareBoth(x, np.vectorize(math.erfc), math_ops.erfc)
        try:
            from scipy import special
            self._compareBoth(x, special.i0e, special_math_ops.bessel_i0e)
            self._compareBoth(x, special.i1e, special_math_ops.bessel_i1e)
        except ImportError as e:
            tf_logging.warn('Cannot test special functions: %s' % str(e))
        self._compareBothSparse(x, np.abs, math_ops.abs)
        self._compareBothSparse(x, np.negative, math_ops.negative)
        self._compareBothSparse(x, np.square, math_ops.square)
        self._compareBothSparse(z, np.sqrt, math_ops.sqrt, tol=0.001)
        self._compareBothSparse(x, np.tanh, math_ops.tanh)
        self._compareBothSparse(y, np.sign, math_ops.sign)
        self._compareBothSparse(x, np.vectorize(math.erf), math_ops.erf)

    @test_util.run_deprecated_v1
    def testFloatTanhEdge(self):
        if False:
            return 10
        x = np.arange(40, 40 + 6).reshape(6).astype(np.float32)
        self._compareBoth(x, np.tanh, math_ops.tanh)
        x = np.arange(-40, -40 + 6).reshape(6).astype(np.float32)
        self._compareBoth(x, np.tanh, math_ops.tanh)

    @test_util.run_deprecated_v1
    def testFloatEmpty(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.empty((2, 0, 5), dtype=np.float32)
        self._compareBoth(x, np.abs, math_ops.abs)
        self._compareBoth(x, np.abs, _ABS)
        self._compareBoth(x, np.negative, math_ops.negative)
        self._compareBoth(x, np.negative, _NEG)
        self._compareBoth(x, self._inv, math_ops.reciprocal)
        self._compareBoth(x, np.square, math_ops.square)
        self._compareBoth(x, np.sqrt, math_ops.sqrt)
        self._compareBoth(x, self._rsqrt, math_ops.rsqrt)
        self._compareBoth(x, np.exp, math_ops.exp)
        self._compareBoth(x, np.expm1, math_ops.expm1)
        self._compareBoth(x, np.log, math_ops.log)
        self._compareBoth(x, np.log1p, math_ops.log1p)
        self._compareBoth(x, np.sinh, math_ops.sinh)
        self._compareBoth(x, np.arcsinh, math_ops.asinh)
        self._compareBoth(x, np.cosh, math_ops.cosh)
        self._compareBoth(x, np.arccosh, math_ops.acosh)
        self._compareBoth(x, np.tanh, math_ops.tanh)
        self._compareBoth(x, np.arctanh, math_ops.atanh)
        self._compareBoth(x, self._sigmoid, math_ops.sigmoid)
        self._compareBoth(x, np.sign, math_ops.sign)
        self._compareBoth(x, np.sin, math_ops.sin)
        self._compareBoth(x, np.cos, math_ops.cos)
        self._compareBoth(x, np.sign, math_ops.lgamma)
        self._compareBoth(x, np.sign, math_ops.erf)
        self._compareBoth(x, np.sign, math_ops.erfc)
        self._compareBoth(x, np.tan, math_ops.tan)
        self._compareBoth(x, np.arcsin, math_ops.asin)
        self._compareBoth(x, np.arccos, math_ops.acos)
        self._compareBoth(x, np.arctan, math_ops.atan)
        try:
            from scipy import special
            self._compareBoth(x, special.i0e, special_math_ops.bessel_i0e)
            self._compareBoth(x, special.i1e, special_math_ops.bessel_i1e)
        except ImportError as e:
            tf_logging.warn('Cannot test special functions: %s' % str(e))
        self._compareBothSparse(x, np.abs, math_ops.abs)
        self._compareBothSparse(x, np.negative, math_ops.negative)
        self._compareBothSparse(x, np.square, math_ops.square)
        self._compareBothSparse(x, np.sqrt, math_ops.sqrt, tol=0.001)
        self._compareBothSparse(x, np.tanh, math_ops.tanh)
        self._compareBothSparse(x, np.sign, math_ops.sign)
        self._compareBothSparse(x, np.sign, math_ops.erf)

    @test_util.run_deprecated_v1
    def testDoubleBasic(self):
        if False:
            return 10
        x = np.arange(-3, 3).reshape(1, 3, 2).astype(np.float64)
        w = x - x.min() + 1.02
        y = (x + 0.5).astype(np.float64)
        z = (x + 15.5).astype(np.float64)
        k = np.arange(-0.9, 0.9, 0.35).reshape(1, 3, 2).astype(np.float64)
        self._compareBoth(x, np.abs, math_ops.abs)
        self._compareBoth(x, np.abs, _ABS)
        self._compareBoth(x, np.negative, math_ops.negative)
        self._compareBoth(x, np.negative, _NEG)
        self._compareBoth(y, self._inv, math_ops.reciprocal)
        self._compareBoth(x, np.square, math_ops.square)
        self._compareBoth(z, np.sqrt, math_ops.sqrt)
        self._compareBoth(z, self._rsqrt, math_ops.rsqrt)
        self._compareBoth(x, np.exp, math_ops.exp)
        self._compareBoth(x, np.expm1, math_ops.expm1)
        self._compareBoth(z, np.log, math_ops.log)
        self._compareBoth(z, np.log1p, math_ops.log1p)
        self._compareBoth(x, np.sinh, math_ops.sinh)
        self._compareBoth(x, np.cosh, math_ops.cosh)
        self._compareBoth(x, np.tanh, math_ops.tanh)
        self._compareBoth(x, np.arcsinh, math_ops.asinh)
        self._compareBoth(w, np.arccosh, math_ops.acosh)
        self._compareBoth(k, np.arctanh, math_ops.atanh)
        self._compareBoth(x, self._sigmoid, math_ops.sigmoid)
        self._compareBoth(y, np.sign, math_ops.sign)
        self._compareBoth(x, np.sin, math_ops.sin)
        self._compareBoth(x, np.cos, math_ops.cos)
        self._compareBoth(y, np.vectorize(self._replace_domain_error_with_inf(math.lgamma)), math_ops.lgamma)
        self._compareBoth(x, np.vectorize(math.erf), math_ops.erf)
        self._compareBoth(x, np.vectorize(math.erfc), math_ops.erfc)
        self._compareBoth(x, np.arctan, math_ops.atan)
        self._compareBoth(k, np.arcsin, math_ops.asin)
        self._compareBoth(k, np.arccos, math_ops.acos)
        self._compareBoth(k, np.tan, math_ops.tan)
        try:
            from scipy import special
            self._compareBoth(x, special.i0e, special_math_ops.bessel_i0e)
            self._compareBoth(x, special.i1e, special_math_ops.bessel_i1e)
        except ImportError as e:
            tf_logging.warn('Cannot test special functions: %s' % str(e))
        self._compareBothSparse(x, np.abs, math_ops.abs)
        self._compareBothSparse(x, np.negative, math_ops.negative)
        self._compareBothSparse(x, np.square, math_ops.square)
        self._compareBothSparse(z, np.sqrt, math_ops.sqrt, tol=0.001)
        self._compareBothSparse(x, np.tanh, math_ops.tanh)
        self._compareBothSparse(y, np.sign, math_ops.sign)
        self._compareBothSparse(x, np.vectorize(math.erf), math_ops.erf)

    @test_util.run_deprecated_v1
    def testHalfBasic(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.arange(-3, 3).reshape(1, 3, 2).astype(np.float16)
        w = x - x.min() + 1.1
        y = (x + 0.5).astype(np.float16)
        z = (x + 15.5).astype(np.float16)
        k = np.arange(-0.9, 0.9, 0.05).astype(np.float16)
        self._compareBoth(x, np.abs, math_ops.abs)
        self._compareBoth(x, np.abs, _ABS)
        self._compareBoth(x, np.negative, math_ops.negative)
        self._compareBoth(x, np.negative, _NEG)
        self._compareBoth(y, self._inv, math_ops.reciprocal)
        self._compareBoth(x, np.square, math_ops.square)
        self._compareBoth(z, np.sqrt, math_ops.sqrt)
        self._compareBoth(z, self._rsqrt, math_ops.rsqrt)
        self._compareBoth(x, np.exp, math_ops.exp)
        self._compareBoth(x, np.expm1, math_ops.expm1)
        self._compareBoth(z, np.log, math_ops.log)
        self._compareBoth(z, np.log1p, math_ops.log1p)
        self._compareBoth(x, np.sinh, math_ops.sinh)
        self._compareBoth(x, np.cosh, math_ops.cosh)
        self._compareBoth(x, np.tanh, math_ops.tanh)
        self._compareBoth(x, self._sigmoid, math_ops.sigmoid)
        self._compareBoth(y, np.sign, math_ops.sign)
        self._compareBoth(x, np.sin, math_ops.sin)
        self._compareBoth(x, np.cos, math_ops.cos)
        self._compareBoth(x, np.tan, math_ops.tan)
        self._compareBoth(k, np.arcsin, math_ops.asin)
        self._compareBoth(k, np.arccos, math_ops.acos)
        self._compareBoth(x, np.arctan, math_ops.atan)
        self._compareBoth(x, np.arcsinh, math_ops.asinh)
        self._compareBoth(w, np.arccosh, math_ops.acosh, grad_tol=0.001)
        self._compareBoth(k, np.arctanh, math_ops.atanh)
        self._compareBoth(y, np.vectorize(self._replace_domain_error_with_inf(math.lgamma)), math_ops.lgamma)
        self._compareBoth(x, np.vectorize(math.erf), math_ops.erf)
        self._compareBoth(x, np.vectorize(math.erfc), math_ops.erfc)
        self._compareBothSparse(x, np.abs, math_ops.abs)
        self._compareBothSparse(x, np.negative, math_ops.negative)
        self._compareBothSparse(x, np.square, math_ops.square)
        self._compareBothSparse(z, np.sqrt, math_ops.sqrt, tol=0.001)
        self._compareBothSparse(x, np.tanh, math_ops.tanh)
        self._compareBothSparse(y, np.sign, math_ops.sign)
        self._compareBothSparse(x, np.vectorize(math.erf), math_ops.erf, tol=0.001)

    @test_util.run_deprecated_v1
    def testBFloat16Basic(self):
        if False:
            while True:
                i = 10

        def compute_f32(np_func):
            if False:
                for i in range(10):
                    print('nop')
            'Decorator to compute Numpy function with float32 math.'

            def f(x):
                if False:
                    while True:
                        i = 10
                y = np_func(x.astype(np.float32))
                return y.astype(x.dtype)
            return f
        bfloat16 = dtypes_lib.bfloat16.as_numpy_dtype
        x = np.arange(-6, 6, 2).reshape(1, 3, 2).astype(bfloat16)
        w = x - x.min() + 1.1
        y = (x + 0.5).astype(bfloat16)
        z = (x + 15.5).astype(bfloat16)
        k = np.arange(-0.9, 0.9, 0.05).astype(bfloat16)
        self._compareBoth(x, np.abs, math_ops.abs)
        self._compareBoth(x, np.abs, _ABS)
        self._compareBoth(x, np.negative, math_ops.negative)
        self._compareBoth(x, np.negative, _NEG)
        self._compareBoth(y, compute_f32(self._inv), math_ops.reciprocal)
        self._compareCpu(x, np.round, math_ops.round)
        self._compareCpu(x, np.exp, math_ops.exp)
        self._compareCpu(x, np.expm1, math_ops.expm1)
        self._compareCpu(z, compute_f32(np.log), math_ops.log)
        self._compareCpu(z, compute_f32(np.log1p), math_ops.log1p)
        self._compareBoth(y, np.sign, math_ops.sign)
        self._compareCpu(z, self._rsqrt, math_ops.rsqrt)
        self._compareCpu(x, np.square, math_ops.square)
        self._compareBoth(x, compute_f32(np.sin), math_ops.sin)
        self._compareBoth(x, compute_f32(np.cos), math_ops.cos)
        self._compareBoth(x, compute_f32(np.tan), math_ops.tan)
        self._compareBoth(x, compute_f32(np.sinh), math_ops.sinh)
        self._compareBoth(x, compute_f32(np.cosh), math_ops.cosh)
        self._compareBoth(x, compute_f32(np.tanh), math_ops.tanh)
        self._compareBoth(k, compute_f32(np.arcsin), math_ops.asin)
        self._compareBoth(k, compute_f32(np.arccos), math_ops.acos)
        self._compareBoth(x, compute_f32(np.arctan), math_ops.atan)
        self._compareBoth(x, compute_f32(np.arcsinh), math_ops.asinh)
        self._compareBoth(w, compute_f32(np.arccosh), math_ops.acosh)
        self._compareBoth(k, compute_f32(np.arctanh), math_ops.atanh, grad_tol=0.01)
        self._compareBoth(x, compute_f32(np.vectorize(math.erf)), math_ops.erf)
        self._compareBoth(x, compute_f32(np.vectorize(math.erfc)), math_ops.erfc)
        self._compareBoth(x, compute_f32(np.square), math_ops.square)

    @test.disable_with_predicate(pred=test.is_built_with_rocm, skip_message='On ROCm this test fails')
    def testInt8Basic(self):
        if False:
            i = 10
            return i + 15
        x = np.arange(-6, 6, 2).reshape(1, 3, 2).astype(np.int8)
        self._compareCpu(x, np.abs, math_ops.abs)
        self._compareCpu(x, np.abs, _ABS)
        self._compareBoth(x, np.negative, math_ops.negative)
        self._compareBoth(x, np.negative, _NEG)
        self._compareBoth(x, np.sign, math_ops.sign)

    @test.disable_with_predicate(pred=test.is_built_with_rocm, skip_message='On ROCm this test fails')
    def testUInt8Basic(self):
        if False:
            while True:
                i = 10
        x = np.arange(6).reshape(1, 3, 2).astype(np.uint8)
        self._compareBoth(x, np.square, math_ops.square)

    @test.disable_with_predicate(pred=test.is_built_with_rocm, skip_message='On ROCm this test fails')
    def testInt16Basic(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.arange(-6, 6, 2).reshape(1, 3, 2).astype(np.int16)
        self._compareCpu(x, np.abs, math_ops.abs)
        self._compareCpu(x, np.abs, _ABS)
        self._compareBoth(x, np.negative, math_ops.negative)
        self._compareBoth(x, np.negative, _NEG)
        self._compareBoth(x, np.sign, math_ops.sign)

    @test.disable_with_predicate(pred=test.is_built_with_rocm, skip_message='On ROCm this test fails')
    def testUInt16Basic(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.arange(6).reshape(1, 3, 2).astype(np.uint16)
        self._compareBoth(x, np.square, math_ops.square)

    def testInt32Basic(self):
        if False:
            print('Hello World!')
        x = np.arange(-6, 6, 2).reshape(1, 3, 2).astype(np.int32)
        self._compareCpu(x, np.abs, math_ops.abs)
        self._compareCpu(x, np.abs, _ABS)
        self._compareBoth(x, np.negative, math_ops.negative)
        self._compareBoth(x, np.negative, _NEG)
        self._compareBoth(x, np.square, math_ops.square)
        self._compareCpu(x, np.sign, math_ops.sign)
        self._compareBothSparse(x, np.abs, math_ops.abs)
        self._compareBothSparse(x, np.negative, math_ops.negative)
        self._compareBothSparse(x, np.square, math_ops.square)
        self._compareBothSparse(x, np.sign, math_ops.sign)

    @test.disable_with_predicate(pred=test.is_built_with_rocm, skip_message='On ROCm this test fails')
    def testUInt32Basic(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.arange(6).reshape(1, 3, 2).astype(np.uint32)
        self._compareBoth(x, np.square, math_ops.square)

    def testInt64Basic(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.arange(-6 << 40, 6 << 40, 2 << 40).reshape(1, 3, 2).astype(np.int64)
        self._compareCpu(x, np.abs, math_ops.abs)
        self._compareCpu(x, np.abs, _ABS)
        self._compareCpu(x, np.negative, math_ops.negative)
        self._compareCpu(x, np.negative, _NEG)
        self._compareCpu(x, np.sign, math_ops.sign)
        self._compareBothSparse(x, np.abs, math_ops.abs)
        self._compareBothSparse(x, np.negative, math_ops.negative)
        self._compareBothSparse(x, np.sign, math_ops.sign)

    def testInt64Square(self):
        if False:
            print('Hello World!')
        x = np.arange(-6 << 20, 6 << 20, 2 << 20).reshape(1, 3, 2).astype(np.int64)
        self._compareCpu(x, np.square, math_ops.square)
        self._compareBothSparse(x, np.square, math_ops.square)

    @test.disable_with_predicate(pred=test.is_built_with_rocm, skip_message='On ROCm this test fails')
    def testUInt64Basic(self):
        if False:
            return 10
        x = np.arange(6).reshape(1, 3, 2).astype(np.uint64)
        self._compareBoth(x, np.square, math_ops.square)

    @test_util.run_deprecated_v1
    def testComplex64Basic(self):
        if False:
            while True:
                i = 10
        x = (1 + 1j) * np.arange(-3, 3).reshape(1, 3, 2).astype(np.complex64)
        y = x + (0.5 + 0.5j)
        self._compareBoth(x, np.abs, math_ops.abs)
        self._compareBoth(x, np.abs, _ABS)
        self._compareBoth(x, np.negative, math_ops.negative)
        self._compareBoth(x, np.negative, _NEG)
        self._compareBoth(y, self._inv, math_ops.reciprocal)
        self._compareCpu(x, np.square, math_ops.square)
        self._compareCpu(y, np.sqrt, math_ops.sqrt)
        self._compareCpu(y, self._rsqrt, math_ops.rsqrt)
        self._compareBoth(x, np.exp, math_ops.exp)
        self._compareCpu(x, np.expm1, math_ops.expm1)
        self._compareCpu(y, np.log, math_ops.log)
        self._compareCpu(y, np.log1p, math_ops.log1p)
        self._compareCpu(x, np.sinh, math_ops.sinh)
        self._compareCpu(x, np.cosh, math_ops.cosh)
        self._compareCpu(x, np.tanh, math_ops.tanh)
        self._compareCpu(x, np.arcsin, math_ops.asin)
        self._compareCpu(x, np.arctan, math_ops.atan)
        self._compareCpu(y, np.arcsinh, math_ops.asinh, grad_rtol=0.01)
        self._compareCpu(y, np.arccosh, math_ops.acosh, grad_rtol=0.01)
        self._compareCpu(y, np.arctanh, math_ops.atanh)
        self._compareCpu(x, self._sigmoid, math_ops.sigmoid)
        self._compareCpu(x, np.sin, math_ops.sin)
        self._compareCpu(x, np.cos, math_ops.cos)
        self._compareBothSparse(x, np.abs, math_ops.abs)
        self._compareBothSparse(x, np.negative, math_ops.negative)
        self._compareBothSparse(x, np.square, math_ops.square)
        self._compareBothSparse(x, np.sqrt, math_ops.sqrt, 0.001)
        self._compareBothSparse(x, np.tanh, math_ops.tanh)

        def complex_sign(x):
            if False:
                for i in range(10):
                    print('nop')
            return x / np.abs(x)
        self._compareBoth(y, complex_sign, math_ops.sign)
        self._compareBothSparse(y, complex_sign, math_ops.sign)

    @test_util.run_deprecated_v1
    def testComplex128Basic(self):
        if False:
            i = 10
            return i + 15
        x = (1 + 1j) * np.arange(-3, 3).reshape(1, 3, 2).astype(np.complex128)
        y = x + (0.5 + 0.5j)
        self._compareBoth(x, np.abs, math_ops.abs)
        self._compareBoth(x, np.abs, _ABS)
        self._compareBoth(x, np.negative, math_ops.negative)
        self._compareBoth(x, np.negative, _NEG)
        self._compareBoth(y, self._inv, math_ops.reciprocal)
        self._compareCpu(x, np.square, math_ops.square)
        self._compareCpu(y, np.sqrt, math_ops.sqrt)
        self._compareCpu(y, self._rsqrt, math_ops.rsqrt)
        self._compareBoth(x, np.exp, math_ops.exp)
        self._compareCpu(x, np.expm1, math_ops.expm1)
        self._compareCpu(y, np.log, math_ops.log)
        self._compareCpu(y, np.log1p, math_ops.log1p)
        self._compareCpu(x, np.sinh, math_ops.sinh)
        self._compareCpu(x, np.cosh, math_ops.cosh)
        self._compareCpu(x, np.tanh, math_ops.tanh)
        self._compareCpu(y, np.arcsinh, math_ops.asinh)
        self._compareCpu(y, np.arccosh, math_ops.acosh)
        self._compareCpu(y, np.arctanh, math_ops.atanh)
        self._compareCpu(x, self._sigmoid, math_ops.sigmoid)
        self._compareCpu(x, np.sin, math_ops.sin)
        self._compareCpu(x, np.cos, math_ops.cos)
        self._compareCpu(x, np.arcsin, math_ops.asin)
        self._compareCpu(x, np.arctan, math_ops.atan)
        self._compareBothSparse(x, np.abs, math_ops.abs)
        self._compareBothSparse(x, np.negative, math_ops.negative)
        self._compareBothSparse(x, np.square, math_ops.square)
        self._compareBothSparse(x, np.sqrt, math_ops.sqrt, 0.001)
        self._compareBothSparse(x, np.tanh, math_ops.tanh)

        def complex_sign(x):
            if False:
                print('Hello World!')
            return x / np.abs(x)
        self._compareBoth(y, complex_sign, math_ops.sign)
        self._compareBothSparse(y, complex_sign, math_ops.sign)

    @test_util.run_deprecated_v1
    def testGradGrad(self):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(7)
        shape = (5,)
        dtype_tols = [(np.float32, 0.0005), (np.float64, 1e-06), (np.complex64, 0.0005), (np.complex128, 1e-06)]
        op_range = [(gen_math_ops.reciprocal_grad, [-2, 2]), (gen_math_ops.rsqrt_grad, [0.1, 3]), (gen_math_ops.sigmoid_grad, [-2, 2]), (gen_math_ops.sqrt_grad, [0.1, 3]), (gen_math_ops.tanh_grad, [-2, 2])]

        def rand(dtype, real_range):
            if False:
                print('Hello World!')
            x = np.random.uniform(real_range[0], real_range[1], size=shape[0]).astype(dtype)
            if dtype in (np.complex64, np.complex128):
                x += 1j * np.random.uniform(-2, 2, size=shape[0]).astype(dtype)
            return x
        for (op, real_range) in op_range:
            with self.cached_session():
                for (dtype, tol) in dtype_tols:
                    x = constant_op.constant(rand(dtype, real_range))
                    y = constant_op.constant(rand(dtype, real_range))
                    z = op(x, y)
                    grads = gradient_checker.compute_gradient([x, y], [shape, shape], z, shape, x_init_value=[rand(dtype, real_range), rand(dtype, real_range)])
                    if isinstance(grads, tuple):
                        grads = [grads]
                    for (analytical, numerical) in grads:
                        self.assertAllClose(analytical, numerical, rtol=tol, atol=tol)

    @test_util.run_in_graph_and_eager_modes
    def testComplexAbsGradGrad(self):
        if False:
            i = 10
            return i + 15

        def f(x):
            if False:
                i = 10
                return i + 15
            real = math_ops.cos(x)
            imag = ops.convert_to_tensor(1.0)
            return math_ops.abs(math_ops.complex(real, imag))

        def g(x):
            if False:
                return 10
            with backprop.GradientTape() as t:
                t.watch(x)
                y = f(x)
            return t.gradient(y, x)
        err = gradient_checker_v2.max_error(*gradient_checker_v2.compute_gradient(g, [ops.convert_to_tensor(2.0)]))
        self.assertLess(err, 0.001)
if __name__ == '__main__':
    test.main()