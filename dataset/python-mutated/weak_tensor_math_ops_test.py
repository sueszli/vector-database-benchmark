"""Tests for tensorflow.ops.math_ops on WeakTensor."""
import itertools
from absl.testing import parameterized
import numpy as np
from tensorflow.core.framework import full_type_pb2
from tensorflow.python import tf2
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import test_util
from tensorflow.python.framework.weak_tensor import WeakTensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import weak_tensor_ops
from tensorflow.python.ops import weak_tensor_test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import googletest
_convert_to_input_type = weak_tensor_test_util.convert_to_input_type
_get_weak_tensor = weak_tensor_test_util.get_weak_tensor

@test_util.run_all_in_graph_and_eager_modes
class ReduceTest(test_util.TensorFlowTestCase, parameterized.TestCase):

    @parameterized.parameters(('WeakTensor', WeakTensor), ('Python', WeakTensor), ('NumPy', tensor.Tensor), ('Tensor', tensor.Tensor))
    def testReduceAllDims(self, input_type, result_type):
        if False:
            return 10
        test_input = _convert_to_input_type([[1, 2, 3], [4, 5, 6]], input_type, dtypes.int32)
        with test_util.device(use_gpu=True):
            res = math_ops.reduce_sum(test_input)
            self.assertIsInstance(res, result_type)
            self.assertEqual(self.evaluate(res), 21)

    def testReduceExtendType(self):
        if False:
            i = 10
            return i + 15
        test_in = np.random.randn(1000, 1000).astype(np.float32)
        in_f32 = _get_weak_tensor(test_in, dtypes.float32)
        in_bfl6 = math_ops.cast(test_in, dtypes.bfloat16)
        out_f32 = self.evaluate(math_ops.reduce_sum(in_f32))
        out_bf16 = self.evaluate(math_ops.reduce_sum(in_bfl6))
        expected = math_ops.cast(out_f32, dtypes.bfloat16)
        self.assertAllClose(out_bf16, expected, 0.001)

    def testCountNonzero(self):
        if False:
            i = 10
            return i + 15
        x = _get_weak_tensor([[0, -2, 0], [4, 0, 0]], dtypes.int32)
        self.assertEqual(self.evaluate(math_ops.count_nonzero(x)), 2)
        x = math_ops.not_equal(x, 0)
        self.assertEqual(self.evaluate(math_ops.count_nonzero(x)), 2)
        x = 2 * np.ones(512, dtype=np.int8)
        self.assertEqual(self.evaluate(math_ops.count_nonzero(x)), 512)

    @parameterized.parameters(('WeakTensor', WeakTensor), ('Python', WeakTensor), ('NumPy', tensor.Tensor), ('Tensor', tensor.Tensor))
    def testReduceExplicitAxes(self, input_type, result_type):
        if False:
            i = 10
            return i + 15
        x = _convert_to_input_type([[1, 2, 3], [4, 5, 6]], input_type, dtypes.int32)
        with test_util.device(use_gpu=True):
            for axis in (0, -2):
                res = math_ops.reduce_sum(x, axis=axis)
                self.assertIsInstance(res, result_type)
                self.assertAllEqual(res, [5, 7, 9])
            for axis in (1, -1):
                res = math_ops.reduce_sum(x, axis=axis)
                self.assertIsInstance(res, result_type)
                self.assertAllEqual(res, [6, 15])
            for axis in (None, (0, 1), (1, 0), (-1, 0), (0, -1), (-2, 1), (1, -2), (-1, -2), (-2, -1)):
                res = math_ops.reduce_sum(x, axis=axis)
                self.assertIsInstance(res, result_type)
                self.assertEqual(self.evaluate(res), 21)

    def testReduceInvalidAxis(self):
        if False:
            while True:
                i = 10
        if context.executing_eagerly():
            return
        x = _get_weak_tensor([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        axis = np.array([[0], [1]])
        with self.assertRaisesRegex(ValueError, 'must be at most rank 1'):
            math_ops.reduce_sum(x, axis)

    def testReduceVar(self):
        if False:
            return 10
        x = _get_weak_tensor([[0, 0, 0], [0, 0, 0]], dtype=dtypes.float32)
        self.assertAllClose(self.evaluate(math_ops.reduce_variance(x)), 0)
        self.assertAllClose(self.evaluate(math_ops.reduce_variance(x, axis=0)), [0, 0, 0])
        x = _get_weak_tensor([[1, 2, 1, 1], [1, 1, 0, 1]])
        with self.assertRaisesRegex(TypeError, 'must be either real or complex'):
            math_ops.reduce_variance(x)
        x = _get_weak_tensor([[1.0, 2.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0]])
        self.assertEqual(self.evaluate(math_ops.reduce_variance(x)), 0.25)
        x_np = np.array([[1, 2, 1, 1], [1, 1, 0, 1]], 'float32')
        self.assertEqual(np.var(x_np), 0.25)
        self.assertEqual(self.evaluate(math_ops.reduce_variance(x_np)), 0.25)
        x = ragged_factory_ops.constant([[5.0, 1.0, 4.0, 1.0], [], [5.0, 9.0, 2.0], [5.0], []])
        self.assertAllClose(math_ops.reduce_variance(x, axis=0), [0.0, 16.0, 1.0, 0.0])

    def testReduceVarComplex(self):
        if False:
            i = 10
            return i + 15
        complex_ys = [([0 - 1j, 0 + 1j], dtypes.float64), (np.array([0 - 1j, 0 + 1j], 'complex64'), dtypes.float32), (np.array([0 - 1j, 0 + 1j], 'complex128'), dtypes.float64)]
        for (y, dtype) in complex_ys:
            y_result = math_ops.reduce_variance(y)
            self.assertEqual(np.var(y), 1.0)
            self.assertEqual(self.evaluate(y_result), 1.0)
            self.assertEqual(y_result.dtype, dtype)

    def testReduceStd(self):
        if False:
            for i in range(10):
                print('nop')
        x = _get_weak_tensor([[0, 0, 0], [0, 0, 0]], dtypes.float32)
        self.assertAllClose(self.evaluate(math_ops.reduce_std(x)), 0)
        self.assertAllClose(self.evaluate(math_ops.reduce_std(x, axis=0)), [0, 0, 0])
        x = _get_weak_tensor([[1, 2, 1, 1], [1, 1, 0, 1]])
        with self.assertRaisesRegex(TypeError, 'must be either real or complex'):
            math_ops.reduce_std(x)
        x = [[1.0, 2.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0]]
        res = math_ops.reduce_std(x)
        self.assertEqual(self.evaluate(res), 0.5)
        self.assertIsInstance(res, WeakTensor)
        x_np = np.array(x)
        self.assertEqual(np.std(x_np), 0.5)
        self.assertEqual(self.evaluate(math_ops.reduce_std(x_np)), 0.5)
        self.assertIsInstance(math_ops.reduce_std(x_np), tensor.Tensor)
        x = ragged_factory_ops.constant([[5.0, 1.0, 4.0, 1.0], [], [5.0, 9.0, 2.0], [5.0], []])
        self.assertAllClose(math_ops.reduce_std(x, axis=0), [0.0, 4.0, 1.0, 0.0])

    def testReduceStdComplex(self):
        if False:
            i = 10
            return i + 15
        complex_ys = [([0 - 1j, 0 + 1j], dtypes.float64), (np.array([0 - 1j, 0 + 1j], 'complex64'), dtypes.float32), (np.array([0 - 1j, 0 + 1j], 'complex128'), dtypes.float64)]
        for (y, dtype) in complex_ys:
            y_result = math_ops.reduce_std(y)
            self.assertEqual(np.std(y), 1.0)
            self.assertEqual(self.evaluate(y_result), 1.0)
            self.assertEqual(y_result.dtype, dtype)

@test_util.run_all_in_graph_and_eager_modes
class LogSumExpTest(test_util.TensorFlowTestCase):

    def testReduceLogSumExp(self):
        if False:
            print('Hello World!')
        for dtype in [np.float16, np.float32, np.double]:
            x_np = np.random.rand(5, 5).astype(dtype)
            with test_util.use_gpu():
                y_tf_np = math_ops.reduce_logsumexp(x_np)
                y_np = np.log(np.sum(np.exp(x_np)))
                self.assertAllClose(y_tf_np, y_np)

    def testReductionIndices(self):
        if False:
            print('Hello World!')
        for dtype in [np.float16, np.float32, np.double]:
            x_np = np.random.rand(5, 5).astype(dtype)
            with test_util.use_gpu():
                y_tf = math_ops.reduce_logsumexp(x_np, axis=[0])
                y_np = np.log(np.sum(np.exp(x_np), axis=0))
                self.assertShapeEqual(y_np, y_tf)
                y_tf_np = self.evaluate(y_tf)
                self.assertAllClose(y_tf_np, y_np)

    def testReductionIndices2(self):
        if False:
            print('Hello World!')
        for dtype in [np.float16, np.float32, np.double]:
            x_np = np.random.rand(5, 5).astype(dtype)
            with test_util.use_gpu():
                y_tf = math_ops.reduce_logsumexp(x_np, axis=0)
                y_np = np.log(np.sum(np.exp(x_np), axis=0))
                self.assertShapeEqual(y_np, y_tf)
                y_tf_np = self.evaluate(y_tf)
                self.assertAllClose(y_tf_np, y_np)

    def testKeepDims(self):
        if False:
            i = 10
            return i + 15
        for dtype in [np.float16, np.float32, np.double]:
            x_np = np.random.rand(5, 5).astype(dtype)
            with test_util.use_gpu():
                y_tf_np = math_ops.reduce_logsumexp(x_np, keepdims=True)
                self.assertEqual(y_tf_np.shape.rank, x_np.ndim)
                y_np = np.log(np.sum(np.exp(x_np), keepdims=True))
                self.assertAllClose(y_tf_np, y_np)

    def testOverflow(self):
        if False:
            return 10
        x = [1000, 1001, 1002, 1003]
        for dtype in [np.float32, np.double]:
            x_np = np.array(x, dtype=dtype)
            max_np = np.max(x_np)
            with self.assertRaisesRegex(RuntimeWarning, 'overflow encountered in exp'):
                out = np.log(np.sum(np.exp(x_np)))
                if out == np.inf:
                    raise RuntimeWarning('overflow encountered in exp')
            with test_util.use_gpu():
                x_tf = _get_weak_tensor(x_np, shape=x_np.shape)
                y_tf_np = math_ops.reduce_logsumexp(x_tf)
                y_np = np.log(np.sum(np.exp(x_np - max_np))) + max_np
                self.assertAllClose(y_tf_np, y_np)

    def testUnderflow(self):
        if False:
            while True:
                i = 10
        x = [-1000, -1001, -1002, -1003]
        for dtype in [np.float32, np.double]:
            x_np = np.array(x, dtype=dtype)
            max_np = np.max(x_np)
            with self.assertRaisesRegex(RuntimeWarning, 'divide by zero encountered in log'):
                out = np.log(np.sum(np.exp(x_np)))
                if out == -np.inf:
                    raise RuntimeWarning('divide by zero encountered in log')
            with test_util.use_gpu():
                x_tf = _get_weak_tensor(x_np, shape=x_np.shape)
                y_tf_np = math_ops.reduce_logsumexp(x_tf)
                y_np = np.log(np.sum(np.exp(x_np - max_np))) + max_np
                self.assertAllClose(y_tf_np, y_np)

    def testInfinity(self):
        if False:
            print('Hello World!')
        with test_util.use_gpu():
            res = math_ops.reduce_logsumexp(-np.inf)
            self.assertEqual(-np.inf, self.evaluate(res))

@test_util.run_all_in_graph_and_eager_modes
class RoundTest(test_util.TensorFlowTestCase):

    def testRounding(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.arange(-5.0, 5.0, 0.25)
        for dtype in [np.float32, np.double, np.int32]:
            x_np = np.array(x, dtype=dtype)
            with test_util.device(use_gpu=True):
                x_tf = _get_weak_tensor(x_np, shape=x_np.shape)
                y_tf = math_ops.round(x_tf)
                y_tf_np = self.evaluate(y_tf)
                y_np = np.round(x_np)
                self.assertAllClose(y_tf_np, y_np, atol=0.01)

class SignTest(test_util.TensorFlowTestCase):

    def test_complex_sign_gradient(self):
        if False:
            return 10
        with context.eager_mode():
            x = math_ops.complex(1.0, 1.0)
            with backprop.GradientTape() as t:
                t.watch(x)
                y = math_ops.sign(x)
            self.assertAllClose(t.gradient(y, x), math_ops.complex(0.353553, -0.353553))

@test_util.run_all_in_graph_and_eager_modes
class ReciprocalNoNanTest(test_util.TensorFlowTestCase):
    allowed_dtypes = [dtypes.float32, dtypes.float64, dtypes.complex128]

    def testBasic(self):
        if False:
            while True:
                i = 10
        for dtype in self.allowed_dtypes:
            x = _get_weak_tensor([1.0, 2.0, 0.0, 4.0], dtype=dtype)
            y = math_ops.reciprocal_no_nan(x)
            target = _get_weak_tensor([1.0, 0.5, 0.0, 0.25], dtype=dtype)
            self.assertAllEqual(y, target)
            self.assertEqual(y.dtype.base_dtype, target.dtype.base_dtype)

    def testInverse(self):
        if False:
            return 10
        for dtype in self.allowed_dtypes:
            x = np.random.choice([0, 1, 2, 4, 5], size=(5, 5, 5))
            x = _get_weak_tensor(x, dtype=dtype)
            y = math_ops.reciprocal_no_nan(math_ops.reciprocal_no_nan(x))
            self.assertAllClose(y, x)
            self.assertEqual(y.dtype.base_dtype, x.dtype.base_dtype)

class EqualityTest(test_util.TensorFlowTestCase, parameterized.TestCase):

    @test_util.run_all_in_graph_and_eager_modes
    def testEqualityNone(self):
        if False:
            for i in range(10):
                print('nop')
        x = _get_weak_tensor([1.0, 2.0, 0.0, 4.0], dtype=dtypes.float32)
        self.assertNotEqual(x, None)
        self.assertNotEqual(None, x)
        self.assertFalse(math_ops.tensor_equals(x, None))
        self.assertTrue(math_ops.tensor_not_equals(x, None))

    @parameterized.named_parameters(((f'-is_equals={is_equals}-float_literal_type={type(float_literal)}-float_literal={float_literal}', is_equals, float_literal) for float_literal in [4.6, np.float32(4.6), 4.4, np.float32(4.4)] for is_equals in [True, False]))
    def testEqualityNoDowncast(self, is_equals, float_literal):
        if False:
            print('Hello World!')
        if tf2.enabled() and isinstance(float_literal, np.float32) or (not tf2.enabled() and isinstance(float_literal, float)):
            self.skipTest('There is a bug in type promotion.')
        if is_equals:
            op = math_ops.tensor_equals
        else:
            op = math_ops.tensor_not_equals
        x = _get_weak_tensor(4)
        try:
            result = op(x, float_literal)
            if isinstance(result, tensor.Tensor):
                result = self.evaluate(result)
        except TypeError:
            return
        self.assertEqual(result, not is_equals)

@test_util.run_all_in_graph_and_eager_modes
class ErfcinvTest(test_util.TensorFlowTestCase):

    def testErfcinv(self):
        if False:
            while True:
                i = 10
        values = _get_weak_tensor(np.random.uniform(0.1, 1.9, size=int(10000.0)).astype(np.float32))
        approx_id = math_ops.erfc(math_ops.erfcinv(values))
        self.assertAllClose(values, self.evaluate(approx_id))

@test_util.run_all_in_graph_and_eager_modes
class ArgMaxMinTest(test_util.TensorFlowTestCase):

    def _generateRandomWeakTensor(self, dtype, shape):
        if False:
            print('Hello World!')
        if dtype.is_integer:
            array = np.random.default_rng().integers(low=dtype.min, high=dtype.max, size=shape, endpoint=True)
            return _get_weak_tensor(array, dtype=dtype)
        else:
            array = np.random.default_rng().uniform(low=-1.0, high=1.0, size=shape)
            return _get_weak_tensor(array, dtype=dtype)

    def _getValidDtypes(self):
        if False:
            return 10
        return (dtypes.float32, dtypes.float64, dtypes.int32, dtypes.int64)

    def testArgMax(self):
        if False:
            i = 10
            return i + 15
        shape = (24, 8)
        for dtype in self._getValidDtypes():
            tf_values = self._generateRandomWeakTensor(dtype, shape)
            np_values = self.evaluate(tf_values)
            for axis in range(0, len(shape)):
                np_max = np.argmax(np_values, axis=axis)
                tf_max = math_ops.argmax(tf_values, axis=axis)
                self.assertAllEqual(tf_max, np_max)

    def testArgMaxReturnsFirstOccurence(self):
        if False:
            return 10
        for dtype in self._getValidDtypes():
            values = _get_weak_tensor([[10, 11, 15, 15, 10], [12, 12, 10, 10, 12]], dtype=dtype)
            self.assertAllEqual(math_ops.argmax(values, axis=1), np.argmax(self.evaluate(values), axis=1))
            values = array_ops.zeros(shape=(193681,), dtype=dtype)
            self.assertAllEqual(math_ops.argmax(values), 0)

    def testArgMaxUint16(self):
        if False:
            print('Hello World!')
        shape = (24, 8)
        for dtype in self._getValidDtypes():
            tf_values = self._generateRandomWeakTensor(dtype, shape)
            np_values = self.evaluate(tf_values)
            for axis in range(0, len(shape)):
                np_max = np.argmax(np_values, axis=axis)
                tf_max = math_ops.argmax(tf_values, axis=axis, output_type=dtypes.uint16)
                self.assertAllEqual(tf_max, np_max)

    def testArgMin(self):
        if False:
            i = 10
            return i + 15
        shape = (24, 8)
        for dtype in self._getValidDtypes():
            tf_values = self._generateRandomWeakTensor(dtype, shape)
            np_values = self.evaluate(tf_values)
            for axis in range(0, len(shape)):
                np_min = np.argmin(np_values, axis=axis)
                tf_min = math_ops.argmin(tf_values, axis=axis)
                self.assertAllEqual(tf_min, np_min)

    def testArgMinReturnsFirstOccurence(self):
        if False:
            while True:
                i = 10
        for dtype in self._getValidDtypes():
            values = _get_weak_tensor([[10, 11, 15, 15, 10], [12, 12, 10, 10, 12]], dtype=dtype)
            self.assertAllEqual(math_ops.argmin(values, axis=1), np.argmin(self.evaluate(values), axis=1))
            values = array_ops.zeros(shape=(193681,), dtype=dtype)
            self.assertAllEqual(math_ops.argmin(values), 0)

class CastTest(test_util.TensorFlowTestCase):

    def testCastWithFullType(self):
        if False:
            for i in range(10):
                print('nop')

        @def_function.function
        def test_fn():
            if False:
                for i in range(10):
                    print('nop')
            ta = tensor_array_ops.TensorArray(dtypes.int32, size=1)
            h = math_ops.cast(ta.flow, dtypes.variant)
            t = full_type_pb2.FullTypeDef(type_id=full_type_pb2.TFT_PRODUCT, args=[full_type_pb2.FullTypeDef(type_id=full_type_pb2.TFT_ARRAY)])
            h.op.experimental_set_type(t)
            ta = tensor_array_ops.TensorArray(dtypes.int32, flow=h)
            ta = ta.write(0, _get_weak_tensor(1))
            return ta.stack()
        self.assertAllEqual(self.evaluate(test_fn()), [1])

@test_util.run_all_in_graph_and_eager_modes
class BinaryOpsTest(test_util.TensorFlowTestCase):

    def testRHSDispatchingAndErrorRaising(self):
        if False:
            return 10
        if context.executing_eagerly():
            error = ValueError
            error_message = 'Attempt to convert a value .* with an unsupported type'
        else:
            error = TypeError
            error_message = 'Failed to convert elements of .* to Tensor'

        class RHSReturnsTrue:

            def __radd__(self, other):
                if False:
                    print('Hello World!')
                return True

            def __rmod__(self, other):
                if False:
                    i = 10
                    return i + 15
                return False
        a = array_ops.ones([1], dtype=dtypes.int32) + RHSReturnsTrue()
        self.assertEqual(a, True)
        a = _get_weak_tensor(5, dtype=dtypes.int32) + RHSReturnsTrue()
        self.assertEqual(a, True)
        a = array_ops.ones([1], dtype=dtypes.float32) % RHSReturnsTrue()
        self.assertEqual(a, False)
        a = _get_weak_tensor(5, dtype=dtypes.float32) % RHSReturnsTrue()
        self.assertEqual(a, False)

        class RHSRaisesError:

            def __radd__(self, other):
                if False:
                    for i in range(10):
                        print('nop')
                raise TypeError('RHS not implemented')
        with self.assertRaisesRegex(error, error_message):
            a = array_ops.ones([1], dtype=dtypes.int32) + RHSRaisesError()
            self.evaluate(a)
        with self.assertRaisesRegex(error, error_message):
            a = _get_weak_tensor([1], dtype=dtypes.int32) + RHSRaisesError()
            self.evaluate(a)

        class RHSReturnsNotImplemented:

            def __radd__(self, other):
                if False:
                    i = 10
                    return i + 15
                return NotImplemented
        with self.assertRaisesRegex(error, error_message):
            a = array_ops.ones([1], dtype=dtypes.int32) + RHSReturnsNotImplemented()
            self.evaluate(a)
            a = _get_weak_tensor([1], dtype=dtypes.int32) + RHSReturnsNotImplemented()
            self.evaluate(a)

        class RHSNotImplemented:
            pass
        with self.assertRaisesRegex(error, error_message):
            a = array_ops.ones([1], dtype=dtypes.int32) + RHSNotImplemented()
            self.evaluate(a)
            a = _get_weak_tensor([1], dtype=dtypes.int32) + RHSNotImplemented()
            self.evaluate(a)

@test_util.run_all_in_graph_and_eager_modes
class ScalarMulTest(test_util.TensorFlowTestCase):

    def testAcceptsRefs(self):
        if False:
            for i in range(10):
                print('nop')
        if context.executing_eagerly():
            var = resource_variable_ops.ResourceVariable(10, name='var')
        else:
            var = variables.Variable(10)
        result = math_ops.scalar_mul(3, var)
        init = variables.global_variables_initializer()
        with test_util.device(use_gpu=True):
            self.evaluate(init)
            self.assertEqual(30, self.evaluate(result))

    def testAcceptsIndexedSlices(self):
        if False:
            for i in range(10):
                print('nop')
        values = constant_op.constant([2, 3, 5, 7, 0, -1], shape=[3, 2])
        indices = constant_op.constant([0, 2, 5])
        x = math_ops.scalar_mul(-3, indexed_slices.IndexedSlices(values, indices))
        with test_util.device(use_gpu=True):
            self.assertAllEqual(self.evaluate(x.values), [[-6, -9], [-15, -21], [0, 3]])
            self.assertAllEqual(self.evaluate(x.indices), [0, 2, 5])

class ComparisonOps(parameterized.TestCase, test_util.TensorFlowTestCase):

    def test_math_equal(self):
        if False:
            i = 10
            return i + 15
        self.assertAllEqual(math_ops.equal(1, constant_op.constant(1)), True)
        self.assertAllEqual(math_ops.equal(np.int_(1), constant_op.constant(1)), True)
        self.assertAllEqual(math_ops.equal(constant_op.constant(1, dtypes.float32), constant_op.constant(1, dtypes.int32)), True)

    def test_math_maximum(self):
        if False:
            while True:
                i = 10
        self.assertAllEqual(math_ops.maximum(1, constant_op.constant(2)), 2)
        self.assertAllEqual(math_ops.maximum(np.int_(1), constant_op.constant(1.5, dtypes.float32)), np.array(1.5, np.float32))
        self.assertAllEqual(math_ops.maximum(constant_op.constant(5, dtypes.float32), constant_op.constant(1, dtypes.int32)), 5)
        self.assertAllEqual(gen_math_ops.maximum(1, constant_op.constant(2)), 2)
        self.assertAllEqual(gen_math_ops.maximum(np.int_(1), constant_op.constant(1.5, dtypes.float32)), np.array(1.5, np.float32))
        self.assertAllEqual(gen_math_ops.maximum(constant_op.constant(5, dtypes.float32), constant_op.constant(1, dtypes.int32)), 5)

    def test_math_minimum(self):
        if False:
            print('Hello World!')
        self.assertAllEqual(math_ops.minimum(1, constant_op.constant(2)), 1)
        self.assertAllEqual(math_ops.minimum(np.int_(1), constant_op.constant(1.1, dtypes.float32)), 1)
        self.assertAllEqual(math_ops.minimum(constant_op.constant(5, dtypes.float32), constant_op.constant(-1, dtypes.int32)), -1)
        self.assertAllEqual(gen_math_ops.minimum(1, constant_op.constant(2)), 1)
        self.assertAllEqual(gen_math_ops.minimum(np.int_(1), constant_op.constant(1.1, dtypes.float32)), 1)
        self.assertAllEqual(gen_math_ops.minimum(constant_op.constant(5, dtypes.float32), constant_op.constant(-1, dtypes.int32)), -1)
allowed_var_op_input_combinations = [(dtypes.uint8, 10), (dtypes.uint8, 'weak_i64'), (dtypes.uint8, dtypes.uint8), (dtypes.uint16, 10), (dtypes.uint16, 'weak_i64'), (dtypes.uint16, dtypes.uint8), (dtypes.uint16, dtypes.uint16), (dtypes.uint32, 10), (dtypes.uint32, 'weak_i64'), (dtypes.uint32, dtypes.uint8), (dtypes.uint32, dtypes.uint32), (dtypes.uint64, 10), (dtypes.uint64, 'weak_i64'), (dtypes.uint64, dtypes.uint32), (dtypes.uint64, dtypes.uint64), (dtypes.int8, 10), (dtypes.int8, 'weak_i64'), (dtypes.int8, dtypes.int8), (dtypes.int16, 10), (dtypes.int16, 'weak_i64'), (dtypes.int16, dtypes.uint8), (dtypes.int16, dtypes.int8), (dtypes.int16, dtypes.int16), (dtypes.int32, 10), (dtypes.int32, 'weak_i64'), (dtypes.int32, dtypes.uint16), (dtypes.int32, dtypes.int16), (dtypes.int32, dtypes.int32), (dtypes.int64, 10), (dtypes.int64, 'weak_i64'), (dtypes.int64, dtypes.uint32), (dtypes.int64, dtypes.int32), (dtypes.int64, dtypes.int64), (dtypes.bfloat16, 10), (dtypes.bfloat16, 'weak_i64'), (dtypes.bfloat16, 1.0), (dtypes.bfloat16, 'weak_f64'), (dtypes.bfloat16, dtypes.int32), (dtypes.bfloat16, dtypes.bfloat16), (dtypes.float16, 10), (dtypes.float16, 'weak_i64'), (dtypes.float16, 1.0), (dtypes.float16, 'weak_f64'), (dtypes.float16, dtypes.int32), (dtypes.float16, dtypes.float16), (dtypes.float32, 10), (dtypes.float32, 'weak_i64'), (dtypes.float32, 1.0), (dtypes.float32, 'weak_f64'), (dtypes.float32, dtypes.int32), (dtypes.float32, dtypes.float32), (dtypes.float64, 10), (dtypes.float64, 'weak_i64'), (dtypes.float64, 1.0), (dtypes.float64, 'weak_f64'), (dtypes.float64, dtypes.int32), (dtypes.float64, dtypes.float64), (dtypes.complex64, 10), (dtypes.complex64, 'weak_i64'), (dtypes.complex64, 1.0), (dtypes.complex64, 'weak_f64'), (dtypes.complex64, 1.0 + 2j), (dtypes.complex64, 'weak_c128'), (dtypes.complex64, dtypes.int32), (dtypes.complex64, dtypes.complex64), (dtypes.complex128, 10), (dtypes.complex128, 'weak_i64'), (dtypes.complex128, 1.0), (dtypes.complex128, 'weak_f64'), (dtypes.complex128, 1.0 + 2j), (dtypes.complex128, 'weak_c128'), (dtypes.complex128, dtypes.int32), (dtypes.complex128, dtypes.float64), (dtypes.complex128, dtypes.complex128)]
disallowed_var_op_input_combinations = [(dtypes.uint8, 1.0), (dtypes.uint8, 'weak_f64'), (dtypes.uint8, dtypes.int8), (dtypes.uint8, dtypes.uint16), (dtypes.uint16, 1.0), (dtypes.uint16, 'weak_f64'), (dtypes.uint16, dtypes.int8), (dtypes.uint16, dtypes.uint32), (dtypes.uint32, 1.0), (dtypes.uint32, 'weak_f64'), (dtypes.uint32, dtypes.int32), (dtypes.uint32, dtypes.uint64), (dtypes.uint64, 1.0), (dtypes.uint64, 'weak_f64'), (dtypes.uint64, dtypes.int8), (dtypes.uint64, dtypes.float16), (dtypes.int8, 1.0), (dtypes.int8, 'weak_f64'), (dtypes.int8, dtypes.int16), (dtypes.int8, dtypes.float32), (dtypes.int8, dtypes.complex64), (dtypes.int16, 1.0), (dtypes.int16, 'weak_f64'), (dtypes.int16, dtypes.int32), (dtypes.int16, dtypes.float32), (dtypes.int16, dtypes.complex64), (dtypes.int32, 1.0), (dtypes.int32, 'weak_f64'), (dtypes.int32, dtypes.int64), (dtypes.int32, dtypes.float16), (dtypes.int32, dtypes.complex64), (dtypes.int64, 1.0), (dtypes.int64, 'weak_f64'), (dtypes.int64, dtypes.uint64), (dtypes.int64, dtypes.float16), (dtypes.int64, dtypes.complex64), (dtypes.bfloat16, 1.0 + 2j), (dtypes.bfloat16, 'weak_c128'), (dtypes.bfloat16, dtypes.float16), (dtypes.bfloat16, dtypes.float32), (dtypes.float16, 1.0 + 2j), (dtypes.float16, 'weak_c128'), (dtypes.float16, dtypes.bfloat16), (dtypes.float16, dtypes.float32), (dtypes.float64, 1.0 + 2j), (dtypes.float64, 'weak_c128'), (dtypes.float64, dtypes.complex64), (dtypes.complex64, dtypes.float64), (dtypes.complex64, dtypes.complex128)]

def _weak_tensor_from_str(s):
    if False:
        for i in range(10):
            print('nop')
    if s == 'weak_i64':
        return _get_weak_tensor(1, dtype=dtypes.int64)
    elif s == 'weak_f64':
        return _get_weak_tensor(1.0, dtype=dtypes.float64)
    elif s == 'weak_c128':
        return _get_weak_tensor(1.0 + 2j, dtype=dtypes.complex128)
    else:
        raise ValueError(f'Unsupported str: {s}')

@test_util.run_all_in_graph_and_eager_modes
class VariableInplaceOpsTest(parameterized.TestCase, test_util.TensorFlowTestCase):

    @parameterized.parameters(itertools.product(allowed_var_op_input_combinations, ('assign', 'assign_add', 'assign_sub')))
    def testAllowedDtypes(self, v_dtype_and_delta, op):
        if False:
            i = 10
            return i + 15
        (v_dtype, delta) = v_dtype_and_delta
        if isinstance(delta, dtypes.DType):
            delta = constant_op.constant(1, delta)
        elif isinstance(delta, str):
            delta = _weak_tensor_from_str(delta)
        var = resource_variable_ops.ResourceVariable(10, dtype=v_dtype)
        result = getattr(var, op)(delta)
        with test_util.device(use_gpu=True):
            self.assertEqual(result.dtype, v_dtype)

    @parameterized.parameters(itertools.product(disallowed_var_op_input_combinations, ('assign', 'assign_add', 'assign_sub')))
    def testDisallowedDtypes(self, v_dtype_and_delta, op):
        if False:
            i = 10
            return i + 15
        (v_dtype, delta) = v_dtype_and_delta
        if isinstance(delta, dtypes.DType):
            delta = constant_op.constant(1, delta)
        elif isinstance(delta, str):
            delta = _weak_tensor_from_str(delta)
        var = resource_variable_ops.ResourceVariable(10, dtype=v_dtype)
        with self.assertRaises(TypeError):
            _ = getattr(var, op)(delta)

@test_util.run_all_in_graph_and_eager_modes
class DivAndModTest(test_util.TensorFlowTestCase):

    def numpySafeFloorDivInt(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        z = x // y
        z[(x == np.iinfo(x.dtype).min) & (y == -1)] = np.iinfo(x.dtype).min
        return z

    def numpySafeFloorModInt(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        z = self.numpySafeFloorDivInt(x, y)
        return x - z * y

    def numpySafeTruncateDivInt(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        z = self.numpySafeFloorDivInt(x, y)
        z[(x != z * y) & ((x < 0) != (y < 0))] += 1
        return z

    def numpySafeTruncateModInt(self, x, y):
        if False:
            print('Hello World!')
        z = self.numpySafeTruncateDivInt(x, y)
        return x - z * y

    def intEdgeTestData(self, dtype):
        if False:
            i = 10
            return i + 15
        'Edge-case test data for integer types.'
        nums = np.array([np.iinfo(dtype).min, -1, 1, np.iinfo(dtype).max], dtype=dtype).reshape([4, 1])
        divs = nums.reshape([1, 4])
        return (nums, divs)

    @test_util.disable_asan('Expected signed integer overflow.')
    @test_util.disable_ubsan('Expected signed integer overflow.')
    def testFloorDivModIntEdges(self):
        if False:
            for i in range(10):
                print('nop')
        for dtype in [np.int32, np.int64]:
            (x, y) = self.intEdgeTestData(dtype)
            (x_weak, y_weak) = (_get_weak_tensor(x), _get_weak_tensor(y))
            tf_floor_div = math_ops.floor_div(x_weak, y_weak)
            np_floor_div = self.numpySafeFloorDivInt(x, y)
            self.assertIsInstance(tf_floor_div, WeakTensor)
            self.assertAllEqual(tf_floor_div, np_floor_div)
            tf_floor_mod = math_ops.floormod(x_weak, y_weak)
            np_floor_mod = self.numpySafeFloorModInt(x, y)
            self.assertIsInstance(tf_floor_div, WeakTensor)
            self.assertAllEqual(tf_floor_mod, np_floor_mod)
            z = math_ops.add(math_ops.multiply(tf_floor_div, y_weak), tf_floor_mod)
            self.assertIsInstance(z, WeakTensor)
            self.assertAllEqual(z, np.broadcast_to(x, z.shape))

    @test_util.disable_asan('Expected signed integer overflow.')
    @test_util.disable_ubsan('Expected signed integer overflow.')
    def testTruncateDivModIntEdges(self):
        if False:
            i = 10
            return i + 15
        for dtype in [np.int32, np.int64]:
            (x, y) = self.intEdgeTestData(dtype)
            (x_weak, y_weak) = (_get_weak_tensor(x), _get_weak_tensor(y))
            tf_truncate_div = math_ops.truncatediv(x_weak, y_weak)
            np_truncate_div = self.numpySafeTruncateDivInt(x, y)
            self.assertIsInstance(tf_truncate_div, WeakTensor)
            self.assertAllEqual(tf_truncate_div, np_truncate_div)
            tf_truncate_mod = math_ops.truncatemod(x_weak, y_weak)
            np_truncate_mod = self.numpySafeTruncateModInt(x, y)
            self.assertIsInstance(tf_truncate_mod, WeakTensor)
            self.assertAllEqual(tf_truncate_mod, np_truncate_mod)
            z = math_ops.add(math_ops.multiply(tf_truncate_div, y_weak), tf_truncate_mod)
            self.assertIsInstance(z, WeakTensor)
            self.assertAllEqual(z, np.broadcast_to(x, z.shape))

@test_util.run_all_in_graph_and_eager_modes
class DivNoNanTest(test_util.TensorFlowTestCase, parameterized.TestCase):
    _SUPPORTED_DTYPES = [dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64, dtypes.complex128]

    @parameterized.parameters(*_SUPPORTED_DTYPES)
    def testBasic(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        if dtype.is_unsigned:
            nums = np.arange(0, 120, 3).reshape(40, 1)
            divs = np.arange(0, 48, 4).reshape(1, 12)
        elif dtype.is_integer:
            nums = np.arange(-120, 120, 3).reshape(80, 1)
            divs = np.arange(-48, 48, 4).reshape(1, 24)
        else:
            nums = np.arange(-10, 10, 0.25).reshape(80, 1)
            divs = np.arange(-3, 3, 0.25).reshape(1, 24)
        assert 0 in divs, 'Bad test set-up'
        tf_nums = _get_weak_tensor(nums, dtype=dtype)
        tf_divs = _get_weak_tensor(divs, dtype=dtype)
        np_nums = self.evaluate(tf_nums)
        np_divs = self.evaluate(tf_divs)
        np_result = np.true_divide(np_nums, np_divs)
        np_result[:, np_divs[0] == 0] = 0
        with test_util.use_gpu():
            tf_result = math_ops.div_no_nan(tf_nums, tf_divs)
            self.assertIsInstance(tf_result, WeakTensor)
            self.assertAllCloseAccordingToType(tf_result, np_result)

    @parameterized.product(type_x=_SUPPORTED_DTYPES + [float, int], type_y=_SUPPORTED_DTYPES + [float, int])
    def testSameSupportedTypesAsDivide(self, type_x, type_y):
        if False:
            for i in range(10):
                print('nop')

        def one(type_):
            if False:
                print('Hello World!')
            if type_ is int:
                return 1
            elif type_ is float:
                return 1.0
            else:
                return _get_weak_tensor(1, dtype=type_)
        x = one(type_x)
        y = one(type_y)
        divide_raises = False
        try:
            divide_result = math_ops.divide(x, y)
        except TypeError:
            divide_raises = True
        if divide_raises:
            with self.assertRaises(TypeError):
                _ = math_ops.div_no_nan(x, y)
        else:
            divide_no_nan_result = math_ops.div_no_nan(x, y)
            self.assertIsInstance(divide_no_nan_result, WeakTensor)
            self.assertIsInstance(divide_result, WeakTensor)
            self.assertEqual(divide_no_nan_result.dtype, divide_result.dtype)
            self.assertAllEqual(divide_no_nan_result, divide_result)

    @parameterized.parameters(dtypes.float32, dtypes.float64, dtypes.complex128)
    def testSmall(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        zero = _get_weak_tensor([0, 0, 0, 0], dtype=dtype)
        divs = _get_weak_tensor([1e-25, -1e-20, 1e-165, -1e-160], dtype=dtype)
        tf_result = math_ops.div_no_nan(zero, divs)
        self.assertAllEqual(tf_result, zero)
        self.assertIsInstance(tf_result, WeakTensor)

    @parameterized.parameters(dtypes.float32, dtypes.float64, dtypes.complex128)
    def testNonFiniteInNumerator(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        nums = _get_weak_tensor([np.nan, np.inf, np.NINF], dtype=dtype)
        zeros = _get_weak_tensor([0, 0, 0], dtype=dtype)
        ones = _get_weak_tensor([1, 1, 1], dtype=dtype)
        with test_util.use_gpu():
            tf_result_zeros = math_ops.div_no_nan(nums, zeros)
            self.assertAllEqual([0, 0, 0], tf_result_zeros)
            self.assertIsInstance(tf_result_zeros, WeakTensor)
            tf_result_ones = math_ops.div_no_nan(nums, ones)
            self.assertAllEqual(nums / ones, tf_result_ones)
            self.assertIsInstance(tf_result_ones, WeakTensor)
if __name__ == '__main__':
    ops.set_dtype_conversion_mode('all')
    googletest.main()