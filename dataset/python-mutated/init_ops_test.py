"""Tests for tensorflow.ops.ops."""
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.layers import convolutional
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

def identicaltest(tc, init1, init2, shape=None):
    if False:
        print('Hello World!')
    'Tests if two initializations are identical to within tiny tolerances.\n\n  Args:\n    tc: An instance of TensorFlowTestCase.\n    init1: An Initializer that generates a tensor of a given shape\n    init2: An Initializer that generates a tensor of a given shape\n    shape: Shape of the tensor to initialize or `None` to use a vector of length\n      100.\n\n  Returns:\n    True or False as determined by test.\n  '
    if shape is None:
        shape = [100]
    with tc.test_session(graph=ops.Graph()):
        t1 = init1(shape).eval()
    with tc.test_session(graph=ops.Graph()):
        t2 = init2(shape).eval()
    return np.allclose(t1, t2, rtol=1e-15, atol=1e-15)

def duplicated_initializer(tc, init, graph_seed, shape=None):
    if False:
        i = 10
        return i + 15
    'Tests duplicated random initializer within the same graph.\n\n  This test generates two random kernels from the same initializer to the same\n  graph, and checks if the results are close enough. Even given the same global,\n  seed, two different instances of random kernels should generate different\n  results.\n\n  Args:\n    tc: An instance of TensorFlowTestCase.\n    init: An Initializer that generates a tensor of a given shape\n    graph_seed: A graph-level seed to use.\n    shape: Shape of the tensor to initialize or `None` to use a vector of length\n      100.\n\n  Returns:\n    True or False as determined by test.\n  '
    if shape is None:
        shape = [100]
    with tc.test_session(graph=ops.Graph()):
        random_seed.set_random_seed(graph_seed)
        t1 = init(shape).eval()
        t2 = init(shape).eval()
        return np.allclose(t1, t2, rtol=1e-15, atol=1e-15)

def _init_sampler(tc, init, num):
    if False:
        i = 10
        return i + 15
    'Returns a func to generate a random tensor of shape [num].\n\n  Args:\n    tc: An instance of TensorFlowTestCase.\n    init: An Initializer that generates a tensor of a given shape\n    num: Size of 1D tensor to create.\n\n  Returns:\n    Function to generate a random tensor.\n  '

    def func():
        if False:
            for i in range(10):
                print('nop')
        with tc.test_session():
            return init([num]).eval()
    return func

class ConstantInitializersTest(test.TestCase):

    @test_util.run_deprecated_v1
    def testZerosInitializer(self):
        if False:
            while True:
                i = 10
        with self.session():
            shape = [2, 3]
            x = variable_scope.get_variable('x', shape=shape, initializer=init_ops.zeros_initializer())
            self.evaluate(x.initializer)
            self.assertAllEqual(x, np.zeros(shape))

    @test_util.run_deprecated_v1
    def testOnesInitializer(self):
        if False:
            while True:
                i = 10
        with self.session():
            shape = [2, 3]
            x = variable_scope.get_variable('x', shape=shape, initializer=init_ops.ones_initializer())
            self.evaluate(x.initializer)
            self.assertAllEqual(x, np.ones(shape))

    @test_util.run_deprecated_v1
    def testConstantZeroInitializer(self):
        if False:
            while True:
                i = 10
        with self.session():
            shape = [2, 3]
            x = variable_scope.get_variable('x', shape=shape, initializer=init_ops.constant_initializer(0.0))
            self.evaluate(x.initializer)
            self.assertAllEqual(x, np.zeros(shape))

    @test_util.run_deprecated_v1
    def testConstantOneInitializer(self):
        if False:
            i = 10
            return i + 15
        with self.session():
            shape = [2, 3]
            x = variable_scope.get_variable('x', shape=shape, initializer=init_ops.constant_initializer(1.0))
            self.evaluate(x.initializer)
            self.assertAllEqual(x, np.ones(shape))

    @test_util.run_deprecated_v1
    def testConstantIntInitializer(self):
        if False:
            print('Hello World!')
        with self.session():
            shape = [2, 3]
            x = variable_scope.get_variable('x', shape=shape, dtype=dtypes.int32, initializer=init_ops.constant_initializer(7))
            self.evaluate(x.initializer)
            self.assertEqual(x.dtype.base_dtype, dtypes.int32)
            self.assertAllEqual(x, 7 * np.ones(shape, dtype=np.int32))

    @test_util.run_deprecated_v1
    def testConstantTupleInitializer(self):
        if False:
            for i in range(10):
                print('nop')
        with self.session():
            shape = [3]
            x = variable_scope.get_variable('x', shape=shape, dtype=dtypes.int32, initializer=init_ops.constant_initializer((10, 20, 30)))
            self.evaluate(x.initializer)
            self.assertEqual(x.dtype.base_dtype, dtypes.int32)
            self.assertAllEqual(x, [10, 20, 30])

    def _testNDimConstantInitializer(self, name, value, shape, expected):
        if False:
            while True:
                i = 10
        with self.cached_session():
            init = init_ops.constant_initializer(value, dtype=dtypes.int32)
            x = variable_scope.get_variable(name, shape=shape, initializer=init)
            self.evaluate(x.initializer)
            actual = array_ops.reshape(x, [-1]).eval()
            self.assertEqual(len(actual), len(expected))
            for (a, e) in zip(actual, expected):
                self.assertEqual(a, e)

    @test_util.run_deprecated_v1
    def testNDimConstantInitializer(self):
        if False:
            for i in range(10):
                print('nop')
        value = [0, 1, 2, 3, 4, 5]
        shape = [2, 3]
        expected = list(value)
        self._testNDimConstantInitializer('list', value, shape, expected)
        self._testNDimConstantInitializer('ndarray', np.asarray(value), shape, expected)
        self._testNDimConstantInitializer('2D-ndarray', np.asarray(value).reshape(tuple(shape)), shape, expected)

    def _testNDimConstantInitializerLessValues(self, name, value, shape, expected):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            init = init_ops.constant_initializer(value, dtype=dtypes.int32)
            x = variable_scope.get_variable(name, shape=shape, initializer=init)
            self.evaluate(x.initializer)
            actual = array_ops.reshape(x, [-1]).eval()
            self.assertGreater(len(actual), len(expected))
            for i in range(len(actual)):
                a = actual[i]
                e = expected[i] if i < len(expected) else expected[-1]
                self.assertEqual(a, e)

    @test_util.run_deprecated_v1
    def testNDimConstantInitializerLessValues(self):
        if False:
            print('Hello World!')
        value = [0, 1, 2, 3, 4, 5]
        shape = [2, 4]
        expected = list(value)
        self._testNDimConstantInitializerLessValues('list', value, shape, expected)
        self._testNDimConstantInitializerLessValues('ndarray', np.asarray(value), shape, expected)
        self._testNDimConstantInitializerLessValues('2D-ndarray', np.asarray(value).reshape(tuple([2, 3])), shape, expected)

    def _testNDimConstantInitializerMoreValues(self, value, shape):
        if False:
            while True:
                i = 10
        ops.reset_default_graph()
        with self.cached_session():
            init = init_ops.constant_initializer(value, dtype=dtypes.int32)
            self.assertRaises(ValueError, variable_scope.get_variable, 'x', shape=shape, initializer=init)

    @test_util.run_deprecated_v1
    def testNDimConstantInitializerMoreValues(self):
        if False:
            print('Hello World!')
        value = [0, 1, 2, 3, 4, 5, 6, 7]
        shape = [2, 3]
        self._testNDimConstantInitializerMoreValues(value, shape)
        self._testNDimConstantInitializerMoreValues(np.asarray(value), shape)
        self._testNDimConstantInitializerMoreValues(np.asarray(value).reshape(tuple([2, 4])), shape)

    def testInvalidValueTypeForConstantInitializerCausesTypeError(self):
        if False:
            return 10
        c = constant_op.constant([1.0, 2.0, 3.0])
        with self.assertRaisesRegex(TypeError, 'Invalid type for initial value=.*Tensor.*'):
            init_ops.constant_initializer(c, dtype=dtypes.float32)
        v = variables.Variable([3.0, 2.0, 1.0])
        with self.assertRaisesRegex(TypeError, 'Invalid type for initial value=.*Variable.*'):
            init_ops.constant_initializer(v, dtype=dtypes.float32)

class RandomNormalInitializationTest(test.TestCase):

    @test_util.run_deprecated_v1
    def testInitializerIdentical(self):
        if False:
            print('Hello World!')
        for dtype in [dtypes.float32, dtypes.float64]:
            init1 = init_ops.random_normal_initializer(0.0, 1.0, seed=1, dtype=dtype)
            init2 = init_ops.random_normal_initializer(0.0, 1.0, seed=1, dtype=dtype)
            self.assertTrue(identicaltest(self, init1, init2))

    @test_util.run_deprecated_v1
    def testInitializerDifferent(self):
        if False:
            while True:
                i = 10
        for dtype in [dtypes.float32, dtypes.float64]:
            init1 = init_ops.random_normal_initializer(0.0, 1.0, seed=1, dtype=dtype)
            init2 = init_ops.random_normal_initializer(0.0, 1.0, seed=2, dtype=dtype)
            self.assertFalse(identicaltest(self, init1, init2))

    @test_util.run_deprecated_v1
    def testDuplicatedInitializer(self):
        if False:
            return 10
        init = init_ops.random_normal_initializer(0.0, 1.0)
        self.assertFalse(duplicated_initializer(self, init, 1))

    def testInvalidDataType(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(ValueError, init_ops.random_normal_initializer, 0.0, 1.0, dtype=dtypes.string)

class TruncatedNormalInitializationTest(test.TestCase):

    @test_util.run_deprecated_v1
    def testInitializerIdentical(self):
        if False:
            return 10
        for dtype in [dtypes.float32, dtypes.float64]:
            init1 = init_ops.truncated_normal_initializer(0.0, 1.0, seed=1, dtype=dtype)
            init2 = init_ops.truncated_normal_initializer(0.0, 1.0, seed=1, dtype=dtype)
            self.assertTrue(identicaltest(self, init1, init2))

    @test_util.run_deprecated_v1
    def testInitializerDifferent(self):
        if False:
            i = 10
            return i + 15
        for dtype in [dtypes.float32, dtypes.float64]:
            init1 = init_ops.truncated_normal_initializer(0.0, 1.0, seed=1, dtype=dtype)
            init2 = init_ops.truncated_normal_initializer(0.0, 1.0, seed=2, dtype=dtype)
            self.assertFalse(identicaltest(self, init1, init2))

    @test_util.run_deprecated_v1
    def testDuplicatedInitializer(self):
        if False:
            while True:
                i = 10
        init = init_ops.truncated_normal_initializer(0.0, 1.0)
        self.assertFalse(duplicated_initializer(self, init, 1))

    def testInvalidDataType(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(ValueError, init_ops.truncated_normal_initializer, 0.0, 1.0, dtype=dtypes.string)

class RandomUniformInitializationTest(test.TestCase):

    @test_util.run_deprecated_v1
    def testInitializerIdentical(self):
        if False:
            while True:
                i = 10
        for dtype in [dtypes.float32, dtypes.float64, dtypes.int64]:
            init1 = init_ops.random_uniform_initializer(0, 7, seed=1, dtype=dtype)
            init2 = init_ops.random_uniform_initializer(0, 7, seed=1, dtype=dtype)
            self.assertTrue(identicaltest(self, init1, init2))

    @test_util.run_deprecated_v1
    def testInitializerDifferent(self):
        if False:
            while True:
                i = 10
        for dtype in [dtypes.float32, dtypes.float64, dtypes.int32, dtypes.int64]:
            init1 = init_ops.random_uniform_initializer(0, 7, seed=1, dtype=dtype)
            init2 = init_ops.random_uniform_initializer(0, 7, seed=2, dtype=dtype)
            self.assertFalse(identicaltest(self, init1, init2))

    @test_util.run_deprecated_v1
    def testDuplicatedInitializer(self):
        if False:
            i = 10
            return i + 15
        init = init_ops.random_uniform_initializer(0.0, 1.0)
        self.assertFalse(duplicated_initializer(self, init, 1))

class UniformUnitScalingInitializationTest(test.TestCase):

    @test_util.run_deprecated_v1
    def testInitializerIdentical(self):
        if False:
            return 10
        for dtype in [dtypes.float32, dtypes.float64]:
            init1 = init_ops.uniform_unit_scaling_initializer(seed=1, dtype=dtype)
            init2 = init_ops.uniform_unit_scaling_initializer(seed=1, dtype=dtype)
            self.assertTrue(identicaltest(self, init1, init2))
            init3 = init_ops.uniform_unit_scaling_initializer(1.5, seed=1, dtype=dtype)
            init4 = init_ops.uniform_unit_scaling_initializer(1.5, seed=1, dtype=dtype)
            self.assertTrue(identicaltest(self, init3, init4))

    @test_util.run_deprecated_v1
    def testInitializerDifferent(self):
        if False:
            while True:
                i = 10
        for dtype in [dtypes.float32, dtypes.float64]:
            init1 = init_ops.uniform_unit_scaling_initializer(seed=1, dtype=dtype)
            init2 = init_ops.uniform_unit_scaling_initializer(seed=2, dtype=dtype)
            init3 = init_ops.uniform_unit_scaling_initializer(1.5, seed=1, dtype=dtype)
            self.assertFalse(identicaltest(self, init1, init2))
            self.assertFalse(identicaltest(self, init1, init3))
            self.assertFalse(identicaltest(self, init2, init3))

    @test_util.run_deprecated_v1
    def testZeroSize(self):
        if False:
            return 10
        shape = [0, 2]
        with self.cached_session():
            x = variable_scope.get_variable('x', shape=shape, initializer=init_ops.uniform_unit_scaling_initializer())
            self.evaluate(variables.global_variables_initializer())
            self.assertAllEqual(shape, self.evaluate(x).shape)

    @test_util.run_deprecated_v1
    def testDuplicatedInitializer(self):
        if False:
            return 10
        init = init_ops.uniform_unit_scaling_initializer()
        self.assertFalse(duplicated_initializer(self, init, 1))

    def testInvalidDataType(self):
        if False:
            return 10
        self.assertRaises(ValueError, init_ops.uniform_unit_scaling_initializer, dtype=dtypes.string)

class VarianceScalingInitializationTest(test.TestCase):

    @test_util.run_deprecated_v1
    def testTruncatedNormalDistribution(self):
        if False:
            for i in range(10):
                print('nop')
        shape = [100, 100]
        expect_mean = 0.0
        expect_var = 1.0 / shape[0]
        init = init_ops.variance_scaling_initializer(distribution='truncated_normal')
        with self.session(), test.mock.patch.object(random_ops, 'truncated_normal', wraps=random_ops.truncated_normal) as mock_truncated_normal:
            x = init(shape).eval()
            self.assertTrue(mock_truncated_normal.called)
        self.assertNear(np.mean(x), expect_mean, err=0.01)
        self.assertNear(np.var(x), expect_var, err=0.01)

    @test_util.run_deprecated_v1
    def testNormalDistribution(self):
        if False:
            for i in range(10):
                print('nop')
        shape = [100, 100]
        expect_mean = 0.0
        expect_var = 1.0 / shape[0]
        init = init_ops.variance_scaling_initializer(distribution='normal')
        with self.session(), test.mock.patch.object(random_ops, 'truncated_normal', wraps=random_ops.truncated_normal) as mock_truncated_normal:
            x = init(shape).eval()
            self.assertTrue(mock_truncated_normal.called)
        self.assertNear(np.mean(x), expect_mean, err=0.01)
        self.assertNear(np.var(x), expect_var, err=0.01)

    @test_util.run_deprecated_v1
    def testUntruncatedNormalDistribution(self):
        if False:
            i = 10
            return i + 15
        shape = [100, 100]
        expect_mean = 0.0
        expect_var = 1.0 / shape[0]
        init = init_ops.variance_scaling_initializer(distribution='untruncated_normal')
        with self.session(), test.mock.patch.object(random_ops, 'random_normal', wraps=random_ops.random_normal) as mock_random_normal:
            x = init(shape).eval()
            self.assertTrue(mock_random_normal.called)
        self.assertNear(np.mean(x), expect_mean, err=0.01)
        self.assertNear(np.var(x), expect_var, err=0.01)

    @test_util.run_deprecated_v1
    def testUniformDistribution(self):
        if False:
            print('Hello World!')
        shape = [100, 100]
        expect_mean = 0.0
        expect_var = 1.0 / shape[0]
        init = init_ops.variance_scaling_initializer(distribution='uniform')
        with self.session():
            x = init(shape).eval()
        self.assertNear(np.mean(x), expect_mean, err=0.01)
        self.assertNear(np.var(x), expect_var, err=0.01)

class RangeTest(test.TestCase):

    def _Range(self, start, limit, delta):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            tf_ans = math_ops.range(start, limit, delta, name='range')
            self.assertEqual([len(np.arange(start, limit, delta))], tf_ans.get_shape())
            return self.evaluate(tf_ans)

    def testBasic(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(np.array_equal(self._Range(0, 5, 1), np.array([0, 1, 2, 3, 4])))
        self.assertTrue(np.array_equal(self._Range(0, 5, 2), np.array([0, 2, 4])))
        self.assertTrue(np.array_equal(self._Range(0, 6, 2), np.array([0, 2, 4])))
        self.assertTrue(np.array_equal(self._Range(13, 32, 7), np.array([13, 20, 27])))
        self.assertTrue(np.array_equal(self._Range(100, 500, 100), np.array([100, 200, 300, 400])))
        self.assertEqual(math_ops.range(0, 5, 1).dtype, dtypes.int32)

    @test_util.run_deprecated_v1
    def testLimitOnly(self):
        if False:
            print('Hello World!')
        with self.session():
            self.assertAllEqual(np.arange(5), math_ops.range(5))

    def testEmpty(self):
        if False:
            print('Hello World!')
        for start in (0, 5):
            self.assertTrue(np.array_equal(self._Range(start, start, 1), []))

    def testNonInteger(self):
        if False:
            while True:
                i = 10
        self.assertTrue(np.allclose(self._Range(0, 2, 0.5), np.array([0, 0.5, 1, 1.5])))
        self.assertTrue(np.allclose(self._Range(0, 5, 2.5), np.array([0, 2.5])))
        self.assertTrue(np.allclose(self._Range(0, 3, 0.9), np.array([0, 0.9, 1.8, 2.7])))
        self.assertTrue(np.allclose(self._Range(100.0, 500.0, 100.0), np.array([100, 200, 300, 400])))
        self.assertEqual(math_ops.range(0.0, 5.0, 1.0).dtype, dtypes.float32)

    def testNegativeDelta(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(np.array_equal(self._Range(5, -1, -1), np.array([5, 4, 3, 2, 1, 0])))
        self.assertTrue(np.allclose(self._Range(2.5, 0, -0.5), np.array([2.5, 2, 1.5, 1, 0.5])))
        self.assertTrue(np.array_equal(self._Range(-5, -10, -3), np.array([-5, -8])))

    def testDType(self):
        if False:
            return 10
        zero_int32 = math_ops.cast(0, dtypes.int32)
        zero_int64 = math_ops.cast(0, dtypes.int64)
        zero_float32 = math_ops.cast(0, dtypes.float32)
        zero_float64 = math_ops.cast(0, dtypes.float64)
        self.assertEqual(math_ops.range(zero_int32, 0, 1).dtype, dtypes.int32)
        self.assertEqual(math_ops.range(zero_int64, 0, 1).dtype, dtypes.int64)
        self.assertEqual(math_ops.range(zero_float32, 0, 1).dtype, dtypes.float32)
        self.assertEqual(math_ops.range(zero_float64, 0, 1).dtype, dtypes.float64)
        self.assertEqual(math_ops.range(zero_int32, zero_int64, 1).dtype, dtypes.int64)
        self.assertEqual(math_ops.range(zero_int64, zero_float32, 1).dtype, dtypes.float32)
        self.assertEqual(math_ops.range(zero_float32, zero_float64, 1).dtype, dtypes.float64)
        self.assertEqual(math_ops.range(zero_float64, zero_int32, 1).dtype, dtypes.float64)
        self.assertEqual(math_ops.range(0, 0, 1, dtype=dtypes.int32).dtype, dtypes.int32)
        self.assertEqual(math_ops.range(0, 0, 1, dtype=dtypes.int64).dtype, dtypes.int64)
        self.assertEqual(math_ops.range(0, 0, 1, dtype=dtypes.float32).dtype, dtypes.float32)
        self.assertEqual(math_ops.range(0, 0, 1, dtype=dtypes.float64).dtype, dtypes.float64)

    def testMixedDType(self):
        if False:
            i = 10
            return i + 15
        tf_ans = math_ops.range(constant_op.constant(4, dtype=dtypes.int32), dtype=dtypes.int64)
        self.assertAllEqual(self.evaluate(tf_ans), np.array([0, 1, 2, 3]))

    def testLargeLimits(self):
        if False:
            for i in range(10):
                print('nop')
        with self.session():
            with self.assertRaises(errors_impl.ResourceExhaustedError):
                v = math_ops.range(0, 9223372036854775807)
                self.evaluate(v)

    def testLargeStarts(self):
        if False:
            while True:
                i = 10
        with self.session():
            with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
                v = math_ops.range(start=-1e+38, limit=1)
                self.evaluate(v)

class LinSpaceTest(test.TestCase):

    def _gpu_modes(self):
        if False:
            i = 10
            return i + 15
        if test.is_gpu_available():
            return [False, True]
        else:
            return [False]

    def _LinSpace(self, start, stop, num):
        if False:
            print('Hello World!')
        with ops.Graph().as_default() as graph:
            with self.session(graph=graph, force_gpu=self.force_gpu):
                tf_ans = math_ops.linspace(start, stop, num, name='linspace')
                self.assertEqual([num], tf_ans.get_shape())
                return self.evaluate(tf_ans)

    def testPositive(self):
        if False:
            return 10
        for self.force_gpu in self._gpu_modes():
            self.assertArrayNear(self._LinSpace(1.0, 5.0, 1), np.array([1.0]), 1e-05)
            self.assertArrayNear(self._LinSpace(1.0, 5.0, 2), np.array([1.0, 5.0]), 1e-05)
            self.assertArrayNear(self._LinSpace(1.0, 5.0, 3), np.array([1.0, 3.0, 5.0]), 1e-05)
            self.assertArrayNear(self._LinSpace(1.0, 5.0, 4), np.array([1.0, 7.0 / 3.0, 11.0 / 3.0, 5.0]), 1e-05)

    def testNegative(self):
        if False:
            for i in range(10):
                print('nop')
        for self.force_gpu in self._gpu_modes():
            self.assertArrayNear(self._LinSpace(-1.0, -5.0, 1), np.array([-1.0]), 1e-05)
            self.assertArrayNear(self._LinSpace(-1.0, -5.0, 2), np.array([-1.0, -5.0]), 1e-05)
            self.assertArrayNear(self._LinSpace(-1.0, -5.0, 3), np.array([-1.0, -3.0, -5.0]), 1e-05)
            self.assertArrayNear(self._LinSpace(-1.0, -5.0, 4), np.array([-1.0, -7.0 / 3.0, -11.0 / 3.0, -5.0]), 1e-05)

    def testNegativeToPositive(self):
        if False:
            for i in range(10):
                print('nop')
        for self.force_gpu in self._gpu_modes():
            self.assertArrayNear(self._LinSpace(-1.0, 5.0, 1), np.array([-1.0]), 1e-05)
            self.assertArrayNear(self._LinSpace(-1.0, 5.0, 2), np.array([-1.0, 5.0]), 1e-05)
            self.assertArrayNear(self._LinSpace(-1.0, 5.0, 3), np.array([-1.0, 2.0, 5.0]), 1e-05)
            self.assertArrayNear(self._LinSpace(-1.0, 5.0, 4), np.array([-1.0, 1.0, 3.0, 5.0]), 1e-05)

    def testPoint(self):
        if False:
            print('Hello World!')
        for self.force_gpu in self._gpu_modes():
            self.assertArrayNear(self._LinSpace(5.0, 5.0, 1), np.array([5.0]), 1e-05)
            self.assertArrayNear(self._LinSpace(5.0, 5.0, 2), np.array([5.0] * 2), 1e-05)
            self.assertArrayNear(self._LinSpace(5.0, 5.0, 3), np.array([5.0] * 3), 1e-05)
            self.assertArrayNear(self._LinSpace(5.0, 5.0, 4), np.array([5.0] * 4), 1e-05)

    def testEndpointsAreExact(self):
        if False:
            return 10
        for self.force_gpu in self._gpu_modes():
            self.assertAllEqual(self._LinSpace(0.0, 1.0, 42)[[0, -1]], np.array([0.0, 1.0], np.float32))
            self.assertAllEqual(self._LinSpace(-1.0, 0.0, 42)[[0, -1]], np.array([-1.0, 0.0], np.float32))
            self.assertAllEqual(self._LinSpace(0.1, 0.2, 4)[[0, -1]], np.array([0.1, 0.2], np.float32))
            self.assertAllEqual(self._LinSpace(np.array(0.0, np.float64), 0.1, 12)[[0, -1]], np.array([0.0, 0.1], np.float64))

class LinSpaceNdTest(test.TestCase):

    def _gpu_modes(self):
        if False:
            i = 10
            return i + 15
        if test.is_gpu_available():
            return [False, True]
        else:
            return [False]

    def _LinSpace(self, start, stop, num, axis=0):
        if False:
            return 10
        with ops.Graph().as_default() as graph:
            with self.session(graph=graph, force_gpu=self.force_gpu):
                tf_ans = math_ops.linspace_nd(start, stop, num, axis=axis)
                return self.evaluate(tf_ans)

    def _LinSpaceNumConstant(self, start, stop, num, axis=0):
        if False:
            return 10
        with ops.Graph().as_default() as graph:
            num_constant = constant_op.constant(num)
            with self.session(graph=graph, force_gpu=self.force_gpu):
                tf_ans = math_ops.linspace_nd(start, stop, num_constant, axis=axis)
                return self.evaluate(tf_ans)

    def _LinspaceNoneShape(self, start, stop, num, graph_shape=None, axis=0):
        if False:
            return 10
        with ops.Graph().as_default() as graph:
            num_tensor = array_ops.placeholder(dtypes.int32)
            start_t = array_ops.placeholder(dtypes.float32, shape=graph_shape)
            stop_t = array_ops.placeholder(dtypes.float32, shape=graph_shape)
            ans_tensor = math_ops.linspace_nd(start_t, stop_t, num_tensor, axis=axis)
            with self.session(graph=graph, force_gpu=self.force_gpu) as sess:
                feed_dict = {start_t: start, stop_t: stop, num_tensor: num}
                return sess.run(ans_tensor, feed_dict=feed_dict)

    def testPositive(self):
        if False:
            print('Hello World!')
        for self.force_gpu in self._gpu_modes():
            self.assertArrayNear(self._LinSpace(1.0, 5.0, 1), np.array([1.0]), 1e-05)
            self.assertArrayNear(self._LinSpace(1.0, 5.0, 2), np.array([1.0, 5.0]), 1e-05)
            self.assertArrayNear(self._LinSpace(1.0, 5.0, 3), np.array([1.0, 3.0, 5.0]), 1e-05)
            self.assertArrayNear(self._LinSpace(1.0, 5.0, 4), np.array([1.0, 7.0 / 3.0, 11.0 / 3.0, 5.0]), 1e-05)

    def testNegative(self):
        if False:
            while True:
                i = 10
        for self.force_gpu in self._gpu_modes():
            self.assertArrayNear(self._LinSpace(-1.0, -5.0, 1), np.array([-1.0]), 1e-05)
            self.assertArrayNear(self._LinSpace(-1.0, -5.0, 2), np.array([-1.0, -5.0]), 1e-05)
            self.assertArrayNear(self._LinSpace(-1.0, -5.0, 3), np.array([-1.0, -3.0, -5.0]), 1e-05)
            self.assertArrayNear(self._LinSpace(-1.0, -5.0, 4), np.array([-1.0, -7.0 / 3.0, -11.0 / 3.0, -5.0]), 1e-05)

    def testNegativeToPositive(self):
        if False:
            print('Hello World!')
        for self.force_gpu in self._gpu_modes():
            self.assertArrayNear(self._LinSpace(-1.0, 5.0, 1), np.array([-1.0]), 1e-05)
            self.assertArrayNear(self._LinSpace(-1.0, 5.0, 2), np.array([-1.0, 5.0]), 1e-05)
            self.assertArrayNear(self._LinSpace(-1.0, 5.0, 3), np.array([-1.0, 2.0, 5.0]), 1e-05)
            self.assertArrayNear(self._LinSpace(-1.0, 5.0, 4), np.array([-1.0, 1.0, 3.0, 5.0]), 1e-05)

    def testPoint(self):
        if False:
            i = 10
            return i + 15
        for self.force_gpu in self._gpu_modes():
            self.assertArrayNear(self._LinSpace(5.0, 5.0, 1), np.array([5.0]), 1e-05)
            self.assertArrayNear(self._LinSpace(5.0, 5.0, 2), np.array([5.0] * 2), 1e-05)
            self.assertArrayNear(self._LinSpace(5.0, 5.0, 3), np.array([5.0] * 3), 1e-05)
            self.assertArrayNear(self._LinSpace(5.0, 5.0, 4), np.array([5.0] * 4), 1e-05)

    def testEndpointsAreExact(self):
        if False:
            return 10
        for self.force_gpu in self._gpu_modes():
            self.assertAllEqual(self._LinSpace(0.0, 1.0, 42)[[0, -1]], np.array([0.0, 1.0], np.float32))
            self.assertAllEqual(self._LinSpace(-1.0, 0.0, 42)[[0, -1]], np.array([-1.0, 0.0], np.float32))
            self.assertAllEqual(self._LinSpace(0.1, 0.2, 4)[[0, -1]], np.array([0.1, 0.2], np.float32))
            self.assertAllEqual(self._LinSpace(np.array(0.0, np.float64), 0.1, 12)[[0, -1]], np.array([0.0, 0.1], np.float64))

    def testScalarsCompareToNumpy(self):
        if False:
            i = 10
            return i + 15
        for self.force_gpu in self._gpu_modes():
            actual = self._LinSpace(0.0, 1.0, 32)
            expected = np.linspace(0.0, 1.0, 32)
            self.assertArrayNear(expected, actual, 1e-05)

    def _baseNDArrayCompareToNumpy(self, axis):
        if False:
            print('Hello World!')
        for self.force_gpu in self._gpu_modes():
            (a, b, expected, num) = self.create_nd_inputs_and_expected_output(axis)
            actual = self._LinSpace(a, b, num, axis=axis)
            self.assert_close(actual, expected)

    def assert_close(self, actual, expected):
        if False:
            i = 10
            return i + 15
        wrong_indices = np.where(~np.allclose(actual, expected))
        mess = 'Wrong float answer. Wrong indices: {}'.format(wrong_indices)
        self.assertTrue(np.allclose(actual, expected), mess)

    def create_nd_inputs_and_expected_output(self, axis):
        if False:
            i = 10
            return i + 15
        a = np.arange(2, dtype=np.float32)
        b = a * 5
        num = 5
        res = np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0, 5.0]])
        expected = res if axis != 0 else res.T
        return (a, b, expected, num)

    def testNDArrayCompareToNumpyDefaultAxis(self):
        if False:
            i = 10
            return i + 15
        self._baseNDArrayCompareToNumpy(0)

    def testNDArrayAxisStrictlyPositive(self):
        if False:
            print('Hello World!')
        self._baseNDArrayCompareToNumpy(1)

    def testNDArrayAxisStrictlyNegative(self):
        if False:
            print('Hello World!')
        self._baseNDArrayCompareToNumpy(-1)

    def testNumConstant(self):
        if False:
            return 10
        for self.force_gpu in self._gpu_modes():
            actual = self._LinSpaceNumConstant(0.0, 1.0, 32)
            expected = np.linspace(0.0, 1.0, 32)
            self.assertArrayNear(expected, actual, 1e-05)

    def testUnknownShapeAtGraphCreationTime(self):
        if False:
            return 10
        self.base_test_unknown_shape(2)

    def testNoneValuesInShapeAtGraphCreationTime(self):
        if False:
            for i in range(10):
                print('nop')
        self.base_test_unknown_shape(None)

    def testNoneShapeAtGraphCreationTime(self):
        if False:
            print('Hello World!')
        self.base_test_unknown_shape(None)

    def base_test_unknown_shape(self, graph_shape):
        if False:
            return 10
        for self.force_gpu in self._gpu_modes():
            axis = 1
            (a, b, expected, num) = self.create_nd_inputs_and_expected_output(axis)
            actual = self._LinspaceNoneShape(a, b, num, graph_shape, axis)
            self.assert_close(actual, expected)

class DeviceTest(test.TestCase):

    def testNoDevice(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.Graph().as_default():
            var = variables.Variable([[1.0, 1.0]])
        self.assertDeviceEqual(None, var.device)
        self.assertDeviceEqual(None, var.initializer.device)

    def testDevice(self):
        if False:
            while True:
                i = 10
        with ops.Graph().as_default():
            with ops.device('/job:ps'):
                var = variables.Variable([[1.0, 1.0]])
        self.assertDeviceEqual('/job:ps', var.device)
        self.assertDeviceEqual('/job:ps', var.initializer.device)

class OrthogonalInitializerTest(test.TestCase):

    @test_util.run_deprecated_v1
    def testInitializerIdentical(self):
        if False:
            while True:
                i = 10
        for dtype in [dtypes.float32, dtypes.float64]:
            init1 = init_ops.orthogonal_initializer(seed=1, dtype=dtype)
            init2 = init_ops.orthogonal_initializer(seed=1, dtype=dtype)
            self.assertTrue(identicaltest(self, init1, init2, (10, 10)))

    @test_util.run_deprecated_v1
    def testInitializerDifferent(self):
        if False:
            print('Hello World!')
        for dtype in [dtypes.float32, dtypes.float64]:
            init1 = init_ops.orthogonal_initializer(seed=1, dtype=dtype)
            init2 = init_ops.orthogonal_initializer(seed=2, dtype=dtype)
            self.assertFalse(identicaltest(self, init1, init2, (10, 10)))

    @test_util.run_deprecated_v1
    def testDuplicatedInitializer(self):
        if False:
            while True:
                i = 10
        init = init_ops.orthogonal_initializer()
        self.assertFalse(duplicated_initializer(self, init, 1, (10, 10)))

    def testInvalidDataType(self):
        if False:
            while True:
                i = 10
        self.assertRaises(ValueError, init_ops.orthogonal_initializer, dtype=dtypes.string)

    def testInvalidShape(self):
        if False:
            for i in range(10):
                print('nop')
        init1 = init_ops.orthogonal_initializer()
        with self.session(graph=ops.Graph(), use_gpu=True):
            self.assertRaises(ValueError, init1, shape=[5])

    @test_util.run_deprecated_v1
    def testGain(self):
        if False:
            for i in range(10):
                print('nop')
        shape = (10, 10)
        for dtype in [dtypes.float32, dtypes.float64]:
            init1 = init_ops.orthogonal_initializer(seed=1, dtype=dtype)
            init2 = init_ops.orthogonal_initializer(gain=3.14, seed=1, dtype=dtype)
            with self.session(graph=ops.Graph(), use_gpu=True):
                t1 = init1(shape).eval()
                t2 = init2(shape).eval()
            self.assertAllClose(t1, t2 / 3.14)

    @test_util.run_deprecated_v1
    def testShapesValues(self):
        if False:
            print('Hello World!')
        for dtype in [dtypes.float32, dtypes.float64]:
            for shape in [(10, 10), (10, 9, 8), (100, 5, 5), (50, 40), (40, 50)]:
                init = init_ops.orthogonal_initializer(dtype=dtype)
                tol = 1e-05 if dtype == dtypes.float32 else 1e-12
                with self.session(graph=ops.Graph(), use_gpu=True):
                    t = init(shape).eval()
                    self.assertAllEqual(shape, t.shape)
                    t = t.reshape((np.prod(t.shape[:-1]), t.shape[-1]))
                    if t.shape[0] > t.shape[1]:
                        self.assertAllClose(np.dot(t.T, t), np.eye(t.shape[1]), rtol=tol, atol=tol)
                    else:
                        self.assertAllClose(np.dot(t, t.T), np.eye(t.shape[0]), rtol=tol, atol=tol)

class ConvolutionDeltaOrthogonalInitializerTest(test.TestCase):

    @test_util.run_deprecated_v1
    def testInitializerIdentical(self):
        if False:
            print('Hello World!')
        for dtype in [dtypes.float32, dtypes.float64]:
            init1 = init_ops.convolutional_delta_orthogonal(seed=1, dtype=dtype)
            init2 = init_ops.convolutional_delta_orthogonal(seed=1, dtype=dtype)
            self.assertTrue(identicaltest(self, init1, init2, (3, 3, 10, 10)))

    @test_util.run_deprecated_v1
    def testInitializerDifferent(self):
        if False:
            for i in range(10):
                print('nop')
        for dtype in [dtypes.float32, dtypes.float64]:
            init1 = init_ops.convolutional_delta_orthogonal(seed=1, dtype=dtype)
            init2 = init_ops.convolutional_delta_orthogonal(seed=2, dtype=dtype)
            self.assertFalse(identicaltest(self, init1, init2, (3, 3, 10, 10)))

    @test_util.run_deprecated_v1
    def testDuplicatedInitializer(self):
        if False:
            i = 10
            return i + 15
        init = init_ops.convolutional_delta_orthogonal()
        self.assertFalse(duplicated_initializer(self, init, 1, (3, 3, 10, 10)))

    def testInvalidDataType(self):
        if False:
            print('Hello World!')
        self.assertRaises(ValueError, init_ops.convolutional_delta_orthogonal, dtype=dtypes.string)

    def testInvalidShape(self):
        if False:
            for i in range(10):
                print('nop')
        init1 = init_ops.convolutional_delta_orthogonal()
        with self.session(graph=ops.Graph(), use_gpu=True):
            self.assertRaises(ValueError, init1, shape=[3, 3, 6, 5])

    @test_util.run_deprecated_v1
    def testGain(self):
        if False:
            i = 10
            return i + 15
        shape = (3, 3, 10, 10)
        for dtype in [dtypes.float32, dtypes.float64]:
            init1 = init_ops.convolutional_delta_orthogonal(seed=1, dtype=dtype)
            init2 = init_ops.convolutional_delta_orthogonal(gain=3.14, seed=1, dtype=dtype)
            with self.session(graph=ops.Graph(), use_gpu=True):
                t1 = init1(shape).eval()
                t2 = init2(shape).eval()
            self.assertAllClose(t1, t2 / 3.14)

    @test_util.run_deprecated_v1
    def testShapesValues(self):
        if False:
            i = 10
            return i + 15
        gain = 3.14
        for dtype in [dtypes.float32]:
            for kernel_size in [[3], [8], [3, 5], [2, 4], [3, 3, 3], [2, 2, 2]]:
                tol = 0.01
                if len(kernel_size) == 1:
                    shape = [4, 32, 64]
                    convolution = convolutional.conv1d
                elif len(kernel_size) == 2:
                    convolution = convolutional.conv2d
                    shape = [4, 32, 32, 64]
                else:
                    shape = [4, 16, 16, 16, 64]
                    convolution = convolutional.conv3d
                inputs = random_ops.random_normal(shape, dtype=dtype)
                inputs_2norm = linalg_ops.norm(inputs)
                outputs = convolution(inputs, padding='same', filters=128, kernel_size=kernel_size, use_bias=False, kernel_initializer=init_ops.convolutional_delta_orthogonal(gain=gain))
                outputs_shape = shape[0:-1] + [128]
                outputs_2norm = linalg_ops.norm(outputs)
                ratio = outputs_2norm / inputs_2norm
                my_ops = variables.global_variables_initializer()
                with self.session():
                    self.evaluate(my_ops)
                    t = self.evaluate(outputs)
                    self.assertAllEqual(t.shape, outputs_shape)
                    self.assertAllClose(self.evaluate(ratio), gain, rtol=tol, atol=tol)

    @test_util.run_deprecated_v1
    def testNonuniformity(self):
        if False:
            return 10
        value = 0
        abs_value = 0
        shape = [3, 3, 10, 10]
        count = 70
        tol = 1e-05
        with self.session():
            for i in range(count):
                x = variable_scope.get_variable('{}'.format(i), shape=shape, initializer=init_ops.convolutional_delta_orthogonal)
                self.evaluate(x.initializer)
                y = self.evaluate(x)[1, 1, :, :]
                determinant = np.linalg.det(y)
                value += determinant
                abs_value += np.abs(determinant)
            self.assertLess(value, count - tol)
            self.assertLess(-count + tol, value)
            self.assertAllClose(abs_value, count, rtol=tol, atol=tol)

@test_util.run_all_without_tensor_float_32('Tests convolutional_orthogonal_1d, which calls matmul')
class ConvolutionOrthogonal1dInitializerTest(test.TestCase):

    @test_util.run_deprecated_v1
    def testInitializerIdentical(self):
        if False:
            for i in range(10):
                print('nop')
        for dtype in [dtypes.float32, dtypes.float64]:
            init1 = init_ops.convolutional_orthogonal_1d(seed=1, dtype=dtype)
            init2 = init_ops.convolutional_orthogonal_1d(seed=1, dtype=dtype)
            self.assertTrue(identicaltest(self, init1, init2, (3, 10, 10)))

    @test_util.run_deprecated_v1
    def testInitializerDifferent(self):
        if False:
            while True:
                i = 10
        for dtype in [dtypes.float32, dtypes.float64]:
            init1 = init_ops.convolutional_orthogonal_1d(seed=1, dtype=dtype)
            init2 = init_ops.convolutional_orthogonal_1d(seed=2, dtype=dtype)
            self.assertFalse(identicaltest(self, init1, init2, (3, 10, 10)))

    @test_util.run_deprecated_v1
    def testDuplicatedInitializer(self):
        if False:
            i = 10
            return i + 15
        init = init_ops.convolutional_orthogonal_1d()
        self.assertFalse(duplicated_initializer(self, init, 1, (3, 10, 10)))

    def testInvalidDataType(self):
        if False:
            print('Hello World!')
        self.assertRaises(ValueError, init_ops.convolutional_orthogonal_1d, dtype=dtypes.string)

    def testInvalidShape(self):
        if False:
            print('Hello World!')
        init1 = init_ops.convolutional_orthogonal_1d()
        with self.session(graph=ops.Graph(), use_gpu=True):
            self.assertRaises(ValueError, init1, shape=[3, 6, 5])

    @test_util.run_deprecated_v1
    def testGain(self):
        if False:
            while True:
                i = 10
        shape = (3, 10, 10)
        for dtype in [dtypes.float32, dtypes.float64]:
            init1 = init_ops.convolutional_orthogonal_1d(seed=1, dtype=dtype)
            init2 = init_ops.convolutional_orthogonal_1d(gain=3.14, seed=1, dtype=dtype)
            with self.session(graph=ops.Graph(), use_gpu=True):
                t1 = init1(shape).eval()
                t2 = init2(shape).eval()
            self.assertAllClose(t1, t2 / 3.14)

    @test_util.run_deprecated_v1
    def testNonuniformity(self):
        if False:
            while True:
                i = 10
        value = 0
        abs_value = 0
        shape = [3, 10, 10]
        count = 70
        tol = 1e-05
        with self.session():
            for i in range(count):
                x = variable_scope.get_variable('{}'.format(i), shape=shape, initializer=init_ops.convolutional_orthogonal_1d)
                self.evaluate(x.initializer)
                y = np.sum(self.evaluate(x), axis=0)
                determinant = np.linalg.det(y)
                value += determinant
                abs_value += np.abs(determinant)
            self.assertLess(value, count - tol)
            self.assertLess(-count + tol, value)
            self.assertAllClose(abs_value, count, rtol=tol, atol=tol)

    @test_util.run_deprecated_v1
    def testShapesValues(self):
        if False:
            i = 10
            return i + 15

        def circular_pad(input_, width, kernel_size):
            if False:
                return 10
            'Pad input_ for computing (circular) convolution.\n\n      Args:\n        input_: the input tensor\n        width: the width of the tensor.\n        kernel_size: the kernel size of the filter.\n\n      Returns:\n        a tensor whose width is (width + kernel_size - 1).\n      '
            beginning = kernel_size // 2
            end = kernel_size - 1 - beginning
            tmp_up = array_ops.slice(input_, [0, width - beginning, 0], [-1, beginning, -1])
            tmp_down = array_ops.slice(input_, [0, 0, 0], [-1, end, -1])
            tmp = array_ops.concat([tmp_up, input_, tmp_down], 1)
            return tmp
        cout = 64
        shape = [10, 20, 32]
        outputs_shape = shape[0:-1] + [cout]
        dtype = dtypes.float32
        tol = 0.001
        gain = 3.14
        for kernel_size in [[1], [2], [3], [4], [5], [6]]:
            convolution = convolutional.conv1d
            inputs = random_ops.random_normal(shape, dtype=dtype)
            inputs_2norm = linalg_ops.norm(inputs)
            input_with_circular_pad = circular_pad(inputs, shape[1], kernel_size[0])
            outputs = convolution(input_with_circular_pad, padding='valid', filters=cout, kernel_size=kernel_size[0], use_bias=False, kernel_initializer=init_ops.convolutional_orthogonal_1d(gain=gain))
            outputs_2norm = linalg_ops.norm(outputs)
            ratio = outputs_2norm / inputs_2norm
            my_ops = variables.global_variables_initializer()
            with self.session():
                self.evaluate(my_ops)
                t = self.evaluate(outputs)
                self.assertAllEqual(t.shape, outputs_shape)
                self.assertAllClose(self.evaluate(ratio), gain, rtol=tol, atol=tol)

class ConvolutionOrthogonal2dInitializerTest(test.TestCase):

    @test_util.run_deprecated_v1
    def testInitializerIdentical(self):
        if False:
            i = 10
            return i + 15
        for dtype in [dtypes.float32, dtypes.float64]:
            init1 = init_ops.convolutional_orthogonal_2d(seed=1, dtype=dtype)
            init2 = init_ops.convolutional_orthogonal_2d(seed=1, dtype=dtype)
            self.assertTrue(identicaltest(self, init1, init2, (3, 3, 10, 10)))

    @test_util.run_deprecated_v1
    def testInitializerDifferent(self):
        if False:
            print('Hello World!')
        for dtype in [dtypes.float32, dtypes.float64]:
            init1 = init_ops.convolutional_orthogonal_2d(seed=1, dtype=dtype)
            init2 = init_ops.convolutional_orthogonal_2d(seed=2, dtype=dtype)
            self.assertFalse(identicaltest(self, init1, init2, (3, 3, 10, 10)))

    @test_util.run_deprecated_v1
    def testDuplicatedInitializer(self):
        if False:
            while True:
                i = 10
        init = init_ops.convolutional_orthogonal_2d()
        self.assertFalse(duplicated_initializer(self, init, 1, (3, 3, 10, 10)))

    def testInvalidDataType(self):
        if False:
            print('Hello World!')
        self.assertRaises(ValueError, init_ops.convolutional_orthogonal_2d, dtype=dtypes.string)

    def testInvalidShape(self):
        if False:
            return 10
        init1 = init_ops.convolutional_orthogonal_2d()
        with self.session(graph=ops.Graph(), use_gpu=True):
            self.assertRaises(ValueError, init1, shape=[3, 3, 6, 5])

    @test_util.run_deprecated_v1
    def testGain(self):
        if False:
            while True:
                i = 10
        shape = (3, 3, 10, 10)
        for dtype in [dtypes.float32, dtypes.float64]:
            init1 = init_ops.convolutional_orthogonal_2d(seed=1, dtype=dtype)
            init2 = init_ops.convolutional_orthogonal_2d(gain=3.14, seed=1, dtype=dtype)
            with self.session(graph=ops.Graph(), use_gpu=True):
                t1 = init1(shape).eval()
                t2 = init2(shape).eval()
            self.assertAllClose(t1, t2 / 3.14)

    @test_util.run_deprecated_v1
    def testShapesValues(self):
        if False:
            for i in range(10):
                print('nop')

        def circular_pad(input_, width, kernel_size):
            if False:
                while True:
                    i = 10
            'Pad input_ for computing (circular) convolution.\n\n      Args:\n        input_: the input tensor\n        width: the width of the tensor.\n        kernel_size: the kernel size of the filter.\n\n      Returns:\n        a tensor whose width is (width + kernel_size - 1).\n      '
            beginning = kernel_size // 2
            end = kernel_size - 1 - beginning
            tmp_up = array_ops.slice(input_, [0, width - beginning, 0, 0], [-1, beginning, width, -1])
            tmp_down = array_ops.slice(input_, [0, 0, 0, 0], [-1, end, width, -1])
            tmp = array_ops.concat([tmp_up, input_, tmp_down], 1)
            new_width = width + kernel_size - 1
            tmp_left = array_ops.slice(tmp, [0, 0, width - beginning, 0], [-1, new_width, beginning, -1])
            tmp_right = array_ops.slice(tmp, [0, 0, 0, 0], [-1, new_width, end, -1])
            final = array_ops.concat([tmp_left, tmp, tmp_right], 2)
            return final
        cout = 45
        shape = [64, 28, 28, 32]
        outputs_shape = shape[0:-1] + [cout]
        dtype = dtypes.float32
        tol = 0.001
        gain = 3.14
        for kernel_size in [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]:
            convolution = convolutional.conv2d
            inputs = random_ops.random_normal(shape, dtype=dtype)
            inputs_2norm = linalg_ops.norm(inputs)
            input_with_circular_pad = circular_pad(inputs, shape[1], kernel_size[0])
            outputs = convolution(input_with_circular_pad, padding='valid', filters=cout, kernel_size=kernel_size, use_bias=False, kernel_initializer=init_ops.convolutional_orthogonal_2d(gain=gain))
            outputs_2norm = linalg_ops.norm(outputs)
            ratio = outputs_2norm / inputs_2norm
            my_ops = variables.global_variables_initializer()
            with self.session():
                self.evaluate(my_ops)
                t = self.evaluate(outputs)
                self.assertAllEqual(t.shape, outputs_shape)
                self.assertAllClose(self.evaluate(ratio), gain, rtol=tol, atol=tol)

@test_util.run_all_without_tensor_float_32('Tests convolutional_orthogonal_3d, which calls matmul')
class ConvolutionOrthogonal3dInitializerTest(test.TestCase):

    @test_util.run_deprecated_v1
    def testInitializerIdentical(self):
        if False:
            print('Hello World!')
        for dtype in [dtypes.float32, dtypes.float64]:
            init1 = init_ops.convolutional_orthogonal_3d(seed=1, dtype=dtype)
            init2 = init_ops.convolutional_orthogonal_3d(seed=1, dtype=dtype)
            self.assertTrue(identicaltest(self, init1, init2, (3, 3, 3, 10, 10)))

    @test_util.run_deprecated_v1
    def testInitializerDifferent(self):
        if False:
            print('Hello World!')
        for dtype in [dtypes.float32, dtypes.float64]:
            init1 = init_ops.convolutional_orthogonal_3d(seed=1, dtype=dtype)
            init2 = init_ops.convolutional_orthogonal_3d(seed=2, dtype=dtype)
            self.assertFalse(identicaltest(self, init1, init2, (3, 3, 3, 10, 10)))

    @test_util.run_deprecated_v1
    def testDuplicatedInitializer(self):
        if False:
            for i in range(10):
                print('nop')
        init = init_ops.convolutional_orthogonal_3d()
        self.assertFalse(duplicated_initializer(self, init, 1, (3, 3, 3, 10, 10)))

    def testInvalidDataType(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(ValueError, init_ops.convolutional_orthogonal_3d, dtype=dtypes.string)

    def testInvalidShape(self):
        if False:
            return 10
        init1 = init_ops.convolutional_orthogonal_3d()
        with self.session(graph=ops.Graph(), use_gpu=True):
            self.assertRaises(ValueError, init1, shape=[3, 3, 3, 6, 5])

    @test_util.run_deprecated_v1
    def testGain(self):
        if False:
            return 10
        shape = (3, 3, 3, 10, 10)
        for dtype in [dtypes.float32, dtypes.float64]:
            init1 = init_ops.convolutional_orthogonal_3d(seed=1, dtype=dtype)
            init2 = init_ops.convolutional_orthogonal_3d(gain=3.14, seed=1, dtype=dtype)
            with self.session(graph=ops.Graph(), use_gpu=True):
                t1 = init1(shape).eval()
                t2 = init2(shape).eval()
            self.assertAllClose(t1, t2 / 3.14)

    @test_util.run_deprecated_v1
    def testNonuniformity(self):
        if False:
            i = 10
            return i + 15
        value = 0
        abs_value = 0
        shape = [3, 3, 3, 5, 5]
        count = 20
        tol = 1e-05
        with self.session():
            for i in range(count):
                x = variable_scope.get_variable('{}'.format(i), shape=shape, initializer=init_ops.convolutional_orthogonal_3d)
                self.evaluate(x.initializer)
                y = np.sum(self.evaluate(x), axis=(0, 1, 2))
                determinant = np.linalg.det(y)
                value += determinant
                abs_value += np.abs(determinant)
            self.assertLess(value, count - tol)
            self.assertLess(-count + tol, value)
            self.assertAllClose(abs_value, count, rtol=tol, atol=tol)

    @test_util.run_deprecated_v1
    def testShapesValues(self):
        if False:
            print('Hello World!')

        def circular_pad(input_, width, kernel_size):
            if False:
                return 10
            'Padding input_ for computing circular convolution.\n\n      Args:\n        input_: the input tensor\n        width: the width of the tensor.\n        kernel_size: the kernel size of the filter.\n\n      Returns:\n        a tensor whose width is (width + kernel_size - 1).\n      '
            beginning = kernel_size // 2
            end = kernel_size - 1 - beginning
            tmp_up = array_ops.slice(input_, [0, width - beginning, 0, 0, 0], [-1, beginning, -1, -1, -1])
            tmp_down = array_ops.slice(input_, [0, 0, 0, 0, 0], [-1, end, -1, -1, -1])
            tmp = array_ops.concat([tmp_up, input_, tmp_down], 1)
            tmp_left = array_ops.slice(tmp, [0, 0, width - beginning, 0, 0], [-1, -1, beginning, -1, -1])
            tmp_right = array_ops.slice(tmp, [0, 0, 0, 0, 0], [-1, -1, end, -1, -1])
            tmp = array_ops.concat([tmp_left, tmp, tmp_right], 2)
            tmp_front = array_ops.slice(tmp, [0, 0, 0, width - beginning, 0], [-1, -1, -1, beginning, -1])
            tmp_back = array_ops.slice(tmp, [0, 0, 0, 0, 0], [-1, -1, -1, end, -1])
            return array_ops.concat([tmp_front, tmp, tmp_back], 3)
        cout = 32
        shape = [1, 7, 7, 7, 16]
        outputs_shape = shape[0:-1] + [cout]
        dtype = dtypes.float32
        tol = 0.001
        gain = 3.14
        for kernel_size in [[1, 1, 1], [2, 2, 2], [3, 3, 3]]:
            convolution = convolutional.conv3d
            inputs = random_ops.random_normal(shape, dtype=dtype)
            inputs_2norm = linalg_ops.norm(inputs)
            input_with_circular_pad = circular_pad(inputs, shape[1], kernel_size[0])
            outputs = convolution(input_with_circular_pad, padding='valid', filters=cout, kernel_size=kernel_size[0], use_bias=False, kernel_initializer=init_ops.convolutional_orthogonal_3d(gain=gain))
            outputs_2norm = linalg_ops.norm(outputs)
            ratio = outputs_2norm / inputs_2norm
            my_ops = variables.global_variables_initializer()
            with self.cached_session():
                self.evaluate(my_ops)
                t = self.evaluate(outputs)
                self.assertAllEqual(t.shape, outputs_shape)
                self.assertAllClose(self.evaluate(ratio), gain, rtol=tol, atol=tol)

class IdentityInitializerTest(test.TestCase):

    def testInvalidDataType(self):
        if False:
            return 10
        self.assertRaises(ValueError, init_ops.orthogonal_initializer, dtype=dtypes.string)

    def testInvalidShape(self):
        if False:
            i = 10
            return i + 15
        init = init_ops.identity_initializer()
        with self.session(graph=ops.Graph(), use_gpu=True):
            self.assertRaises(ValueError, init, shape=[5, 7, 7])
            self.assertRaises(ValueError, init, shape=[5])
            self.assertRaises(ValueError, init, shape=[])

    @test_util.run_deprecated_v1
    def testNonSquare(self):
        if False:
            while True:
                i = 10
        init = init_ops.identity_initializer()
        shape = (10, 5)
        with self.session(graph=ops.Graph(), use_gpu=True):
            self.assertAllClose(init(shape), np.eye(*shape))

    @test_util.run_deprecated_v1
    def testGain(self):
        if False:
            while True:
                i = 10
        shape = (10, 10)
        for dtype in [dtypes.float32, dtypes.float64]:
            init_default = init_ops.identity_initializer(dtype=dtype)
            init_custom = init_ops.identity_initializer(gain=0.9, dtype=dtype)
            with self.session(graph=ops.Graph(), use_gpu=True):
                self.assertAllClose(init_default(shape), np.eye(*shape))
            with self.session(graph=ops.Graph(), use_gpu=True):
                self.assertAllClose(init_custom(shape), np.eye(*shape) * 0.9)

    @test_util.run_deprecated_v1
    def testPartitions(self):
        if False:
            for i in range(10):
                print('nop')
        shape = (10, 10)
        init = init_ops.identity_initializer()
        partitioner = partitioned_variables.variable_axis_size_partitioner(1)
        with self.session(graph=ops.Graph(), use_gpu=True):
            with variable_scope.variable_scope('foo', partitioner=partitioner, initializer=init):
                v = array_ops.identity(variable_scope.get_variable('bar', shape=shape))
            self.evaluate(variables.global_variables_initializer())
            self.assertAllClose(v, np.eye(*shape))
if __name__ == '__main__':
    test.main()