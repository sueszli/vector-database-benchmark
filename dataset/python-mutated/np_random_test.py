"""Tests for tf numpy random number methods."""
from absl.testing import parameterized
import numpy as onp
from tensorflow.python.framework import ops
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_math_ops
from tensorflow.python.ops.numpy_ops import np_random
from tensorflow.python.platform import test

class SeedTest(test.TestCase):

    def test(self):
        if False:
            while True:
                i = 10
        np_random.seed(1)
        np_random.seed(np_dtypes.int32(1))
        with self.assertRaises(ValueError):
            np_random.seed((1, 3))

class RandomTestBase(test.TestCase, parameterized.TestCase):

    def _test(self, *args, **kw_args):
        if False:
            while True:
                i = 10
        onp_dtype = kw_args.pop('onp_dtype', None)
        allow_float64 = kw_args.pop('allow_float64', True)
        old_allow_float64 = np_dtypes.is_allow_float64()
        np_dtypes.set_allow_float64(allow_float64)
        old_func = getattr(self, 'onp_func', None)
        self.onp_func = lambda *args, **kwargs: onp.asarray(old_func(*args, **kwargs))
        np_out = self.np_func(*args, **kw_args)
        onp_out = onp.asarray(self.onp_func(*args, **kw_args))
        if onp_dtype is not None:
            onp_out = onp_out.astype(onp_dtype)
        self.assertEqual(np_out.shape, onp_out.shape)
        self.assertEqual(np_out.dtype, onp_out.dtype)
        np_dtypes.set_allow_float64(old_allow_float64)

class RandNTest(RandomTestBase):

    def setUp(self):
        if False:
            return 10
        self.np_func = np_random.randn
        self.onp_func = onp.random.randn
        super(RandNTest, self).setUp()

    @parameterized.parameters((), 2, (2, 3))
    def test_float64(self, *dims):
        if False:
            i = 10
            return i + 15
        self._test(*dims)

    @parameterized.parameters((), 2, (2,), (2, 3))
    def test_float32(self, *dims):
        if False:
            return 10
        self._test(*dims, allow_float64=False, onp_dtype=np_dtypes.float32)

class StandardNormalTest(RandomTestBase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.np_func = np_random.standard_normal
        self.onp_func = onp.random.standard_normal
        super(StandardNormalTest, self).setUp()

    @parameterized.parameters((None,), ((),), ((1,),), ((1, 2),))
    def test(self, size):
        if False:
            while True:
                i = 10
        self._test(size)

class UniformTest(RandomTestBase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.np_func = np_random.uniform
        self.onp_func = onp.random.uniform
        super(UniformTest, self).setUp()

    @parameterized.parameters(((), (), None), (1, (), None), ((), 1, None), (1, 1, None), ((1, 2), (2, 1), None), ((1, 2, 1), (2, 1, 1), (2, 2, 2)), ((), (), (2, 2, 2)))
    def test_broadcast(self, low_shape, high_shape, size):
        if False:
            i = 10
            return i + 15
        low = np_array_ops.zeros(low_shape).astype(np_dtypes.float64)
        high = np_array_ops.ones(high_shape).astype(np_dtypes.float64)
        self._test(low=low, high=high, size=size)

    def test_float32(self):
        if False:
            print('Hello World!')
        self._test(0, 1, (1, 2), allow_float64=False, onp_dtype=np_dtypes.float32)

    def test_dtype_cast(self):
        if False:
            while True:
                i = 10
        self._test(np_dtypes.int8(0), np_dtypes.uint8(1), (1, 2))

class PoissonTest(RandomTestBase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.np_func = np_random.poisson
        self.onp_func = onp.random.poisson
        super(PoissonTest, self).setUp()

    @parameterized.parameters((1.0, None), (1.0, 1), (2.0, (3, 3)))
    def test(self, lam, size):
        if False:
            return 10
        self._test(lam, size)

class RandomTest(RandomTestBase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.np_func = np_random.random
        self.onp_func = onp.random.random
        super(RandomTest, self).setUp()

    @parameterized.parameters((None,), ((),), ((1,),), ((1, 2),))
    def test(self, size):
        if False:
            while True:
                i = 10
        self._test(size)

class RandTest(RandomTestBase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.np_func = np_random.rand
        self.onp_func = onp.random.rand
        super(RandTest, self).setUp()

    @parameterized.parameters((), (1,), (1, 2))
    def test(self, *size):
        if False:
            while True:
                i = 10
        self._test(*size)

class RandIntTest(RandomTestBase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.np_func = np_random.randint
        self.onp_func = onp.random.randint
        super(RandIntTest, self).setUp()

    @parameterized.parameters((0, 1, None, 'l'), (0, 1, None, np_dtypes.int64), (0, 1, 2, np_dtypes.int32), (0, 1, (), np_dtypes.int32), (0, 1, 2, np_dtypes.int64), (0, 1, (2, 2), 'l'))
    def test(self, low, high, size, dtype):
        if False:
            return 10
        self._test(low, high, size=size, dtype=dtype)

class RandNDistriutionTest(test.TestCase):

    def assertNotAllClose(self, a, b, **kwargs):
        if False:
            i = 10
            return i + 15
        try:
            self.assertAllClose(a, b, **kwargs)
        except AssertionError:
            return
        raise AssertionError('The two values are close at all %d elements' % np_array_ops.size(a))

    def testDistribution(self):
        if False:
            i = 10
            return i + 15

        def run_test(*args):
            if False:
                return 10
            num_samples = 1000
            tol = 0.1
            np_random.seed(10)
            outputs = [np_random.randn(*args) for _ in range(num_samples)]
            for output in outputs:
                self.assertEqual(output.shape, tuple(args))
                default_dtype = np_dtypes.float64 if np_dtypes.is_allow_float64() else np_dtypes.float32
                self.assertEqual(output.dtype.as_numpy_dtype, default_dtype)
            if np_array_ops.prod(args):
                outputs = [output.tolist() for output in outputs]
                mean = np_array_ops.mean(outputs, axis=0)
                stddev = np_array_ops.std(outputs, axis=0)
                self.assertAllClose(mean, np_array_ops.zeros(args), atol=tol)
                self.assertAllClose(stddev, np_array_ops.ones(args), atol=tol)
                np_random.seed(20)
                diff_seed_outputs = [np_random.randn(*args).tolist() for _ in range(num_samples)]
                self.assertNotAllClose(outputs, diff_seed_outputs)
                np_random.seed(10)
                same_seed_outputs = [np_random.randn(*args).tolist() for _ in range(num_samples)]
                self.assertAllClose(outputs, same_seed_outputs)
        run_test()
        run_test(0)
        run_test(1)
        run_test(5)
        run_test(2, 3)
        run_test(0, 2, 3)
        run_test(2, 0, 3)
        run_test(2, 3, 0)
        run_test(2, 3, 5)
if __name__ == '__main__':
    ops.enable_eager_execution()
    np_math_ops.enable_numpy_methods_on_tensor()
    test.main()