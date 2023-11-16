"""Tests for tensorflow.ops.random_ops.random_gamma."""
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.kernel_tests.random import util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging

class RandomGammaTest(test.TestCase):
    """This is a medium test due to the moments computation taking some time."""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(137)
        random_seed.set_random_seed(137)

    def _Sampler(self, num, alpha, beta, dtype, use_gpu=True, seed=None):
        if False:
            return 10

        def func():
            if False:
                return 10
            with self.session(use_gpu=use_gpu, graph=ops.Graph()) as sess:
                rng = random_ops.random_gamma([num], alpha, beta=beta, dtype=dtype, seed=seed)
                ret = np.empty([10, num])
                for i in range(10):
                    ret[i, :] = self.evaluate(rng)
            return ret
        return func

    def testNpDtypes(self):
        if False:
            for i in range(10):
                print('nop')
        self.evaluate(random_ops.random_gamma([5], alpha=np.ones([2, 1, 3]), beta=np.ones([3]), dtype=np.float32))

    def testEmptySamplingNoError(self):
        if False:
            i = 10
            return i + 15
        self.evaluate(random_ops.random_gamma([5], alpha=np.ones([2, 0, 3]), beta=np.ones([3]), dtype=dtypes.float32))

    @test_util.run_deprecated_v1
    def testMomentsFloat32(self):
        if False:
            while True:
                i = 10
        self._testMoments(dtypes.float32)

    @test_util.run_deprecated_v1
    def testMomentsFloat64(self):
        if False:
            i = 10
            return i + 15
        self._testMoments(dtypes.float64)

    def _testMoments(self, dt):
        if False:
            while True:
                i = 10
        try:
            from scipy import stats
        except ImportError as e:
            tf_logging.warn('Cannot test moments: %s' % e)
            return
        z_limit = 6.0
        for stride in (0, 1, 4, 17):
            alphas = [0.2, 1.0, 3.0]
            if dt == dtypes.float64:
                alphas = [0.01] + alphas
            for alpha in alphas:
                for scale in (9, 17):
                    max_moment = min(6, scale // 2)
                    sampler = self._Sampler(20000, alpha, 1 / scale, dt, seed=12345)
                    z_scores = util.test_moment_matching(sampler(), max_moment, stats.gamma(alpha, scale=scale), stride=stride)
                    self.assertAllLess(z_scores, z_limit)

    def _testZeroDensity(self, alpha):
        if False:
            return 10
        "Zero isn't in the support of the gamma distribution.\n\n    But quantized floating point math has its limits.\n    TODO(bjp): Implement log-gamma sampler for small-shape distributions.\n\n    Args:\n      alpha: float shape value to test\n    "
        try:
            from scipy import stats
        except ImportError as e:
            tf_logging.warn('Cannot test zero density proportions: %s' % e)
            return
        allowable_zeros = {dtypes.float16: stats.gamma(alpha).cdf(np.finfo(np.float16).tiny), dtypes.float32: stats.gamma(alpha).cdf(np.finfo(np.float32).tiny), dtypes.float64: stats.gamma(alpha).cdf(np.finfo(np.float64).tiny)}
        failures = []
        for dt in (dtypes.float16, dtypes.float32, dtypes.float64):
            sampler = self._Sampler(10000, alpha, 1.0, dt, seed=12345)
            x = sampler()
            allowable = allowable_zeros[dt] * x.size
            allowable = allowable * 2 if allowable < 10 else allowable * 1.05
            if np.sum(x <= 0) > allowable:
                failures += [dt]
        self.assertEqual([], failures)

    def testNonZeroSmallShape(self):
        if False:
            while True:
                i = 10
        self._testZeroDensity(0.01)

    def testNonZeroSmallishShape(self):
        if False:
            for i in range(10):
                print('nop')
        self._testZeroDensity(0.35)

    def testDistinct(self):
        if False:
            return 10
        for dt in (dtypes.float16, dtypes.float32, dtypes.float64):
            sampler = self._Sampler(1000, 2.0, 1.0, dt)
            x = sampler()
            y = sampler()
            count = (x == y).sum()
            count_limit = 20 if dt == dtypes.float16 else 10
            self.assertLess(count, count_limit)

    @test_util.run_deprecated_v1
    def testCPUGPUMatch(self):
        if False:
            print('Hello World!')
        for dt in (dtypes.float16, dtypes.float32, dtypes.float64):
            results = {}
            for use_gpu in [False, True]:
                sampler = self._Sampler(1000, 0.0, 1.0, dt, use_gpu=use_gpu, seed=12345)
                results[use_gpu] = sampler()
            if dt == dtypes.float16:
                self.assertAllClose(results[False], results[True], rtol=0.001, atol=0.001)
            else:
                self.assertAllClose(results[False], results[True], rtol=1e-06, atol=1e-06)

    def testSeed(self):
        if False:
            while True:
                i = 10
        for dt in (dtypes.float16, dtypes.float32, dtypes.float64):
            sx = self._Sampler(1000, 0.0, 1.0, dt, seed=345)
            sy = self._Sampler(1000, 0.0, 1.0, dt, seed=345)
            self.assertAllEqual(sx(), sy())

    @test_util.run_deprecated_v1
    def testNoCSE(self):
        if False:
            for i in range(10):
                print('nop')
        'CSE = constant subexpression eliminator.\n\n    SetIsStateful() should prevent two identical random ops from getting\n    merged.\n    '
        for dtype in (dtypes.float16, dtypes.float32, dtypes.float64):
            with self.cached_session():
                rnd1 = random_ops.random_gamma([24], 2.0, dtype=dtype)
                rnd2 = random_ops.random_gamma([24], 2.0, dtype=dtype)
                diff = rnd2 - rnd1
                self.assertGreater(np.linalg.norm(diff.eval()), 0.1)

    @test_util.run_deprecated_v1
    def testShape(self):
        if False:
            while True:
                i = 10
        rnd = random_ops.random_gamma([150], 2.0)
        self.assertEqual([150], rnd.get_shape().as_list())
        rnd = random_ops.random_gamma([150], 2.0, beta=[3.0, 4.0])
        self.assertEqual([150, 2], rnd.get_shape().as_list())
        rnd = random_ops.random_gamma([150], array_ops.ones([1, 2, 3]))
        self.assertEqual([150, 1, 2, 3], rnd.get_shape().as_list())
        rnd = random_ops.random_gamma([20, 30], array_ops.ones([1, 2, 3]))
        self.assertEqual([20, 30, 1, 2, 3], rnd.get_shape().as_list())
        rnd = random_ops.random_gamma([123], array_ops.placeholder(dtypes.float32, shape=(2,)))
        self.assertEqual([123, 2], rnd.get_shape().as_list())
        rnd = random_ops.random_gamma(array_ops.placeholder(dtypes.int32, shape=(1,)), array_ops.ones([7, 3]))
        self.assertEqual([None, 7, 3], rnd.get_shape().as_list())
        rnd = random_ops.random_gamma(array_ops.placeholder(dtypes.int32, shape=(3,)), array_ops.ones([9, 6]))
        self.assertEqual([None, None, None, 9, 6], rnd.get_shape().as_list())
        rnd = random_ops.random_gamma(array_ops.placeholder(dtypes.int32), array_ops.placeholder(dtypes.float32))
        self.assertIs(None, rnd.get_shape().ndims)
        rnd = random_ops.random_gamma([50], array_ops.placeholder(dtypes.float32))
        self.assertIs(None, rnd.get_shape().ndims)

    @test_util.run_deprecated_v1
    def testPositive(self):
        if False:
            for i in range(10):
                print('nop')
        n = int(10000.0)
        for dt in [dtypes.float16, dtypes.float32, dtypes.float64]:
            with self.cached_session():
                x = random_ops.random_gamma(shape=[n], alpha=0.001, dtype=dt, seed=0)
                self.assertEqual(0, math_ops.reduce_sum(math_ops.cast(math_ops.less_equal(x, 0.0), dtype=dtypes.int64)).eval())

    def testSizeTooLarge(self):
        if False:
            print('Hello World!')
        if context.executing_eagerly():
            with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError), 'overflow'):
                rate = constant_op.constant(1.0, shape=(4, 4, 4, 4, 4))
                self.evaluate(random_ops.random_gamma(shape=[46902, 51188, 34063, 59195], alpha=rate))
if __name__ == '__main__':
    test.main()