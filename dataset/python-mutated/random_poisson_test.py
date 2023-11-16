"""Tests for tensorflow.ops.random_ops.random_poisson."""
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.kernel_tests.random import util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
_SUPPORTED_DTYPES = (dtypes.float16, dtypes.float32, dtypes.float64, dtypes.int32, dtypes.int64)

class RandomPoissonTest(test.TestCase):
    """This is a large test due to the moments computation taking some time."""

    def _Sampler(self, num, lam, dtype, use_gpu, seed=None):
        if False:
            i = 10
            return i + 15

        def func():
            if False:
                i = 10
                return i + 15
            with self.session(use_gpu=use_gpu, graph=ops.Graph()) as sess:
                rng = random_ops.random_poisson(lam, [num], dtype=dtype, seed=seed)
                ret = np.empty([10, num])
                for i in range(10):
                    ret[i, :] = self.evaluate(rng)
            return ret
        return func

    def testMoments(self):
        if False:
            while True:
                i = 10
        try:
            from scipy import stats
        except ImportError as e:
            tf_logging.warn('Cannot test moments: %s', e)
            return
        z_limit = 6.0
        for dt in _SUPPORTED_DTYPES:
            for stride in (0, 4, 10):
                for lam in (3.0, 20):
                    max_moment = 5
                    sampler = self._Sampler(10000, lam, dt, use_gpu=False, seed=12345)
                    z_scores = util.test_moment_matching(sampler(), max_moment, stats.poisson(lam), stride=stride)
                    self.assertAllLess(z_scores, z_limit)

    @test_util.run_deprecated_v1
    def testCPUGPUMatch(self):
        if False:
            for i in range(10):
                print('nop')
        for dt in _SUPPORTED_DTYPES:
            results = {}
            for use_gpu in [False, True]:
                sampler = self._Sampler(1000, 1.0, dt, use_gpu=use_gpu, seed=12345)
                results[use_gpu] = sampler()
            if dt == dtypes.float16:
                self.assertAllClose(results[False], results[True], rtol=0.001, atol=0.001)
            else:
                self.assertAllClose(results[False], results[True], rtol=1e-06, atol=1e-06)

    @test_util.run_deprecated_v1
    def testSeed(self):
        if False:
            return 10
        for dt in (dtypes.float16, dtypes.float32, dtypes.float64):
            sx = self._Sampler(1000, 1.0, dt, use_gpu=True, seed=345)
            sy = self._Sampler(1000, 1.0, dt, use_gpu=True, seed=345)
            self.assertAllEqual(sx(), sy())

    @test_util.run_deprecated_v1
    def testNoCSE(self):
        if False:
            while True:
                i = 10
        'CSE = constant subexpression eliminator.\n\n    SetIsStateful() should prevent two identical random ops from getting\n    merged.\n    '
        for dtype in (dtypes.float16, dtypes.float32, dtypes.float64):
            with self.cached_session():
                rnd1 = random_ops.random_poisson(2.0, [24], dtype=dtype)
                rnd2 = random_ops.random_poisson(2.0, [24], dtype=dtype)
                diff = rnd2 - rnd1
                self.assertGreaterEqual(np.linalg.norm(diff.eval()), 1)

    def testZeroShape(self):
        if False:
            while True:
                i = 10
        with self.cached_session():
            rnd = random_ops.random_poisson([], [], seed=12345)
            self.assertEqual([0], rnd.get_shape().as_list())
            self.assertAllClose(np.array([], dtype=np.float32), self.evaluate(rnd))

    @test_util.run_deprecated_v1
    def testShape(self):
        if False:
            for i in range(10):
                print('nop')
        rnd = random_ops.random_poisson(2.0, [150], seed=12345)
        self.assertEqual([150], rnd.get_shape().as_list())
        rnd = random_ops.random_poisson(lam=array_ops.ones([1, 2, 3]), shape=[150], seed=12345)
        self.assertEqual([150, 1, 2, 3], rnd.get_shape().as_list())
        rnd = random_ops.random_poisson(lam=array_ops.ones([1, 2, 3]), shape=[20, 30], seed=12345)
        self.assertEqual([20, 30, 1, 2, 3], rnd.get_shape().as_list())
        rnd = random_ops.random_poisson(lam=array_ops.placeholder(dtypes.float32, shape=(2,)), shape=[12], seed=12345)
        self.assertEqual([12, 2], rnd.get_shape().as_list())
        rnd = random_ops.random_poisson(lam=array_ops.ones([7, 3]), shape=array_ops.placeholder(dtypes.int32, shape=(1,)), seed=12345)
        self.assertEqual([None, 7, 3], rnd.get_shape().as_list())
        rnd = random_ops.random_poisson(lam=array_ops.ones([9, 6]), shape=array_ops.placeholder(dtypes.int32, shape=(3,)), seed=12345)
        self.assertEqual([None, None, None, 9, 6], rnd.get_shape().as_list())
        rnd = random_ops.random_poisson(lam=array_ops.placeholder(dtypes.float32), shape=array_ops.placeholder(dtypes.int32), seed=12345)
        self.assertIs(None, rnd.get_shape().ndims)
        rnd = random_ops.random_poisson(lam=array_ops.placeholder(dtypes.float32), shape=[50], seed=12345)
        self.assertIs(None, rnd.get_shape().ndims)

    @test_util.run_deprecated_v1
    def testDTypeCombinationsV2(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests random_poisson_v2() for all supported dtype combinations.'
        with self.cached_session():
            for lam_dt in _SUPPORTED_DTYPES:
                for out_dt in _SUPPORTED_DTYPES:
                    random_ops.random_poisson(constant_op.constant([1], dtype=lam_dt), [10], dtype=out_dt).eval()

    @test_util.run_deprecated_v1
    def testInfRate(self):
        if False:
            return 10
        sample = random_ops.random_poisson(shape=[2], lam=np.inf)
        self.assertAllEqual([np.inf, np.inf], self.evaluate(sample))

    def testSizeTooLarge(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError), 'overflow'):
            rate = constant_op.constant(1.0, shape=(4, 4, 4, 4, 4))
            self.evaluate(random_ops.random_poisson(shape=[46902, 51188, 34063, 59195], lam=rate))
if __name__ == '__main__':
    test.main()