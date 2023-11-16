"""Tests for special math operations."""
import os
from absl import flags
from absl.testing import parameterized
import numpy as np
import scipy.special as sps
from tensorflow.compiler.tests import xla_test
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
flags.DEFINE_bool('vary_seed', False, 'Whether to vary the PRNG seed unpredictably.  With --runs_per_test=N, produces N iid runs.')
NUM_SAMPLES = int(1000.0)

@def_function.function(jit_compile=True)
def _igamma(a, x):
    if False:
        for i in range(10):
            print('nop')
    return math_ops.igamma(a, x)

@def_function.function(jit_compile=True)
def _igammac(a, x):
    if False:
        i = 10
        return i + 15
    return math_ops.igammac(a, x)

@def_function.function(jit_compile=True)
def _polygamma(n, x):
    if False:
        print('Hello World!')
    return math_ops.polygamma(n, x)

@def_function.function(jit_compile=True)
def _zeta(a, q):
    if False:
        i = 10
        return i + 15
    return math_ops.zeta(a, q)

def implicit_reparameterization_grad(a, x):
    if False:
        i = 10
        return i + 15
    log_prob = math_ops.xlogy(a - 1.0, x) - math_ops.lgamma(a) - x
    prob = math_ops.exp(log_prob)
    return -gen_math_ops.igamma_grad_a(a, x) / prob

@def_function.function(jit_compile=True)
def _log1p(x):
    if False:
        i = 10
        return i + 15
    return math_ops.log1p(x)

class Log1pTest(xla_test.XLATestCase, parameterized.TestCase):

    def setUp(self):
        if False:
            return 10
        if flags.FLAGS.vary_seed:
            entropy = os.urandom(64)
            answer = int.from_bytes(entropy, 'big')
            np.random.seed(answer % (2 ** 32 - 1))
        super(Log1pTest, self).setUp()

    def adjust_tolerance_for_tpu(self, dtype, rtol, atol):
        if False:
            i = 10
            return i + 15
        if self.device not in ['TPU']:
            return (rtol, atol)
        if dtype == np.float32:
            return (0.0004, 0.0)
        return (1e-10, 0.0)

    def _test_range(self, low, high, dtype, rtol, atol, is_negative=False):
        if False:
            return 10
        (rtol, atol) = self.adjust_tolerance_for_tpu(dtype, rtol, atol)
        x = np.exp(np.random.uniform(low=low, high=high, size=[NUM_SAMPLES])).astype(dtype)
        if is_negative:
            x = -x
        expected_values = np.log1p(x)
        with self.session() as sess:
            with self.test_scope():
                actual = _log1p(x)
            actual = sess.run(actual)
        self.assertAllClose(expected_values, actual, atol=atol, rtol=rtol)

    @parameterized.parameters((np.float32, 1e-07, 0.0), (np.float64, 1e-15, 0.0))
    def testSmallX(self, dtype, rtol, atol):
        if False:
            print('Hello World!')
        self._test_range(-40.0, -20.0, dtype, rtol, atol, is_negative=False)
        self._test_range(-40.0, -20.0, dtype, rtol, atol, is_negative=True)

    @parameterized.parameters((np.float32, 2e-07, 0.0), (np.float64, 1e-15, 0.0))
    def testGreaterThanNegativeTwentyExponent(self, dtype, rtol, atol):
        if False:
            i = 10
            return i + 15
        self._test_range(-20.0, -10.0, dtype, rtol, atol, is_negative=False)
        self._test_range(-20.0, -10.0, dtype, rtol, atol, is_negative=True)

    @parameterized.parameters((np.float32, 2e-07, 0.0), (np.float64, 1e-15, 0.0))
    def testGreaterThanNegativeTenExponent(self, dtype, rtol, atol):
        if False:
            for i in range(10):
                print('nop')
        self._test_range(-10.0, -5.0, dtype, rtol, atol, is_negative=False)
        self._test_range(-10.0, -5.0, dtype, rtol, atol, is_negative=True)

    @parameterized.parameters((np.float32, 2e-07, 0.0), (np.float64, 1e-15, 0.0))
    def testGreaterThanNegativeFiveExponent(self, dtype, rtol, atol):
        if False:
            return 10
        self._test_range(-5.0, -1.0, dtype, rtol, atol, is_negative=False)
        self._test_range(-5.0, -1.0, dtype, rtol, atol, is_negative=True)

    @parameterized.parameters((np.float32, 4e-07, 0.0), (np.float64, 3e-14, 0.0))
    def testXGreaterThanOneTenth(self, dtype, rtol, atol):
        if False:
            print('Hello World!')
        self._test_range(-1.0, 0.0, dtype, rtol, atol, is_negative=False)
        self._test_range(-1.0, 0.0, dtype, rtol, atol, is_negative=True)

    @parameterized.parameters((np.float32, 2e-07, 0.0), (np.float64, 2e-15, 0.0))
    def testXGreaterThanOne(self, dtype, rtol, atol):
        if False:
            i = 10
            return i + 15
        self._test_range(0.0, 3.0, dtype, rtol, atol, is_negative=False)

class ZetaTest(xla_test.XLATestCase, parameterized.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        if flags.FLAGS.vary_seed:
            entropy = os.urandom(64)
            answer = int.from_bytes(entropy, 'big')
            np.random.seed(answer % (2 ** 32 - 1))
        super(ZetaTest, self).setUp()

    def adjust_tolerance_for_tpu(self, dtype, rtol, atol):
        if False:
            print('Hello World!')
        if self.device not in ['TPU']:
            return (rtol, atol)
        if dtype == np.float32:
            return (0.02, 1e-07)
        return (0.0002, 1e-20)

    def testBadValues(self):
        if False:
            return 10
        q = np.random.uniform(low=0.3, high=20.0, size=[10])
        with self.session() as sess:
            with self.test_scope():
                y = _zeta(np.float64(1.0), q)
            actual = sess.run(y)
        self.assertTrue(np.all(np.isinf(actual)))
        with self.session() as sess:
            with self.test_scope():
                y = _zeta(np.float64(0.1), q)
            actual = sess.run(y)
        self.assertTrue(np.all(np.isnan(actual)))
        with self.session() as sess:
            with self.test_scope():
                y = _zeta([1.1, 1.2, 2.1, 2.2, 3.1], [-2.0, -1.1, -1.0, -0.5, -0.1])
            actual = sess.run(y)
        self.assertTrue(np.all(np.isnan(actual)))
        with self.session() as sess:
            with self.test_scope():
                y = _zeta([2.0, 4.0, 6.0], [0.0, -1.0, -2.0])
            actual = sess.run(y)
        self.assertTrue(np.all(np.isinf(actual)))
        with self.session() as sess:
            with self.test_scope():
                y = _zeta([3.0, 5.0, 7.0], [0.0, -1.0, -2.0])
            actual = sess.run(y)
        self.assertTrue(np.all(np.isnan(actual)))
        with self.session() as sess:
            with self.test_scope():
                y = _zeta([1.1, 2.2, 3.3], [-1.1, -1.0, 0.0])
            actual = sess.run(y)
        self.assertTrue(np.all(np.isnan(actual)))

    @parameterized.parameters((np.float32, 0.01, 1e-11), (np.float64, 0.0001, 1e-30))
    def testLargeXSmallQ(self, dtype, rtol, atol):
        if False:
            print('Hello World!')
        (rtol, atol) = self.adjust_tolerance_for_tpu(dtype, rtol, atol)
        if self.device not in ['XLA_GPU', 'XLA_CPU'] and dtype == np.float64:
            self.skipTest('Skipping test because some F64 operations are numerically unstable on TPU.')
        x = np.random.uniform(low=100.0, high=200.0, size=[NUM_SAMPLES]).astype(dtype)
        q = np.random.uniform(low=0.3, high=1.0, size=[NUM_SAMPLES]).astype(dtype)
        expected_values = sps.zeta(x, q)
        with self.session() as sess:
            with self.test_scope():
                y = _zeta(x, q)
            actual = sess.run(y)
        self.assertAllClose(expected_values, actual, atol=atol, rtol=rtol)

    @parameterized.parameters((np.float32, 0.01, 1e-11), (np.float64, 0.0001, 1e-30))
    def testSmallValues(self, dtype, rtol, atol):
        if False:
            return 10
        (rtol, atol) = self.adjust_tolerance_for_tpu(dtype, rtol, atol)
        x = np.random.uniform(low=1.1, high=10.0, size=[NUM_SAMPLES]).astype(dtype)
        q = np.random.uniform(low=np.finfo(dtype).tiny, high=1.0, size=[NUM_SAMPLES]).astype(dtype)
        expected_values = sps.zeta(x, q)
        with self.session() as sess:
            with self.test_scope():
                actual = sess.run(_zeta(x, q))
        self.assertAllClose(expected_values, actual, atol=atol, rtol=rtol)

    @parameterized.parameters((np.float32, 0.01, 1e-11), (np.float64, 0.0001, 1e-30))
    def testMediumValues(self, dtype, rtol, atol):
        if False:
            return 10
        (rtol, atol) = self.adjust_tolerance_for_tpu(dtype, rtol, atol)
        x = np.random.uniform(low=1.1, high=100.0, size=[NUM_SAMPLES]).astype(dtype)
        q = np.random.uniform(low=1.0, high=10.0, size=[NUM_SAMPLES]).astype(dtype)
        expected_values = sps.zeta(x, q)
        with self.session() as sess:
            with self.test_scope():
                actual = sess.run(_zeta(x, q))
        self.assertAllClose(expected_values, actual, atol=atol, rtol=rtol)

    @parameterized.parameters((np.float32, 0.02, 1e-05), (np.float64, 0.0001, 1e-30))
    def testLargeValues(self, dtype, rtol, atol):
        if False:
            while True:
                i = 10
        x = np.random.uniform(low=100.0, high=int(1000.0), size=[NUM_SAMPLES]).astype(dtype)
        q = np.random.uniform(low=1.0, high=int(10.0), size=[NUM_SAMPLES]).astype(dtype)
        expected_values = sps.zeta(x, q)
        with self.session() as sess:
            with self.test_scope():
                actual = sess.run(_zeta(x, q))
        self.assertAllClose(expected_values, actual, atol=atol, rtol=rtol)

class PolygammaTest(xla_test.XLATestCase, parameterized.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        if flags.FLAGS.vary_seed:
            entropy = os.urandom(64)
            answer = int.from_bytes(entropy, 'big')
            np.random.seed(answer % (2 ** 32 - 1))
        super(PolygammaTest, self).setUp()

    def adjust_tolerance_for_tpu(self, dtype, rtol, atol):
        if False:
            while True:
                i = 10
        if self.device not in ['TPU']:
            return (rtol, atol)
        if dtype == np.float32:
            return (0.02, 1e-07)
        return (0.0002, 1e-20)

    def testBadValues(self):
        if False:
            print('Hello World!')
        x = np.random.uniform(low=0.3, high=20.0, size=[10])
        with self.session() as sess:
            with self.test_scope():
                y = _polygamma(np.float64(-1.0), x)
            actual = sess.run(y)
        self.assertTrue(np.all(np.isnan(actual)))
        with self.session() as sess:
            with self.test_scope():
                y = _polygamma(np.float64(0.1), x)
            actual = sess.run(y)
        self.assertTrue(np.all(np.isnan(actual)))

    @parameterized.parameters((np.float32, 0.01, 1e-11), (np.float64, 0.0001, 1e-30))
    def testRecoverDigamma(self, dtype, rtol, atol):
        if False:
            print('Hello World!')
        (rtol, atol) = self.adjust_tolerance_for_tpu(dtype, rtol, atol)
        if self.device not in ['XLA_GPU', 'XLA_CPU'] and dtype == np.float64:
            self.skipTest('Skipping test because some F64 operations are numerically unstable on TPU.')
        x = np.random.uniform(low=0.1, high=50.0, size=[NUM_SAMPLES]).astype(dtype)
        expected_values = sps.digamma(x)
        with self.session() as sess:
            with self.test_scope():
                y = _polygamma(dtype(0.0), x)
            actual = sess.run(y)
        self.assertAllClose(expected_values, actual, atol=atol, rtol=rtol)

    @parameterized.parameters((np.float32, 0.01, 1e-11), (np.float64, 0.0001, 1e-30))
    def testSmallN(self, dtype, rtol, atol):
        if False:
            i = 10
            return i + 15
        (rtol, atol) = self.adjust_tolerance_for_tpu(dtype, rtol, atol)
        n = np.random.randint(low=1, high=5, size=[NUM_SAMPLES]).astype(dtype)
        x = np.random.uniform(low=np.finfo(dtype).tiny, high=1.0, size=[NUM_SAMPLES]).astype(dtype)
        expected_values = sps.polygamma(n, x)
        with self.session() as sess:
            with self.test_scope():
                actual = sess.run(_polygamma(n, x))
        self.assertAllClose(expected_values, actual, atol=atol, rtol=rtol)

    @parameterized.parameters((np.float32, 0.01, 1e-11), (np.float64, 0.0001, 1e-30))
    def testMediumLargeN(self, dtype, rtol, atol):
        if False:
            while True:
                i = 10
        (rtol, atol) = self.adjust_tolerance_for_tpu(dtype, rtol, atol)
        n = np.random.randint(low=5, high=10, size=[NUM_SAMPLES]).astype(dtype)
        x = np.random.uniform(low=1.0, high=10.0, size=[NUM_SAMPLES]).astype(dtype)
        expected_values = sps.polygamma(n, x)
        with self.session() as sess:
            with self.test_scope():
                actual = sess.run(_polygamma(n, x))
        self.assertAllClose(expected_values, actual, atol=atol, rtol=rtol)

class IgammaTest(xla_test.XLATestCase, parameterized.TestCase):

    def setUp(self):
        if False:
            return 10
        if flags.FLAGS.vary_seed:
            entropy = os.urandom(64)
            answer = int.from_bytes(entropy, 'big')
            np.random.seed(answer % (2 ** 32 - 1))
        super(IgammaTest, self).setUp()

    def maybe_skip_test(self, dtype):
        if False:
            return 10
        if self.device not in ['XLA_GPU', 'XLA_CPU'] and dtype == np.float64:
            self.skipTest('Skipping test because some F64 operations not supported on TPU.')

    def adjust_tolerance_for_tpu(self, dtype, rtol, atol):
        if False:
            for i in range(10):
                print('nop')
        if self.device not in ['TPU']:
            return (rtol, atol)
        if dtype == np.float32:
            return (0.02, 1e-07)
        return (0.0002, 1e-20)

    @parameterized.parameters((np.float32, 0.01, 1e-11), (np.float64, 0.0001, 1e-30))
    def testLargeXSmallA(self, dtype, rtol, atol):
        if False:
            for i in range(10):
                print('nop')
        self.maybe_skip_test(dtype)
        (rtol, atol) = self.adjust_tolerance_for_tpu(dtype, rtol, atol)
        x = np.random.uniform(low=100.0, high=200.0, size=[NUM_SAMPLES]).astype(dtype)
        a = np.random.uniform(low=0.3, high=1.0, size=[NUM_SAMPLES]).astype(dtype)
        expected_values = sps.gammainc(a, x)
        with self.session() as sess:
            with self.test_scope():
                y = _igamma(a, x)
            actual = sess.run(y)
        self.assertAllClose(expected_values, actual, atol=atol, rtol=rtol)

    @parameterized.parameters((np.float32, 0.01, 1e-11), (np.float64, 0.0001, 1e-30))
    def testSmallValues(self, dtype, rtol, atol):
        if False:
            i = 10
            return i + 15
        self.maybe_skip_test(dtype)
        (rtol, atol) = self.adjust_tolerance_for_tpu(dtype, rtol, atol)
        x = np.random.uniform(low=np.finfo(dtype).tiny, high=1.0, size=[NUM_SAMPLES]).astype(dtype)
        a = np.random.uniform(low=np.finfo(dtype).tiny, high=1.0, size=[NUM_SAMPLES]).astype(dtype)
        expected_values = sps.gammainc(a, x)
        with self.session() as sess:
            with self.test_scope():
                actual = sess.run(_igamma(a, x))
        self.assertAllClose(expected_values, actual, atol=atol, rtol=rtol)

    @parameterized.parameters((np.float32, 0.01, 1e-11), (np.float64, 0.0001, 1e-30))
    def testMediumValues(self, dtype, rtol, atol):
        if False:
            print('Hello World!')
        self.maybe_skip_test(dtype)
        (rtol, atol) = self.adjust_tolerance_for_tpu(dtype, rtol, atol)
        x = np.random.uniform(low=1.0, high=100.0, size=[NUM_SAMPLES]).astype(dtype)
        a = np.random.uniform(low=1.0, high=100.0, size=[NUM_SAMPLES]).astype(dtype)
        expected_values = sps.gammainc(a, x)
        with self.session() as sess:
            with self.test_scope():
                actual = sess.run(_igamma(a, x))
        self.assertAllClose(expected_values, actual, atol=atol, rtol=rtol)

    @parameterized.parameters((np.float32, 0.02, 1e-05), (np.float64, 0.0001, 1e-30))
    def testLargeValues(self, dtype, rtol, atol):
        if False:
            for i in range(10):
                print('nop')
        if self.device == 'TPU':
            self.skipTest('Skipping test since numerically unstable on TPU.')
        x = np.random.uniform(low=100.0, high=int(10000.0), size=[NUM_SAMPLES]).astype(dtype)
        a = np.random.uniform(low=100.0, high=int(10000.0), size=[NUM_SAMPLES]).astype(dtype)
        expected_values = sps.gammainc(a, x)
        with self.session() as sess:
            with self.test_scope():
                actual = sess.run(_igamma(a, x))
        self.assertAllClose(expected_values, actual, atol=atol, rtol=rtol)

    @parameterized.parameters((np.float32, 0.09), (np.float64, 1e-07))
    def testGradMediumValues(self, dtype, tolerance):
        if False:
            return 10
        self.maybe_skip_test(dtype)
        with self.session():
            with self.test_scope():
                x = constant_op.constant(np.random.uniform(low=1.0, high=100.0, size=[NUM_SAMPLES]).astype(dtype))
                a = constant_op.constant(np.random.uniform(low=1.0, high=100.0, size=[NUM_SAMPLES]).astype(dtype))
                f = lambda b: _igamma(b, x)
                max_error = gradient_checker_v2.max_error(*gradient_checker_v2.compute_gradient(f, x=[a], delta=0.001))
        self.assertLessEqual(max_error, tolerance)

    @parameterized.parameters((np.float32, 0.5), (np.float64, 1e-07))
    def testGradLargeValues(self, dtype, tolerance):
        if False:
            while True:
                i = 10
        self.maybe_skip_test(dtype)
        with self.session():
            with self.test_scope():
                x = constant_op.constant(np.random.uniform(low=100.0, high=int(10000.0), size=[NUM_SAMPLES]).astype(dtype))
                a = constant_op.constant(np.random.uniform(low=100.0, high=int(10000.0), size=[NUM_SAMPLES]).astype(dtype))
                f = lambda b: _igamma(b, x)
                max_error = gradient_checker_v2.max_error(*gradient_checker_v2.compute_gradient(f, x=[a], delta=0.01))
        self.assertLessEqual(max_error, tolerance)

    @parameterized.parameters((np.float32, 0.01, 1e-11), (np.float64, 0.0001, 1e-30))
    def testRandomGammaGradSmallValues(self, dtype, rtol, atol):
        if False:
            for i in range(10):
                print('nop')
        self.maybe_skip_test(dtype)
        (rtol, atol) = self.adjust_tolerance_for_tpu(dtype, rtol, atol)
        with self.session() as sess:
            with self.test_scope():
                x = constant_op.constant(np.random.uniform(low=np.finfo(dtype).tiny, high=1.0, size=[NUM_SAMPLES]).astype(dtype))
                a = constant_op.constant(np.random.uniform(low=np.finfo(dtype).tiny, high=1.0, size=[NUM_SAMPLES]).astype(dtype))
                gamma_sample_grad = gen_random_ops.random_gamma_grad(a, x)
                actual_grad = implicit_reparameterization_grad(a, x)
                (gamma_sample_grad, actual_grad) = sess.run([gamma_sample_grad, actual_grad])
                gamma_sample_grad = gamma_sample_grad[~np.logical_or(np.isnan(actual_grad), np.isinf(actual_grad))]
                actual_grad = actual_grad[~np.logical_or(np.isnan(actual_grad), np.isinf(actual_grad))]
        self.assertAllClose(actual_grad, gamma_sample_grad, atol=atol, rtol=rtol)

    @parameterized.parameters((np.float32, 0.01, 1e-11), (np.float64, 0.0001, 1e-30))
    def testRandomGammaGradMediumValues(self, dtype, rtol, atol):
        if False:
            while True:
                i = 10
        self.maybe_skip_test(dtype)
        (rtol, atol) = self.adjust_tolerance_for_tpu(dtype, rtol, atol)
        with self.session() as sess:
            with self.test_scope():
                x = constant_op.constant(np.random.uniform(low=1.0, high=10.0, size=[NUM_SAMPLES]).astype(dtype))
                a = constant_op.constant(np.random.uniform(low=1.0, high=10.0, size=[NUM_SAMPLES]).astype(dtype))
                gamma_sample_grad = gen_random_ops.random_gamma_grad(a, x)
                actual_grad = implicit_reparameterization_grad(a, x)
                (gamma_sample_grad, actual_grad) = sess.run([gamma_sample_grad, actual_grad])
                gamma_sample_grad = gamma_sample_grad[~np.logical_or(np.isnan(actual_grad), np.isinf(actual_grad))]
                actual_grad = actual_grad[~np.logical_or(np.isnan(actual_grad), np.isinf(actual_grad))]
        self.assertAllClose(actual_grad, gamma_sample_grad, atol=atol, rtol=rtol)

class IgammacTest(xla_test.XLATestCase, parameterized.TestCase):

    def setUp(self):
        if False:
            return 10
        if flags.FLAGS.vary_seed:
            entropy = os.urandom(64)
            answer = int.from_bytes(entropy, 'big')
            np.random.seed(answer % (2 ** 32 - 1))
        super(IgammacTest, self).setUp()

    def maybe_skip_test(self, dtype):
        if False:
            print('Hello World!')
        if self.device not in ['XLA_GPU', 'XLA_CPU'] and dtype == np.float64:
            self.skipTest('Skipping test because some F64 operations not supported on TPU.')

    def adjust_tolerance_for_tpu(self, dtype, rtol, atol):
        if False:
            print('Hello World!')
        if self.device not in ['TPU']:
            return (rtol, atol)
        if dtype == np.float32:
            return (0.02, 1e-07)
        return (0.0002, 1e-20)

    @parameterized.parameters((np.float32, 0.01, 1e-11), (np.float64, 0.0001, 1e-30))
    def testLargeXSmallA(self, dtype, rtol, atol):
        if False:
            return 10
        self.maybe_skip_test(dtype)
        (rtol, atol) = self.adjust_tolerance_for_tpu(dtype, rtol, atol)
        x = np.random.uniform(low=100.0, high=200.0, size=[NUM_SAMPLES]).astype(dtype)
        a = np.random.uniform(low=0.3, high=1.0, size=[NUM_SAMPLES]).astype(dtype)
        expected_values = sps.gammaincc(a, x)
        with self.session() as sess:
            with self.test_scope():
                y = _igammac(a, x)
            actual = sess.run(y)
        self.assertAllClose(expected_values, actual, atol=atol, rtol=rtol)

    @parameterized.parameters((np.float32, 0.01, 1e-11), (np.float64, 0.0001, 1e-30))
    def testSmallValues(self, dtype, rtol, atol):
        if False:
            print('Hello World!')
        self.maybe_skip_test(dtype)
        (rtol, atol) = self.adjust_tolerance_for_tpu(dtype, rtol, atol)
        x = np.random.uniform(low=np.finfo(dtype).tiny, high=1.0, size=[NUM_SAMPLES]).astype(dtype)
        a = np.random.uniform(low=np.finfo(dtype).tiny, high=1.0, size=[NUM_SAMPLES]).astype(dtype)
        expected_values = sps.gammaincc(a, x)
        with self.session() as sess:
            with self.test_scope():
                actual = sess.run(_igammac(a, x))
        self.assertAllClose(expected_values, actual, atol=atol, rtol=rtol)

    @parameterized.parameters((np.float32, 0.01, 1e-11), (np.float64, 0.0001, 1e-30))
    def testMediumValues(self, dtype, rtol, atol):
        if False:
            print('Hello World!')
        self.maybe_skip_test(dtype)
        (rtol, atol) = self.adjust_tolerance_for_tpu(dtype, rtol, atol)
        x = np.random.uniform(low=1.0, high=100.0, size=[NUM_SAMPLES]).astype(dtype)
        a = np.random.uniform(low=1.0, high=100.0, size=[NUM_SAMPLES]).astype(dtype)
        expected_values = sps.gammaincc(a, x)
        with self.session() as sess:
            with self.test_scope():
                actual = sess.run(_igammac(a, x))
        self.assertAllClose(expected_values, actual, atol=atol, rtol=rtol)

    @parameterized.parameters((np.float32, 0.02, 1e-05), (np.float64, 0.0001, 1e-30))
    def testLargeValues(self, dtype, rtol, atol):
        if False:
            for i in range(10):
                print('nop')
        if self.device == 'TPU':
            self.skipTest('Skipping test since numerically unstable on TPU.')
        x = np.random.uniform(low=100.0, high=int(10000.0), size=[NUM_SAMPLES]).astype(dtype)
        a = np.random.uniform(low=100.0, high=int(10000.0), size=[NUM_SAMPLES]).astype(dtype)
        expected_values = sps.gammaincc(a, x)
        with self.session() as sess:
            with self.test_scope():
                actual = sess.run(_igammac(a, x))
        self.assertAllClose(expected_values, actual, atol=atol, rtol=rtol)
if __name__ == '__main__':
    os.environ['XLA_FLAGS'] = '--xla_cpu_enable_fast_math=false'
    test.main()