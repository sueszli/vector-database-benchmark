"""Tests for tensorflow.ops.random_grad."""
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_stateless_random_ops_v2
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_grad
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging

class AddLeadingUnitDimensionsTest(test.TestCase):

    def testBasic(self):
        if False:
            return 10
        ret = random_grad.add_leading_unit_dimensions(array_ops.ones([3, 2, 1]), 3)
        self.assertAllEqual(ret.shape, [1, 1, 1, 3, 2, 1])

    def testZeroExtraDimensions(self):
        if False:
            i = 10
            return i + 15
        ret = random_grad.add_leading_unit_dimensions(array_ops.ones([3, 2, 1]), 0)
        self.assertAllEqual(ret.shape, [3, 2, 1])

    def testScalarInput(self):
        if False:
            for i in range(10):
                print('nop')
        ret = random_grad.add_leading_unit_dimensions(1.0, 2)
        self.assertAllEqual(ret.shape, [1, 1])

    @test_util.run_deprecated_v1
    def testUnknownShape(self):
        if False:
            print('Hello World!')
        x = array_ops.placeholder(dtypes.float32)
        num_dimensions = array_ops.placeholder(dtypes.int32)
        ret = random_grad.add_leading_unit_dimensions(x, num_dimensions)
        with self.cached_session() as sess:
            ret_val = sess.run(ret, {x: np.ones([2, 2]), num_dimensions: 2})
        self.assertAllEqual(ret_val.shape, [1, 1, 2, 2])

class RandomGammaGradTest(test.TestCase):
    """Tests for derivative of a sample ~ Gamma(alpha, beta) wrt alpha and beta.

  The sample is an "implicit" function of alpha, beta and the independent random
  noise u. The derivatives we are looking for are
  d sample(alpha, beta, u) / dalpha (and dbeta).

  The derivative w.r.t. beta is computed by the standard automatic
  differentiation, so we trust that it is computed correctly.

  The derivative w.r.t. alpha is computed by Eigen function, so we test it in
  several ways. Unfortunately, the standard derivative checking by perturbing
  the parameter is impossible here, because we cannot fix the value of u
  in the random sampler. Instead, we compare the derivative for the given pair
  of (sample, alpha) to the values computed in various ways, and also check
  some statistical properties of the derivative.
  """

    @test_util.run_deprecated_v1
    def testGradientsShape(self):
        if False:
            for i in range(10):
                print('nop')
        shape = [2, 3]
        alpha = array_ops.ones([2, 2])
        beta = array_ops.ones([1, 2])
        sample = random_ops.random_gamma(shape, alpha, beta, seed=12345)
        (grads_alpha, grads_beta) = gradients_impl.gradients(sample, [alpha, beta])
        self.assertAllEqual(grads_alpha.shape, alpha.shape)
        self.assertAllEqual(grads_beta.shape, beta.shape)

    @test_util.run_deprecated_v1
    def testGradientsShapeWithOneSamplePerParameter(self):
        if False:
            return 10
        shape = []
        alpha = array_ops.ones([2, 2])
        beta = array_ops.ones([1, 2])
        sample = random_ops.random_gamma(shape, alpha, beta, seed=12345)
        (grads_alpha, grads_beta) = gradients_impl.gradients(sample, [alpha, beta])
        self.assertAllEqual(grads_alpha.shape, alpha.shape)
        self.assertAllEqual(grads_beta.shape, beta.shape)

    @test_util.run_deprecated_v1
    def testGradientsUnknownShape(self):
        if False:
            for i in range(10):
                print('nop')
        shape = array_ops.placeholder(dtypes.int32)
        alpha = array_ops.placeholder(dtypes.float32)
        beta = array_ops.placeholder(dtypes.float32)
        sample = random_ops.random_gamma(shape, alpha, beta, seed=12345)
        (grads_alpha, grads_beta) = gradients_impl.gradients(sample, [alpha, beta])
        alpha_val = np.ones([1, 2])
        beta_val = np.ones([2, 1])
        with self.cached_session() as sess:
            (grads_alpha_val, grads_beta_val) = sess.run([grads_alpha, grads_beta], {alpha: alpha_val, beta: beta_val, shape: [2, 1]})
        self.assertAllEqual(grads_alpha_val.shape, alpha_val.shape)
        self.assertAllEqual(grads_beta_val.shape, beta_val.shape)

    def _testCompareToExplicitDerivative(self, dtype):
        if False:
            print('Hello World!')
        'Compare to the explicit reparameterization derivative.\n\n    Verifies that the computed derivative satisfies\n    dsample / dalpha = d igammainv(alpha, u) / dalpha,\n    where u = igamma(alpha, sample).\n\n    Args:\n      dtype: TensorFlow dtype to perform the computations in.\n    '
        delta = 0.001
        np_dtype = dtype.as_numpy_dtype
        try:
            from scipy import misc
            from scipy import special
            alpha_val = np.logspace(-2, 3, dtype=np_dtype)
            alpha = constant_op.constant(alpha_val)
            sample = random_ops.random_gamma([], alpha, np_dtype(1.0), dtype=dtype, seed=12345)
            actual = gradients_impl.gradients(sample, alpha)[0]
            (sample_val, actual_val) = self.evaluate((sample, actual))
            u = special.gammainc(alpha_val, sample_val)
            expected_val = misc.derivative(lambda alpha_prime: special.gammaincinv(alpha_prime, u), alpha_val, dx=delta * alpha_val)
            self.assertAllClose(actual_val, expected_val, rtol=0.001, atol=0.001)
        except ImportError as e:
            tf_logging.warn('Cannot use special functions in a test: %s' % str(e))

    @test_util.run_deprecated_v1
    def testCompareToExplicitDerivativeFloat(self):
        if False:
            while True:
                i = 10
        self._testCompareToExplicitDerivative(dtypes.float32)

    @test_util.run_deprecated_v1
    def testCompareToExplicitDerivativeDouble(self):
        if False:
            while True:
                i = 10
        self._testCompareToExplicitDerivative(dtypes.float64)

    def _testCompareToImplicitDerivative(self, dtype):
        if False:
            print('Hello World!')
        "Compare to the implicit reparameterization derivative.\n\n    Let's derive the formula we compare to.\n\n    Start from the fact that CDF maps a random variable to the Uniform\n    random variable:\n      igamma(alpha, sample) = u, where u ~ Uniform(0, 1).\n\n    Apply d / dalpha to both sides:\n      d igamma(alpha, sample) / dalpha\n          + d igamma(alpha, sample) / dsample * dsample/dalpha  = 0\n      d igamma(alpha, sample) / dalpha\n          + d igamma(alpha, sample) / dsample * dsample / dalpha = 0\n      dsample/dalpha = - (d igamma(alpha, sample) / dalpha)\n                        / d igamma(alpha, sample) / dsample\n\n    This is the equation (8) of https://arxiv.org/abs/1805.08498\n\n    Args:\n      dtype: TensorFlow dtype to perform the computations in.\n    "
        np_dtype = dtype.as_numpy_dtype
        alpha = constant_op.constant(np.logspace(-2, 3, dtype=np_dtype))
        sample = random_ops.random_gamma([], alpha, np_dtype(1.0), dtype=dtype, seed=12345)
        actual = gradients_impl.gradients(sample, alpha)[0]
        sample_sg = array_ops.stop_gradient(sample)
        cdf = math_ops.igamma(alpha, sample_sg)
        (dcdf_dalpha, dcdf_dsample) = gradients_impl.gradients(cdf, [alpha, sample_sg])
        expected = -dcdf_dalpha / dcdf_dsample
        (actual_val, expected_val) = self.evaluate((actual, expected))
        self.assertAllClose(actual_val, expected_val, rtol=0.001, atol=0.001)

    @test_util.run_deprecated_v1
    def testCompareToImplicitDerivativeFloat(self):
        if False:
            while True:
                i = 10
        self._testCompareToImplicitDerivative(dtypes.float32)

    @test_util.run_deprecated_v1
    def testCompareToImplicitDerivativeDouble(self):
        if False:
            while True:
                i = 10
        self._testCompareToImplicitDerivative(dtypes.float64)

    @test_util.run_deprecated_v1
    def testAverageAlphaGradient(self):
        if False:
            for i in range(10):
                print('nop')
        'Statistical test for the gradient.\n\n    Using the equation (5) of https://arxiv.org/abs/1805.08498, we have\n      1 = d/dalpha E_{sample ~ Gamma(alpha, 1)} sample\n        = E_{sample ~ Gamma(alpha, 1)} dsample/dalpha.\n    Here we verify that the rhs is fairly close to one.\n    The convergence speed is not great, so we use many samples and loose bounds.\n    '
        num_samples = 10000
        alpha = constant_op.constant([0.8, 10.0, 1000.0], dtype=dtypes.float32)
        sample = random_ops.random_gamma([num_samples], alpha, seed=12345)
        mean_sample = math_ops.reduce_mean(sample, axis=0)
        dsample_dalpha = gradients_impl.gradients(mean_sample, alpha)[0]
        dsample_dalpha_val = self.evaluate(dsample_dalpha)
        self.assertAllClose(dsample_dalpha_val, [1.0] * 3, atol=0.1, rtol=0.1)

    @test_util.run_deprecated_v1
    def testQuadraticLoss(self):
        if False:
            while True:
                i = 10
        'Statistical test for the gradient.\n\n    The equation (5) of https://arxiv.org/abs/1805.08498 says\n      d/dalpha E_{sample ~ Gamma(alpha, 1)} f(sample)\n        = E_{sample ~ Gamma(alpha, 1)} df(sample)/dalpha.\n\n    Choose a quadratic loss function f(sample) = (sample - t)^2.\n    Then, the lhs can be computed analytically:\n      d/dalpha E_{sample ~ Gamma(alpha, 1)} f(sample)\n        = d/dalpha [ (alpha + alpha^2) - 2 * t * alpha + t^2 ]\n        = 1 + 2 * alpha - 2 * t.\n\n    We compare the Monte-Carlo estimate of the expectation with the\n    true gradient.\n    '
        num_samples = 10000
        t = 0.3
        alpha = 0.5
        expected = 1 + 2 * alpha - 2 * t
        alpha = constant_op.constant(alpha)
        sample = random_ops.random_gamma([num_samples], alpha, 1.0, seed=12345)
        loss = math_ops.reduce_mean(math_ops.square(sample - t))
        dloss_dalpha = gradients_impl.gradients(loss, alpha)[0]
        dloss_dalpha_val = self.evaluate(dloss_dalpha)
        self.assertAllClose(expected, dloss_dalpha_val, atol=0.1, rtol=0.1)

    @test_util.run_deprecated_v1
    def testQuadraticLossV3(self):
        if False:
            return 10
        'Statistical test for the gradient.\n\n    This is the same test as in testQuadraticLoss but for\n    StatelessRandomGammaV3.\n    '
        shape = constant_op.constant([10000])
        t = 0.3
        alpha = constant_op.constant(0.5, dtype=dtypes.float32)
        key = constant_op.constant([0], dtype=dtypes.uint64)
        counter = constant_op.constant([10, 20], dtype=dtypes.uint64)
        alg = constant_op.constant(1)
        expected = 1 + 2 * alpha - 2 * t
        sample = gen_stateless_random_ops_v2.stateless_random_gamma_v3(shape=shape, key=key, counter=counter, alg=alg, alpha=alpha)
        loss = math_ops.reduce_mean(math_ops.square(sample - t))
        dloss_dalpha = gradients_impl.gradients(loss, alpha)[0]
        dloss_dalpha_val = self.evaluate(dloss_dalpha)
        self.assertAllClose(expected, dloss_dalpha_val, atol=0.1, rtol=0.1)
if __name__ == '__main__':
    test.main()