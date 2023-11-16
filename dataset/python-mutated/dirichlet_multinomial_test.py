import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import dirichlet_multinomial
from tensorflow.python.platform import test
ds = dirichlet_multinomial

class DirichletMultinomialTest(test.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self._rng = np.random.RandomState(42)

    @test_util.run_deprecated_v1
    def testSimpleShapes(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            alpha = np.random.rand(3)
            dist = ds.DirichletMultinomial(1.0, alpha)
            self.assertEqual(3, dist.event_shape_tensor().eval())
            self.assertAllEqual([], dist.batch_shape_tensor())
            self.assertEqual(tensor_shape.TensorShape([3]), dist.event_shape)
            self.assertEqual(tensor_shape.TensorShape([]), dist.batch_shape)

    @test_util.run_deprecated_v1
    def testComplexShapes(self):
        if False:
            while True:
                i = 10
        with self.cached_session():
            alpha = np.random.rand(3, 2, 2)
            n = [[3.0, 2], [4, 5], [6, 7]]
            dist = ds.DirichletMultinomial(n, alpha)
            self.assertEqual(2, dist.event_shape_tensor().eval())
            self.assertAllEqual([3, 2], dist.batch_shape_tensor())
            self.assertEqual(tensor_shape.TensorShape([2]), dist.event_shape)
            self.assertEqual(tensor_shape.TensorShape([3, 2]), dist.batch_shape)

    @test_util.run_deprecated_v1
    def testNproperty(self):
        if False:
            return 10
        alpha = [[1.0, 2, 3]]
        n = [[5.0]]
        with self.cached_session():
            dist = ds.DirichletMultinomial(n, alpha)
            self.assertEqual([1, 1], dist.total_count.get_shape())
            self.assertAllClose(n, dist.total_count)

    @test_util.run_deprecated_v1
    def testAlphaProperty(self):
        if False:
            return 10
        alpha = [[1.0, 2, 3]]
        with self.cached_session():
            dist = ds.DirichletMultinomial(1, alpha)
            self.assertEqual([1, 3], dist.concentration.get_shape())
            self.assertAllClose(alpha, dist.concentration)

    @test_util.run_deprecated_v1
    def testPmfNandCountsAgree(self):
        if False:
            while True:
                i = 10
        alpha = [[1.0, 2, 3]]
        n = [[5.0]]
        with self.cached_session():
            dist = ds.DirichletMultinomial(n, alpha, validate_args=True)
            dist.prob([2.0, 3, 0]).eval()
            dist.prob([3.0, 0, 2]).eval()
            with self.assertRaisesOpError('must be non-negative'):
                dist.prob([-1.0, 4, 2]).eval()
            with self.assertRaisesOpError('last-dimension must sum to `self.total_count`'):
                dist.prob([3.0, 3, 0]).eval()

    @test_util.run_deprecated_v1
    def testPmfNonIntegerCounts(self):
        if False:
            while True:
                i = 10
        alpha = [[1.0, 2, 3]]
        n = [[5.0]]
        with self.cached_session():
            dist = ds.DirichletMultinomial(n, alpha, validate_args=True)
            dist.prob([2.0, 3, 0]).eval()
            dist.prob([3.0, 0, 2]).eval()
            dist.prob([3.0, 0, 2.0]).eval()
            placeholder = array_ops.placeholder(dtypes.float32)
            with self.assertRaisesOpError('cannot contain fractional components'):
                dist.prob(placeholder).eval(feed_dict={placeholder: [1.0, 2.5, 1.5]})
            dist = ds.DirichletMultinomial(n, alpha, validate_args=False)
            dist.prob([1.0, 2.0, 3.0]).eval()
            dist.prob([1.0, 2.5, 1.5]).eval()

    def testPmfBothZeroBatches(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session():
            alpha = [1.0, 2]
            counts = [1.0, 0]
            dist = ds.DirichletMultinomial(1.0, alpha)
            pmf = dist.prob(counts)
            self.assertAllClose(1 / 3.0, self.evaluate(pmf))
            self.assertEqual((), pmf.get_shape())

    def testPmfBothZeroBatchesNontrivialN(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            alpha = [1.0, 2]
            counts = [3.0, 2]
            dist = ds.DirichletMultinomial(5.0, alpha)
            pmf = dist.prob(counts)
            self.assertAllClose(1 / 7.0, self.evaluate(pmf))
            self.assertEqual((), pmf.get_shape())

    def testPmfBothZeroBatchesMultidimensionalN(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session():
            alpha = [1.0, 2]
            counts = [3.0, 2]
            n = np.full([4, 3], 5.0, dtype=np.float32)
            dist = ds.DirichletMultinomial(n, alpha)
            pmf = dist.prob(counts)
            self.assertAllClose([[1 / 7.0, 1 / 7.0, 1 / 7.0]] * 4, self.evaluate(pmf))
            self.assertEqual((4, 3), pmf.get_shape())

    def testPmfAlphaStretchedInBroadcastWhenSameRank(self):
        if False:
            return 10
        with self.cached_session():
            alpha = [[1.0, 2]]
            counts = [[1.0, 0], [0.0, 1]]
            dist = ds.DirichletMultinomial([1.0], alpha)
            pmf = dist.prob(counts)
            self.assertAllClose([1 / 3.0, 2 / 3.0], self.evaluate(pmf))
            self.assertAllEqual([2], pmf.get_shape())

    def testPmfAlphaStretchedInBroadcastWhenLowerRank(self):
        if False:
            return 10
        with self.cached_session():
            alpha = [1.0, 2]
            counts = [[1.0, 0], [0.0, 1]]
            pmf = ds.DirichletMultinomial(1.0, alpha).prob(counts)
            self.assertAllClose([1 / 3.0, 2 / 3.0], self.evaluate(pmf))
            self.assertAllEqual([2], pmf.get_shape())

    def testPmfCountsStretchedInBroadcastWhenSameRank(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            alpha = [[1.0, 2], [2.0, 3]]
            counts = [[1.0, 0]]
            pmf = ds.DirichletMultinomial([1.0, 1.0], alpha).prob(counts)
            self.assertAllClose([1 / 3.0, 2 / 5.0], self.evaluate(pmf))
            self.assertAllEqual([2], pmf.get_shape())

    def testPmfCountsStretchedInBroadcastWhenLowerRank(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            alpha = [[1.0, 2], [2.0, 3]]
            counts = [1.0, 0]
            pmf = ds.DirichletMultinomial(1.0, alpha).prob(counts)
            self.assertAllClose([1 / 3.0, 2 / 5.0], self.evaluate(pmf))
            self.assertAllEqual([2], pmf.get_shape())

    @test_util.run_deprecated_v1
    def testPmfForOneVoteIsTheMeanWithOneRecordInput(self):
        if False:
            i = 10
            return i + 15
        alpha = [1.0, 2, 3]
        with self.cached_session():
            for class_num in range(3):
                counts = np.zeros([3], dtype=np.float32)
                counts[class_num] = 1
                dist = ds.DirichletMultinomial(1.0, alpha)
                mean = dist.mean().eval()
                pmf = dist.prob(counts).eval()
                self.assertAllClose(mean[class_num], pmf)
                self.assertAllEqual([3], mean.shape)
                self.assertAllEqual([], pmf.shape)

    @test_util.run_deprecated_v1
    def testMeanDoubleTwoVotes(self):
        if False:
            while True:
                i = 10
        alpha = [1.0, 2, 3]
        with self.cached_session():
            for class_num in range(3):
                counts_one = np.zeros([3], dtype=np.float32)
                counts_one[class_num] = 1.0
                counts_two = np.zeros([3], dtype=np.float32)
                counts_two[class_num] = 2
                dist1 = ds.DirichletMultinomial(1.0, alpha)
                dist2 = ds.DirichletMultinomial(2.0, alpha)
                mean1 = dist1.mean().eval()
                mean2 = dist2.mean().eval()
                self.assertAllClose(mean2[class_num], 2 * mean1[class_num])
                self.assertAllEqual([3], mean1.shape)

    @test_util.run_deprecated_v1
    def testCovarianceFromSampling(self):
        if False:
            return 10
        alpha = np.array([[1.0, 2, 3], [2.5, 4, 0.01]], dtype=np.float32)
        n = np.float32(5)
        with self.cached_session() as sess:
            dist = ds.DirichletMultinomial(n, alpha)
            x = dist.sample(int(250000.0), seed=1)
            sample_mean = math_ops.reduce_mean(x, 0)
            x_centered = x - sample_mean[array_ops.newaxis, ...]
            sample_cov = math_ops.reduce_mean(math_ops.matmul(x_centered[..., array_ops.newaxis], x_centered[..., array_ops.newaxis, :]), 0)
            sample_var = array_ops.matrix_diag_part(sample_cov)
            sample_stddev = math_ops.sqrt(sample_var)
            [sample_mean_, sample_cov_, sample_var_, sample_stddev_, analytic_mean, analytic_cov, analytic_var, analytic_stddev] = sess.run([sample_mean, sample_cov, sample_var, sample_stddev, dist.mean(), dist.covariance(), dist.variance(), dist.stddev()])
            self.assertAllClose(sample_mean_, analytic_mean, atol=0.04, rtol=0.0)
            self.assertAllClose(sample_cov_, analytic_cov, atol=0.05, rtol=0.0)
            self.assertAllClose(sample_var_, analytic_var, atol=0.05, rtol=0.0)
            self.assertAllClose(sample_stddev_, analytic_stddev, atol=0.02, rtol=0.0)

    @test_util.run_without_tensor_float_32('Tests DirichletMultinomial.covariance, which calls matmul')
    def testCovariance(self):
        if False:
            i = 10
            return i + 15
        alpha = [1.0, 2]
        ns = [2.0, 3.0, 4.0, 5.0]
        alpha_0 = np.sum(alpha)
        variance_entry = lambda a, a_sum: a / a_sum * (1 - a / a_sum)
        covariance_entry = lambda a, b, a_sum: -a * b / a_sum ** 2
        shared_matrix = np.array([[variance_entry(alpha[0], alpha_0), covariance_entry(alpha[0], alpha[1], alpha_0)], [covariance_entry(alpha[1], alpha[0], alpha_0), variance_entry(alpha[1], alpha_0)]])
        with self.cached_session():
            for n in ns:
                dist = ds.DirichletMultinomial(n, alpha)
                covariance = dist.covariance()
                expected_covariance = n * (n + alpha_0) / (1 + alpha_0) * shared_matrix
                self.assertEqual([2, 2], covariance.get_shape())
                self.assertAllClose(expected_covariance, self.evaluate(covariance))

    def testCovarianceNAlphaBroadcast(self):
        if False:
            while True:
                i = 10
        alpha_v = [1.0, 2, 3]
        alpha_0 = 6.0
        alpha = np.array(4 * [alpha_v], dtype=np.float32)
        ns = np.array([[2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
        variance_entry = lambda a, a_sum: a / a_sum * (1 - a / a_sum)
        covariance_entry = lambda a, b, a_sum: -a * b / a_sum ** 2
        shared_matrix = np.array(4 * [[[variance_entry(alpha_v[0], alpha_0), covariance_entry(alpha_v[0], alpha_v[1], alpha_0), covariance_entry(alpha_v[0], alpha_v[2], alpha_0)], [covariance_entry(alpha_v[1], alpha_v[0], alpha_0), variance_entry(alpha_v[1], alpha_0), covariance_entry(alpha_v[1], alpha_v[2], alpha_0)], [covariance_entry(alpha_v[2], alpha_v[0], alpha_0), covariance_entry(alpha_v[2], alpha_v[1], alpha_0), variance_entry(alpha_v[2], alpha_0)]]], dtype=np.float32)
        with self.cached_session():
            dist = ds.DirichletMultinomial(ns, alpha)
            covariance = dist.covariance()
            expected_covariance = shared_matrix * (ns * (ns + alpha_0) / (1 + alpha_0))[..., array_ops.newaxis]
            self.assertEqual([4, 3, 3], covariance.get_shape())
            self.assertAllClose(expected_covariance, self.evaluate(covariance))

    def testCovarianceMultidimensional(self):
        if False:
            while True:
                i = 10
        alpha = np.random.rand(3, 5, 4).astype(np.float32)
        alpha2 = np.random.rand(6, 3, 3).astype(np.float32)
        ns = np.random.randint(low=1, high=11, size=[3, 5, 1]).astype(np.float32)
        ns2 = np.random.randint(low=1, high=11, size=[6, 1, 1]).astype(np.float32)
        with self.cached_session():
            dist = ds.DirichletMultinomial(ns, alpha)
            dist2 = ds.DirichletMultinomial(ns2, alpha2)
            covariance = dist.covariance()
            covariance2 = dist2.covariance()
            self.assertEqual([3, 5, 4, 4], covariance.get_shape())
            self.assertEqual([6, 3, 3, 3], covariance2.get_shape())

    def testZeroCountsResultsInPmfEqualToOne(self):
        if False:
            for i in range(10):
                print('nop')
        alpha = [5, 0.5]
        counts = [0.0, 0]
        with self.cached_session():
            dist = ds.DirichletMultinomial(0.0, alpha)
            pmf = dist.prob(counts)
            self.assertAllClose(1.0, self.evaluate(pmf))
            self.assertEqual((), pmf.get_shape())

    def testLargeTauGivesPreciseProbabilities(self):
        if False:
            for i in range(10):
                print('nop')
        mu = np.array([0.1, 0.1, 0.8], dtype=np.float32)
        tau = np.array([100.0], dtype=np.float32)
        alpha = tau * mu
        counts = [0.0, 0, 1]
        with self.cached_session():
            dist = ds.DirichletMultinomial(1.0, alpha)
            pmf = dist.prob(counts)
            self.assertAllClose(0.8, self.evaluate(pmf), atol=0.0001)
            self.assertEqual((), pmf.get_shape())
        counts = [0.0, 0, 2]
        with self.cached_session():
            dist = ds.DirichletMultinomial(2.0, alpha)
            pmf = dist.prob(counts)
            self.assertAllClose(0.8 ** 2, self.evaluate(pmf), atol=0.01)
            self.assertEqual((), pmf.get_shape())
        counts = [1.0, 0, 2]
        with self.cached_session():
            dist = ds.DirichletMultinomial(3.0, alpha)
            pmf = dist.prob(counts)
            self.assertAllClose(3 * 0.1 * 0.8 * 0.8, self.evaluate(pmf), atol=0.01)
            self.assertEqual((), pmf.get_shape())

    def testSmallTauPrefersCorrelatedResults(self):
        if False:
            while True:
                i = 10
        mu = np.array([0.5, 0.5], dtype=np.float32)
        tau = np.array([0.1], dtype=np.float32)
        alpha = tau * mu
        counts = [1.0, 0]
        with self.cached_session():
            dist = ds.DirichletMultinomial(1.0, alpha)
            pmf = dist.prob(counts)
            self.assertAllClose(0.5, self.evaluate(pmf))
            self.assertEqual((), pmf.get_shape())
        counts_same = [2.0, 0]
        counts_different = [1, 1.0]
        with self.cached_session():
            dist = ds.DirichletMultinomial(2.0, alpha)
            pmf_same = dist.prob(counts_same)
            pmf_different = dist.prob(counts_different)
            self.assertLess(5 * self.evaluate(pmf_different), self.evaluate(pmf_same))
            self.assertEqual((), pmf_same.get_shape())

    @test_util.run_deprecated_v1
    def testNonStrictTurnsOffAllChecks(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            alpha = [[-1.0, 2]]
            counts = [[1.0, 0], [0.0, -1]]
            n = [-5.3]
            dist = ds.DirichletMultinomial(n, alpha, validate_args=False)
            dist.prob(counts).eval()

    @test_util.run_deprecated_v1
    def testSampleUnbiasedNonScalarBatch(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session() as sess:
            dist = ds.DirichletMultinomial(total_count=5.0, concentration=1.0 + 2.0 * self._rng.rand(4, 3, 2).astype(np.float32))
            n = int(3000.0)
            x = dist.sample(n, seed=0)
            sample_mean = math_ops.reduce_mean(x, 0)
            x_centered = array_ops.transpose(x - sample_mean, [1, 2, 3, 0])
            sample_covariance = math_ops.matmul(x_centered, x_centered, adjoint_b=True) / n
            [sample_mean_, sample_covariance_, actual_mean_, actual_covariance_] = sess.run([sample_mean, sample_covariance, dist.mean(), dist.covariance()])
            self.assertAllEqual([4, 3, 2], sample_mean.get_shape())
            self.assertAllClose(actual_mean_, sample_mean_, atol=0.0, rtol=0.2)
            self.assertAllEqual([4, 3, 2, 2], sample_covariance.get_shape())
            self.assertAllClose(actual_covariance_, sample_covariance_, atol=0.0, rtol=0.2)

    @test_util.run_deprecated_v1
    def testSampleUnbiasedScalarBatch(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session() as sess:
            dist = ds.DirichletMultinomial(total_count=5.0, concentration=1.0 + 2.0 * self._rng.rand(4).astype(np.float32))
            n = int(5000.0)
            x = dist.sample(n, seed=0)
            sample_mean = math_ops.reduce_mean(x, 0)
            x_centered = x - sample_mean
            sample_covariance = math_ops.matmul(x_centered, x_centered, adjoint_a=True) / n
            [sample_mean_, sample_covariance_, actual_mean_, actual_covariance_] = sess.run([sample_mean, sample_covariance, dist.mean(), dist.covariance()])
            self.assertAllEqual([4], sample_mean.get_shape())
            self.assertAllClose(actual_mean_, sample_mean_, atol=0.0, rtol=0.2)
            self.assertAllEqual([4, 4], sample_covariance.get_shape())
            self.assertAllClose(actual_covariance_, sample_covariance_, atol=0.0, rtol=0.2)

    def testNotReparameterized(self):
        if False:
            i = 10
            return i + 15
        total_count = constant_op.constant(5.0)
        concentration = constant_op.constant([0.1, 0.1, 0.1])
        with backprop.GradientTape() as tape:
            tape.watch(total_count)
            tape.watch(concentration)
            dist = ds.DirichletMultinomial(total_count=total_count, concentration=concentration)
            samples = dist.sample(100)
        (grad_total_count, grad_concentration) = tape.gradient(samples, [total_count, concentration])
        self.assertIsNone(grad_total_count)
        self.assertIsNone(grad_concentration)
if __name__ == '__main__':
    test.main()