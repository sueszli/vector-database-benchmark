import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import multinomial
from tensorflow.python.platform import test

class MultinomialTest(test.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self._rng = np.random.RandomState(42)

    @test_util.run_v1_only('b/120545219')
    def testSimpleShapes(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            p = [0.1, 0.3, 0.6]
            dist = multinomial.Multinomial(total_count=1.0, probs=p)
            self.assertEqual(3, dist.event_shape_tensor().eval())
            self.assertAllEqual([], dist.batch_shape_tensor())
            self.assertEqual(tensor_shape.TensorShape([3]), dist.event_shape)
            self.assertEqual(tensor_shape.TensorShape([]), dist.batch_shape)

    @test_util.run_v1_only('b/120545219')
    def testComplexShapes(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session():
            p = 0.5 * np.ones([3, 2, 2], dtype=np.float32)
            n = [[3.0, 2], [4, 5], [6, 7]]
            dist = multinomial.Multinomial(total_count=n, probs=p)
            self.assertEqual(2, dist.event_shape_tensor().eval())
            self.assertAllEqual([3, 2], dist.batch_shape_tensor())
            self.assertEqual(tensor_shape.TensorShape([2]), dist.event_shape)
            self.assertEqual(tensor_shape.TensorShape([3, 2]), dist.batch_shape)

    @test_util.run_v1_only('b/120545219')
    def testN(self):
        if False:
            return 10
        p = [[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]]
        n = [[3.0], [4]]
        with self.cached_session():
            dist = multinomial.Multinomial(total_count=n, probs=p)
            self.assertEqual((2, 1), dist.total_count.get_shape())
            self.assertAllClose(n, dist.total_count)

    @test_util.run_v1_only('b/120545219')
    def testP(self):
        if False:
            i = 10
            return i + 15
        p = [[0.1, 0.2, 0.7]]
        with self.cached_session():
            dist = multinomial.Multinomial(total_count=3.0, probs=p)
            self.assertEqual((1, 3), dist.probs.get_shape())
            self.assertEqual((1, 3), dist.logits.get_shape())
            self.assertAllClose(p, dist.probs)

    @test_util.run_v1_only('b/120545219')
    def testLogits(self):
        if False:
            print('Hello World!')
        p = np.array([[0.1, 0.2, 0.7]], dtype=np.float32)
        logits = np.log(p) - 50.0
        with self.cached_session():
            multinom = multinomial.Multinomial(total_count=3.0, logits=logits)
            self.assertEqual((1, 3), multinom.probs.get_shape())
            self.assertEqual((1, 3), multinom.logits.get_shape())
            self.assertAllClose(p, multinom.probs)
            self.assertAllClose(logits, multinom.logits)

    @test_util.run_v1_only('b/120545219')
    def testPmfUnderflow(self):
        if False:
            for i in range(10):
                print('nop')
        logits = np.array([[-200, 0]], dtype=np.float32)
        with self.cached_session():
            dist = multinomial.Multinomial(total_count=1.0, logits=logits)
            lp = dist.log_prob([1.0, 0.0]).eval()[0]
            self.assertAllClose(-200, lp, atol=0, rtol=1e-06)

    @test_util.run_v1_only('b/120545219')
    def testPmfandCountsAgree(self):
        if False:
            i = 10
            return i + 15
        p = [[0.1, 0.2, 0.7]]
        n = [[5.0]]
        with self.cached_session():
            dist = multinomial.Multinomial(total_count=n, probs=p, validate_args=True)
            dist.prob([2.0, 3, 0]).eval()
            dist.prob([3.0, 0, 2]).eval()
            with self.assertRaisesOpError('must be non-negative'):
                dist.prob([-1.0, 4, 2]).eval()
            with self.assertRaisesOpError('counts must sum to `self.total_count`'):
                dist.prob([3.0, 3, 0]).eval()

    @test_util.run_v1_only('b/120545219')
    def testPmfNonIntegerCounts(self):
        if False:
            for i in range(10):
                print('nop')
        p = [[0.1, 0.2, 0.7]]
        n = [[5.0]]
        with self.cached_session():
            multinom = multinomial.Multinomial(total_count=n, probs=p, validate_args=True)
            multinom.prob([2.0, 1, 2]).eval()
            multinom.prob([3.0, 0, 2]).eval()
            with self.assertRaisesOpError('counts must sum to `self.total_count`'):
                multinom.prob([2.0, 3, 2]).eval()
            x = array_ops.placeholder(dtypes.float32)
            with self.assertRaisesOpError('cannot contain fractional components.'):
                multinom.prob(x).eval(feed_dict={x: [1.0, 2.5, 1.5]})
            multinom = multinomial.Multinomial(total_count=n, probs=p, validate_args=False)
            multinom.prob([1.0, 2.0, 2.0]).eval()
            multinom.prob([1.0, 2.5, 1.5]).eval()

    def testPmfBothZeroBatches(self):
        if False:
            while True:
                i = 10
        with self.cached_session():
            p = [0.5, 0.5]
            counts = [1.0, 0]
            pmf = multinomial.Multinomial(total_count=1.0, probs=p).prob(counts)
            self.assertAllClose(0.5, self.evaluate(pmf))
            self.assertEqual((), pmf.get_shape())

    def testPmfBothZeroBatchesNontrivialN(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session():
            p = [0.1, 0.9]
            counts = [3.0, 2]
            dist = multinomial.Multinomial(total_count=5.0, probs=p)
            pmf = dist.prob(counts)
            self.assertAllClose(81.0 / 10000, self.evaluate(pmf))
            self.assertEqual((), pmf.get_shape())

    def testPmfPStretchedInBroadcastWhenSameRank(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            p = [[0.1, 0.9]]
            counts = [[1.0, 0], [0, 1]]
            pmf = multinomial.Multinomial(total_count=1.0, probs=p).prob(counts)
            self.assertAllClose([0.1, 0.9], self.evaluate(pmf))
            self.assertEqual(2, pmf.get_shape())

    def testPmfPStretchedInBroadcastWhenLowerRank(self):
        if False:
            return 10
        with self.cached_session():
            p = [0.1, 0.9]
            counts = [[1.0, 0], [0, 1]]
            pmf = multinomial.Multinomial(total_count=1.0, probs=p).prob(counts)
            self.assertAllClose([0.1, 0.9], self.evaluate(pmf))
            self.assertEqual(2, pmf.get_shape())

    @test_util.run_v1_only('b/120545219')
    def testPmfCountsStretchedInBroadcastWhenSameRank(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session():
            p = [[0.1, 0.9], [0.7, 0.3]]
            counts = [[1.0, 0]]
            pmf = multinomial.Multinomial(total_count=1.0, probs=p).prob(counts)
            self.assertAllClose(pmf, [0.1, 0.7])
            self.assertEqual(2, pmf.get_shape())

    @test_util.run_v1_only('b/120545219')
    def testPmfCountsStretchedInBroadcastWhenLowerRank(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            p = [[0.1, 0.9], [0.7, 0.3]]
            counts = [1.0, 0]
            pmf = multinomial.Multinomial(total_count=1.0, probs=p).prob(counts)
            self.assertAllClose(pmf, [0.1, 0.7])
            self.assertEqual(pmf.get_shape(), 2)

    def testPmfShapeCountsStretchedN(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            p = [[[0.1, 0.9], [0.1, 0.9]], [[0.7, 0.3], [0.7, 0.3]]]
            n = [[3.0, 3], [3, 3]]
            counts = [2.0, 1]
            pmf = multinomial.Multinomial(total_count=n, probs=p).prob(counts)
            self.evaluate(pmf)
            self.assertEqual(pmf.get_shape(), (2, 2))

    def testPmfShapeCountsPStretchedN(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            p = [0.1, 0.9]
            counts = [3.0, 2]
            n = np.full([4, 3], 5.0, dtype=np.float32)
            pmf = multinomial.Multinomial(total_count=n, probs=p).prob(counts)
            self.evaluate(pmf)
            self.assertEqual((4, 3), pmf.get_shape())

    @test_util.run_v1_only('b/120545219')
    def testMultinomialMean(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            n = 5.0
            p = [0.1, 0.2, 0.7]
            dist = multinomial.Multinomial(total_count=n, probs=p)
            expected_means = 5 * np.array(p, dtype=np.float32)
            self.assertEqual((3,), dist.mean().get_shape())
            self.assertAllClose(expected_means, dist.mean())

    @test_util.run_v1_only('b/120545219')
    def testMultinomialCovariance(self):
        if False:
            return 10
        with self.cached_session():
            n = 5.0
            p = [0.1, 0.2, 0.7]
            dist = multinomial.Multinomial(total_count=n, probs=p)
            expected_covariances = [[9.0 / 20, -1 / 10, -7 / 20], [-1 / 10, 4 / 5, -7 / 10], [-7 / 20, -7 / 10, 21 / 20]]
            self.assertEqual((3, 3), dist.covariance().get_shape())
            self.assertAllClose(expected_covariances, dist.covariance())

    @test_util.run_v1_only('b/120545219')
    def testMultinomialCovarianceBatch(self):
        if False:
            return 10
        with self.cached_session():
            n = [5.0] * 2
            p = [[[0.1, 0.9]], [[0.1, 0.9]]] * 2
            dist = multinomial.Multinomial(total_count=n, probs=p)
            inner_var = [[9.0 / 20, -9 / 20], [-9 / 20, 9 / 20]]
            expected_covariances = [[inner_var, inner_var]] * 4
            self.assertEqual((4, 2, 2, 2), dist.covariance().get_shape())
            self.assertAllClose(expected_covariances, dist.covariance())

    def testCovarianceMultidimensional(self):
        if False:
            for i in range(10):
                print('nop')
        p = np.random.dirichlet([0.25, 0.25, 0.25, 0.25], [3, 5]).astype(np.float32)
        p2 = np.random.dirichlet([0.3, 0.3, 0.4], [6, 3]).astype(np.float32)
        ns = np.random.randint(low=1, high=11, size=[3, 5]).astype(np.float32)
        ns2 = np.random.randint(low=1, high=11, size=[6, 1]).astype(np.float32)
        with self.cached_session():
            dist = multinomial.Multinomial(ns, p)
            dist2 = multinomial.Multinomial(ns2, p2)
            covariance = dist.covariance()
            covariance2 = dist2.covariance()
            self.assertEqual((3, 5, 4, 4), covariance.get_shape())
            self.assertEqual((6, 3, 3, 3), covariance2.get_shape())

    @test_util.run_v1_only('b/120545219')
    def testCovarianceFromSampling(self):
        if False:
            i = 10
            return i + 15
        theta = np.array([[1.0, 2, 3], [2.5, 4, 0.01]], dtype=np.float32)
        theta /= np.sum(theta, 1)[..., array_ops.newaxis]
        n = np.array([[10.0, 9.0], [8.0, 7.0], [6.0, 5.0]], dtype=np.float32)
        with self.cached_session() as sess:
            dist = multinomial.Multinomial(n, theta)
            x = dist.sample(int(1000000.0), seed=1)
            sample_mean = math_ops.reduce_mean(x, 0)
            x_centered = x - sample_mean[array_ops.newaxis, ...]
            sample_cov = math_ops.reduce_mean(math_ops.matmul(x_centered[..., array_ops.newaxis], x_centered[..., array_ops.newaxis, :]), 0)
            sample_var = array_ops.matrix_diag_part(sample_cov)
            sample_stddev = math_ops.sqrt(sample_var)
            [sample_mean_, sample_cov_, sample_var_, sample_stddev_, analytic_mean, analytic_cov, analytic_var, analytic_stddev] = sess.run([sample_mean, sample_cov, sample_var, sample_stddev, dist.mean(), dist.covariance(), dist.variance(), dist.stddev()])
            self.assertAllClose(sample_mean_, analytic_mean, atol=0.01, rtol=0.01)
            self.assertAllClose(sample_cov_, analytic_cov, atol=0.01, rtol=0.01)
            self.assertAllClose(sample_var_, analytic_var, atol=0.01, rtol=0.01)
            self.assertAllClose(sample_stddev_, analytic_stddev, atol=0.01, rtol=0.01)

    @test_util.run_v1_only('b/120545219')
    def testSampleUnbiasedNonScalarBatch(self):
        if False:
            return 10
        with self.cached_session() as sess:
            dist = multinomial.Multinomial(total_count=[7.0, 6.0, 5.0], logits=math_ops.log(2.0 * self._rng.rand(4, 3, 2).astype(np.float32)))
            n = int(30000.0)
            x = dist.sample(n, seed=0)
            sample_mean = math_ops.reduce_mean(x, 0)
            x_centered = array_ops.transpose(x - sample_mean, [1, 2, 3, 0])
            sample_covariance = math_ops.matmul(x_centered, x_centered, adjoint_b=True) / n
            [sample_mean_, sample_covariance_, actual_mean_, actual_covariance_] = sess.run([sample_mean, sample_covariance, dist.mean(), dist.covariance()])
            self.assertAllEqual([4, 3, 2], sample_mean.get_shape())
            self.assertAllClose(actual_mean_, sample_mean_, atol=0.0, rtol=0.1)
            self.assertAllEqual([4, 3, 2, 2], sample_covariance.get_shape())
            self.assertAllClose(actual_covariance_, sample_covariance_, atol=0.0, rtol=0.2)

    @test_util.run_v1_only('b/120545219')
    def testSampleUnbiasedScalarBatch(self):
        if False:
            print('Hello World!')
        with self.cached_session() as sess:
            dist = multinomial.Multinomial(total_count=5.0, logits=math_ops.log(2.0 * self._rng.rand(4).astype(np.float32)))
            n = int(5000.0)
            x = dist.sample(n, seed=0)
            sample_mean = math_ops.reduce_mean(x, 0)
            x_centered = x - sample_mean
            sample_covariance = math_ops.matmul(x_centered, x_centered, adjoint_a=True) / n
            [sample_mean_, sample_covariance_, actual_mean_, actual_covariance_] = sess.run([sample_mean, sample_covariance, dist.mean(), dist.covariance()])
            self.assertAllEqual([4], sample_mean.get_shape())
            self.assertAllClose(actual_mean_, sample_mean_, atol=0.0, rtol=0.1)
            self.assertAllEqual([4, 4], sample_covariance.get_shape())
            self.assertAllClose(actual_covariance_, sample_covariance_, atol=0.0, rtol=0.2)

    def testNotReparameterized(self):
        if False:
            print('Hello World!')
        total_count = constant_op.constant(5.0)
        p = constant_op.constant([0.2, 0.6])
        with backprop.GradientTape() as tape:
            tape.watch(total_count)
            tape.watch(p)
            dist = multinomial.Multinomial(total_count=total_count, probs=p)
            samples = dist.sample(100)
        (grad_total_count, grad_p) = tape.gradient(samples, [total_count, p])
        self.assertIsNone(grad_total_count)
        self.assertIsNone(grad_p)
if __name__ == '__main__':
    test.main()