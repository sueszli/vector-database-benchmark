"""Tests for random.multivariate_normal."""
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
import tf_quant_finance as tff
from tensorflow.python.framework import test_util
tff_rnd = tff.math.random

@test_util.run_all_in_graph_and_eager_modes
class RandomTest(parameterized.TestCase, tf.test.TestCase):

    def test_shapes(self):
        if False:
            while True:
                i = 10
        'Tests the sample shapes.'
        sample_no_batch = self.evaluate(tff_rnd.mv_normal_sample([2, 4], mean=[0.2, 0.1]))
        self.assertEqual(sample_no_batch.shape, (2, 4, 2))
        sample_batch = self.evaluate(tff_rnd.mv_normal_sample([2, 4], mean=[[0.2, 0.1], [0.0, -0.1], [0.0, 0.1]]))
        self.assertEqual(sample_batch.shape, (2, 4, 3, 2))

    def test_mean_default(self):
        if False:
            print('Hello World!')
        'Tests that the default value of mean is 0.'
        covar = np.array([[1.0, 0.1], [0.1, 1.0]])
        sample = self.evaluate(tff_rnd.mv_normal_sample([40000], covariance_matrix=covar, seed=1234))
        with self.subTest('Shape'):
            np.testing.assert_array_equal(sample.shape, [40000, 2])
        with self.subTest('Mean'):
            self.assertArrayNear(np.mean(sample, axis=0), [0.0, 0.0], 0.01)
        with self.subTest('Covariance'):
            self.assertArrayNear(np.cov(sample, rowvar=False).reshape([-1]), covar.reshape([-1]), 0.02)

    def test_covariance_default(self):
        if False:
            return 10
        'Tests that the default value of the covariance matrix is identity.'
        mean = np.array([[1.0, 0.1], [0.1, 1.0]])
        sample = self.evaluate(tff_rnd.mv_normal_sample([10000], mean=mean))
        np.testing.assert_array_equal(sample.shape, [10000, 2, 2])
        np.testing.assert_array_almost_equal(np.mean(sample, axis=0), mean, decimal=1)
        np.testing.assert_array_almost_equal(np.cov(sample[:, 0, :], rowvar=False), np.eye(2), decimal=1)
        np.testing.assert_array_almost_equal(np.cov(sample[:, 1, :], rowvar=False), np.eye(2), decimal=1)

    @parameterized.named_parameters({'testcase_name': 'PSEUDO', 'random_type': tff_rnd.RandomType.PSEUDO, 'seed': 4567}, {'testcase_name': 'STATELESS', 'random_type': tff_rnd.RandomType.STATELESS, 'seed': [1, 4567]})
    def test_general_mean_covariance(self, random_type, seed):
        if False:
            return 10
        'Tests that the sample is correctly generated for pseudo and stateless.'
        mean = np.array([[1.0, 0.1], [0.1, 1.0]])
        covar = np.array([[[0.9, -0.1], [-0.1, 1.0]], [[1.1, -0.3], [-0.3, 0.6]]])
        size = 30000
        sample = self.evaluate(tff_rnd.mv_normal_sample([size], mean=mean, covariance_matrix=covar, random_type=random_type, seed=seed))
        np.testing.assert_array_equal(sample.shape, [size, 2, 2])
        np.testing.assert_array_almost_equal(np.mean(sample, axis=0), mean, decimal=1)
        np.testing.assert_array_almost_equal(np.cov(sample[:, 0, :], rowvar=False), covar[0], decimal=1)
        np.testing.assert_array_almost_equal(np.cov(sample[:, 1, :], rowvar=False), covar[1], decimal=1)

    @parameterized.named_parameters({'testcase_name': 'HALTON', 'random_type': tff_rnd.RandomType.HALTON, 'seed': None}, {'testcase_name': 'STATELESS', 'random_type': tff_rnd.RandomType.STATELESS, 'seed': [1, 4567]})
    def test_dynamic_shapes(self, random_type, seed):
        if False:
            while True:
                i = 10
        'Tests that the sample is correctly generated for dynamic shape.'
        mean = np.array([[1.0, 0.1], [0.1, 1.0]])
        covar = np.array([[[0.9, -0.1], [-0.1, 1.0]], [[1.1, -0.3], [-0.3, 0.6]]])
        size = 30000

        @tf.function(input_signature=[tf.TensorSpec([None], dtype=tf.int32), tf.TensorSpec([None, None]), tf.TensorSpec([None, None, None])])
        def sampler(sample_shape, mean, covar):
            if False:
                while True:
                    i = 10
            return tff_rnd.mv_normal_sample(sample_shape, mean=mean, covariance_matrix=covar, random_type=random_type, seed=seed)
        sample = self.evaluate(sampler([size], mean, covar))
        np.testing.assert_array_equal(sample.shape, [size, 2, 2])
        np.testing.assert_array_almost_equal(np.mean(sample, axis=0), mean, decimal=1)
        np.testing.assert_array_almost_equal(np.cov(sample[:, 0, :], rowvar=False), covar[0], decimal=1)
        np.testing.assert_array_almost_equal(np.cov(sample[:, 1, :], rowvar=False), covar[1], decimal=1)

    def test_mean_and_scale(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests sample for scale specification.'
        mean = np.array([[1.0, 0.1], [0.1, 1.0]])
        scale = np.array([[0.4, -0.1], [0.22, 1.38]])
        covariance = np.matmul(scale, scale.transpose())
        size = 30000
        sample = self.evaluate(tff_rnd.mv_normal_sample([size], mean=mean, scale_matrix=scale, seed=7534))
        np.testing.assert_array_equal(sample.shape, [size, 2, 2])
        np.testing.assert_array_almost_equal(np.mean(sample, axis=0), mean, decimal=1)
        np.testing.assert_array_almost_equal(np.cov(sample[:, 0, :], rowvar=False), covariance, decimal=1)
        np.testing.assert_array_almost_equal(np.cov(sample[:, 1, :], rowvar=False), covariance, decimal=1)

    def test_mean_default_sobol(self):
        if False:
            print('Hello World!')
        'Tests that the default value of mean is 0.'
        covar = np.array([[1.0, 0.1], [0.1, 1.0]])
        skip = 1000
        sample = self.evaluate(tff_rnd.mv_normal_sample([10000], covariance_matrix=covar, random_type=tff_rnd.RandomType.SOBOL, skip=skip))
        np.testing.assert_array_equal(sample.shape, [10000, 2])
        self.assertArrayNear(np.mean(sample, axis=0), [0.0, 0.0], 0.01)
        self.assertArrayNear(np.cov(sample, rowvar=False).reshape([-1]), covar.reshape([-1]), 0.02)

    def test_mean_and_scale_sobol(self):
        if False:
            while True:
                i = 10
        'Tests sample for scale specification.'
        mean = np.array([[1.0, 0.1], [0.1, 1.0], [2.0, 0.3], [0.0, 0.0]])
        scale = np.array([[0.4, -0.1], [0.22, 1.38]])
        covariance = np.matmul(scale, scale.transpose())
        sample_shape = [2, 3, 5000]
        sample = self.evaluate(tff_rnd.mv_normal_sample(sample_shape, mean=mean, scale_matrix=scale, random_type=tff_rnd.RandomType.SOBOL))
        np.testing.assert_array_equal(sample.shape, sample_shape + [4, 2])
        np.testing.assert_array_almost_equal(np.mean(sample, axis=(0, 1, 2)), mean, decimal=1)
        for i in range(4):
            np.testing.assert_array_almost_equal(np.cov(sample[0, 1, :, i, :], rowvar=False), covariance, decimal=1)

    def test_mean_default_halton(self):
        if False:
            print('Hello World!')
        'Tests that the default value of mean is 0.'
        covar = np.array([[1.0, 0.1], [0.1, 1.0]])
        skip = 1000
        sample = self.evaluate(tff_rnd.mv_normal_sample([10000], covariance_matrix=covar, random_type=tff_rnd.RandomType.HALTON_RANDOMIZED, skip=skip))
        np.testing.assert_array_equal(sample.shape, [10000, 2])
        self.assertArrayNear(np.mean(sample, axis=0), [0.0, 0.0], 0.01)
        self.assertArrayNear(np.cov(sample, rowvar=False).reshape([-1]), covar.reshape([-1]), 0.02)

    def test_mean_default_halton_randomization_params(self):
        if False:
            while True:
                i = 10
        'Tests that the default value of mean is 0.'
        dtype = np.float32
        covar = np.array([[1.0, 0.1], [0.1, 1.0]], dtype=dtype)
        num_samples = 10000
        randomization_params = tff.math.random.halton.sample(2, num_samples, randomized=True, seed=42)[1]
        sample = self.evaluate(tff_rnd.mv_normal_sample([num_samples], covariance_matrix=covar, random_type=tff_rnd.RandomType.HALTON_RANDOMIZED, randomization_params=randomization_params))
        np.testing.assert_array_equal(sample.shape, [num_samples, 2])
        self.assertArrayNear(np.mean(sample, axis=0), [0.0, 0.0], 0.01)
        self.assertArrayNear(np.cov(sample, rowvar=False).reshape([-1]), covar.reshape([-1]), 0.02)

    def test_mean_and_scale_halton(self):
        if False:
            while True:
                i = 10
        'Tests sample for scale specification.'
        mean = np.array([[1.0, 0.1], [0.1, 1.0], [2.0, 0.3], [0.0, 0.0]])
        scale = np.array([[0.4, -0.1], [0.22, 1.38]])
        covariance = np.matmul(scale, scale.transpose())
        sample_shape = [2, 3, 5000]
        sample = self.evaluate(tff_rnd.mv_normal_sample(sample_shape, mean=mean, scale_matrix=scale, random_type=tff_rnd.RandomType.HALTON))
        np.testing.assert_array_equal(sample.shape, sample_shape + [4, 2])
        np.testing.assert_array_almost_equal(np.mean(sample, axis=(0, 1, 2)), mean, decimal=1)
        for i in range(4):
            np.testing.assert_array_almost_equal(np.cov(sample[0, 2, :, i, :], rowvar=False), covariance, decimal=1)

    @parameterized.named_parameters({'testcase_name': 'PSEUDO', 'random_type': tff_rnd.RandomType.PSEUDO_ANTITHETIC, 'seed': 42}, {'testcase_name': 'STATELESS', 'random_type': tff_rnd.RandomType.STATELESS_ANTITHETIC, 'seed': [1, 42]})
    def test_mean_and_scale_antithetic(self, random_type, seed):
        if False:
            i = 10
            return i + 15
        'Tests antithetic sampler for scale specification.'
        mean = np.array([[1.0, 0.1], [0.1, 1.0]])
        scale = np.array([[0.4, -0.1], [0.22, 1.38]])
        covariance = np.matmul(scale, scale.transpose())
        size = 30000
        sample = self.evaluate(tff_rnd.mv_normal_sample([size], mean=mean, scale_matrix=scale, random_type=random_type, seed=seed))
        with self.subTest('Shape'):
            np.testing.assert_array_equal(sample.shape, [size, 2, 2])
        antithetic_size = size // 2
        antithetic_combination = (sample[:antithetic_size, ...] + sample[antithetic_size:, ...]) / 2
        with self.subTest('Partition'):
            np.testing.assert_allclose(antithetic_combination, mean + np.zeros([antithetic_size, 2, 2]), 1e-10, 1e-10)
        with self.subTest('Mean'):
            np.testing.assert_array_almost_equal(np.mean(sample[:antithetic_size, ...], axis=0), mean, decimal=1)
        with self.subTest('CovariancePart1'):
            np.testing.assert_array_almost_equal(np.cov(sample[:antithetic_size, 0, :], rowvar=False), covariance, decimal=1)
        with self.subTest('CovariancePart2'):
            np.testing.assert_array_almost_equal(np.cov(sample[:antithetic_size, 1, :], rowvar=False), covariance, decimal=1)

    def test_antithetic_sample_requires_even_dim(self):
        if False:
            while True:
                i = 10
        'Error is triggered if the first dim of sample_shape is odd.'
        mean = np.array([[1.0, 0.1], [0.1, 1.0]])
        scale = np.array([[0.4, -0.1], [0.22, 1.38]])
        sample_shape = [11, 100]
        with self.assertRaises(tf.errors.InvalidArgumentError):
            self.evaluate(tff_rnd.mv_normal_sample(sample_shape, mean=mean, scale_matrix=scale, random_type=tff_rnd.RandomType.PSEUDO_ANTITHETIC))

    @parameterized.named_parameters({'testcase_name': 'STATELESS', 'random_type': tff_rnd.RandomType.STATELESS, 'seed': [1, 4567]}, {'testcase_name': 'STATELESS_2', 'random_type': tff_rnd.RandomType.STATELESS, 'seed': [456, 91011]}, {'testcase_name': 'STATELESS_ANTITHETIC', 'random_type': tff_rnd.RandomType.STATELESS_ANTITHETIC, 'seed': [1, 4567]}, {'testcase_name': 'SOBOL', 'random_type': tff_rnd.RandomType.SOBOL, 'seed': [1, 4567]}, {'testcase_name': 'HALTON', 'random_type': tff_rnd.RandomType.HALTON, 'seed': [1, 4567]}, {'testcase_name': 'HALTON_RANDOMIZED', 'random_type': tff_rnd.RandomType.HALTON_RANDOMIZED, 'seed': 7889})
    def test_results_are_repeatable(self, random_type, seed):
        if False:
            print('Hello World!')
        'Tests the sample is repeatably generated for pseudo and stateless.'
        mean = np.array([[1.0, 0.1], [0.1, 1.0]])
        covar = np.array([[[0.9, -0.1], [-0.1, 1.0]], [[1.1, -0.3], [-0.3, 0.6]]])
        size = 10
        sample1 = self.evaluate(tff_rnd.mv_normal_sample([size], mean=mean, covariance_matrix=covar, random_type=random_type, seed=seed))
        sample2 = self.evaluate(tff_rnd.mv_normal_sample([size], mean=mean, covariance_matrix=covar, random_type=random_type, seed=seed))
        np.testing.assert_array_almost_equal(sample1, sample2, decimal=6)

    @parameterized.named_parameters({'testcase_name': 'STATELESS', 'random_type': tff_rnd.RandomType.STATELESS, 'mean': np.array([[1.0, 0.1], [0.1, 1.0]]), 'covar': np.array([[[0.9, -0.1], [-0.1, 1.0]], [[1.1, -0.3], [-0.3, 0.6]]]), 'seed': [1, 4567]}, {'testcase_name': 'STATELESS_2', 'random_type': tff_rnd.RandomType.STATELESS, 'mean': np.array([[1.0, 0.1, 2.1], [0.1, 1.0, -0.5]]), 'covar': None, 'seed': [456, 91011]})
    def test_adding_time_steps(self, random_type, mean, covar, seed):
        if False:
            print('Hello World!')
        "Tests that adding additional draws doesn't change previous draws."
        size = 10
        sample1 = self.evaluate(tff_rnd.mv_normal_sample([size], mean=mean, covariance_matrix=covar, random_type=random_type, seed=seed))
        size2 = 16
        sample2 = self.evaluate(tff_rnd.mv_normal_sample([size2], mean=mean, covariance_matrix=covar, random_type=random_type, seed=seed))
        np.testing.assert_array_almost_equal(sample1, sample2[:size], decimal=6)
if __name__ == '__main__':
    tf.test.main()