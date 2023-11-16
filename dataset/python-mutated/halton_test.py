"""Tests for random.halton."""
import numpy as np
from six.moves import range
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow.python.framework import test_util
from tf_quant_finance.math import random
tfd = tfp.distributions

@test_util.run_all_in_graph_and_eager_modes
class HaltonSequenceTest(tf.test.TestCase):

    def test_known_values_small_bases(self):
        if False:
            for i in range(10):
                print('nop')
        expected = np.array([[1.0 / 2, 1.0 / 3], [1.0 / 4, 2.0 / 3], [3.0 / 4, 1.0 / 9], [1.0 / 8, 4.0 / 9], [5.0 / 8, 7.0 / 9]], dtype=np.float32)
        (sample, _) = random.halton.sample(2, num_results=5, randomized=False)
        self.assertAllClose(expected, self.evaluate(sample), rtol=1e-06)

    def test_dynamic_num_samples(self):
        if False:
            print('Hello World!')
        'Tests that num_samples argument supports Tensors.'
        expected = np.array([[1.0 / 2, 1.0 / 3], [1.0 / 4, 2.0 / 3], [3.0 / 4, 1.0 / 9], [1.0 / 8, 4.0 / 9], [5.0 / 8, 7.0 / 9]], dtype=np.float32)
        (sample, _) = random.halton.sample(2, num_results=tf.constant(5), randomized=False)
        self.assertAllClose(expected, self.evaluate(sample), rtol=1e-06)

    def test_sequence_indices(self):
        if False:
            while True:
                i = 10
        'Tests access of sequence elements by index.'
        dim = 5
        indices = tf.range(10, dtype=tf.int32)
        (sample_direct, _) = random.halton.sample(dim, num_results=10, randomized=False)
        (sample_from_indices, _) = random.halton.sample(dim, sequence_indices=indices, randomized=False)
        self.assertAllClose(self.evaluate(sample_direct), self.evaluate(sample_from_indices), rtol=1e-06)

    def test_dtypes_works_correctly(self):
        if False:
            i = 10
            return i + 15
        'Tests that all supported dtypes work without error.'
        dim = 3
        (sample_float32, _) = random.halton.sample(dim, num_results=10, dtype=tf.float32, seed=11)
        (sample_float64, _) = random.halton.sample(dim, num_results=10, dtype=tf.float64, seed=21)
        self.assertEqual(self.evaluate(sample_float32).dtype, np.float32)
        self.assertEqual(self.evaluate(sample_float64).dtype, np.float64)

    def test_normal_integral_mean_and_var_correctly_estimated(self):
        if False:
            print('Hello World!')
        n = int(1000)
        mu_p = tf.constant([-1.0, 1.0], dtype=tf.float64)
        mu_q = tf.constant([0.0, 0.0], dtype=tf.float64)
        sigma_p = tf.constant([0.5, 0.5], dtype=tf.float64)
        sigma_q = tf.constant([1.0, 1.0], dtype=tf.float64)
        p = tfd.Normal(loc=mu_p, scale=sigma_p)
        q = tfd.Normal(loc=mu_q, scale=sigma_q)
        (cdf_sample, _) = random.halton.sample(2, num_results=n, dtype=tf.float64, seed=1729)
        q_sample = q.quantile(cdf_sample)
        e_x = tf.reduce_mean(q_sample * p.prob(q_sample) / q.prob(q_sample), 0)
        e_x2 = tf.reduce_mean(q_sample ** 2 * p.prob(q_sample) / q.prob(q_sample) - e_x ** 2, 0)
        stddev = tf.sqrt(e_x2)
        self.assertEqual(p.batch_shape, e_x.shape)
        self.assertAllClose(self.evaluate(p.mean()), self.evaluate(e_x), rtol=0.01)
        self.assertAllClose(self.evaluate(p.stddev()), self.evaluate(stddev), rtol=0.02)

    def test_docstring_example(self):
        if False:
            print('Hello World!')
        num_results = 1000
        dim = 3
        (sample, params) = random.halton.sample(dim, num_results=num_results, seed=127)
        powers = tf.range(1.0, limit=dim + 1)
        integral = tf.reduce_mean(input_tensor=tf.reduce_prod(input_tensor=sample ** powers, axis=-1))
        true_value = 1.0 / tf.reduce_prod(input_tensor=powers + 1.0)
        self.assertAllClose(self.evaluate(integral), self.evaluate(true_value), rtol=0.02)
        sequence_indices = tf.range(start=1000, limit=1000 + num_results, dtype=tf.int32)
        (sample_leaped, _) = random.halton.sample(dim, sequence_indices=sequence_indices, randomization_params=params)
        integral_leaped = tf.reduce_mean(input_tensor=tf.reduce_prod(input_tensor=sample_leaped ** powers, axis=-1))
        self.assertAllClose(self.evaluate(integral_leaped), self.evaluate(true_value), rtol=0.05)

    def test_randomized_qmc_basic(self):
        if False:
            return 10
        'Tests the randomization of the random.halton sequences.'
        dim = 20
        num_results = 2000
        replica = 5
        seed = 121117
        values = []
        for i in range(replica):
            (sample, _) = random.halton.sample(dim, num_results=num_results, seed=seed + i)
            f = tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=sample, axis=1) ** 2)
            values.append(self.evaluate(f))
        self.assertAllClose(np.mean(values), 101.6667, atol=np.std(values) * 2)

    def test_partial_sum_func_qmc(self):
        if False:
            print('Hello World!')
        'Tests the QMC evaluation of (x_j + x_{j+1} ...+x_{n})^2.\n\n    A good test of QMC is provided by the function:\n\n      f(x_1,..x_n, x_{n+1}, ..., x_{n+m}) = (x_{n+1} + ... x_{n+m} - m / 2)^2\n\n    with the coordinates taking values in the unit interval. The mean and\n    variance of this function (with the uniform distribution over the\n    unit-hypercube) is exactly calculable:\n\n      <f> = m / 12, Var(f) = m (5m - 3) / 360\n\n    The purpose of the "shift" (if n > 0) in the coordinate dependence of the\n    function is to provide a test for Halton sequence which exhibit more\n    dependence in the higher axes.\n\n    This test confirms that the mean squared error of RQMC estimation falls\n    as O(N^(2-e)) for any e>0.\n    '
        (n, m) = (5, 5)
        dim = n + m
        (num_results_lo, num_results_hi) = (500, 5000)
        replica = 10
        true_mean = m / 12.0
        seed_lo = 1925
        seed_hi = 898128

        def func_estimate(x):
            if False:
                print('Hello World!')
            return tf.reduce_mean(input_tensor=tf.math.squared_difference(tf.reduce_sum(input_tensor=x[:, -m:], axis=-1), m / 2.0))
        estimates = []
        for i in range(replica):
            (sample_lo, _) = random.halton.sample(dim, num_results=num_results_lo, seed=seed_lo + i)
            (sample_hi, _) = random.halton.sample(dim, num_results=num_results_hi, seed=seed_hi + i)
            (f_lo, f_hi) = (func_estimate(sample_lo), func_estimate(sample_hi))
            estimates.append((self.evaluate(f_lo), self.evaluate(f_hi)))
        (var_lo, var_hi) = np.mean((np.array(estimates) - true_mean) ** 2, axis=0)
        log_rel_err = np.log(100 * var_hi / var_lo)
        self.assertAllClose(log_rel_err, 0.0, atol=1.2)

    def test_seed_implies_deterministic_results(self):
        if False:
            while True:
                i = 10
        dim = 20
        num_results = 100
        seed = 1925
        (sample1, _) = random.halton.sample(dim, num_results=num_results, seed=seed)
        (sample2, _) = random.halton.sample(dim, num_results=num_results, seed=seed)
        [sample1_, sample2_] = self.evaluate([sample1, sample2])
        self.assertAllClose(sample1_, sample2_, atol=0.0, rtol=1e-06)

    def test_randomization_does_not_depend_on_sequence_indices(self):
        if False:
            for i in range(10):
                print('nop')
        dim = 2
        seed = 9427
        (sample1, _) = random.halton.sample(dim, sequence_indices=[0], seed=seed)
        (sample2, _) = random.halton.sample(dim, sequence_indices=[0, 1000], seed=seed)
        sample2 = sample2[:1, :]
        self.assertAllClose(self.evaluate(sample1), self.evaluate(sample2), rtol=1e-06)

    def test_many_small_batches_same_as_one_big_batch(self):
        if False:
            i = 10
            return i + 15
        dim = 2
        num_results_per_batch = 1
        num_batches = 3
        seed = 1925
        (sample1, _) = random.halton.sample(dim, num_results_per_batch * num_batches, seed=seed)
        batch_indices = (tf.range(i * num_results_per_batch, (i + 1) * num_results_per_batch) for i in range(num_batches))
        sample2 = (random.halton.sample(dim, sequence_indices=sequence_indices, seed=seed) for sequence_indices in batch_indices)
        result_set1 = set((tuple(row) for row in self.evaluate(sample1)))
        result_set2 = set()
        for (batch, _) in sample2:
            result_set2.update((tuple(row) for row in self.evaluate(batch)))
        self.assertEqual(result_set1, result_set2)

    def test_max_index_exceeded_raises(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(tf.errors.InvalidArgumentError):
            (sample, _) = random.halton.sample(1, sequence_indices=[2 ** 30], dtype=tf.float32, randomized=False, validate_args=True)
            self.evaluate(sample)

    def test_dim_is_negative(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(tf.errors.InvalidArgumentError):
            (sample, _) = random.halton.sample(-1, num_results=10, dtype=tf.float32, randomized=False, validate_args=True)
            self.evaluate(sample)

    def test_dim_too_big(self):
        if False:
            print('Hello World!')
        with self.assertRaises(tf.errors.InvalidArgumentError):
            (sample, _) = random.halton.sample(1001, num_results=10, dtype=tf.float32, randomized=False, validate_args=True)
            self.evaluate(sample)

    def test_reusing_params_returns_same_points(self):
        if False:
            return 10
        dim = 20
        num_results = 100
        (seed1, seed2) = (1925, 62278)
        (sample1, params) = random.halton.sample(dim, num_results=num_results, seed=seed1)
        (sample2, _) = random.halton.sample(dim, num_results=num_results, seed=seed2, randomization_params=params)
        [sample1_, sample2_] = self.evaluate([sample1, sample2])
        self.assertAllClose(sample1_, sample2_, atol=0.0, rtol=1e-06)

    def test_using_params_with_randomization_false_does_not_randomize(self):
        if False:
            i = 10
            return i + 15
        dim = 20
        num_results = 100
        (sample_plain, _) = random.halton.sample(dim, num_results=num_results, randomized=False)
        seed = 87226
        (_, params) = random.halton.sample(dim, num_results=num_results, seed=seed)
        (sample_with_params, _) = random.halton.sample(dim, num_results=num_results, randomized=False, randomization_params=params)
        self.assertAllClose(self.evaluate(sample_plain), self.evaluate(sample_with_params), atol=0.0, rtol=1e-06)
if __name__ == '__main__':
    tf.test.main()