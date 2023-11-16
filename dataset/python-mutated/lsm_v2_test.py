"""Tests for the regression Monte Carlo algorithm."""
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
import tf_quant_finance as tff
from tensorflow.python.framework import test_util
lsm_algorithm = tff.experimental.lsm_algorithm
_SAMPLES = [[1.0, 1.09, 1.08, 1.34], [1.0, 1.16, 1.26, 1.54], [1.0, 1.22, 1.07, 1.03], [1.0, 0.93, 0.97, 0.92], [1.0, 1.11, 1.56, 1.52], [1.0, 0.76, 0.77, 0.9], [1.0, 0.92, 0.84, 1.01], [1.0, 0.88, 1.22, 1.34]]

@test_util.run_all_in_graph_and_eager_modes
class LsmTest(parameterized.TestCase, tf.test.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        'Sets `samples` as in the Longstaff-Schwartz paper.'
        super(LsmTest, self).setUp()
        self.samples = np.expand_dims(_SAMPLES, -1)
        interest_rates = [0.06, 0.06, 0.06]
        self.discount_factors = np.exp(-np.cumsum(interest_rates))

    @parameterized.named_parameters({'testcase_name': 'SinglePrecision', 'dtype': np.float32}, {'testcase_name': 'DoublePrecision', 'dtype': np.float64})
    def test_european_option_put(self, dtype):
        if False:
            print('Hello World!')
        'Tests that LSM price of European put option is computed as expected.'
        basis_fn = lsm_algorithm.make_polynomial_basis_v2(2)
        payoff_fn = lsm_algorithm.make_basket_put_payoff([1.1], dtype=dtype)
        european_put_price = lsm_algorithm.least_square_mc_v2(self.samples, [3], payoff_fn, basis_fn, discount_factors=[self.discount_factors[-1]], dtype=dtype)
        self.assertAllClose(european_put_price, [0.0564], rtol=0.0001, atol=0.0001)

    @parameterized.named_parameters({'testcase_name': 'SinglePrecision', 'dtype': np.float32}, {'testcase_name': 'DoublePrecision', 'dtype': np.float64}, {'testcase_name': 'DoublePrecisionPassCalibrationSamples', 'dtype': np.float64})
    def test_american_option_put(self, dtype):
        if False:
            i = 10
            return i + 15
        'Tests that LSM price of American put option is computed as expected.'
        basis_fn = lsm_algorithm.make_polynomial_basis_v2(2)
        payoff_fn = lsm_algorithm.make_basket_put_payoff([1.1], dtype=dtype)
        american_put_price = lsm_algorithm.least_square_mc_v2(self.samples, [1, 2, 3], payoff_fn, basis_fn, discount_factors=self.discount_factors, dtype=dtype)
        self.assertAllClose(american_put_price, [0.1144], rtol=0.0001, atol=0.0001)

    @parameterized.named_parameters({'testcase_name': 'SinglePrecision', 'num_calibration_samples': 4, 'dtype': np.float32}, {'testcase_name': 'DoublePrecision', 'num_calibration_samples': 4, 'dtype': np.float64})
    def test_american_option_put_calibration(self, num_calibration_samples, dtype):
        if False:
            return 10
        'Tests that LSM price of American put option is computed as expected.'
        basis_fn = lsm_algorithm.make_polynomial_basis_v2(2)
        payoff_fn = lsm_algorithm.make_basket_put_payoff([1.1], dtype=dtype)
        american_put_price = lsm_algorithm.least_square_mc_v2(self.samples, [1, 2, 3], payoff_fn, basis_fn, discount_factors=self.discount_factors, num_calibration_samples=num_calibration_samples, dtype=dtype)
        self.assertAllClose(american_put_price, [0.174226], rtol=0.0001, atol=0.0001)

    def test_american_basket_option_put(self):
        if False:
            i = 10
            return i + 15
        'Tests the LSM price of American Basket put option.'
        basis_fn = lsm_algorithm.make_polynomial_basis_v2(10)
        exercise_times = [1, 2, 3]
        dtype = np.float64
        payoff_fn = lsm_algorithm.make_basket_put_payoff([1.1, 1.2, 1.3], dtype=dtype)
        samples = tf.convert_to_tensor(self.samples, dtype=dtype)
        samples_2d = tf.concat([samples, samples], -1)
        american_basket_put_price = lsm_algorithm.least_square_mc_v2(samples_2d, exercise_times, payoff_fn, basis_fn, discount_factors=self.discount_factors, dtype=dtype)
        american_put_price = lsm_algorithm.least_square_mc_v2(self.samples, exercise_times, payoff_fn, basis_fn, discount_factors=self.discount_factors, dtype=dtype)
        with self.subTest(name='Price'):
            self.assertAllClose(american_basket_put_price, american_put_price, rtol=0.0001, atol=0.0001)
        with self.subTest(name='Shape'):
            self.assertAllEqual(american_basket_put_price.shape, [3])

    @parameterized.named_parameters({'testcase_name': 'SinglePrecision', 'dtype': np.float32}, {'testcase_name': 'DoublePrecision', 'dtype': np.float64})
    def test_american_option_put_batch_payoff(self, dtype):
        if False:
            while True:
                i = 10
        'Tests that LSM price of American put option is computed as expected.'
        basis_fn = lsm_algorithm.make_polynomial_basis_v2(2)
        payoff_fn = lsm_algorithm.make_basket_put_payoff([1.1, 1.2], dtype=dtype)
        interest_rates = [[0.06, 0.06, 0.06], [0.05, 0.05, 0.05]]
        discount_factors = np.exp(-np.cumsum(interest_rates, -1))
        discount_factors = np.expand_dims(discount_factors, 0)
        american_put_price = lsm_algorithm.least_square_mc_v2(self.samples, [1, 2, 3], payoff_fn, basis_fn, discount_factors=discount_factors, dtype=dtype)
        self.assertAllClose(american_put_price, [0.1144, 0.199], rtol=0.0001, atol=0.0001)

    @parameterized.named_parameters({'testcase_name': 'SinglePrecision', 'dtype': np.float32}, {'testcase_name': 'DoublePrecision', 'dtype': np.float64})
    def test_american_option_put_batch_samples(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        'Tests LSM price of a batch of American put options.'
        basis_fn = lsm_algorithm.make_polynomial_basis_v2(2)
        payoff_fn = lsm_algorithm.make_basket_put_payoff([1.1, 1.2], dtype=dtype)
        interest_rates = [[0.06, 0.06, 0.06], [0.05, 0.05, 0.05]]
        discount_factors = np.exp(-np.cumsum(interest_rates, -1))
        discount_factors = np.expand_dims(discount_factors, 0)
        sample_paths1 = tf.convert_to_tensor(self.samples, dtype=dtype)
        sample_paths2 = sample_paths1 + 0.1
        sample_paths = tf.stack([sample_paths1, sample_paths2], axis=0)
        american_put_price = lsm_algorithm.least_square_mc_v2(sample_paths, [1, 2, 3], payoff_fn, basis_fn, discount_factors=discount_factors, dtype=dtype)
        self.assertAllClose(american_put_price, [0.1144, 0.1157], rtol=0.0001, atol=0.0001)
if __name__ == '__main__':
    tf.test.main()