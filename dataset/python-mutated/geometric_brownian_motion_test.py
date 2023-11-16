"""Tests for Geometric Brownian Motion."""
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
import tf_quant_finance as tff
from tensorflow.python.framework import test_util
from tf_quant_finance.models.geometric_brownian_motion import geometric_brownian_motion_test_utils
from tf_quant_finance.models.geometric_brownian_motion import univariate_geometric_brownian_motion
arrays_all_close = geometric_brownian_motion_test_utils.arrays_all_close
calculate_sample_paths_mean_and_variance = geometric_brownian_motion_test_utils.calculate_sample_paths_mean_and_variance
NUM_SAMPLES = 100000
NUM_STDERRS = 3.0

def _tolerance_by_dtype(dtype):
    if False:
        while True:
            i = 10
    'Returns the expected tolerance based on dtype.'
    return 1e-08 if dtype == np.float64 else 0.005

@test_util.run_all_in_graph_and_eager_modes
class GeometricBrownianMotionTest(parameterized.TestCase, tf.test.TestCase):

    @parameterized.named_parameters({'testcase_name': 'SinglePrecision', 'dtype': np.float32}, {'testcase_name': 'DoublePrecision', 'dtype': np.float64})
    def test_univariate_constant_drift_and_volatility(self, dtype):
        if False:
            return 10
        'Tests univariate GBM constant drift and volatility functions.'
        drift_in = 0.05
        vol_in = 0.5
        process = tff.models.GeometricBrownianMotion(drift_in, vol_in, dtype=dtype)
        drift_fn = process.drift_fn()
        volatility_fn = process.volatility_fn()
        state = np.array([[1.0], [2.0], [3.0]], dtype=dtype)
        with self.subTest('Drift'):
            drift = drift_fn(0.2, state)
            expected_drift = state * drift_in
            self.assertAllClose(drift, expected_drift, atol=1e-08, rtol=1e-08)
        with self.subTest('Volatility'):
            vol = volatility_fn(0.2, state)
            expected_vol = state * vol_in
            self.assertAllClose(vol, np.expand_dims(expected_vol, axis=-1), atol=1e-08, rtol=1e-08)

    def test_univariate_default_initialization(self):
        if False:
            return 10
        'Tests default initialization behavior of univariate sample_paths.'
        drift_in = 0.05
        vol_in = 0.5
        times = [0, 1, 2, 3]
        num_samples = 2
        dtype = tf.float64
        process = tff.models.GeometricBrownianMotion(drift_in, vol_in, dtype=dtype)
        sample_paths = process.sample_paths(times=times, num_samples=num_samples)
        self.assertAllEqual(sample_paths[:, 0, 0], np.ones((num_samples,)))

    @parameterized.named_parameters({'testcase_name': 'SinglePrecision', 'dtype': np.float32}, {'testcase_name': 'DoublePrecision', 'dtype': np.float64})
    def test_univariate_time_varying_drift_and_volatility(self, dtype):
        if False:
            print('Hello World!')
        'Tests univariate GBM time varying drift and volatility functions.'
        times = np.linspace(0.0, 10.0, 6, dtype=dtype)
        drift = np.append([0.0], np.sin(times, dtype=dtype))
        sigma = np.append([0.0], np.cos(times, dtype=dtype)) ** 2.0
        drift_in = tff.math.piecewise.PiecewiseConstantFunc(jump_locations=times, values=drift, dtype=dtype)
        sigma_in = tff.math.piecewise.PiecewiseConstantFunc(jump_locations=times, values=sigma, dtype=dtype)
        process = tff.models.GeometricBrownianMotion(drift_in, sigma_in, dtype=dtype)
        drift_fn = process.drift_fn()
        volatility_fn = process.volatility_fn()
        state = np.array([[1.0], [2.0], [3.0]], dtype=dtype)
        test_times = np.array([1.0, 3.5, 7.5, 9.8, 12], dtype=dtype)
        expected_drift = drift_in(test_times)
        expected_sigma = sigma_in(test_times)
        with self.subTest('Drift'):
            drift = drift_fn(test_times, state)
            self.assertAllClose(drift, expected_drift * state, atol=1e-08, rtol=1e-08)
        with self.subTest('Volatility'):
            vol = volatility_fn(test_times, state)
            self.assertAllClose(vol, expected_sigma * tf.expand_dims(state, -1), atol=1e-08, rtol=1e-08)

    @parameterized.named_parameters({'testcase_name': 'SinglePrecision', 'dtype': np.float32}, {'testcase_name': 'DoublePrecision', 'dtype': np.float64})
    def test_univariate_integrate_parameter(self, dtype):
        if False:
            while True:
                i = 10
        'Tests univariate GBM integrate parameter.'
        times = np.linspace(0.0, 10.0, 6, dtype=dtype)
        a_constant = 0.3
        constant_batched = np.random.uniform(size=(2, 3, 4)).astype(dtype)
        time_varying = np.append([0], np.cos(times, dtype=dtype))
        time_varying_fn = tff.math.piecewise.PiecewiseConstantFunc(jump_locations=times, values=time_varying, dtype=dtype)
        process = tff.models.GeometricBrownianMotion(a_constant, a_constant, dtype=dtype)
        start_times = np.array([1.0, 3.5, 7.5, 12], dtype=dtype)
        end_times = np.array([2.0, 6.0, 8.5, 16], dtype=dtype)
        rtol = atol = _tolerance_by_dtype(dtype)
        with self.subTest('Constant'):
            integral = process._integrate_parameter(a_constant, True, start_times, end_times)
            expected = a_constant * (end_times - start_times)
            self.assertAllClose(integral, expected, atol=atol, rtol=rtol)
        with self.subTest('Batched'):
            integral = process._integrate_parameter(constant_batched, True, start_times, end_times)
            expected = constant_batched * (end_times - start_times)
            self.assertAllClose(integral, expected, atol=atol, rtol=rtol)
        with self.subTest('Time dependent'):
            integral = process._integrate_parameter(time_varying_fn, False, start_times, end_times)
            expected = np.array([1.0 * np.cos(0.0, dtype=dtype), 0.5 * np.cos(2.0, dtype=dtype) + 2.0 * np.cos(4.0, dtype=dtype), 0.5 * np.cos(6.0, dtype=dtype) + 0.5 * np.cos(8.0, dtype=dtype), 4.0 * np.cos(10.0, dtype=dtype)], dtype=dtype)
            self.assertAllClose(integral, expected, atol=atol, rtol=rtol)

    @parameterized.named_parameters({'testcase_name': 'SinglePrecision', 'dtype': np.float32}, {'testcase_name': 'DoublePrecision', 'dtype': np.float64})
    def test_multivariate_drift_and_volatility(self, dtype):
        if False:
            while True:
                i = 10
        'Tests multivariate GBM drift and volatility functions.'
        means = [0.05, 0.02]
        volatilities = [0.1, 0.2]
        corr_matrix = [[1, 0.1], [0.1, 1]]
        process = tff.models.MultivariateGeometricBrownianMotion(dim=2, means=means, volatilities=volatilities, corr_matrix=corr_matrix, dtype=tf.float64)
        drift_fn = process.drift_fn()
        volatility_fn = process.volatility_fn()
        state = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=dtype)
        with self.subTest('Drift'):
            drift = drift_fn(0.2, state)
            expected_drift = np.array(means) * state
            self.assertAllClose(drift, expected_drift, atol=1e-08, rtol=1e-08)
        with self.subTest('Volatility'):
            vol = volatility_fn(0.2, state)
            expected_vol = np.expand_dims(np.array(volatilities) * state, axis=-1) * np.linalg.cholesky(corr_matrix)
            self.assertAllClose(vol, expected_vol, atol=1e-08, rtol=1e-08)

    @parameterized.named_parameters({'testcase_name': 'SinglePrecision', 'dtype': np.float32}, {'testcase_name': 'DoublePrecision', 'dtype': np.float64})
    def test_multivariate_drift_and_volatility_no_corr(self, dtype):
        if False:
            return 10
        'Tests multivariate GBM drift and volatility functions.'
        means = [0.05, 0.02]
        volatilities = [0.1, 0.2]
        corr_matrix = [[1, 0.0], [0.0, 1]]
        process = tff.models.MultivariateGeometricBrownianMotion(dim=2, means=means, volatilities=volatilities, dtype=tf.float64)
        drift_fn = process.drift_fn()
        volatility_fn = process.volatility_fn()
        state = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=dtype)
        with self.subTest('Drift'):
            drift = drift_fn(0.2, state)
            expected_drift = np.array(means) * state
            self.assertAllClose(drift, expected_drift, atol=1e-08, rtol=1e-08)
        with self.subTest('Volatility'):
            vol = volatility_fn(0.2, state)
            expected_vol = np.expand_dims(np.array(volatilities) * state, axis=-1) * np.linalg.cholesky(corr_matrix)
            self.assertAllClose(vol, expected_vol, atol=1e-08, rtol=1e-08)

    @parameterized.named_parameters({'testcase_name': 'SinglePrecisionNoDraws', 'supply_draws': False, 'dtype': np.float32}, {'testcase_name': 'DoublePrecisionNoDraws', 'supply_draws': False, 'dtype': np.float64}, {'testcase_name': 'DoublePrecisionWithDraws', 'supply_draws': True, 'dtype': np.float64})
    def test_univariate_sample_mean_and_variance_constant_parameters(self, supply_draws, dtype):
        if False:
            i = 10
            return i + 15
        'Tests the mean and vol of the univariate GBM sampled paths.'
        mu = 0.05
        sigma = 0.05
        times = np.array([0.1, 0.5, 1.0], dtype=dtype)
        initial_state = 2.0
        (mean, var, se_mean, se_var) = calculate_sample_paths_mean_and_variance(self, mu, sigma, times, initial_state, supply_draws, NUM_SAMPLES, dtype)
        expected_mean = (mu - sigma ** 2 / 2) * np.array(times) + np.log(initial_state)
        expected_var = sigma ** 2 * times
        atol_mean = se_mean * NUM_STDERRS
        atol_var = se_var * NUM_STDERRS
        with self.subTest('Drift'):
            arrays_all_close(self, tf.squeeze(mean), expected_mean, atol_mean, msg='comparing means')
        with self.subTest('Variance'):
            arrays_all_close(self, tf.squeeze(var), expected_var, atol_var, msg='comparing variances')

    @parameterized.named_parameters({'testcase_name': 'SinglePrecisionNoDraws', 'supply_draws': False, 'dtype': np.float32}, {'testcase_name': 'DoublePrecisionNoDraws', 'supply_draws': False, 'dtype': np.float64}, {'testcase_name': 'DoublePrecisionWithDraws', 'supply_draws': True, 'dtype': np.float64})
    def test_univariate_sample_mean_constant_parameters_batched(self, supply_draws, dtype):
        if False:
            i = 10
            return i + 15
        'Tests the mean and vol of the batched univariate GBM sampled paths.'
        mu = np.array([[0.05], [0.06], [0.04], [0.02]], dtype=dtype)
        sigma = np.array([[0.05], [0.1], [0.15], [0.2]], dtype=dtype)
        times = np.array([0.1, 0.5, 1.0], dtype=dtype)
        initial_state = np.array([[2.0], [10.0], [5.0], [25.0]], dtype=dtype)
        (mean, var, se_mean, se_var) = calculate_sample_paths_mean_and_variance(self, mu, sigma, times, initial_state, supply_draws, NUM_SAMPLES, dtype)
        expected_mean = (mu - sigma ** 2 / 2) * times + np.log(initial_state)
        expected_var = sigma ** 2 * times
        with self.subTest('Drift'):
            arrays_all_close(self, tf.squeeze(mean), expected_mean, se_mean * NUM_STDERRS, 'comparing means')
        with self.subTest('Variance'):
            arrays_all_close(self, tf.squeeze(var), expected_var, se_var * NUM_STDERRS, 'comparing variances')

    @parameterized.named_parameters({'testcase_name': 'SinglePrecisionNoDraws', 'supply_draws': False, 'dtype': np.float32}, {'testcase_name': 'DoublePrecisionNoDraws', 'supply_draws': False, 'dtype': np.float64}, {'testcase_name': 'DoublePrecisionWithDraws', 'supply_draws': True, 'dtype': np.float64})
    def test_univariate_sample_mean_constant_parameters_batched_time(self, supply_draws, dtype):
        if False:
            while True:
                i = 10
        'Tests the mean and vol of the batched univariate GBM sampled paths.'
        mu = np.array([[0.05], [0.06], [0.04], [0.03]], dtype=dtype)
        sigma = np.array([[0.05], [0.1], [0.15], [0.2]], dtype=dtype)
        times = np.array([[0.1, 0.5, 1.0], [0.2, 0.4, 2.0], [0.3, 0.6, 5.0], [0.4, 0.9, 7.0]], dtype=dtype)
        initial_state = np.array([[2.0], [10.0], [5.0], [25.0]], dtype=dtype)
        (mean, var, se_mean, se_var) = calculate_sample_paths_mean_and_variance(self, mu, sigma, times, initial_state, supply_draws, NUM_SAMPLES, dtype)
        expected_mean = (mu - sigma ** 2 / 2) * times + np.log(initial_state)
        expected_var = sigma ** 2 * times
        with self.subTest('Drift'):
            arrays_all_close(self, tf.squeeze(mean), expected_mean, se_mean * NUM_STDERRS, 'comparing means')
        with self.subTest('Variance'):
            arrays_all_close(self, tf.squeeze(var), expected_var, se_var * NUM_STDERRS, 'comparing variances')

    @parameterized.named_parameters({'testcase_name': 'SinglePrecisionNoDraws', 'supply_draws': False, 'dtype': np.float32}, {'testcase_name': 'DoublePrecisionNoDraws', 'supply_draws': False, 'dtype': np.float64}, {'testcase_name': 'DoublePrecisionWithDraws', 'supply_draws': True, 'dtype': np.float64})
    def test_univariate_sample_mean_var_constant_parameters_batched2(self, supply_draws, dtype):
        if False:
            for i in range(10):
                print('nop')
        'Tests the mean and vol of the batched univariate GBM sampled paths.'
        mu = (3.0 * np.random.uniform(size=(2, 3, 4, 1))).astype(dtype)
        sigma = (2.0 * np.random.uniform(size=(2, 3, 4, 1))).astype(dtype)
        times = np.array([1.4, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=dtype)
        initial_state = np.ones_like(mu, dtype=dtype) * 100.0
        (mean, var, se_mean, se_var) = calculate_sample_paths_mean_and_variance(self, mu, sigma, times, initial_state, supply_draws, NUM_SAMPLES, dtype)
        expected_mean = (mu - sigma ** 2 / 2) * times + np.log(initial_state)
        expected_var = sigma ** 2 * times
        with self.subTest('Drift'):
            arrays_all_close(self, tf.squeeze(mean), expected_mean, se_mean * NUM_STDERRS, msg='comparing means')
        with self.subTest('Variance'):
            arrays_all_close(self, tf.squeeze(var), expected_var, se_var * NUM_STDERRS, msg='comparing variances')

    @parameterized.named_parameters({'testcase_name': 'SinglePrecisionNoDraws', 'supply_draws': False, 'dtype': np.float32}, {'testcase_name': 'DoublePrecisionNoDraws', 'supply_draws': False, 'dtype': np.float64}, {'testcase_name': 'DoublePrecisionWithDraws', 'supply_draws': True, 'dtype': np.float64})
    def test_univariate_sample_mean_var_constant_parameters_batched_time2(self, supply_draws, dtype):
        if False:
            print('Hello World!')
        'Tests the mean and vol of the batched univariate GBM sampled paths.'
        mu = (3.0 * np.random.uniform(size=(2, 3, 4, 1))).astype(dtype)
        sigma = (2.0 * np.random.uniform(size=(2, 3, 4, 1))).astype(dtype)
        times = np.reshape(np.arange(1.0, 1.0 + 2 * 3 * 4 * 7, 1.0, dtype=dtype), (2, 3, 4, 7))
        initial_state = np.ones_like(mu, dtype=dtype) * 100.0
        (mean, var, se_mean, se_var) = calculate_sample_paths_mean_and_variance(self, mu, sigma, times, initial_state, supply_draws, NUM_SAMPLES, dtype)
        expected_mean = (mu - sigma ** 2 / 2) * times + np.log(initial_state)
        expected_var = sigma ** 2 * times
        with self.subTest('Drift'):
            arrays_all_close(self, tf.squeeze(mean), expected_mean, se_mean * NUM_STDERRS, msg='comparing means')
        with self.subTest('Variance'):
            arrays_all_close(self, tf.squeeze(var), expected_var, se_var * NUM_STDERRS, msg='comparing variances')

    @parameterized.named_parameters({'testcase_name': 'SinglePrecisionNoDraws', 'supply_draws': False, 'dtype': np.float32}, {'testcase_name': 'DoublePrecisionNoDraws', 'supply_draws': False, 'dtype': np.float64}, {'testcase_name': 'DoublePrecisionWithDraws', 'supply_draws': True, 'dtype': np.float64})
    def test_univariate_sample_mean_and_variance_time_varying_drift(self, supply_draws, dtype):
        if False:
            while True:
                i = 10
        'Tests the mean and vol of the univariate GBM sampled paths.'
        initial_state = 2.0
        min_tol = _tolerance_by_dtype(dtype)
        with self.subTest('Drift as a step function, sigma = 0.0'):
            mu_times = np.array([0.0, 5.0, 10.0], dtype=dtype)
            mu_values = np.array([0.0, 0.0, 0.05, 0.05], dtype=dtype)
            mu = tff.math.piecewise.PiecewiseConstantFunc(jump_locations=mu_times, values=mu_values, dtype=dtype)
            sigma = 0.0
            times = np.array([0.0, 1.0, 5.0, 7.0, 10.0], dtype=dtype)
            (mean, var, se_mean, se_var) = calculate_sample_paths_mean_and_variance(self, mu, sigma, times, initial_state, supply_draws, NUM_SAMPLES, dtype)
            expected_mean = np.array([0.0, 0.0, 0.0, 2.0 * 0.05, 5.0 * 0.05], dtype=dtype) + np.log(initial_state)
            expected_var = sigma * np.sqrt(times)
            mean_tol = np.maximum(se_mean * NUM_STDERRS, min_tol)
            var_tol = np.maximum(se_var * NUM_STDERRS, min_tol)
            arrays_all_close(self, tf.squeeze(mean), expected_mean, mean_tol, msg='comparing means')
            arrays_all_close(self, tf.squeeze(var), expected_var, var_tol, msg='comparing variances')
        with self.subTest('Drift = 0.05, sigma = step function'):
            mu = 0.05
            sigma_times = np.array([0.0, 5.0, 10.0], dtype=dtype)
            sigma_values = np.array([0.0, 0.2, 0.4, 0.6], dtype=dtype)
            sigma = tff.math.piecewise.PiecewiseConstantFunc(jump_locations=sigma_times, values=sigma_values, dtype=dtype)
            times = np.array([0.0, 1.0, 5.0, 7.0, 10.0], dtype=dtype)
            (mean, var, se_mean, se_var) = calculate_sample_paths_mean_and_variance(self, mu, sigma, times, initial_state, supply_draws, NUM_SAMPLES, dtype)
            expected_mean = np.array([0.0, 1.0 * mu - 0.5 * 1.0 * 0.2 ** 2, 5.0 * mu - 0.5 * 5.0 * 0.2 ** 2, 7.0 * mu - 0.5 * (5.0 * 0.2 ** 2 + 2.0 * 0.4 ** 2), 10.0 * mu - 0.5 * (5.0 * 0.2 ** 2 + 5.0 * 0.4 ** 2)], dtype=dtype) + np.log(initial_state)
            expected_var = np.array([0.0, 1.0 * 0.2 ** 2, 5.0 * 0.2 ** 2, 5.0 * 0.2 ** 2 + 2.0 * 0.4 ** 2, 5.0 * 0.2 ** 2 + 5.0 * 0.4 ** 2], dtype=dtype)
            mean_tol = np.maximum(se_mean * NUM_STDERRS, min_tol)
            var_tol = np.maximum(se_var * NUM_STDERRS, min_tol)
            arrays_all_close(self, tf.squeeze(mean), expected_mean, mean_tol, msg='comparing means')
            arrays_all_close(self, tf.squeeze(var), expected_var, var_tol, msg='comparing variances')

    @parameterized.named_parameters({'testcase_name': 'SinglePrecisionNoDraws', 'supply_draws': False, 'dtype': np.float32}, {'testcase_name': 'DoublePrecisionNoDraws', 'supply_draws': False, 'dtype': np.float64}, {'testcase_name': 'DoublePrecisionWithDraws', 'supply_draws': True, 'dtype': np.float64})
    def test_univariate_time_varying_drift_batched(self, supply_draws, dtype):
        if False:
            print('Hello World!')
        'Tests the mean and vol of the univariate GBM sampled paths.'
        initial_state = 2.0
        min_tol = _tolerance_by_dtype(dtype)
        mu_times = np.array([[0.0, 5.0, 10.0], [0.0, 7.0, 10.0]], dtype=dtype)
        mu_values = np.array([[0.0, 0.0, 0.05, 0.05], [0.01, 0.01, 0.07, 0.07]], dtype=dtype)
        mu = tff.math.piecewise.PiecewiseConstantFunc(jump_locations=mu_times, values=mu_values, dtype=dtype)
        sigma = 0.0
        times = np.array([[0.0, 1.0, 5.0, 7.0, 10.0], [0.0, 1.5, 3.2, 4.8, 25.3]], dtype=dtype)
        (mean, var, se_mean, _) = calculate_sample_paths_mean_and_variance(self, mu, sigma, times, initial_state, supply_draws, NUM_SAMPLES, dtype)
        expected_mean = np.array([[0.0, 0.0, 0.0, 2.0 * 0.05, 5.0 * 0.05], [0.0, 1.5 * 0.01, 3.2 * 0.01, 4.8 * 0.01, 7.0 * 0.01 + 18.3 * 0.07]], dtype=dtype) + np.log(initial_state)
        expected_var = np.zeros((2, 5), dtype=dtype)
        mean_tol = np.maximum(se_mean * NUM_STDERRS, min_tol)
        var_tol = np.ones_like(expected_var) * min_tol
        arrays_all_close(self, tf.squeeze(mean), expected_mean, atol=mean_tol, msg='comparing means')
        arrays_all_close(self, tf.squeeze(var), expected_var, var_tol, msg='comparing variances')

    @parameterized.named_parameters({'testcase_name': 'SinglePrecisionNoDraws', 'supply_draws': False, 'dtype': np.float32}, {'testcase_name': 'DoublePrecisionNoDraws', 'supply_draws': False, 'dtype': np.float64}, {'testcase_name': 'DoublePrecisionWithDraws', 'supply_draws': True, 'dtype': np.float64})
    def test_univariate_sample_mean_and_variance_time_varying_vol(self, supply_draws, dtype):
        if False:
            while True:
                i = 10
        'Tests the mean and vol of the univariate GBM sampled paths.'
        initial_state = 2.0
        min_tol = _tolerance_by_dtype(dtype)
        mu = 0.05
        sigma_times = np.array([0.0, 5.0, 10.0], dtype=dtype)
        sigma_values = np.array([0.0, 0.2, 0.4, 0.6], dtype=dtype)
        sigma = tff.math.piecewise.PiecewiseConstantFunc(jump_locations=sigma_times, values=sigma_values, dtype=dtype)
        times = np.array([0.0, 1.0, 5.0, 7.0, 10.0], dtype=dtype)
        (mean, var, se_mean, se_var) = calculate_sample_paths_mean_and_variance(self, mu, sigma, times, initial_state, supply_draws, NUM_SAMPLES, dtype)
        expected_mean = np.array([0.0, 1.0 * mu - 0.5 * 1.0 * 0.2 ** 2, 5.0 * mu - 0.5 * 5.0 * 0.2 ** 2, 7.0 * mu - 0.5 * (5.0 * 0.2 ** 2 + 2.0 * 0.4 ** 2), 10.0 * mu - 0.5 * (5.0 * 0.2 ** 2 + 5.0 * 0.4 ** 2)], dtype=dtype) + np.log(initial_state)
        expected_var = np.array([0.0, 1.0 * 0.2 ** 2, 5.0 * 0.2 ** 2, 5.0 * 0.2 ** 2 + 2.0 * 0.4 ** 2, 5.0 * 0.2 ** 2 + 5.0 * 0.4 ** 2], dtype=dtype)
        mean_tol = np.maximum(se_mean * NUM_STDERRS, min_tol)
        var_tol = np.maximum(se_var * NUM_STDERRS, min_tol)
        arrays_all_close(self, tf.squeeze(mean), expected_mean, mean_tol, msg='comparing means')
        arrays_all_close(self, tf.squeeze(var), expected_var, var_tol, msg='comparing variances')

    @parameterized.named_parameters({'testcase_name': 'SinglePrecisionNoDraws', 'supply_draws': False, 'dtype': np.float32}, {'testcase_name': 'DoublePrecisionNoDraws', 'supply_draws': False, 'dtype': np.float64}, {'testcase_name': 'DoublePrecisionWithDraws', 'supply_draws': True, 'dtype': np.float64})
    def test_univariate_time_varying_vol_batched(self, supply_draws, dtype):
        if False:
            i = 10
            return i + 15
        'Tests the mean and vol of the univariate GBM sampled paths.'
        initial_state = 2.0
        min_tol = 0.005
        mu = 0.05
        sigma_times = np.array([[0.0, 5.0, 10.0], [0.0, 7.0, 10.0]], dtype=dtype)
        sigma_values = np.array([[0.4, 0.6, 0.8, 0.4], [0.5, 0.1, 0.3, 0.1]], dtype=dtype)
        sigma = tff.math.piecewise.PiecewiseConstantFunc(jump_locations=sigma_times, values=sigma_values, dtype=dtype)
        times = np.array([[0.0, 1.3, 4.5, 7.5, 19.0]], dtype=dtype)
        (mean, var, se_mean, se_var) = calculate_sample_paths_mean_and_variance(self, mu, sigma, times, initial_state, supply_draws, NUM_SAMPLES, dtype)
        expected_mean = np.array([[0.0, 1.3 * mu - 0.5 * 1.3 * 0.6 ** 2, 4.5 * mu - 0.5 * 4.5 * 0.6 ** 2, 7.5 * mu - 0.5 * (5.0 * 0.6 ** 2 + 2.5 * 0.8 ** 2), 19.0 * mu - 0.5 * (5.0 * 0.6 ** 2 + 5.0 * 0.8 ** 2 + 9.0 * 0.4 ** 2)], [0.0, 1.3 * mu - 0.5 * 1.3 * 0.1 ** 2, 4.5 * mu - 0.5 * 4.5 * 0.1 ** 2, 7.5 * mu - 0.5 * (7.0 * 0.1 ** 2 + 0.5 * 0.3 ** 2), 19.0 * mu - 0.5 * (7.0 * 0.1 ** 2 + 3.0 * 0.3 ** 2 + 9.0 * 0.1 ** 2)]], dtype=dtype) + np.log(initial_state)
        expected_var = np.array([[0.0, 1.3 * 0.6 ** 2, 4.5 * 0.6 ** 2, 5.0 * 0.6 ** 2 + 2.5 * 0.8 ** 2, 5.0 * 0.6 ** 2 + 5.0 * 0.8 ** 2 + 9.0 * 0.4 ** 2], [0.0, 1.3 * 0.1 ** 2, 4.5 * 0.1 ** 2, 7.0 * 0.1 ** 2 + 0.5 * 0.3 ** 2, 7.0 * 0.1 ** 2 + 3.0 * 0.3 ** 2 + 9.0 * 0.1 ** 2]], dtype=dtype)
        mean_tol = np.maximum(se_mean * NUM_STDERRS, min_tol)
        var_tol = np.maximum(se_var * NUM_STDERRS, min_tol)
        arrays_all_close(self, tf.squeeze(mean), expected_mean, mean_tol, msg='comparing means')
        arrays_all_close(self, tf.squeeze(var), expected_var, var_tol, msg='comparing variances')

    @parameterized.named_parameters({'testcase_name': 'SinglePrecisionNoDraws', 'supply_draws': False, 'dtype': np.float32}, {'testcase_name': 'DoublePrecisionNoDraws', 'supply_draws': False, 'dtype': np.float64}, {'testcase_name': 'DoublePrecisionWithDraws', 'supply_draws': True, 'dtype': np.float64})
    def test_univariate_time_varying_vol_batched_time(self, supply_draws, dtype):
        if False:
            while True:
                i = 10
        'Tests the mean and vol of the univariate GBM sampled paths.'
        initial_state = 2.0
        min_tol = 0.005
        mu = 0.05
        sigma_times = np.array([[0.0, 5.0, 10.0], [0.0, 7.0, 10.0]], dtype=dtype)
        sigma_values = np.array([[0.2, 0.2, 0.4, 0.4], [0.5, 0.5, 0.3, 0.1]], dtype=dtype)
        sigma = tff.math.piecewise.PiecewiseConstantFunc(jump_locations=sigma_times, values=sigma_values, dtype=dtype)
        times = np.array([[0.0, 1.0, 5.0, 7.0, 12.0], [0.0, 1.5, 3.5, 9.0, 17.0]], dtype=dtype)
        (mean, var, se_mean, se_var) = calculate_sample_paths_mean_and_variance(self, mu, sigma, times, initial_state, supply_draws, NUM_SAMPLES, dtype)
        expected_mean = np.array([[0.0, 1.0 * mu - 0.5 * 1.0 * 0.2 ** 2, 5.0 * mu - 0.5 * 5.0 * 0.2 ** 2, 7.0 * mu - 0.5 * (5.0 * 0.2 ** 2 + 2.0 * 0.4 ** 2), 12.0 * mu - 0.5 * (5.0 * 0.2 ** 2 + 7.0 * 0.4 ** 2)], [0.0, 1.5 * mu - 0.5 * 1.5 * 0.5 ** 2, 3.5 * mu - 0.5 * 3.5 * 0.5 ** 2, 9.0 * mu - 0.5 * (7.0 * 0.5 ** 2 + 2.0 * 0.3 ** 2), 17.0 * mu - 0.5 * (7.0 * 0.5 ** 2 + 3.0 * 0.3 ** 2 + 7.0 * 0.1 ** 2)]], dtype=dtype) + np.log(initial_state)
        expected_var = np.array([[0.0, 1.0 * 0.2 ** 2, 5.0 * 0.2 ** 2, 5.0 * 0.2 ** 2 + 2.0 * 0.4 ** 2, 5.0 * 0.2 ** 2 + 7.0 * 0.4 ** 2], [0.0, 1.5 * 0.5 ** 2, 3.5 * 0.5 ** 2, 7.0 * 0.5 ** 2 + 2.0 * 0.3 ** 2, 7.0 * 0.5 ** 2 + 3.0 * 0.3 ** 2 + 7.0 * 0.1 ** 2]], dtype=dtype)
        mean_tol = np.maximum(se_mean * NUM_STDERRS, min_tol)
        var_tol = np.maximum(se_var * NUM_STDERRS, min_tol)
        arrays_all_close(self, tf.squeeze(mean), expected_mean, mean_tol, msg='comparing means')
        arrays_all_close(self, tf.squeeze(var), expected_var, var_tol, msg='comparing variances')

    @parameterized.named_parameters({'testcase_name': 'SinglePrecisionNoDraws', 'supply_draws': False, 'dtype': np.float32}, {'testcase_name': 'DoublePrecisionNoDraws', 'supply_draws': False, 'dtype': np.float64})
    def test_univariate_time_varying_vol_batched_time_broadcast(self, supply_draws, dtype):
        if False:
            i = 10
            return i + 15
        'Tests the mean of a univariate GBM sampled paths has the right shape.'
        initial_state = 2.0
        mu = 0.05
        expected_shape = (2, 3, 4, 5)
        sigma_times = np.array([0.0, 5.0, 10.0, 15.0, 20.0], dtype=dtype) * np.ones(expected_shape, dtype=dtype)
        sigma_values = np.array([0.2, 0.2, 0.4, 0.4, 0.3, 0.1], dtype=dtype) * np.ones((2, 3, 4, 6), dtype=dtype)
        sigma = tff.math.piecewise.PiecewiseConstantFunc(jump_locations=sigma_times, values=sigma_values, dtype=dtype)
        times = np.array([0.0, 1.0, 5.0, 7.0, 12.0], dtype=dtype) * np.ones((4, 5), dtype=dtype)
        (mean, var, _, _) = calculate_sample_paths_mean_and_variance(self, mu, sigma, times, initial_state, supply_draws, NUM_SAMPLES, dtype)
        with self.subTest('Means'):
            self.assertAllClose(mean.shape, expected_shape, atol=0.001)
        with self.subTest('Variances'):
            self.assertAllClose(var.shape, expected_shape, atol=0.001)

    @parameterized.named_parameters({'testcase_name': 'SinglePrecisionNoDraws', 'supply_draws': False, 'dtype': np.float32}, {'testcase_name': 'DoublePrecisionNoDraws', 'supply_draws': False, 'dtype': np.float64}, {'testcase_name': 'DoublePrecisionWithDraws', 'supply_draws': True, 'dtype': np.float64})
    def test_multivariate_sample_mean_and_variance(self, supply_draws, dtype):
        if False:
            while True:
                i = 10
        'Tests the mean and vol of the univariate GBM sampled paths.'
        means = 0.05
        volatilities = [0.1, 0.2]
        corr_matrix = [[1, 0.1], [0.1, 1]]
        process = tff.models.MultivariateGeometricBrownianMotion(dim=2, means=means, volatilities=volatilities, corr_matrix=corr_matrix, dtype=dtype)
        times = [0.1, 0.5, 1.0]
        initial_state = [1.0, 2.0]
        num_samples_local = 10000
        normal_draws = None
        if supply_draws:
            num_samples_local = 50000
            normal_draws = tf.random.stateless_normal([num_samples_local // 2, 3, 2], seed=[4, 2], dtype=dtype)
            normal_draws = tf.concat([normal_draws, -normal_draws], axis=0)
        samples = process.sample_paths(times=times, initial_state=initial_state, random_type=tff.math.random.RandomType.SOBOL, num_samples=num_samples_local, normal_draws=normal_draws)
        log_s = tf.math.log(samples)
        mean = tf.reduce_mean(log_s, axis=0, keepdims=True)
        var = tf.reduce_mean((log_s - mean) ** 2, axis=0)
        expected_mean = (process._means - process._vols ** 2 / 2) * np.array(np.expand_dims(times, -1)) + np.log(initial_state)
        expected_var = process._vols ** 2 * np.array(np.expand_dims(times, -1))
        with self.subTest('Drift'):
            self.assertAllClose(tf.squeeze(mean), expected_mean, atol=0.001, rtol=0.001)
        with self.subTest('Variance'):
            self.assertAllClose(tf.squeeze(var), expected_var, atol=0.001, rtol=0.001)
        with self.subTest('Correlations'):
            samples = self.evaluate(samples)
            for i in range(len(times)):
                corr = np.corrcoef(samples[:, i, :], rowvar=False)
            self.assertAllClose(corr, corr_matrix, atol=0.01, rtol=0.01)

    def test_univariate_xla_compatible(self):
        if False:
            print('Hello World!')
        'Tests that univariate GBM sampling is XLA-compatible.'
        process = tff.models.GeometricBrownianMotion(0.05, 0.5, dtype=tf.float64)

        @tf.function
        def sample_fn():
            if False:
                while True:
                    i = 10
            return process.sample_paths(times=[0.1, 0.5, 1.0], initial_state=2.0, num_samples=10000)
        samples = tf.function(sample_fn, jit_compile=True)()
        log_s = tf.math.log(samples)
        mean = tf.reduce_mean(log_s, axis=0)
        expected_mean = (process._mean - process._volatility ** 2 / 2) * np.array([0.1, 0.5, 1.0]) + np.log(2.0)
        self.assertAllClose(tf.squeeze(mean), expected_mean, atol=0.01, rtol=0.01)

    def test_multivariate_xla_compatible(self):
        if False:
            return 10
        'Tests that multivariate GBM sampling is XLA-compatible.'
        corr_matrix = [[1, 0.1], [0.1, 1]]
        process = tff.models.MultivariateGeometricBrownianMotion(dim=2, means=0.05, volatilities=[0.1, 0.2], corr_matrix=corr_matrix, dtype=tf.float64)
        times = [0.1, 0.5, 1.0]
        initial_state = [1.0, 2.0]

        @tf.function(jit_compile=True)
        def sample_fn():
            if False:
                for i in range(10):
                    print('nop')
            return process.sample_paths(times=times, initial_state=initial_state, num_samples=10000)
        samples = sample_fn()
        log_s = tf.math.log(samples)
        mean = tf.reduce_mean(log_s, axis=0)
        expected_mean = (process._means - process._vols ** 2 / 2) * np.array(np.expand_dims(times, -1)) + np.log(initial_state)
        self.assertAllClose(mean, expected_mean, atol=0.01, rtol=0.01)

    def test_multivariate_dynamic_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests that GBM sampling can accept dynamically shaped inputs.'
        corr_matrix = [[1, 0.1], [0.1, 1]]
        times = [0.1, 0.5, 1.0]
        initial_state = [1.0, 2.0]

        @tf.function(input_signature=[tf.TensorSpec([None], dtype=tf.float64, name='initial_state'), tf.TensorSpec([None], dtype=tf.float64, name='means'), tf.TensorSpec([None], dtype=tf.float64, name='volatilities'), tf.TensorSpec([2, 2], dtype=tf.float64, name='corr_matrix'), tf.TensorSpec([None], dtype=tf.float64, name='times')])
        def sample_fn(initial_state, means, volatilities, corr_matrix, times):
            if False:
                print('Hello World!')
            process = tff.models.MultivariateGeometricBrownianMotion(dim=2, means=means, volatilities=volatilities, corr_matrix=corr_matrix, dtype=tf.float64)
            return process.sample_paths(times=times, initial_state=initial_state, num_samples=10000)
        means = np.array([0.05], dtype=np.float64)
        volatilities = np.array([0.1, 0.2], dtype=np.float64)
        samples = sample_fn(initial_state=initial_state, means=means, volatilities=volatilities, corr_matrix=corr_matrix, times=times)
        log_s = tf.math.log(samples)
        expected_means = (means - volatilities ** 2 / 2) * np.array(np.expand_dims(times, -1)) + np.log(initial_state)
        actual_means = tf.reduce_mean(log_s, axis=0)
        self.assertAllClose(actual_means, expected_means, atol=0.01, rtol=0.01)

    def test_univariate_dynamic_inputs(self):
        if False:
            return 10
        'Tests that GBM sampling can accept dynamically shaped inputs.'
        times = [0.1, 0.5, 1.0]
        initial_state = [[1.0], [2.0]]

        @tf.function(input_signature=[tf.TensorSpec([None, None], dtype=tf.float64, name='initial_state'), tf.TensorSpec([None, 1], dtype=tf.float64, name='mean'), tf.TensorSpec([None, 1], dtype=tf.float64, name='volatility'), tf.TensorSpec([None], dtype=tf.float64, name='times')])
        def sample_fn(initial_state, mean, volatility, times):
            if False:
                print('Hello World!')
            process = tff.models.GeometricBrownianMotion(mean=mean, volatility=volatility, dtype=tf.float64)
            return process.sample_paths(times=times, initial_state=initial_state, num_samples=10000)
        mean = np.array([[0.05], [0.1]], dtype=np.float64)
        volatility = np.array([[0.1]], dtype=np.float64)
        samples = sample_fn(initial_state=initial_state, mean=mean, volatility=volatility, times=times)
        log_s = tf.math.log(samples)
        expected_mean = (mean - volatility ** 2 / 2) * np.array(np.expand_dims(times, 0)) + np.log(initial_state)
        actual_mean = tf.reduce_mean(log_s, axis=1)
        self.assertAllClose(actual_mean, expected_mean[..., np.newaxis], atol=0.01, rtol=0.01)

    def test_normal_draws_shape_mismatch_2d(self):
        if False:
            print('Hello World!')
        'Error is raised if `dim` is mismatched with the one from normal_draws.'
        dtype = tf.float64
        process = tff.models.MultivariateGeometricBrownianMotion(dim=2, means=0.05, volatilities=[0.1, 0.2], dtype=dtype)
        with self.subTest('WrongDim'):
            with self.assertRaises(ValueError):
                normal_draws = tf.random.normal(shape=[100, 3, 3], dtype=dtype)
                process.sample_paths(times=[0.1, 0.5, 1.0], normal_draws=normal_draws)

    def test_normal_draws_shape_mismatch_1d(self):
        if False:
            i = 10
            return i + 15
        'Error is raised if `dim` is mismatched with the one from normal_draws.'
        dtype = tf.float64
        process = tff.models.GeometricBrownianMotion(mean=0.05, volatility=0.1, dtype=dtype)
        with self.subTest('WrongDim'):
            with self.assertRaises(ValueError):
                normal_draws = tf.random.normal(shape=[100, 3, 2], dtype=dtype)
                process.sample_paths(times=[0.1, 0.5, 1.0], normal_draws=normal_draws)

    def test_gbm_normal_draws_gradient(self):
        if False:
            for i in range(10):
                print('nop')
        'Gradient through paths wrt volatility can be computed.'
        dtype = tf.float64
        volatility = tf.constant(0.1, dtype=dtype)
        with tf.GradientTape() as tape:
            tape.watch(volatility)
            process = tff.models.GeometricBrownianMotion(mean=0.05, volatility=volatility, dtype=dtype)
            samples = process.sample_paths(times=[0.0, 0.1, 0.5, 1.0], num_samples=10000, seed=[4, 2], random_type=tff.math.random.RandomType.STATELESS_ANTITHETIC)
            grad = tape.gradient(tf.reduce_mean(samples), volatility)
        grad = self.evaluate(grad)
        self.assertAlmostEqual(grad, 0.0, delta=0.001)

    def test_sqrt_grad(self):
        if False:
            i = 10
            return i + 15
        'Test for the custom square root.'
        dtype = tf.float64
        x1 = tf.random.stateless_uniform(shape=[10, 5], seed=[1, 2], dtype=dtype)
        x2 = tf.constant([0.0, 1.0], dtype=dtype)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x1, x2])
            y1 = univariate_geometric_brownian_motion._sqrt_no_nan(x1)
            y2 = univariate_geometric_brownian_motion._sqrt_no_nan(x2)
            y1_true = tf.sqrt(x1)
        with self.subTest('Value'):
            self.assertAllClose(y1, y1_true)
        with self.subTest('GradientCorrect'):
            self.assertAllClose(tape.gradient(y1, x1), tape.gradient(y1_true, x1))
        with self.subTest('ZeroGradientCorrect'):
            self.assertAllClose(tape.gradient(y2, x2), [0.0, 0.5])
if __name__ == '__main__':
    tf.test.main()