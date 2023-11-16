"""Tests for `sample_paths` of `ItoProcess`."""
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.framework import test_util
from tf_quant_finance.models.legacy.brownian_motion import BrownianMotion

@test_util.run_all_in_graph_and_eager_modes
class BrownianMotionTest(tf.test.TestCase):

    def assertArrayEqual(self, arg1, arg2):
        if False:
            i = 10
            return i + 15
        self.assertArrayNear(arg1, arg2, 0.0)

    def test_default_construction_1d(self):
        if False:
            return 10
        'Tests the default parameters.'
        process = BrownianMotion()
        self.assertEqual(process.dim(), 1)
        drift_fn = process.drift_fn()
        t0 = np.array([0.2, 0.7, 0.9])
        t1 = t0 + [0.1, 0.8, 0.3]
        drifts = self.evaluate(drift_fn(t0, None))
        total_drift_fn = process.total_drift_fn()
        self.assertAlmostEqual(self.evaluate(total_drift_fn(0.4, 0.5)), 0.0, places=7)
        self.assertArrayNear(drifts, np.zeros([3]), 1e-10)
        variances = self.evaluate(process.total_covariance_fn()(t0, t1))
        self.assertArrayNear(variances, t1 - t0, 1e-10)
        self.assertAlmostEqual(self.evaluate(process.total_covariance_fn()(0.41, 0.55)), 0.14, places=7)

    def test_default_construction_2d(self):
        if False:
            i = 10
            return i + 15
        'Tests the default parameters for 2 dimensional Brownian Motion.'
        process = BrownianMotion(dim=2)
        self.assertEqual(process.dim(), 2)
        drift_fn = process.total_drift_fn()
        t0 = np.array([0.2, 0.7, 0.9])
        delta_t = np.array([0.1, 0.8, 0.3])
        t1 = t0 + delta_t
        drifts = self.evaluate(drift_fn(t0, t1))
        self.assertEqual(drifts.shape, (3, 2))
        self.assertArrayNear(drifts.reshape([-1]), np.zeros([3 * 2]), 1e-10)
        variances = self.evaluate(process.total_covariance_fn()(t0, t1))
        self.assertEqual(variances.shape, (3, 2, 2))
        expected_variances = np.eye(2) * delta_t.reshape([-1, 1, 1])
        print(variances, expected_variances)
        self.assertArrayNear(variances.reshape([-1]), expected_variances.reshape([-1]), 1e-10)

    def test_path_properties_1d(self):
        if False:
            print('Hello World!')
        'Tests path samples have the right properties.'
        process = BrownianMotion()
        times = np.array([0.2, 0.33, 0.7, 0.9, 1.88])
        num_samples = 10000
        paths = self.evaluate(process.sample_paths(times, num_samples=num_samples, initial_state=np.array(0.1), seed=1234))
        self.assertArrayEqual(paths.shape, (num_samples, 5, 1))
        self.assertArrayNear(np.mean(paths, axis=0).reshape([-1]), np.zeros(5) + 0.1, 0.05)
        covars = np.cov(paths.reshape([num_samples, 5]), rowvar=False)
        expected = np.minimum(times.reshape([-1, 1]), times.reshape([1, -1]))
        self.assertArrayNear(covars.reshape([-1]), expected.reshape([-1]), 0.05)

    def test_time_dependent_construction(self):
        if False:
            print('Hello World!')
        'Tests with time dependent drift and variance.'

        def vol_fn(t):
            if False:
                while True:
                    i = 10
            return tf.expand_dims(0.2 - 0.1 * tf.exp(-t), axis=-1)

        def variance_fn(t0, t1):
            if False:
                for i in range(10):
                    print('nop')
            tot_var = (t1 - t0) * 0.04 - (tf.exp(-2 * t1) - tf.exp(-2 * t0)) * 0.005
            tot_var += 0.04 * (tf.exp(-t1) - tf.exp(-t0))
            return tf.reshape(tot_var, [-1, 1, 1])
        process = BrownianMotion(dim=1, drift=0.1, volatility=vol_fn, total_covariance_fn=variance_fn)
        t0 = np.array([0.2, 0.7, 0.9])
        delta_t = np.array([0.1, 0.8, 0.3])
        t1 = t0 + delta_t
        drifts = self.evaluate(process.total_drift_fn()(t0, t1))
        self.assertArrayNear(drifts, 0.1 * delta_t, 1e-10)
        variances = self.evaluate(process.total_covariance_fn()(t0, t1))
        self.assertArrayNear(variances.reshape([-1]), [0.00149104, 0.02204584, 0.00815789], 1e-08)

    def test_paths_time_dependent(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests path properties with time dependent drift and variance.'

        def vol_fn(t):
            if False:
                print('Hello World!')
            return tf.expand_dims(0.2 - 0.1 * tf.exp(-t), axis=-1)

        def variance_fn(t0, t1):
            if False:
                for i in range(10):
                    print('nop')
            tot_var = (t1 - t0) * 0.04 - (tf.exp(-2 * t1) - tf.exp(-2 * t0)) * 0.005
            tot_var += 0.04 * (tf.exp(-t1) - tf.exp(-t0))
            return tf.reshape(tot_var, [-1, 1, 1])
        process = BrownianMotion(dim=1, drift=0.1, volatility=vol_fn, total_covariance_fn=variance_fn)
        times = np.array([0.2, 0.33, 0.7, 0.9, 1.88])
        num_samples = 10000
        paths = self.evaluate(process.sample_paths(times, num_samples=num_samples, initial_state=np.array(0.1), seed=12134))
        self.assertArrayEqual(paths.shape, (num_samples, 5, 1))
        self.assertArrayNear(np.mean(paths, axis=0).reshape([-1]), 0.1 + times * 0.1, 0.05)
        covars = np.cov(paths.reshape([num_samples, 5]), rowvar=False)
        min_times = np.minimum(times.reshape([-1, 1]), times.reshape([1, -1])).reshape([-1])
        expected_covars = self.evaluate(variance_fn(tf.zeros_like(min_times), min_times))
        self.assertArrayNear(covars.reshape([-1]), expected_covars, 0.005)

    def test_paths_multi_dim(self):
        if False:
            for i in range(10):
                print('nop')
        "Tests path properties for 2 dimensional brownian motion.\n\n    We construct the following 2 dimensional time dependent brownian motion.\n\n    dX_1 = mu_1 dt + s11 dW_1 + s12 dW_2\n    dX_2 = mu_2 dt + s21 dW_1 + s22 dW_2\n\n    mu_1, mu_2 are constants. s11, s12, s21, s22 are all linear functions of\n    time. Let s11 = a11 t + b11 and similarly for the other three coefficients.\n    Define the matrices:\n      A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]\n\n    Then the total covariance from 0 to time T is:\n\n    Total Covariance(0,T) = A.A' T**3 / 3 + (A.B'+B.A') T**2 / 2 + B.B' T\n\n    where A', B' are the transposes of A and B.\n    "
        mu = np.array([0.2, 0.7])
        a_mat = np.array([[0.4, 0.1], [0.3, 0.2]])
        b_mat = np.array([[0.33, -0.03], [0.21, 0.5]])
        c1 = np.matmul(a_mat, a_mat.transpose()) / 3
        c2 = np.matmul(a_mat, b_mat.transpose()) / 2
        c2 += c2.transpose()
        c3 = np.matmul(b_mat, b_mat.transpose())

        def vol_fn(t):
            if False:
                while True:
                    i = 10
            return a_mat * tf.reshape(t, [-1, 1, 1]) + b_mat

        def tot_cov_fn(t0, t1):
            if False:
                for i in range(10):
                    print('nop')
            t0 = tf.reshape(t0, [-1, 1, 1])
            t1 = tf.reshape(t1, [-1, 1, 1])
            return c1 * (t1 ** 3 - t0 ** 3) + c2 * (t1 ** 2 - t0 ** 2) + c3 * (t1 - t0)
        process = BrownianMotion(dim=2, drift=mu, volatility=vol_fn, total_covariance_fn=tot_cov_fn)
        times = np.array([0.1, 0.21, 0.32, 0.43, 0.55])
        num_samples = 10000
        initial_state = np.array([0.1, -1.1])
        paths = self.evaluate(process.sample_paths(times, num_samples=num_samples, initial_state=initial_state, seed=12134))
        self.assertArrayEqual(paths.shape, (num_samples, 5, 2))
        expected_means = np.reshape(times, [-1, 1]) * mu + initial_state
        self.assertArrayNear(np.mean(paths, axis=0).reshape([-1]), expected_means.reshape([-1]), 0.005)
        paths = np.transpose(paths, [0, 2, 1])
        covars = np.cov(paths.reshape([num_samples, -1]), rowvar=False)
        min_times = np.minimum(times.reshape([-1, 1]), times.reshape([1, -1])).reshape([-1])
        expected_covars = self.evaluate(tot_cov_fn(tf.zeros_like(min_times), min_times))
        expected_covars = np.transpose(expected_covars, (1, 2, 0))
        xx_actual = covars[:5, :5].reshape([-1])
        xx_expected = expected_covars[0, 0, :]
        self.assertArrayNear(xx_actual, xx_expected, 0.005)
        xy_actual = covars[:5, 5:].reshape([-1])
        xy_expected = expected_covars[0, 1, :]
        self.assertArrayNear(xy_actual, xy_expected, 0.005)
        yy_actual = covars[5:, 5:].reshape([-1])
        yy_expected = expected_covars[1, 1, :]
        self.assertArrayNear(yy_actual, yy_expected, 0.005)

    def test_unsorted_times_is_error(self):
        if False:
            return 10
        'Tests that supplying unsorted times in sample_paths is an error.'
        process = BrownianMotion()
        with self.assertRaises(tf.errors.InvalidArgumentError):
            self.evaluate(process.sample_paths([0.1, 0.09, 1.0], num_samples=1))

    def test_negative_times_is_error(self):
        if False:
            while True:
                i = 10
        'Tests that supplying negative times in sample_paths is an error.'
        process = BrownianMotion()
        with self.assertRaises(tf.errors.InvalidArgumentError):
            self.evaluate(process.sample_paths([-0.1, 0.09, 1.0], num_samples=1))
if __name__ == '__main__':
    tf.test.main()