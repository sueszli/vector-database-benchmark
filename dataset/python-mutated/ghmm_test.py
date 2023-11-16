"""Tests for fivo.models.ghmm"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from fivo.models.ghmm import GaussianHMM
from fivo.models.ghmm import TrainableGaussianHMM

class GHMMTest(tf.test.TestCase):

    def test_transition_no_weights(self):
        if False:
            for i in range(10):
                print('nop')
        with self.test_session() as sess:
            ghmm = GaussianHMM(3, transition_variances=[1.0, 2.0, 3.0])
            prev_z = tf.constant([1.0, 2.0], dtype=tf.float32)
            z0 = ghmm.transition(0, prev_z)
            z1 = ghmm.transition(1, prev_z)
            z2 = ghmm.transition(2, prev_z)
            outs = sess.run([z0.mean(), z0.variance(), z1.mean(), z1.variance(), z2.mean(), z2.variance()])
            self.assertAllClose(outs, [[0.0, 0.0], [1.0, 1.0], [1.0, 2.0], [2.0, 2.0], [1.0, 2.0], [3.0, 3.0]])

    def test_transition_with_weights(self):
        if False:
            return 10
        with self.test_session() as sess:
            ghmm = GaussianHMM(3, transition_variances=[1.0, 2.0, 3.0], transition_weights=[2.0, 3.0])
            prev_z = tf.constant([1.0, 2.0], dtype=tf.float32)
            z0 = ghmm.transition(0, prev_z)
            z1 = ghmm.transition(1, prev_z)
            z2 = ghmm.transition(2, prev_z)
            outs = sess.run([z0.mean(), z0.variance(), z1.mean(), z1.variance(), z2.mean(), z2.variance()])
            self.assertAllClose(outs, [[0.0, 0.0], [1.0, 1.0], [2.0, 4.0], [2.0, 2.0], [3.0, 6.0], [3.0, 3.0]])

    def test_emission_no_weights(self):
        if False:
            for i in range(10):
                print('nop')
        with self.test_session() as sess:
            ghmm = GaussianHMM(3, emission_variances=[1.0, 2.0, 3.0])
            z = tf.constant([1.0, 2.0], dtype=tf.float32)
            x0 = ghmm.emission(0, z)
            x1 = ghmm.emission(1, z)
            x2 = ghmm.emission(2, z)
            outs = sess.run([x0.mean(), x0.variance(), x1.mean(), x1.variance(), x2.mean(), x2.variance()])
            self.assertAllClose(outs, [[1.0, 2.0], [1.0, 1.0], [1.0, 2.0], [2.0, 2.0], [1.0, 2.0], [3.0, 3.0]])

    def test_emission_with_weights(self):
        if False:
            i = 10
            return i + 15
        with self.test_session() as sess:
            ghmm = GaussianHMM(3, emission_variances=[1.0, 2.0, 3.0], emission_weights=[1.0, 2.0, 3.0])
            z = tf.constant([1.0, 2.0], dtype=tf.float32)
            x0 = ghmm.emission(0, z)
            x1 = ghmm.emission(1, z)
            x2 = ghmm.emission(2, z)
            outs = sess.run([x0.mean(), x0.variance(), x1.mean(), x1.variance(), x2.mean(), x2.variance()])
            self.assertAllClose(outs, [[1.0, 2.0], [1.0, 1.0], [2.0, 4.0], [2.0, 2.0], [3.0, 6.0], [3.0, 3.0]])

    def test_filtering_no_weights(self):
        if False:
            while True:
                i = 10
        with self.test_session() as sess:
            ghmm = GaussianHMM(3, transition_variances=[1.0, 2.0, 3.0], emission_variances=[4.0, 5.0, 6.0])
            z_prev = tf.constant([1.0, 2.0], dtype=tf.float32)
            x_cur = tf.constant([3.0, 4.0], dtype=tf.float32)
            expected_outs = [[[3.0 / 5.0, 4.0 / 5.0], [4.0 / 5.0, 4.0 / 5.0]], [[11.0 / 7.0, 18.0 / 7.0], [10.0 / 7.0, 10.0 / 7.0]], [[5.0 / 3.0, 8.0 / 3.0], [2.0, 2.0]]]
            f_post_0 = ghmm.filtering(0, z_prev, x_cur)
            f_post_1 = ghmm.filtering(1, z_prev, x_cur)
            f_post_2 = ghmm.filtering(2, z_prev, x_cur)
            outs = sess.run([[f_post_0.mean(), f_post_0.variance()], [f_post_1.mean(), f_post_1.variance()], [f_post_2.mean(), f_post_2.variance()]])
            self.assertAllClose(expected_outs, outs)

    def test_filtering_with_weights(self):
        if False:
            print('Hello World!')
        with self.test_session() as sess:
            ghmm = GaussianHMM(3, transition_variances=[1.0, 2.0, 3.0], emission_variances=[4.0, 5.0, 6.0], transition_weights=[7.0, 8.0], emission_weights=[9.0, 10.0, 11])
            z_prev = tf.constant([1.0, 2.0], dtype=tf.float32)
            x_cur = tf.constant([3.0, 4.0], dtype=tf.float32)
            expected_outs = [[[27.0 / 85.0, 36.0 / 85.0], [4.0 / 85.0, 4.0 / 85.0]], [[95.0 / 205.0, 150.0 / 205.0], [10.0 / 205.0, 10.0 / 205.0]], [[147.0 / 369.0, 228.0 / 369.0], [18.0 / 369.0, 18.0 / 369.0]]]
            f_post_0 = ghmm.filtering(0, z_prev, x_cur)
            f_post_1 = ghmm.filtering(1, z_prev, x_cur)
            f_post_2 = ghmm.filtering(2, z_prev, x_cur)
            outs = sess.run([[f_post_0.mean(), f_post_0.variance()], [f_post_1.mean(), f_post_1.variance()], [f_post_2.mean(), f_post_2.variance()]])
            self.assertAllClose(expected_outs, outs)

    def test_smoothing(self):
        if False:
            for i in range(10):
                print('nop')
        with self.test_session() as sess:
            ghmm = GaussianHMM(3, transition_variances=[1.0, 2.0, 3.0], emission_variances=[4.0, 5.0, 6.0])
            z_prev = tf.constant([1.0, 2.0], dtype=tf.float32)
            xs = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=tf.float32)
            s_post1 = ghmm.smoothing(0, z_prev, xs)
            outs = sess.run([s_post1.mean(), s_post1.variance()])
            expected_outs = [[281.0 / 421.0, 410.0 / 421.0], [292.0 / 421.0, 292.0 / 421.0]]
            self.assertAllClose(expected_outs, outs)
            expected_outs = [[149.0 / 73.0, 222.0 / 73.0], [90.0 / 73.0, 90.0 / 73.0]]
            s_post2 = ghmm.smoothing(1, z_prev, xs[1:])
            outs = sess.run([s_post2.mean(), s_post2.variance()])
            self.assertAllClose(expected_outs, outs)
            s_post3 = ghmm.smoothing(2, z_prev, xs[2:])
            outs = sess.run([s_post3.mean(), s_post3.variance()])
            expected_outs = [[7.0 / 3.0, 10.0 / 3.0], [2.0, 2.0]]
            self.assertAllClose(expected_outs, outs)

    def test_smoothing_with_weights(self):
        if False:
            print('Hello World!')
        with self.test_session() as sess:
            x_weight = np.array([4, 5, 6, 7], dtype=np.float32)
            sigma_x = np.array([5, 6, 7, 8], dtype=np.float32)
            z_weight = np.array([1, 2, 3], dtype=np.float32)
            sigma_z = np.array([1, 2, 3, 4], dtype=np.float32)
            z_prev = np.array([1, 2], dtype=np.float32)
            batch_size = 2
            xs = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
            (z_cov, x_cov, z_x_cov) = self._compute_covariance_matrices(x_weight, z_weight, sigma_x, sigma_z)
            expected_outs = []
            sigma_12 = z_x_cov[0, :]
            sigma_12_22 = np.dot(sigma_12, np.linalg.inv(x_cov))
            mean = np.dot(sigma_12_22, xs)
            variance = np.squeeze(z_cov[0, 0] - np.dot(sigma_12_22, sigma_12))
            expected_outs.append([mean, np.tile(variance, [batch_size])])
            for t in xrange(1, 4):
                sigma_12 = np.concatenate([[z_cov[t, t - 1]], z_x_cov[t, t:]])
                sigma_22 = np.vstack((np.hstack((z_cov[t - 1, t - 1], z_x_cov[t - 1, t:])), np.hstack((np.transpose([z_x_cov[t - 1, t:]]), x_cov[t:, t:]))))
                sigma_12_22 = np.dot(sigma_12, np.linalg.inv(sigma_22))
                mean = np.dot(sigma_12_22, np.vstack((z_prev, xs[t:])))
                variance = np.squeeze(z_cov[t, t] - np.dot(sigma_12_22, sigma_12))
                expected_outs.append([mean, np.tile(variance, [batch_size])])
            ghmm = GaussianHMM(4, transition_variances=sigma_z, emission_variances=sigma_x, transition_weights=z_weight, emission_weights=x_weight)
            out_dists = [ghmm.smoothing(t, z_prev, xs[t:]) for t in range(0, 4)]
            outs = [[d.mean(), d.variance()] for d in out_dists]
            run_outs = sess.run(outs)
            self.assertAllClose(expected_outs, run_outs)

    def test_covariance_matrices(self):
        if False:
            return 10
        with self.test_session() as sess:
            x_weight = np.array([4, 5, 6, 7], dtype=np.float32)
            sigma_x = np.array([5, 6, 7, 8], dtype=np.float32)
            z_weight = np.array([1, 2, 3], dtype=np.float32)
            sigma_z = np.array([1, 2, 3, 4], dtype=np.float32)
            (z_cov, x_cov, z_x_cov) = self._compute_covariance_matrices(x_weight, z_weight, sigma_x, sigma_z)
            ghmm = GaussianHMM(4, transition_variances=sigma_z, emission_variances=sigma_x, transition_weights=z_weight, emission_weights=x_weight)
            self.assertAllClose(z_cov, sess.run(ghmm.sigma_z))
            self.assertAllClose(x_cov, sess.run(ghmm.sigma_x))
            self.assertAllClose(z_x_cov, sess.run(ghmm.sigma_zx))

    def _compute_covariance_matrices(self, x_weight, z_weight, sigma_x, sigma_z):
        if False:
            while True:
                i = 10
        z_cov = np.zeros([4, 4])
        z_cov[0, 0] = sigma_z[0]
        for i in range(1, 4):
            z_cov[i, i] = z_cov[i - 1, i - 1] * np.square(z_weight[i - 1]) + sigma_z[i]
        for i in range(4):
            for j in range(4):
                if i == j:
                    continue
                min_ind = min(i, j)
                max_ind = max(i, j)
                weights = np.prod(z_weight[min_ind:max_ind])
                z_cov[i, j] = z_cov[min_ind, min_ind] * weights
        x_weights_outer = np.outer(x_weight, x_weight)
        x_cov = x_weights_outer * z_cov + np.diag(sigma_x)
        z_x_cov = x_weight * z_cov
        return (z_cov, x_cov, z_x_cov)

    def test_lookahead(self):
        if False:
            for i in range(10):
                print('nop')
        x_weight = np.array([4, 5, 6, 7], dtype=np.float32)
        sigma_x = np.array([5, 6, 7, 8], dtype=np.float32)
        z_weight = np.array([1, 2, 3], dtype=np.float32)
        sigma_z = np.array([1, 2, 3, 4], dtype=np.float32)
        z_prev = np.array([1, 2], dtype=np.float32)
        with self.test_session() as sess:
            (z_cov, x_cov, z_x_cov) = self._compute_covariance_matrices(x_weight, z_weight, sigma_x, sigma_z)
            expected_outs = []
            for t in range(1, 4):
                sigma_12 = z_x_cov[t - 1, t:]
                z_var = z_cov[t - 1, t - 1]
                mean = np.outer(z_prev, sigma_12 / z_var)
                variance = x_cov[t:, t:] - np.outer(sigma_12, sigma_12) / z_var
                expected_outs.append([mean, variance])
            ghmm = GaussianHMM(4, transition_variances=sigma_z, emission_variances=sigma_x, transition_weights=z_weight, emission_weights=x_weight)
            out_dists = [ghmm.lookahead(t, z_prev) for t in range(1, 4)]
            outs = [[d.mean(), d.covariance()] for d in out_dists]
            run_outs = sess.run(outs)
            self.assertAllClose(expected_outs, run_outs)

class TrainableGHMMTest(tf.test.TestCase):

    def test_filtering_proposal(self):
        if False:
            print('Hello World!')
        "Check that stashing the xs doesn't change the filtering distributions."
        with self.test_session() as sess:
            ghmm = TrainableGaussianHMM(3, 'filtering', transition_variances=[1.0, 2.0, 3.0], emission_variances=[4.0, 5.0, 6.0], transition_weights=[7.0, 8.0], emission_weights=[9.0, 10.0, 11])
            observations = tf.constant([[3.0, 4.0], [3.0, 4.0], [3.0, 4.0]], dtype=tf.float32)
            ghmm.set_observations(observations, [3, 3])
            z_prev = tf.constant([1.0, 2.0], dtype=tf.float32)
            proposals = [ghmm._filtering_proposal(t, z_prev) for t in range(3)]
            dist_params = [[p.mean(), p.variance()] for p in proposals]
            expected_outs = [[[27.0 / 85.0, 36.0 / 85.0], [4.0 / 85.0, 4.0 / 85.0]], [[95.0 / 205.0, 150.0 / 205.0], [10.0 / 205.0, 10.0 / 205.0]], [[147.0 / 369.0, 228.0 / 369.0], [18.0 / 369.0, 18.0 / 369.0]]]
            self.assertAllClose(expected_outs, sess.run(dist_params))

    def test_smoothing_proposal(self):
        if False:
            while True:
                i = 10
        with self.test_session() as sess:
            ghmm = TrainableGaussianHMM(3, 'smoothing', transition_variances=[1.0, 2.0, 3.0], emission_variances=[4.0, 5.0, 6.0])
            xs = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=tf.float32)
            ghmm.set_observations(xs, [3, 3])
            z_prev = tf.constant([1.0, 2.0], dtype=tf.float32)
            proposals = [ghmm._smoothing_proposal(t, z_prev) for t in range(3)]
            dist_params = [[p.mean(), p.variance()] for p in proposals]
            expected_outs = [[[281.0 / 421.0, 410.0 / 421.0], [292.0 / 421.0, 292.0 / 421.0]], [[149.0 / 73.0, 222.0 / 73.0], [90.0 / 73.0, 90.0 / 73.0]], [[7.0 / 3.0, 10.0 / 3.0], [2.0, 2.0]]]
            self.assertAllClose(expected_outs, sess.run(dist_params))
if __name__ == '__main__':
    tf.test.main()