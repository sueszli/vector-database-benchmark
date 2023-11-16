"""Tests for the join of Ito processes."""
import numpy as np
import tensorflow.compat.v2 as tf
import tf_quant_finance as tff
from tensorflow.python.framework import test_util

@test_util.run_all_in_graph_and_eager_modes
class JoinedItoProcessTest(tf.test.TestCase):

    def test_join_hull_white(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests that join of Hull White is the same as VectorHullWhite.'
        tf.random.set_seed(42)
        dtype = np.float64
        instant_forward_rate_fn_1 = lambda t: 2 * [0.2]
        process_1 = tff.models.hull_white.VectorHullWhiteModel(dim=2, mean_reversion=[0.1, 0.2], volatility=[0.1, 0.2], initial_discount_rate_fn=instant_forward_rate_fn_1, dtype=dtype)
        instant_forward_rate_fn_2 = lambda t: 3 * [0.1]
        process_2 = tff.models.hull_white.VectorHullWhiteModel(dim=3, mean_reversion=[0.3, 0.4, 0.5], volatility=[0.1, 0.1, 0.1], initial_discount_rate_fn=instant_forward_rate_fn_2, dtype=dtype)
        corr_1 = [[1.0, 0.3, 0.2], [0.3, 1.0, 0.5], [0.2, 0.5, 1.0]]

        def corr_2(t):
            if False:
                i = 10
                return i + 15
            del t
            return [[1.0, 0.1], [0.1, 1.0]]
        matrices = [corr_1, corr_2]
        process_join = tff.models.JoinedItoProcess([process_1, process_2], matrices)
        expected_corr_matrix = np.array([[1.0, 0.3, 0.2, 0.0, 0.0], [0.3, 1.0, 0.5, 0.0, 0.0], [0.2, 0.5, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.1], [0.0, 0.0, 0.0, 0.1, 1.0]])
        expected_mean = [0.0109434, 0.02356047, 0.01500711, 0.01915375, 0.0230985]
        expected_var = [0.00475813, 0.01812692, 0.0043197, 0.004121, 0.00393469]
        num_samples = 110000
        samples = process_join.sample_paths(times=[0.1, 0.5], time_step=0.01, num_samples=num_samples, random_type=tff.math.random.RandomType.PSEUDO_ANTITHETIC, seed=42)
        self.assertEqual(samples.dtype, dtype)
        self.assertEqual(samples.shape, [num_samples, 2, 5])
        samples = self.evaluate(samples)
        self.assertAllClose(np.corrcoef(samples[:, -1, :], rowvar=False), expected_corr_matrix, rtol=0.01, atol=0.01)
        self.assertAllClose(np.mean(samples[:, -1, :], axis=0), expected_mean, rtol=0.001, atol=0.001)
        self.assertAllClose(np.var(samples[:, -1, :], axis=0), expected_var, rtol=0.001, atol=0.001)

    def test_invalid_processes(self):
        if False:
            while True:
                i = 10
        'Tests that all processes should be `ItoProcess`es.'

        def drift_fn(t, x):
            if False:
                while True:
                    i = 10
            del t, x
            return -1.0 / 2

        def vol_fn(t, x):
            if False:
                return 10
            del t
            return tf.ones([1, 1], dtype=x.dtype)
        process = tff.models.GenericItoProcess(dim=1, drift_fn=drift_fn, volatility_fn=vol_fn)
        with self.assertRaises(ValueError):
            tff.models.JoinedItoProcess([process, lambda x: x], [[1.0], [1.0]])

    def test_inconsistent_dtype(self):
        if False:
            print('Hello World!')
        'Tests that all processes should have the same dtype.'

        def drift_fn(t, x):
            if False:
                print('Hello World!')
            del t, x
            return -1.0 / 2

        def vol_fn(t, x):
            if False:
                print('Hello World!')
            del t
            return tf.ones([1, 1], dtype=x.dtype)
        process_1 = tff.models.GenericItoProcess(dim=1, drift_fn=drift_fn, volatility_fn=vol_fn, dtype=np.float32)
        process_2 = tff.models.GenericItoProcess(dim=1, drift_fn=drift_fn, volatility_fn=vol_fn, dtype=np.float64)
        with self.assertRaises(ValueError):
            tff.models.JoinedItoProcess([process_1, process_2], [[1.0], [1.0]])
if __name__ == '__main__':
    tf.test.main()