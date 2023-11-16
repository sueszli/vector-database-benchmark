"""Tests for `sample_paths` of `ItoProcess`."""
import os
from unittest import mock
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
import tf_quant_finance as tff
from tensorflow.python.framework import test_util

@test_util.run_all_in_graph_and_eager_modes
class GenericItoProcessTest(tf.test.TestCase, parameterized.TestCase):

    @parameterized.named_parameters({'testcase_name': 'no_xla', 'use_xla': False}, {'testcase_name': 'xla', 'use_xla': True})
    def test_sample_paths_wiener(self, use_xla):
        if False:
            i = 10
            return i + 15
        'Tests paths properties for Wiener process (dX = dW).'

        def drift_fn(_, x):
            if False:
                print('Hello World!')
            return tf.zeros_like(x)

        def vol_fn(_, x):
            if False:
                print('Hello World!')
            return tf.expand_dims(tf.ones_like(x), -1)
        process = tff.models.GenericItoProcess(dim=1, drift_fn=drift_fn, volatility_fn=vol_fn)
        times = np.array([0.1, 0.2, 0.3])
        num_samples = 10000

        @tf.function
        def fn():
            if False:
                return 10
            return process.sample_paths(times=times, num_samples=num_samples, seed=42, time_step=0.01)
        if use_xla:
            paths = self.evaluate(tf.xla.experimental.compile(fn))[0]
        else:
            paths = self.evaluate(fn())
        means = np.mean(paths, axis=0).reshape([-1])
        covars = np.cov(paths.reshape([num_samples, -1]), rowvar=False)
        expected_means = np.zeros((3,))
        expected_covars = np.minimum(times.reshape([-1, 1]), times.reshape([1, -1]))
        with self.subTest(name='Means'):
            self.assertAllClose(means, expected_means, rtol=0.01, atol=0.01)
        with self.subTest(name='Covar'):
            self.assertAllClose(covars, expected_covars, rtol=0.01, atol=0.01)

    @parameterized.named_parameters({'testcase_name': 'NoGridNoDraws', 'use_time_grid': False, 'supply_normal_draws': False}, {'testcase_name': 'WithGridWithDraws', 'use_time_grid': True, 'supply_normal_draws': True})
    def test_sample_paths_2d(self, use_time_grid, supply_normal_draws):
        if False:
            i = 10
            return i + 15
        'Tests path properties for 2-dimentional Ito process.\n\n    We construct the following Ito processes.\n\n    dX_1 = mu_1 sqrt(t) dt + s11 dW_1 + s12 dW_2\n    dX_2 = mu_2 sqrt(t) dt + s21 dW_1 + s22 dW_2\n\n    mu_1, mu_2 are constants.\n    s_ij = a_ij t + b_ij\n\n    For this process expected value at time t is (x_0)_i + 2/3 * mu_i * t^1.5.\n\n    Args:\n      use_time_grid: A boolean to indicate whther `times_grid` is supplied.\n      supply_normal_draws: A boolean to indicate whether `normal_draws` is\n        supplied.\n    '
        dtype = tf.float64
        mu = np.array([0.2, 0.7])
        a = np.array([[0.4, 0.1], [0.3, 0.2]])
        b = np.array([[0.33, -0.03], [0.21, 0.5]])

        def drift_fn(t, x):
            if False:
                print('Hello World!')
            return mu * tf.sqrt(t) * tf.ones_like(x, dtype=t.dtype)

        def vol_fn(t, x):
            if False:
                print('Hello World!')
            del x
            return (a * t + b) * tf.ones([2, 2], dtype=t.dtype)
        process = tff.models.GenericItoProcess(dim=2, drift_fn=drift_fn, volatility_fn=vol_fn)
        times = np.array([0.1, 0.21, 0.32, 0.43, 0.55])
        x0 = np.array([0.1, -1.1])
        if use_time_grid:
            times_grid = tf.linspace(tf.constant(0.0, dtype=dtype), 0.55, 56)
            time_step = None
        else:
            times_grid = None
            time_step = 0.01
        if supply_normal_draws:
            num_samples = 1
            normal_draws = tf.random.normal(shape=[5000, times_grid.shape[0] - 1, 2], dtype=dtype)
            normal_draws = tf.concat([normal_draws, -normal_draws], axis=0)
        else:
            num_samples = 10000
            normal_draws = None
        paths = self.evaluate(process.sample_paths(times, num_samples=num_samples, initial_state=x0, time_step=time_step, times_grid=times_grid, normal_draws=normal_draws, seed=12134))
        num_samples = 10000
        self.assertAllClose(paths.shape, (num_samples, 5, 2), atol=0)
        means = np.mean(paths, axis=0)
        times = np.reshape(times, [-1, 1])
        expected_means = x0 + 2.0 / 3.0 * mu * np.power(times, 1.5)
        self.assertAllClose(means, expected_means, rtol=0.01, atol=0.01)

    @parameterized.named_parameters({'testcase_name': '1DBatch', 'batch_rank': 1}, {'testcase_name': '2DBatch', 'batch_rank': 2})
    def test_batch_sample_paths_2d(self, batch_rank):
        if False:
            i = 10
            return i + 15
        'Tests path properties for a batch of 2-dimentional Ito process.\n\n    We construct the following Ito processes.\n\n    dX_1 = mu_1 sqrt(t) dt + s11 dW_1 + s12 dW_2\n    dX_2 = mu_2 sqrt(t) dt + s21 dW_1 + s22 dW_2\n\n    mu_1, mu_2 are constants.\n    s_ij = a_ij t + b_ij\n\n    For this process expected value at time t is (x_0)_i + 2/3 * mu_i * t^1.5.\n\n    Args:\n      batch_rank: The rank of the batch of processes being simulated.\n    '
        dtype = tf.float64
        mu = np.array([0.2, 0.7])
        a = np.array([[0.4, 0.1], [0.3, 0.2]])
        b = np.array([[0.33, -0.03], [0.21, 0.5]])

        def drift_fn(t, x):
            if False:
                while True:
                    i = 10
            return mu * tf.sqrt(t) * tf.ones_like(x, dtype=t.dtype)

        def vol_fn(t, x):
            if False:
                print('Hello World!')
            return (a * t + b) * tf.ones(x.shape.as_list() + [2], dtype=t.dtype)
        process = tff.models.GenericItoProcess(dim=2, drift_fn=drift_fn, volatility_fn=vol_fn, dtype=dtype)
        times = np.array([0.1, 0.21, 0.32, 0.43, 0.55])
        x0 = np.array([0.1, -1.1]) * np.ones([2] * batch_rank + [1, 2])
        times_grid = None
        time_step = 0.01
        num_samples = 10000
        normal_draws = None
        paths = self.evaluate(process.sample_paths(times, num_samples=num_samples, initial_state=x0, time_step=time_step, times_grid=times_grid, normal_draws=normal_draws, seed=12134))
        num_samples = 10000
        self.assertAllClose(list(paths.shape), [2] * batch_rank + [num_samples, 5, 2], atol=0)
        means = np.mean(paths, axis=batch_rank)
        times = np.reshape(times, [1] * batch_rank + [-1, 1])
        expected_means = np.reshape(x0, [2] * batch_rank + [1, 2]) + 2.0 / 3.0 * mu * np.power(times, 1.5)
        self.assertAllClose(means, expected_means, rtol=0.01, atol=0.01)

    def test_sample_paths_dtypes(self):
        if False:
            for i in range(10):
                print('nop')
        'Sampled paths have the expected dtypes.'
        for dtype in [np.float32, np.float64]:
            drift_fn = lambda t, x: tf.sqrt(t) * tf.ones_like(x, dtype=t.dtype)
            vol_fn = lambda t, x: t * tf.ones([1, 1], dtype=t.dtype)
            process = tff.models.GenericItoProcess(dim=1, drift_fn=drift_fn, volatility_fn=vol_fn, dtype=dtype)
            paths = self.evaluate(process.sample_paths(times=[0.1, 0.2], num_samples=10, initial_state=[0.1], time_step=0.01, seed=123))
            self.assertEqual(paths.dtype, dtype)

    def test_backward_pde_coeffs_with_constant_params_1d(self):
        if False:
            i = 10
            return i + 15
        vol = 2
        drift = 1
        discounting = 3
        vol_fn = lambda t, x: tf.constant([[[vol]]], dtype=tf.float32)
        drift_fn = lambda t, x: tf.constant([[drift]], dtype=tf.float32)
        discounting_fn = lambda t, x: tf.constant([discounting], dtype=tf.float32)
        process = tff.models.GenericItoProcess(dim=1, drift_fn=drift_fn, volatility_fn=vol_fn, dtype=tf.float32)
        pde_solver_fn = mock.Mock()
        coord_grid = [tf.constant([0])]
        process.fd_solver_backward(start_time=0, end_time=0, coord_grid=coord_grid, values_grid=tf.constant([0]), discounting=discounting_fn, pde_solver_fn=pde_solver_fn)
        kwargs = pde_solver_fn.call_args[1]
        second_order_coeff = self.evaluate(kwargs['second_order_coeff_fn'](0, coord_grid))
        self.assertAllClose(np.array([[[vol ** 2 / 2]]]), second_order_coeff)
        first_order_coeff = self.evaluate(kwargs['first_order_coeff_fn'](0, coord_grid))
        self.assertAllClose(np.array([[drift]]), first_order_coeff)
        zeroth_order_coeff = self.evaluate(kwargs['zeroth_order_coeff_fn'](0, coord_grid))
        self.assertAllClose(np.array([-discounting]), zeroth_order_coeff)

    def test_backward_pde_coeffs_with_batching_1d(self):
        if False:
            print('Hello World!')
        dtype = tf.float64
        volatilities = tf.constant([[0.3], [0.15], [0.1]], dtype)
        rates = tf.constant([[0.01], [0.03], [0.01]], dtype)
        expiries = 1.0
        dim = 1
        num_processes = 3

        def drift_fn(t, x):
            if False:
                i = 10
                return i + 15
            del t
            expand_rank = x.shape.rank - 2
            rates_expand = tf.reshape(rates, [num_processes] + (expand_rank + 1) * [1])
            return rates_expand * x

        def vol_fn(t, x):
            if False:
                i = 10
                return i + 15
            del t
            expand_rank = x.shape.rank - 2
            volatilities_expand = tf.reshape(volatilities, [num_processes] + (expand_rank + 1) * [1])
            return tf.expand_dims(volatilities_expand * x, axis=-1) * tf.eye(dim, batch_shape=x.shape.as_list()[:-1], dtype=x.dtype)
        process = tff.models.GenericItoProcess(dim=dim, drift_fn=drift_fn, volatility_fn=vol_fn, dtype=dtype)
        num_strikes = 2
        strikes = tf.constant([[[50], [60]], [[100], [90]], [[120], [90]]], dtype)

        @tff.math.pde.boundary_conditions.dirichlet
        def upper_boundary_fn(t, grid):
            if False:
                i = 10
                return i + 15
            del grid
            return tf.squeeze(s_max - strikes * tf.exp(-rates * (expiries - t)))

        def discounting(t, x):
            if False:
                return 10
            del t, x
            rates_expand = tf.expand_dims(rates, axis=-1)
            return rates_expand
        s_min = 0
        s_max = 200
        num_grid_points = 256
        grid = tff.math.pde.grids.uniform_grid(minimums=[s_min], maximums=[s_max], sizes=[num_grid_points], dtype=dtype)
        final_value_grid = tf.nn.relu(grid[0] - strikes)
        prices_estimated = process.fd_solver_backward(start_time=expiries, end_time=0, time_step=0.1, coord_grid=grid, values_grid=final_value_grid, discounting=discounting, boundary_condtions=[(None, upper_boundary_fn)])[0]
        with self.subTest('OutputShape'):
            self.assertAllEqual(prices_estimated.shape.as_list(), (num_processes, num_strikes, num_grid_points))
        loc_1 = 95
        loc_2 = 123
        loc_3 = 155
        inds = tf.stack([loc_1, loc_2, loc_3])
        gather_ind1 = tf.stack([[0, 1, 2], [0] * 3, inds], axis=-1)
        gather_ind2 = tf.stack([[0, 1, 2], [1] * 3, inds], axis=-1)
        gather_ind = tf.concat([gather_ind1, gather_ind2], axis=0)
        prices_at_locs = tf.gather_nd(prices_estimated, gather_ind)
        expected_prices = [25.726, 5.502, 6.3, 17.684, 11.221, 32.467]
        with self.subTest('Prices'):
            self.assertAllClose(prices_at_locs, expected_prices, rtol=0.01, atol=0.01)

    def test_backward_pde_vs_mc_coeffs_with_batching_2d(self):
        if False:
            print('Hello World!')
        'Monte Carlo and PDE pricing of a basket option.'
        num_grid_points = 256
        dtype = tf.float64
        dim = 2
        s_min = 0
        s_max = 200
        grid = tff.math.pde.grids.uniform_grid(minimums=dim * [s_min], maximums=dim * [s_max], sizes=dim * [num_grid_points], dtype=dtype)
        strikes = tf.constant([[50], [100], [90]], dtype)
        volatilities = tf.constant([[0.3, 0.1], [0.15, 0.1], [0.1, 0.1]], dtype)
        rates = tf.constant([[0.01], [0.03], [0.01]], dtype)
        expiries = 1.0
        batch_size = 3

        def drift_fn(t, x):
            if False:
                return 10
            del t
            expand_rank = x.shape.rank - 2
            rates_expand = tf.reshape(rates, [batch_size] + expand_rank * [1] + [1])
            return rates_expand * x

        def vol_fn(t, x):
            if False:
                print('Hello World!')
            del t
            expand_rank = x.shape.rank - 2
            volatilities_expand = tf.reshape(volatilities, [batch_size] + expand_rank * [1] + [dim])
            return tf.expand_dims(volatilities_expand * x, axis=-1) * tf.eye(dim, batch_shape=x.shape.as_list()[:-1], dtype=x.dtype)
        process = tff.models.GenericItoProcess(dim=dim, drift_fn=drift_fn, volatility_fn=vol_fn, dtype=dtype)

        def discounting(t, x):
            if False:
                return 10
            del t, x
            return tf.expand_dims(rates, axis=-1)
        (x, y) = tf.meshgrid(*grid, indexing='ij')
        final_value_grid = tf.broadcast_to(tf.nn.relu((x + y) / 2 - tf.reshape(strikes, [-1, 1, 1])), [batch_size] + dim * [num_grid_points])

        @tff.math.pde.boundary_conditions.dirichlet
        def upper_boundary_fn(t, coord_grid):
            if False:
                for i in range(10):
                    print('nop')
            del coord_grid
            return (s_max + grid[0]) / 2 - strikes * tf.exp(-rates * (expiries - t))
        pde_prices = process.fd_solver_backward(start_time=expiries, end_time=0, time_step=0.1, coord_grid=grid, values_grid=final_value_grid, discounting=discounting, boundary_condtions=[(None, upper_boundary_fn), (None, upper_boundary_fn)])[0]
        with self.subTest('PDEOutputShape'):
            self.assertAllEqual(pde_prices.shape.as_list(), [batch_size, num_grid_points, num_grid_points])
        loc_1 = 95
        loc_2 = 123
        loc_3 = 155
        spots = tf.stack([[grid[0][loc_1]], [grid[0][loc_2]], [grid[0][loc_3]]], axis=0)
        inds = tf.stack([loc_1, loc_2, loc_3])
        inds = tf.stack([[0, 1, 2], inds, inds], axis=1)
        pde_prices = tf.gather_nd(pde_prices, inds)
        x0 = tf.expand_dims(spots, axis=-2) + np.zeros([batch_size, 1, dim])
        times_grid = None
        time_step = 0.01
        num_samples = 10000
        normal_draws = None
        paths = process.sample_paths([expiries], num_samples=num_samples, initial_state=x0, time_step=time_step, times_grid=times_grid, normal_draws=normal_draws, random_type=tff.math.random.RandomType.SOBOL, seed=[1, 42])
        mc_prices = tf.reduce_mean(tf.nn.relu((paths[..., 0] + paths[..., 1]) / 2 - tf.reshape(strikes, [-1, 1, 1])), -2)
        mc_prices *= tf.exp(-rates * expiries)
        with self.subTest('Prices'):
            self.assertAllClose(mc_prices[..., 0], pde_prices, rtol=0.1, atol=0.1)

    def test_backward_pde_coeffs_with_nonconstant_params_1d(self):
        if False:
            i = 10
            return i + 15

        def vol_fn(t, x):
            if False:
                return 10
            del t
            x = x[:, 0]
            return tf.reshape(2 * x ** 2, (-1, 1, 1))

        def drift_fn(t, x):
            if False:
                while True:
                    i = 10
            del t
            x = x[:, 0]
            return tf.reshape(3 * x, (-1, 1))

        def discounting_fn(t, x):
            if False:
                return 10
            del t
            x = x[:, 0]
            return 6 / x
        coord_grid = [tf.constant([1, 2, 3])]
        process = tff.models.GenericItoProcess(dim=1, drift_fn=drift_fn, volatility_fn=vol_fn, dtype=tf.float32)
        pde_solver_fn = mock.Mock()
        process.fd_solver_backward(start_time=0, end_time=0, coord_grid=coord_grid, values_grid=tf.constant([0]), discounting=discounting_fn, pde_solver_fn=pde_solver_fn)
        kwargs = pde_solver_fn.call_args[1]
        second_order_coeff = self.evaluate(kwargs['second_order_coeff_fn'](0, coord_grid))
        self.assertAllClose(np.array([[[2, 8 ** 2 / 2, 18 ** 2 / 2]]]), second_order_coeff)
        first_order_coeff = self.evaluate(kwargs['first_order_coeff_fn'](0, coord_grid))
        self.assertAllClose(np.array([[3, 6, 9]]), first_order_coeff)
        zeroth_order_coeff = self.evaluate(kwargs['zeroth_order_coeff_fn'](0, coord_grid))
        self.assertAllClose(np.array([-6, -3, -2]), zeroth_order_coeff)

    def test_backward_pde_coeffs_with_constant_params_2d(self):
        if False:
            print('Hello World!')
        vol = [[1, 2], [3, 4]]
        drift = [1, 2]
        discounting = 3

        def vol_fn(t, x):
            if False:
                print('Hello World!')
            del t, x
            return tf.expand_dims(tf.constant(vol, dtype=tf.float32), 0)

        def drift_fn(t, x):
            if False:
                for i in range(10):
                    print('nop')
            del t, x
            return tf.expand_dims(tf.constant(drift, dtype=tf.float32), 0)
        discounting_fn = lambda t, x: tf.constant([discounting], dtype=tf.float32)
        process = tff.models.GenericItoProcess(dim=2, drift_fn=drift_fn, volatility_fn=vol_fn, dtype=tf.float32)
        pde_solver_fn = mock.Mock()
        coord_grid = [tf.constant([0])]
        process.fd_solver_backward(start_time=0, end_time=0, coord_grid=coord_grid, values_grid=tf.constant([0]), discounting=discounting_fn, pde_solver_fn=pde_solver_fn)
        kwargs = pde_solver_fn.call_args[1]
        second_order_coeff = self.evaluate(kwargs['second_order_coeff_fn'](0, coord_grid))
        self.assertAllClose(np.array([[[2.5], [5.5]], [[5.5], [12.5]]]), second_order_coeff)
        first_order_coeff = self.evaluate(kwargs['first_order_coeff_fn'](0, coord_grid))
        self.assertAllClose(np.array([[1], [2]]), first_order_coeff)
        zeroth_order_coeff = self.evaluate(kwargs['zeroth_order_coeff_fn'](0, coord_grid))
        self.assertAllClose(np.array([-3]), zeroth_order_coeff)

    def test_backward_pde_coeffs_with_nonconstant_params_2d(self):
        if False:
            while True:
                i = 10

        def vol_fn(t, grid):
            if False:
                return 10
            del t
            self.assertEqual((3, 3, 2), grid.shape)
            ys = grid[..., 0]
            xs = grid[..., 1]
            vol_yy = xs * ys
            vol_yx = 2 * xs
            vol_xy = xs + ys
            vol_xx = 3 * tf.ones_like(xs)
            vol = tf.stack((tf.stack((vol_yy, vol_xy), axis=-1), tf.stack((vol_yx, vol_xx), axis=-1)), axis=-1)
            self.assertEqual((3, 3, 2, 2), vol.shape)
            return vol

        def drift_fn(t, grid):
            if False:
                return 10
            del t
            self.assertEqual((3, 3, 2), grid.shape)
            ys = grid[..., 0]
            xs = grid[..., 1]
            drift_y = xs + ys
            drift_x = xs - ys
            drift = tf.stack((drift_y, drift_x), axis=-1)
            self.assertEqual((3, 3, 2), drift.shape)
            return drift

        def discounting_fn(t, grid):
            if False:
                return 10
            del t
            self.assertEqual((3, 3, 2), grid.shape)
            ys = grid[..., 0]
            xs = grid[..., 1]
            return 3 * xs * ys
        process = tff.models.GenericItoProcess(dim=2, drift_fn=drift_fn, volatility_fn=vol_fn, dtype=tf.float32)
        pde_solver_fn = mock.Mock()
        coord_grid = [tf.constant([-1, 0, 1]), tf.constant([0, 1, 2])]
        process.fd_solver_backward(start_time=0, end_time=0, coord_grid=coord_grid, values_grid=tf.zeros((3, 3)), discounting=discounting_fn, pde_solver_fn=pde_solver_fn)
        kwargs = pde_solver_fn.call_args[1]
        second_order_coeff = self.evaluate(kwargs['second_order_coeff_fn'](0, coord_grid))
        expected_coeff_yy = np.array([[0, 2.5, 10], [0, 2, 8], [0, 2.5, 10]])
        self.assertAllClose(expected_coeff_yy, second_order_coeff[0][0])
        expected_coeff_xy = np.array([[0, 3, 5], [0, 3, 6], [0, 4, 9]])
        self.assertAllClose(expected_coeff_xy, second_order_coeff[0][1])
        self.assertAllClose(expected_coeff_xy, second_order_coeff[1][0])
        expected_coeff_xx = np.array([[5, 4.5, 5], [4.5, 5, 6.5], [5, 6.5, 9]])
        self.assertAllClose(expected_coeff_xx, second_order_coeff[1][1])
        first_order_coeff = self.evaluate(kwargs['first_order_coeff_fn'](0, coord_grid))
        expected_coeff_y = np.array([[-1, 0, 1], [0, 1, 2], [1, 2, 3]])
        self.assertAllClose(expected_coeff_y, first_order_coeff[0])
        expected_coeff_x = np.array([[1, 2, 3], [0, 1, 2], [-1, 0, 1]])
        self.assertAllClose(expected_coeff_x, first_order_coeff[1])
        zeroth_order_coeff = self.evaluate(kwargs['zeroth_order_coeff_fn'](0, coord_grid))
        self.assertAllClose(np.array([[0, 3, 6], [0, 0, 0], [0, -3, -6]]), zeroth_order_coeff)

    def test_solving_backward_pde_for_sde_with_const_coeffs(self):
        if False:
            return 10

        def vol_fn(t, grid):
            if False:
                for i in range(10):
                    print('nop')
            del t
            xs = grid[..., 1]
            vol_elem = tf.ones_like(xs) / np.sqrt(2)
            return tf.stack((tf.stack((vol_elem, vol_elem), axis=-1), tf.stack((vol_elem, vol_elem), axis=-1)), axis=-1)
        drift_fn = lambda t, grid: tf.zeros(grid.shape)
        process = tff.models.GenericItoProcess(dim=2, volatility_fn=vol_fn, drift_fn=drift_fn, dtype=tf.float32)
        grid = tff.math.pde.grids.uniform_grid(minimums=[-10, -20], maximums=[10, 20], sizes=[201, 301], dtype=tf.float32)
        ys = self.evaluate(grid[0])
        xs = self.evaluate(grid[1])
        diff_coeff = 1
        time_step = 0.1
        final_t = 3
        final_variance = 1
        variance_along_diagonal = final_variance + 2 * diff_coeff * final_t

        def expected_fn(x, y):
            if False:
                i = 10
                return i + 15
            return _gaussian((x + y) / np.sqrt(2), variance_along_diagonal) * _gaussian((x - y) / np.sqrt(2), final_variance)
        expected = np.array([[expected_fn(x, y) for x in xs] for y in ys])
        final_values = tf.expand_dims(tf.constant(np.outer(_gaussian(ys, final_variance), _gaussian(xs, final_variance)), dtype=tf.float32), axis=0)
        result = self.evaluate(process.fd_solver_backward(start_time=final_t, end_time=0, coord_grid=grid, values_grid=final_values, time_step=time_step, dtype=tf.float32)[0])
        self.assertLess(np.max(np.abs(result - expected)) / np.max(expected), 0.01)

def _gaussian(xs, variance):
    if False:
        while True:
            i = 10
    return np.exp(-np.square(xs) / (2 * variance)) / np.sqrt(2 * np.pi * variance)
if __name__ == '__main__':
    os.environ['TF_XLA_FLAGS'] = '--tf_mlir_enable_mlir_bridge=true'
    tf.test.main()