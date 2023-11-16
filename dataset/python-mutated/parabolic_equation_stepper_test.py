"""Tests for 1-D parabolic PDE solvers."""
import math
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
import tf_quant_finance as tff
from tensorflow.python.framework import test_util
fd_solvers = tff.math.pde.fd_solvers
dirichlet = tff.math.pde.boundary_conditions.dirichlet
neumann = tff.math.pde.boundary_conditions.neumann
grids = tff.math.pde.grids
crank_nicolson_step = tff.math.pde.steppers.crank_nicolson.crank_nicolson_step
explicit_step = tff.math.pde.steppers.explicit.explicit_step
extrapolation_step = tff.math.pde.steppers.extrapolation.extrapolation_step
implicit_step = tff.math.pde.steppers.implicit.implicit_step
crank_nicolson_with_oscillation_damping_step = tff.math.pde.steppers.oscillation_damped_crank_nicolson.oscillation_damped_crank_nicolson_step
weighted_implicit_explicit_step = tff.math.pde.steppers.weighted_implicit_explicit.weighted_implicit_explicit_step

@test_util.run_all_in_graph_and_eager_modes
class ParabolicEquationStepperTest(tf.test.TestCase, parameterized.TestCase):

    @parameterized.named_parameters({'testcase_name': 'Implicit', 'one_step_fn': implicit_step(), 'time_step': 0.001}, {'testcase_name': 'Explicit', 'one_step_fn': explicit_step(), 'time_step': 0.001}, {'testcase_name': 'Weighted', 'one_step_fn': weighted_implicit_explicit_step(theta=0.3), 'time_step': 0.001}, {'testcase_name': 'CrankNicolson', 'one_step_fn': crank_nicolson_step(), 'time_step': 0.01}, {'testcase_name': 'Extrapolation', 'one_step_fn': extrapolation_step(), 'time_step': 0.01}, {'testcase_name': 'CrankNicolsonWithOscillationDamping', 'one_step_fn': crank_nicolson_with_oscillation_damping_step(), 'time_step': 0.01})
    def testHeatEquationWithVariousSchemes(self, one_step_fn, time_step):
        if False:
            return 10
        'Test solving heat equation with various time marching schemes.\n\n    Tests solving heat equation with the boundary conditions\n    `u(x, t=1) = e * sin(x)`, `u(-2 pi n - pi / 2, t) = -e^t`, and\n    `u(2 pi n + pi / 2, t) = -e^t` with some integer `n` for `u(x, t=0)`.\n\n    The exact solution is `u(x, t=0) = sin(x)`.\n\n    All time marching schemes should yield reasonable results given small enough\n    time steps. First-order accurate schemes (explicit, implicit, weighted with\n    theta != 0.5) require smaller time step than second-order accurate ones\n    (Crank-Nicolson, Extrapolation).\n\n    Args:\n      one_step_fn: one_step_fn representing a time marching scheme to use.\n      time_step: time step for given scheme.\n    '

        def final_cond_fn(x):
            if False:
                i = 10
                return i + 15
            return math.e * math.sin(x)

        def expected_result_fn(x):
            if False:
                i = 10
                return i + 15
            return tf.sin(x)

        @dirichlet
        def lower_boundary_fn(t, x):
            if False:
                while True:
                    i = 10
            del x
            return -tf.math.exp(t)

        @dirichlet
        def upper_boundary_fn(t, x):
            if False:
                for i in range(10):
                    print('nop')
            del x
            return tf.math.exp(t)
        grid = grids.uniform_grid(minimums=[-10.5 * math.pi], maximums=[10.5 * math.pi], sizes=[1000], dtype=np.float32)
        self._testHeatEquation(grid=grid, final_t=1, time_step=time_step, final_cond_fn=final_cond_fn, expected_result_fn=expected_result_fn, one_step_fn=one_step_fn, lower_boundary_fn=lower_boundary_fn, upper_boundary_fn=upper_boundary_fn, error_tolerance=0.001)

    def testHeatEquation_WithNeumannBoundaryConditions(self):
        if False:
            for i in range(10):
                print('nop')
        'Test for Neumann boundary conditions.\n\n    Tests solving heat equation with the following boundary conditions:\n    `u(x, t=1) = e * sin(x)`, `u_x(0, t) = e^t`, and\n    `u_x(2 pi n + pi/2, t) = 0`, where `n` is some integer.\n\n    The exact solution `u(x, t=0) = e^t sin(x)`.\n    '

        def final_cond_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return math.e * math.sin(x)

        def expected_result_fn(x):
            if False:
                while True:
                    i = 10
            return tf.sin(x)

        @neumann
        def lower_boundary_fn(t, x):
            if False:
                print('Hello World!')
            del x
            return -tf.math.exp(t)

        @neumann
        def upper_boundary_fn(t, x):
            if False:
                print('Hello World!')
            del t, x
            return 0
        grid = grids.uniform_grid(minimums=[0.0], maximums=[10.5 * math.pi], sizes=[1000], dtype=np.float32)
        self._testHeatEquation(grid, final_t=1, time_step=0.01, final_cond_fn=final_cond_fn, expected_result_fn=expected_result_fn, one_step_fn=crank_nicolson_step(), lower_boundary_fn=lower_boundary_fn, upper_boundary_fn=upper_boundary_fn, error_tolerance=0.001)

    def testHeatEquation_WithMixedBoundaryConditions(self):
        if False:
            for i in range(10):
                print('nop')
        'Test for mixed boundary conditions.\n\n    Tests solving heat equation with the following boundary conditions:\n    `u(x, t=1) = e * sin(x)`, `u_x(0, t) = e^t`, and\n    `u(2 pi n + pi/2, t) = e^t`, where `n` is some integer.\n\n    The exact solution `u(x, t=0) = e^t sin(x)`.\n    '

        def final_cond_fn(x):
            if False:
                while True:
                    i = 10
            return math.e * math.sin(x)

        def expected_result_fn(x):
            if False:
                i = 10
                return i + 15
            return tf.sin(x)

        @neumann
        def lower_boundary_fn(t, x):
            if False:
                return 10
            del x
            return -tf.math.exp(t)

        @dirichlet
        def upper_boundary_fn(t, x):
            if False:
                while True:
                    i = 10
            del x
            return tf.math.exp(t)
        grid = grids.uniform_grid(minimums=[0], maximums=[10.5 * math.pi], sizes=[1000], dtype=np.float32)
        self._testHeatEquation(grid, final_t=1, time_step=0.01, final_cond_fn=final_cond_fn, expected_result_fn=expected_result_fn, one_step_fn=crank_nicolson_step(), lower_boundary_fn=lower_boundary_fn, upper_boundary_fn=upper_boundary_fn, error_tolerance=0.001)

    def testHeatEquation_WithRobinBoundaryConditions(self):
        if False:
            for i in range(10):
                print('nop')
        'Test for Robin boundary conditions.\n\n    Tests solving heat equation with the following boundary conditions:\n    `u(x, t=1) = e * sin(x)`, `u_x(0, t) + 2u(0, t) = e^t`, and\n    `2u(x_max, t) + u_x(x_max, t) = 2*e^t`, where `x_max = 2 pi n + pi/2` with\n    some integer `n`.\n\n    The exact solution `u(x, t=0) = e^t sin(x)`.\n    '

        def final_cond_fn(x):
            if False:
                return 10
            return math.e * math.sin(x)

        def expected_result_fn(x):
            if False:
                i = 10
                return i + 15
            return tf.sin(x)

        def lower_boundary_fn(t, x):
            if False:
                for i in range(10):
                    print('nop')
            del x
            return (2, -1, tf.math.exp(t))

        def upper_boundary_fn(t, x):
            if False:
                i = 10
                return i + 15
            del x
            return (2, 1, 2 * tf.math.exp(t))
        grid = grids.uniform_grid(minimums=[0], maximums=[4.5 * math.pi], sizes=[1000], dtype=np.float64)
        self._testHeatEquation(grid, final_t=1, time_step=0.01, final_cond_fn=final_cond_fn, expected_result_fn=expected_result_fn, one_step_fn=crank_nicolson_step(), lower_boundary_fn=lower_boundary_fn, upper_boundary_fn=upper_boundary_fn, error_tolerance=0.01)

    def testHeatEquation_WithRobinBoundaryConditions_AndLogUniformGrid(self):
        if False:
            print('Hello World!')
        'Same as previous, but with log-uniform grid.'

        def final_cond_fn(x):
            if False:
                return 10
            return math.e * math.sin(x)

        def expected_result_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return tf.sin(x)

        def lower_boundary_fn(t, x):
            if False:
                for i in range(10):
                    print('nop')
            del x
            return (2, -1, tf.math.exp(t))

        def upper_boundary_fn(t, x):
            if False:
                for i in range(10):
                    print('nop')
            del x
            return (2, 1, 2 * tf.math.exp(t))
        grid = grids.log_uniform_grid(minimums=[2 * math.pi], maximums=[4.5 * math.pi], sizes=[1000], dtype=np.float64)
        self._testHeatEquation(grid=grid, final_t=1, time_step=0.01, final_cond_fn=final_cond_fn, expected_result_fn=expected_result_fn, one_step_fn=crank_nicolson_step(), lower_boundary_fn=lower_boundary_fn, upper_boundary_fn=upper_boundary_fn, error_tolerance=0.01)

    def testHeatEquation_WithRobinBoundaryConditions_AndExtraPointInGrid(self):
        if False:
            return 10
        'Same as previous, but with grid with an extra point.\n\n    We add an extra point in a uniform grid so that grid[1]-grid[0] and\n    grid[2]-grid[1] are significantly different. This is important for testing\n    the discretization of boundary conditions: both deltas participate there.\n    '

        def final_cond_fn(x):
            if False:
                print('Hello World!')
            return math.e * math.sin(x)

        def expected_result_fn(x):
            if False:
                print('Hello World!')
            return tf.sin(x)

        def lower_boundary_fn(t, x):
            if False:
                i = 10
                return i + 15
            del x
            return (2, -1, tf.math.exp(t))

        def upper_boundary_fn(t, x):
            if False:
                for i in range(10):
                    print('nop')
            del x
            return (2, 1, 2 * tf.math.exp(t))
        x_min = 0
        x_max = 4.5 * math.pi
        num_points = 1001
        locations = np.linspace(x_min, x_max, num=num_points - 1)
        delta = locations[1] - locations[0]
        locations = np.insert(locations, 1, locations[0] + delta / 3)
        grid = [tf.constant(locations)]
        self._testHeatEquation(grid=grid, final_t=1, time_step=0.01, final_cond_fn=final_cond_fn, expected_result_fn=expected_result_fn, one_step_fn=crank_nicolson_step(), lower_boundary_fn=lower_boundary_fn, upper_boundary_fn=upper_boundary_fn, error_tolerance=0.01)

    def testCrankNicolsonOscillationDamping(self):
        if False:
            while True:
                i = 10
        'Tests the Crank-Nicolson oscillation damping.\n\n    Oscillations arise in Crank-Nicolson scheme when the initial (or final)\n    conditions have discontinuities. We use Heaviside step function as initial\n    conditions. The exact solution of the heat equation with unbounded x is\n    ```None\n    u(x, t) = (1 + erf(x/2sqrt(t))/2\n    ```\n    We take large enough x_min, x_max to be able to use this as a reference\n    solution.\n\n    CrankNicolsonWithOscillationDamping produces much smaller error than\n    the usual crank_nicolson_scheme.\n    '
        final_t = 1
        x_min = -10
        x_max = 10
        dtype = np.float32

        def final_cond_fn(x):
            if False:
                i = 10
                return i + 15
            return 0.0 if x < 0 else 1.0

        def expected_result_fn(x):
            if False:
                return 10
            return 1 / 2 + tf.math.erf(x / (2 * tf.sqrt(dtype(final_t)))) / 2

        @dirichlet
        def lower_boundary_fn(t, x):
            if False:
                for i in range(10):
                    print('nop')
            del t, x
            return 0

        @dirichlet
        def upper_boundary_fn(t, x):
            if False:
                while True:
                    i = 10
            del t, x
            return 1
        grid = grids.uniform_grid(minimums=[x_min], maximums=[x_max], sizes=[1000], dtype=dtype)
        self._testHeatEquation(grid=grid, final_t=final_t, time_step=0.01, final_cond_fn=final_cond_fn, expected_result_fn=expected_result_fn, one_step_fn=crank_nicolson_with_oscillation_damping_step(), lower_boundary_fn=lower_boundary_fn, upper_boundary_fn=upper_boundary_fn, error_tolerance=0.001)

    @parameterized.named_parameters({'testcase_name': 'DefaultBC', 'lower_bc_type': 'Default', 'upper_bc_type': 'Default'}, {'testcase_name': 'DefaultNeumanBC', 'lower_bc_type': 'Default', 'upper_bc_type': 'Neumann'}, {'testcase_name': 'NeumanDefaultBC', 'lower_bc_type': 'Neumann', 'upper_bc_type': 'Default'})
    def testHeatEquation_WithDefaultBoundaryCondtion(self, lower_bc_type, upper_bc_type):
        if False:
            i = 10
            return i + 15
        'Test for Default boundary conditions.\n\n    Tests solving heat equation with the following boundary conditions involving\n    default boundary `u_xx(0, t) = 0` or `u_xx(5 pi, t) = 0`.\n\n    The exact solution `u(x, t=0) = e^t sin(x)`.\n    Args:\n      lower_bc_type: Lower boundary condition type.\n      upper_bc_type: Upper boundary condition type.\n    '

        def final_cond_fn(x):
            if False:
                print('Hello World!')
            return math.e * math.sin(x)

        def expected_result_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return tf.sin(x)

        @neumann
        def boundary_fn(t, x):
            if False:
                print('Hello World!')
            del x
            return -tf.exp(t)
        lower_boundary_fn = boundary_fn if lower_bc_type == 'Neumann' else None
        upper_boundary_fn = boundary_fn if upper_bc_type == 'Neumann' else None
        grid = grids.uniform_grid(minimums=[0.0], maximums=[5 * math.pi], sizes=[1000], dtype=np.float32)
        self._testHeatEquation(grid, final_t=1, time_step=0.01, final_cond_fn=final_cond_fn, expected_result_fn=expected_result_fn, one_step_fn=crank_nicolson_step(), lower_boundary_fn=lower_boundary_fn, upper_boundary_fn=upper_boundary_fn, error_tolerance=0.001)

    def _testHeatEquation(self, grid, final_t, time_step, final_cond_fn, expected_result_fn, one_step_fn, lower_boundary_fn, upper_boundary_fn, error_tolerance=0.001):
        if False:
            i = 10
            return i + 15
        'Helper function with details of testing heat equation solving.'

        def second_order_coeff_fn(t, x):
            if False:
                print('Hello World!')
            del t, x
            return [[1]]
        xs = self.evaluate(grid)[0]
        final_values = tf.constant([final_cond_fn(x) for x in xs], dtype=grid[0].dtype)
        result = fd_solvers.solve_backward(start_time=final_t, end_time=0, coord_grid=grid, values_grid=final_values, num_steps=None, start_step_count=0, time_step=time_step, one_step_fn=one_step_fn, boundary_conditions=[(lower_boundary_fn, upper_boundary_fn)], values_transform_fn=None, second_order_coeff_fn=second_order_coeff_fn, dtype=grid[0].dtype)
        actual = self.evaluate(result[0])
        expected = self.evaluate(expected_result_fn(xs))
        self.assertLess(np.max(np.abs(actual - expected)), error_tolerance)

    @parameterized.named_parameters({'testcase_name': 'DirichletBC', 'bc_type': 'Dirichlet', 'batch_grid': False}, {'testcase_name': 'DefaultBC', 'bc_type': 'Default', 'batch_grid': False}, {'testcase_name': 'DirichletBC_BatchGrid', 'bc_type': 'Dirichlet', 'batch_grid': True}, {'testcase_name': 'DefaultBC_BatchGrid', 'bc_type': 'Default', 'batch_grid': True})
    def testDocStringExample(self, bc_type, batch_grid):
        if False:
            for i in range(10):
                print('nop')
        'Tests that the European Call option price is computed correctly.'
        num_equations = 2
        num_grid_points = 1024
        dtype = np.float64
        if batch_grid:
            s_min = [0.01, 0.05]
            s_max = [200.0, 220]
            sizes = [num_grid_points, num_grid_points]
        else:
            s_min = [0.01]
            s_max = [200.0]
            sizes = [num_grid_points]
        grid = grids.uniform_grid(minimums=s_min, maximums=s_max, sizes=sizes, dtype=dtype)
        grid = [tf.stack(grid, axis=0)]
        volatility = np.array([0.3, 0.15], dtype=dtype).reshape([-1, 1])
        rate = np.array([0.01, 0.03], dtype=dtype).reshape([-1, 1])
        expiry = 1.0
        strike = np.array([50, 100], dtype=dtype).reshape([-1, 1])

        def second_order_coeff_fn(t, location_grid):
            if False:
                while True:
                    i = 10
            del t
            return [[tf.square(volatility) * tf.square(location_grid[0]) / 2]]

        def first_order_coeff_fn(t, location_grid):
            if False:
                while True:
                    i = 10
            del t
            return [rate * location_grid[0]]

        def zeroth_order_coeff_fn(t, location_grid):
            if False:
                while True:
                    i = 10
            del t, location_grid
            return -rate

        @dirichlet
        def lower_boundary_fn(t, location_grid):
            if False:
                i = 10
                return i + 15
            del t, location_grid
            return 0

        @dirichlet
        def upper_boundary_fn(t, location_grid):
            if False:
                print('Hello World!')
            return location_grid[0][..., -1] + tf.squeeze(-strike * tf.math.exp(-rate * (expiry - t)))
        final_values = tf.nn.relu(grid[0] - strike)
        final_values += tf.zeros([num_equations, num_grid_points], dtype=dtype)
        if bc_type == 'Default':
            boundary_conditions = [(None, upper_boundary_fn)]
        else:
            boundary_conditions = [(lower_boundary_fn, upper_boundary_fn)]
        estimate = fd_solvers.solve_backward(start_time=expiry, end_time=0, coord_grid=grid, values_grid=final_values, num_steps=None, start_step_count=0, time_step=0.001, one_step_fn=crank_nicolson_step(), boundary_conditions=boundary_conditions, values_transform_fn=None, second_order_coeff_fn=second_order_coeff_fn, first_order_coeff_fn=first_order_coeff_fn, zeroth_order_coeff_fn=zeroth_order_coeff_fn, dtype=dtype)[0]
        estimate = self.evaluate(estimate)
        value_grid_first_option = estimate[0, :]
        value_grid_second_option = estimate[1, :]
        loc_1 = 256
        loc_2 = 512
        if batch_grid:
            spots = tf.stack([grid[0][0][loc_1], grid[0][-1][loc_2]])
        else:
            spots = tf.stack([grid[0][0][loc_1], grid[0][0][loc_2]])
        call_price = tff.black_scholes.option_price(volatilities=volatility[..., 0], strikes=strike[..., 0], expiries=expiry, discount_rates=rate[..., 0], spots=spots)
        self.assertAllClose(call_price, [value_grid_first_option[loc_1], value_grid_second_option[loc_2]], rtol=0.001, atol=0.001)

    def testEuropeanCallDynamicVol(self):
        if False:
            while True:
                i = 10
        'Price for the European Call option with time-dependent volatility.'
        num_equations = 1
        num_grid_points = 1024
        dtype = np.float64
        s_max = 300.0
        grid = grids.log_uniform_grid(minimums=[0.01], maximums=[s_max], sizes=[num_grid_points], dtype=dtype)
        expiry = 1.0
        strike = 50.0

        def second_order_coeff_fn(t, location_grid):
            if False:
                return 10
            return [[(1.0 / 6 + t ** 2 / 2) * tf.square(location_grid[0]) / 2]]

        @dirichlet
        def lower_boundary_fn(t, location_grid):
            if False:
                print('Hello World!')
            del t, location_grid
            return 0

        @dirichlet
        def upper_boundary_fn(t, location_grid):
            if False:
                return 10
            del t
            return location_grid[0][-1] - strike
        final_values = tf.nn.relu(grid[0] - strike)
        final_values += tf.zeros([num_equations, num_grid_points], dtype=dtype)
        estimate = fd_solvers.solve_backward(start_time=expiry, end_time=0, coord_grid=grid, values_grid=final_values, num_steps=None, start_step_count=0, time_step=tf.constant(0.01, dtype=dtype), one_step_fn=crank_nicolson_step(), boundary_conditions=[(lower_boundary_fn, upper_boundary_fn)], values_transform_fn=None, second_order_coeff_fn=second_order_coeff_fn, dtype=dtype)[0]
        value_grid = self.evaluate(estimate)[0, :]
        loc_1 = 849
        call_price = 12.582092
        self.assertAllClose(call_price, value_grid[loc_1], rtol=0.01, atol=0.01)

    def testHeatEquation_InForwardDirection(self):
        if False:
            print('Hello World!')
        'Test solving heat equation with various time marching schemes.\n\n    Tests solving heat equation with the boundary conditions\n    `u(x, t=1) = e * sin(x)`, `u(-2 pi n - pi / 2, t) = -e^t`, and\n    `u(2 pi n + pi / 2, t) = -e^t` with some integer `n` for `u(x, t=0)`.\n\n    The exact solution is `u(x, t=0) = sin(x)`.\n\n    All time marching schemes should yield reasonable results given small enough\n    time steps. First-order accurate schemes (explicit, implicit, weighted with\n    theta != 0.5) require smaller time step than second-order accurate ones\n    (Crank-Nicolson, Extrapolation).\n    '
        final_time = 1.0

        def initial_cond_fn(x):
            if False:
                while True:
                    i = 10
            return tf.sin(x)

        def expected_result_fn(x):
            if False:
                return 10
            return np.exp(-final_time) * tf.sin(x)

        @dirichlet
        def lower_boundary_fn(t, x):
            if False:
                i = 10
                return i + 15
            del x
            return -tf.math.exp(-t)

        @dirichlet
        def upper_boundary_fn(t, x):
            if False:
                return 10
            del x
            return tf.math.exp(-t)
        grid = grids.uniform_grid(minimums=[-10.5 * math.pi], maximums=[10.5 * math.pi], sizes=[1000], dtype=np.float32)

        def second_order_coeff_fn(t, x):
            if False:
                i = 10
                return i + 15
            del t, x
            return [[-1]]
        final_values = initial_cond_fn(grid[0])
        result = fd_solvers.solve_forward(start_time=0.0, end_time=final_time, coord_grid=grid, values_grid=final_values, time_step=0.01, boundary_conditions=[(lower_boundary_fn, upper_boundary_fn)], second_order_coeff_fn=second_order_coeff_fn)[0]
        actual = self.evaluate(result)
        expected = self.evaluate(expected_result_fn(grid[0]))
        self.assertLess(np.max(np.abs(actual - expected)), 0.001)

    def testReferenceEquation(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests the equation used as reference for a few further tests.\n\n    We solve the diffusion equation `u_t = u_xx` on x = [0...1] with boundary\n    conditions `u(x<=1/2, t=0) = x`, `u(x>1/2, t=0) = 1 - x`,\n    `u(x=0, t) = u(x=1, t) = 0`.\n\n    The exact solution of the diffusion equation with zero-Dirichlet boundaries\n    is:\n    `u(x, t) = sum_{n=1..inf} b_n sin(pi n x) exp(-n^2 pi^2 t)`,\n    `b_n = 2 integral_{0..1} sin(pi n x) u(x, t=0) dx.`\n\n    The initial conditions are taken so that the integral easily calculates, and\n    the sum can be approximated by a few first terms (given large enough `t`).\n    See the result in _reference_heat_equation_solution.\n\n    Using this solution helps to simplify the tests, as we don\'t have to\n    maintain complicated boundary conditions in each test or tweak the\n    parameters to keep the "support" of the function far from boundaries.\n    '
        grid = grids.uniform_grid(minimums=[0], maximums=[1], sizes=[501], dtype=tf.float32)
        xs = grid[0]
        final_t = 0.1
        time_step = 0.001

        def second_order_coeff_fn(t, coord_grid):
            if False:
                i = 10
                return i + 15
            del t, coord_grid
            return [[-1]]
        initial = _reference_pde_initial_cond(xs)
        expected = _reference_pde_solution(xs, final_t)
        actual = fd_solvers.solve_forward(start_time=0, end_time=final_t, coord_grid=grid, values_grid=initial, time_step=time_step, second_order_coeff_fn=second_order_coeff_fn)[0]
        self.assertAllClose(expected, actual, atol=0.001, rtol=0.001)

    def testReference_WithExponentMultiplier(self):
        if False:
            print('Hello World!')
        'Tests solving diffusion equation with an exponent multiplier.\n\n    Take the heat equation `v_{t} - v_{xx} = 0` and substitute `v = exp(x) u`.\n    This yields `u_{t} - u_{xx} - 2u_{x} - u = 0`. The test compares numerical\n    solution of this equation to the exact one, which is the diffusion equation\n    solution times `exp(-x)`.\n    '
        grid = grids.uniform_grid(minimums=[0], maximums=[1], sizes=[501], dtype=tf.float32)
        xs = grid[0]
        final_t = 0.1
        time_step = 0.001

        def second_order_coeff_fn(t, coord_grid):
            if False:
                while True:
                    i = 10
            del t, coord_grid
            return [[-1]]

        def first_order_coeff_fn(t, coord_grid):
            if False:
                i = 10
                return i + 15
            del t, coord_grid
            return [-2]

        def zeroth_order_coeff_fn(t, coord_grid):
            if False:
                return 10
            del t, coord_grid
            return -1
        initial = tf.math.exp(-xs) * _reference_pde_initial_cond(xs)
        expected = tf.math.exp(-xs) * _reference_pde_solution(xs, final_t)
        actual = fd_solvers.solve_forward(start_time=0, end_time=final_t, coord_grid=grid, values_grid=initial, time_step=time_step, second_order_coeff_fn=second_order_coeff_fn, first_order_coeff_fn=first_order_coeff_fn, zeroth_order_coeff_fn=zeroth_order_coeff_fn)[0]
        self.assertAllClose(expected, actual, atol=0.001, rtol=0.001)

    def testInnerSecondOrderCoeff(self):
        if False:
            return 10
        'Tests handling inner_second_order_coeff.\n\n    As in previous test, take the diffusion equation `v_{t} - v_{xx} = 0` and\n    substitute `v = exp(x) u`, but this time keep exponent under the derivative:\n    `u_{t} - exp(-x)[exp(x)u]_{xx} = 0`. Expect the same solution as in\n    previous test.\n    '
        grid = grids.uniform_grid(minimums=[0], maximums=[1], sizes=[501], dtype=tf.float32)
        xs = grid[0]
        final_t = 0.1
        time_step = 0.001

        def second_order_coeff_fn(t, coord_grid):
            if False:
                i = 10
                return i + 15
            del t
            x = coord_grid[0]
            return [[-tf.math.exp(-x)]]

        def inner_second_order_coeff_fn(t, coord_grid):
            if False:
                while True:
                    i = 10
            del t
            x = coord_grid[0]
            return [[tf.math.exp(x)]]
        initial = tf.math.exp(-xs) * _reference_pde_initial_cond(xs)
        expected = tf.math.exp(-xs) * _reference_pde_solution(xs, final_t)
        actual = fd_solvers.solve_forward(start_time=0, end_time=final_t, coord_grid=grid, values_grid=initial, time_step=time_step, second_order_coeff_fn=second_order_coeff_fn, inner_second_order_coeff_fn=inner_second_order_coeff_fn)[0]
        self.assertAllClose(expected, actual, atol=0.001, rtol=0.001)

    def testInnerFirstAndSecondOrderCoeff(self):
        if False:
            print('Hello World!')
        'Tests handling both inner_first_order_coeff and inner_second_order_coeff.\n\n    We saw previously that the solution of `u_{t} - u_{xx} - 2u_{x} - u = 0` is\n    `u = exp(-x) v`, where v solves the diffusion equation. Substitute now\n    `u = exp(-x) v` without expanding the derivatives:\n    `v_{t} - exp(x)[exp(-x)v]_{xx} - 2exp(x)[exp(-x)v]_{x} - v = 0`.\n    Solve this equation and expect the solution of the diffusion equation.\n    '
        grid = grids.uniform_grid(minimums=[0], maximums=[1], sizes=[501], dtype=tf.float32)
        xs = grid[0]
        final_t = 0.1
        time_step = 0.001

        def second_order_coeff_fn(t, coord_grid):
            if False:
                i = 10
                return i + 15
            del t
            x = coord_grid[0]
            return [[-tf.math.exp(x)]]

        def inner_second_order_coeff_fn(t, coord_grid):
            if False:
                i = 10
                return i + 15
            del t
            x = coord_grid[0]
            return [[tf.math.exp(-x)]]

        def first_order_coeff_fn(t, coord_grid):
            if False:
                i = 10
                return i + 15
            del t
            x = coord_grid[0]
            return [-2 * tf.math.exp(x)]

        def inner_first_order_coeff_fn(t, coord_grid):
            if False:
                while True:
                    i = 10
            del t
            x = coord_grid[0]
            return [tf.math.exp(-x)]

        def zeroth_order_coeff_fn(t, coord_grid):
            if False:
                while True:
                    i = 10
            del t, coord_grid
            return -1
        initial = _reference_pde_initial_cond(xs)
        expected = _reference_pde_solution(xs, final_t)
        actual = fd_solvers.solve_forward(start_time=0, end_time=final_t, coord_grid=grid, values_grid=initial, time_step=time_step, second_order_coeff_fn=second_order_coeff_fn, first_order_coeff_fn=first_order_coeff_fn, zeroth_order_coeff_fn=zeroth_order_coeff_fn, inner_second_order_coeff_fn=inner_second_order_coeff_fn, inner_first_order_coeff_fn=inner_first_order_coeff_fn)[0]
        self.assertAllClose(expected, actual, atol=0.001, rtol=0.001)

    def testCompareExpandedAndNotExpandedPdes(self):
        if False:
            print('Hello World!')
        'Tests comparing PDEs with expanded derivatives and without.\n\n    Take equation `u_{t} - [x^2 u]_{xx} + [x u]_{x} = 0`.\n    Expanding the derivatives yields `u_{t} - x^2 u_{xx} - 3x u_{x} - u = 0`.\n    Solve both equations and expect the results to be equal.\n    '
        grid = grids.uniform_grid(minimums=[0], maximums=[1], sizes=[501], dtype=tf.float32)
        xs = grid[0]
        final_t = 0.1
        time_step = 0.001
        initial = _reference_pde_initial_cond(xs)

        def inner_second_order_coeff_fn(t, coord_grid):
            if False:
                while True:
                    i = 10
            del t
            x = coord_grid[0]
            return [[-tf.square(x)]]

        def inner_first_order_coeff_fn(t, coord_grid):
            if False:
                i = 10
                return i + 15
            del t
            x = coord_grid[0]
            return [x]
        result_not_expanded = fd_solvers.solve_forward(start_time=0, end_time=final_t, coord_grid=grid, values_grid=initial, time_step=time_step, inner_second_order_coeff_fn=inner_second_order_coeff_fn, inner_first_order_coeff_fn=inner_first_order_coeff_fn)[0]

        def second_order_coeff_fn(t, coord_grid):
            if False:
                print('Hello World!')
            del t
            x = coord_grid[0]
            return [[-tf.square(x)]]

        def first_order_coeff_fn(t, coord_grid):
            if False:
                print('Hello World!')
            del t
            x = coord_grid[0]
            return [-3 * x]

        def zeroth_order_coeff_fn(t, coord_grid):
            if False:
                return 10
            del t, coord_grid
            return -1
        result_expanded = fd_solvers.solve_forward(start_time=0, end_time=final_t, coord_grid=grid, values_grid=initial, time_step=time_step, second_order_coeff_fn=second_order_coeff_fn, first_order_coeff_fn=first_order_coeff_fn, zeroth_order_coeff_fn=zeroth_order_coeff_fn)[0]
        self.assertAllClose(result_not_expanded, result_expanded, atol=0.001, rtol=0.001)

    def testDefaultBoundaryConditions(self):
        if False:
            for i in range(10):
                print('nop')
        'Test for PDE with default boundary condition and no inner term.\n\n    Take equation `u_{t} - x u_{xx} + (x - 1) u_{x} = 0` with boundary\n    conditions `u_{t} + (x - 1) u_{x} = 0` at x = 0 and `u(t, 1) = exp(t + 1)`\n    with an initial condition `u(0, x) = exp(x)`.\n\n    Solve this equation and compare the result to `u(t, x) = exp(t + x)`.\n    '

        @dirichlet
        def upper_boundary_fn(t, x):
            if False:
                for i in range(10):
                    print('nop')
            del x
            return tf.math.exp(t + 1)

        def second_order_coeff_fn(t, coord_grid):
            if False:
                while True:
                    i = 10
            del t
            x = coord_grid[0]
            return [[-x]]

        def first_order_coeff_fn(t, coord_grid):
            if False:
                i = 10
                return i + 15
            del t
            x = coord_grid[0]
            return [x - 1]
        grid = self.evaluate(grids.uniform_grid(minimums=[0], maximums=[1], sizes=[1000], dtype=np.float64))
        initial = tf.math.exp(grid[0])
        time_step = 0.01
        final_t = 0.5
        est_values = fd_solvers.solve_forward(start_time=0, end_time=final_t, coord_grid=grid, values_grid=initial, time_step=time_step, one_step_fn=crank_nicolson_step(), second_order_coeff_fn=second_order_coeff_fn, first_order_coeff_fn=first_order_coeff_fn, boundary_conditions=[(None, upper_boundary_fn)])[0]
        true_values = tf.math.exp(final_t + grid[0])
        self.assertAllClose(est_values, true_values, atol=0.01, rtol=0.01)

    @parameterized.named_parameters({'testcase_name': 'LeftDefault', 'default_bc': 'left'}, {'testcase_name': 'RightDefault', 'default_bc': 'right'}, {'testcase_name': 'BothDefault', 'default_bc': 'both'})
    def testDefaultBoundaryConditionsWithInnerTerm(self, default_bc):
        if False:
            i = 10
            return i + 15
        "Test for PDE with default boundary condition with inner term.\n\n    Take equation\n    `u_{t} - (x - x**3)[u]_{xx} + (1 + x) * [(1 - x**2) u]_{x}\n     + (2 * x**2 - 1 + 2 *x - (1 - x**2))u = 0` with\n    boundary conditions `u_{t} + (x - 1) u_{x} = 0` at x = 0\n    and `u(t, 1) = exp(t + 1)`, and an initial condition `u(0, x) = exp(x)`.\n\n    Solve this equation and compare the result to `u(t, x) = exp(t + x)`.\n\n    Args:\n      default_bc: A string to indicate which boundary condition is 'default'.\n        Can be either 'left', 'right', or 'both'.\n    "

        def second_order_coeff_fn(t, coord_grid):
            if False:
                while True:
                    i = 10
            del t
            x = coord_grid[0]
            return [[-(-x ** 3 + x)]]

        def first_order_coeff_fn(t, coord_grid):
            if False:
                return 10
            del t
            x = coord_grid[0]
            return [1 + x]

        def inner_first_order_coeff_fn(t, coord_grid):
            if False:
                for i in range(10):
                    print('nop')
            del t
            x = coord_grid[0]
            return [-x ** 2 + 1]

        @dirichlet
        def lower_boundary_fn(t, x):
            if False:
                print('Hello World!')
            del x
            return tf.math.exp(t)

        @dirichlet
        def upper_boundary_fn(t, x):
            if False:
                return 10
            del x
            return tf.math.exp(1.0 + t)

        def zeroth_order_coeff_fn(t, coord_grid):
            if False:
                for i in range(10):
                    print('nop')
            del t
            x = coord_grid[0]
            return 2 * x ** 2 - 1 + 2 * x - (1 - x ** 2)
        grid = self.evaluate(grids.uniform_grid(minimums=[0], maximums=[1], sizes=[100], dtype=np.float64))
        initial_values = tf.math.exp(grid[0])
        time_step = 0.001
        final_t = 0.1
        if default_bc == 'left':
            boundary_conditions = [(None, upper_boundary_fn)]
        elif default_bc == 'right':
            boundary_conditions = [(lower_boundary_fn, None)]
        else:
            boundary_conditions = [(None, None)]
        est_values = fd_solvers.solve_forward(start_time=0, end_time=final_t, coord_grid=grid, values_grid=initial_values, time_step=time_step, boundary_conditions=boundary_conditions, second_order_coeff_fn=second_order_coeff_fn, first_order_coeff_fn=first_order_coeff_fn, inner_first_order_coeff_fn=inner_first_order_coeff_fn, zeroth_order_coeff_fn=zeroth_order_coeff_fn)[0]
        true_values = tf.math.exp(final_t + grid[0])
        self.assertAllClose(est_values, true_values, atol=0.01, rtol=0.01)

    def testDefaultBoundaryConditionsInnerTermNoOuterLower(self):
        if False:
            while True:
                i = 10
        'Test for PDE with default boundary condition with inner term.\n\n    Take equation\n    `u_{t} - (x - x**3)[u]_{xx} + [(x - x**3) u]_{x} + (3 * x**2 - 2)u = 0` with\n    boundary conditions `u_{t} + (x - 1) u_{x} = 0` at x = 0\n    and `u(t, 1) = exp(t + 1)`, and an initial condition `u(0, x) = exp(x)`.\n\n    Solve this equation and compare the result to `u(t, x) = exp(t + x)`.\n    '

        def second_order_coeff_fn(t, coord_grid):
            if False:
                i = 10
                return i + 15
            del t
            x = coord_grid[0]
            return [[-(x - x ** 3)]]

        def first_order_coeff_fn(t, coord_grid):
            if False:
                return 10
            del t
            x = coord_grid[0]
            return [tf.ones_like(x)]

        def inner_first_order_coeff_fn(t, coord_grid):
            if False:
                while True:
                    i = 10
            del t
            x = coord_grid[0]
            return [x - x ** 3]

        @dirichlet
        def upper_boundary_fn(t, x):
            if False:
                for i in range(10):
                    print('nop')
            del x
            return tf.math.exp(1 + t)

        def zeroth_order_coeff_fn(t, coord_grid):
            if False:
                i = 10
                return i + 15
            del t
            x = coord_grid[0]
            return 3 * x ** 2 - 2
        grid = self.evaluate(grids.uniform_grid(minimums=[0], maximums=[1], sizes=[100], dtype=np.float64))
        initial_values = tf.expand_dims(tf.math.exp(grid[0]), axis=0)
        final_t = 0.1
        time_step = 0.001
        est_values = fd_solvers.solve_forward(start_time=0, end_time=final_t, coord_grid=grid, values_grid=initial_values, time_step=time_step, boundary_conditions=[(None, upper_boundary_fn)], second_order_coeff_fn=second_order_coeff_fn, first_order_coeff_fn=first_order_coeff_fn, inner_first_order_coeff_fn=inner_first_order_coeff_fn, zeroth_order_coeff_fn=zeroth_order_coeff_fn)[0]
        true_values = tf.expand_dims(tf.math.exp(final_t + grid[0]), axis=0)
        self.assertAllClose(est_values, true_values, atol=0.01, rtol=0.01)

    def testDefaultBoundaryConditionsInnerTermNoOuterUpper(self):
        if False:
            print('Hello World!')
        'Test for PDE with default boundary condition with inner term.\n\n    Take equation\n    `u_{t} - (1 - x)[u]_{xx} + [(1 - x) u]_{x} = 0` with\n    boundary conditions `u_{t} + (x - 1) u_{x} = 0` at x = 0\n    and `u(t, 1) = exp(t + 1)`, and an initial condition `u(0, x) = exp(x)`.\n\n    Solve this equation and compare the result to `u(t, x) = exp(t + x)`.\n    '

        def second_order_coeff_fn(t, coord_grid):
            if False:
                while True:
                    i = 10
            del t
            x = coord_grid[0]
            return [[-(1 - x)]]

        def inner_first_order_coeff_fn(t, coord_grid):
            if False:
                while True:
                    i = 10
            del t
            x = coord_grid[0]
            return [1 - x]

        @dirichlet
        def lower_boundary_fn(t, x):
            if False:
                for i in range(10):
                    print('nop')
            del x
            return tf.math.exp(t)
        grid = self.evaluate(grids.uniform_grid(minimums=[0], maximums=[1], sizes=[100], dtype=np.float64))
        initial_values = tf.expand_dims(tf.math.exp(grid[0]), axis=0)
        final_t = 0.1
        time_step = 0.001
        est_values = fd_solvers.solve_forward(start_time=0, end_time=final_t, coord_grid=grid, values_grid=initial_values, time_step=time_step, boundary_conditions=[(lower_boundary_fn, None)], second_order_coeff_fn=second_order_coeff_fn, inner_first_order_coeff_fn=inner_first_order_coeff_fn)[0]
        true_values = tf.expand_dims(tf.math.exp(final_t + grid[0]), axis=0)
        self.assertAllClose(est_values, true_values, atol=0.01, rtol=0.01)

def _reference_pde_initial_cond(xs):
    if False:
        print('Hello World!')
    'Initial conditions for the reference diffusion equation.'
    return -tf.math.abs(xs - 0.5) + 0.5

def _reference_pde_solution(xs, t, num_terms=5):
    if False:
        return 10
    'Solution for the reference diffusion equation.'
    u = tf.zeros_like(xs)
    for k in range(num_terms):
        n = 2 * k + 1
        term = tf.math.sin(np.pi * n * xs) * tf.math.exp(-n ** 2 * np.pi ** 2 * t)
        term *= 4 / (np.pi ** 2 * n ** 2)
        if k % 2 == 1:
            term *= -1
        u += term
    return u
if __name__ == '__main__':
    tf.test.main()