"""Tests for multidimensional parabolic PDE solvers."""
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
import tf_quant_finance as tff
from tensorflow.python.framework import test_util
fd_solvers = tff.math.pde.fd_solvers
dirichlet = tff.math.pde.boundary_conditions.dirichlet
neumann = tff.math.pde.boundary_conditions.neumann
grids = tff.math.pde.grids
douglas_adi_step = tff.math.pde.steppers.douglas_adi.douglas_adi_step
_SQRT2 = np.sqrt(2)

@test_util.run_all_in_graph_and_eager_modes
class MultidimParabolicEquationStepperTest(tf.test.TestCase, parameterized.TestCase):

    @parameterized.named_parameters({'testcase_name': 'DefaultBC', 'boundary_condition': 'default'}, {'testcase_name': 'MixedBC', 'boundary_condition': 'mixed'}, {'testcase_name': 'DirichletBC', 'boundary_condition': 'dirichlet'})
    def testAnisotropicDiffusion(self, boundary_condition):
        if False:
            for i in range(10):
                print('nop')
        'Tests solving 2d diffusion equation.'
        grid = grids.uniform_grid(minimums=[-10, -20], maximums=[10, 20], sizes=[201, 301], dtype=tf.float32)
        ys = self.evaluate(grid[0])
        xs = self.evaluate(grid[1])
        diff_coeff_x = 0.4
        diff_coeff_y = 0.25
        time_step = 0.1
        final_t = 1
        final_variance = 1

        def quadratic_coeff_fn(t, location_grid):
            if False:
                for i in range(10):
                    print('nop')
            del t, location_grid
            u_xx = diff_coeff_x
            u_yy = diff_coeff_y
            u_xy = None
            return [[u_yy, u_xy], [u_xy, u_xx]]
        final_values = tf.expand_dims(tf.constant(np.outer(_gaussian(ys, final_variance), _gaussian(xs, final_variance)), dtype=tf.float32), axis=0)
        if boundary_condition == 'default':
            bound_cond = [(None, None), (None, None)]
        elif boundary_condition == 'dirichlet':
            bound_cond = [(_zero_boundary, _zero_boundary), (_zero_boundary, _zero_boundary)]
        else:
            bound_cond = [(_zero_boundary, None), (None, _zero_grad_boundary)]
        step_fn = douglas_adi_step(theta=0.5)
        result = fd_solvers.solve_backward(start_time=final_t, end_time=0, coord_grid=grid, values_grid=final_values, time_step=time_step, one_step_fn=step_fn, boundary_conditions=bound_cond, second_order_coeff_fn=quadratic_coeff_fn, dtype=grid[0].dtype)[0]
        variance_x = final_variance + 2 * diff_coeff_x * final_t
        variance_y = final_variance + 2 * diff_coeff_y * final_t
        expected = np.outer(_gaussian(ys, variance_y), _gaussian(xs, variance_x))
        self._assertClose(expected, result)

    @parameterized.named_parameters({'testcase_name': 'DefaultBC', 'boundary_condition': 'default'}, {'testcase_name': 'MixedBC', 'boundary_condition': 'mixed'}, {'testcase_name': 'DirichletBC', 'boundary_condition': 'dirichlet'})
    def testAnisotropicDiffusion3d(self, boundary_condition):
        if False:
            print('Hello World!')
        'Tests solving 3d diffusion equation.'
        grid = grids.uniform_grid(minimums=[-10, -20, -10], maximums=[10, 20, 10], sizes=[101, 111, 121], dtype=tf.float32)
        zs = self.evaluate(grid[0])
        ys = self.evaluate(grid[1])
        xs = self.evaluate(grid[2])
        diff_coeff_x = 0.4
        diff_coeff_y = 0.25
        diff_coeff_z = 0.1
        time_step = 0.1
        final_t = 1
        final_variance = 1

        def quadratic_coeff_fn(t, location_grid):
            if False:
                i = 10
                return i + 15
            del t, location_grid
            u_xx = diff_coeff_x
            u_yy = diff_coeff_y
            u_zz = diff_coeff_z
            u_xy = 0
            u_zy = 0
            u_zx = 0
            return [[u_zz, u_zy, u_zx], [u_zy, u_yy, u_xy], [u_zx, u_xy, u_xx]]
        final_values = tf.expand_dims(tf.reshape(_gaussian(zs, final_variance), [-1, 1, 1]) * tf.reshape(_gaussian(ys, final_variance), [1, -1, 1]) * tf.reshape(_gaussian(xs, final_variance), [1, 1, -1]), axis=0)
        if boundary_condition == 'default':
            bound_cond = [(None, None), (None, None), (None, None)]
        elif boundary_condition == 'dirichlet':
            bound_cond = [(_zero_boundary, _zero_boundary), (_zero_boundary, _zero_boundary), (_zero_boundary, _zero_boundary)]
        else:
            bound_cond = [(_zero_boundary, None), (None, _zero_grad_boundary), (_zero_boundary, _zero_grad_boundary)]
        step_fn = douglas_adi_step(theta=0.5)
        result = fd_solvers.solve_backward(start_time=final_t, end_time=0, coord_grid=grid, values_grid=final_values, time_step=time_step, one_step_fn=step_fn, boundary_conditions=bound_cond, second_order_coeff_fn=quadratic_coeff_fn, dtype=grid[0].dtype)[0]
        variance_x = final_variance + 2 * diff_coeff_x * final_t
        variance_y = final_variance + 2 * diff_coeff_y * final_t
        variance_z = final_variance + 2 * diff_coeff_z * final_t
        expected = tf.expand_dims(tf.reshape(_gaussian(zs, variance_z), [-1, 1, 1]) * tf.reshape(_gaussian(ys, variance_y), [1, -1, 1]) * tf.reshape(_gaussian(xs, variance_x), [1, 1, -1]), axis=0)
        self._assertClose(self.evaluate(expected), result)

    def testSimpleDrift(self):
        if False:
            while True:
                i = 10
        'Tests solving 2d drift equation.\n\n    The equation is `u_{t} + vx u_{x} + vy u_{y} = 0`.\n    The final condition is a gaussian centered at (0, 0) with variance sigma.\n    The gaussian should drift with velocity `[vx, vy]`.\n    '
        grid = grids.uniform_grid(minimums=[-10, -20], maximums=[10, 20], sizes=[201, 301], dtype=tf.float32)
        ys = self.evaluate(grid[0])
        xs = self.evaluate(grid[1])
        time_step = 0.01
        final_t = 3
        variance = 1
        vx = 0.1
        vy = 0.3

        def first_order_coeff_fn(t, location_grid):
            if False:
                print('Hello World!')
            del t, location_grid
            return [vy, vx]
        final_values = tf.expand_dims(tf.constant(np.outer(_gaussian(ys, variance), _gaussian(xs, variance)), dtype=tf.float32), axis=0)
        bound_cond = [(_zero_boundary, _zero_boundary), (_zero_boundary, _zero_boundary)]
        result = fd_solvers.solve_backward(start_time=final_t, end_time=0, coord_grid=grid, values_grid=final_values, time_step=time_step, one_step_fn=douglas_adi_step(theta=0.5), boundary_conditions=bound_cond, first_order_coeff_fn=first_order_coeff_fn, dtype=grid[0].dtype)
        expected = np.outer(_gaussian(ys + vy * final_t, variance), _gaussian(xs + vx * final_t, variance))
        self._assertClose(expected, result)

    def testAnisotropicDiffusion_TwoDimList(self):
        if False:
            return 10

        def pack_second_order_coeff_fn(u_yy, u_xy, u_xx):
            if False:
                i = 10
                return i + 15
            return [[u_yy, u_xy], [u_xy, u_xx]]
        self._testDiffusionInDiagonalDirection(pack_second_order_coeff_fn)

    def testAnisotropicDiffusion_TwoDimList_WithoutRedundantElement(self):
        if False:
            i = 10
            return i + 15

        def pack_second_order_coeff_fn(u_yy, u_xy, u_xx):
            if False:
                print('Hello World!')
            return [[u_yy, u_xy], [None, u_xx]]
        self._testDiffusionInDiagonalDirection(pack_second_order_coeff_fn)

    def testAnisotropicDiffusion_ListOfTensors(self):
        if False:
            while True:
                i = 10

        def pack_second_order_coeff_fn(u_yy, u_xy, u_xx):
            if False:
                return 10
            return [tf.constant([u_yy, u_xy], dtype=tf.float32), tf.constant([u_xy, u_xx], dtype=tf.float32)]
        self._testDiffusionInDiagonalDirection(pack_second_order_coeff_fn)

    def testAnisotropicDiffusion_2DTensor(self):
        if False:
            while True:
                i = 10

        def pack_second_order_coeff_fn(u_yy, u_xy, u_xx):
            if False:
                i = 10
                return i + 15
            return tf.convert_to_tensor([[u_yy, u_xy], [u_xy, u_xx]], dtype=tf.float32)
        self._testDiffusionInDiagonalDirection(pack_second_order_coeff_fn)

    @parameterized.named_parameters({'testcase_name': 'DefaultBC', 'boundary_condition': 'default'}, {'testcase_name': 'MixedBC', 'boundary_condition': 'mixed'})
    def testAnisotropicDiffusion_mixed_term_default_boundary(self, boundary_condition):
        if False:
            return 10

        def pack_second_order_coeff_fn(u_yy, u_xy, u_xx):
            if False:
                print('Hello World!')
            return [[u_yy, u_xy], [u_xy, u_xx]]
        self._testDiffusionInDiagonalDirection(pack_second_order_coeff_fn, boundary_condition=boundary_condition)

    def _testDiffusionInDiagonalDirection(self, pack_second_order_coeff_fn, boundary_condition='dirichlet'):
        if False:
            while True:
                i = 10
        'Tests solving 2d diffusion equation involving mixed terms.\n\n    The equation is `u_{t} + D u_{xx} / 2 +  D u_{yy} / 2 + D u_{xy} = 0`.\n    The final condition is a gaussian centered at (0, 0) with variance sigma.\n\n    The equation can be rewritten as `u_{t} + D u_{zz} = 0`, where\n    `z = (x + y) / sqrt(2)`.\n\n    Thus variance should evolve as `sigma + 2D(t_final - t)` along z dimension\n    and stay unchanged in the orthogonal dimension:\n    `u(x, y, t) = gaussian((x + y)/sqrt(2), sigma) + 2D * (t_final - t)) *\n    gaussian((x - y)/sqrt(2), sigma)`.\n    '
        dtype = tf.float32
        grid = grids.uniform_grid(minimums=[-10, -20], maximums=[10, 20], sizes=[201, 301], dtype=dtype)
        ys = self.evaluate(grid[0])
        xs = self.evaluate(grid[1])
        diff_coeff = 1
        time_step = 0.1
        final_t = 3
        final_variance = 1

        def second_order_coeff_fn(t, location_grid):
            if False:
                print('Hello World!')
            del t, location_grid
            return pack_second_order_coeff_fn(diff_coeff / 2, diff_coeff / 2, diff_coeff / 2)
        variance_along_diagonal = final_variance + 2 * diff_coeff * final_t

        def expected_fn(x, y):
            if False:
                i = 10
                return i + 15
            return _gaussian((x + y) / _SQRT2, variance_along_diagonal) * _gaussian((x - y) / _SQRT2, final_variance)
        expected = np.array([[expected_fn(x, y) for x in xs] for y in ys])
        final_values = tf.expand_dims(tf.constant(np.outer(_gaussian(ys, final_variance), _gaussian(xs, final_variance)), dtype=dtype), axis=0)
        if boundary_condition == 'dirichlet':
            bound_cond = [(_zero_boundary, _zero_boundary), (_zero_boundary, _zero_boundary)]
        elif boundary_condition == 'mixed':
            bound_cond = [(_zero_boundary, None), (None, _zero_grad_boundary)]
        elif boundary_condition == 'default':
            bound_cond = [(None, None), (None, None)]
        else:
            raise ValueError('`boundary_cond` should be either `dirichlet`, `mixed` or `default`.')
        step_fn = douglas_adi_step(theta=0.5)
        result = fd_solvers.solve_backward(start_time=final_t, end_time=0, coord_grid=grid, values_grid=final_values, time_step=time_step, one_step_fn=step_fn, boundary_conditions=bound_cond, second_order_coeff_fn=second_order_coeff_fn, dtype=grid[0].dtype)
        self._assertClose(expected, result)

    def testShiftTerm(self):
        if False:
            while True:
                i = 10
        'Simple test for the shift term.\n\n    The equation is `u_{t} + a u = 0`, the solution is\n    `u(x, y, t) = exp(-a(t - t_final)) u(x, y, t_final)`\n    '
        grid = grids.uniform_grid(minimums=[-10, -20], maximums=[10, 20], sizes=[201, 301], dtype=tf.float32)
        ys = self.evaluate(grid[0])
        xs = self.evaluate(grid[1])
        time_step = 0.1
        final_t = 1
        variance = 1
        a = 2

        def zeroth_order_coeff_fn(t, location_grid):
            if False:
                while True:
                    i = 10
            del t, location_grid
            return a
        expected = np.outer(_gaussian(ys, variance), _gaussian(xs, variance)) * np.exp(a * final_t)
        final_values = tf.expand_dims(tf.constant(np.outer(_gaussian(ys, variance), _gaussian(xs, variance)), dtype=tf.float32), axis=0)
        bound_cond = [(_zero_boundary, _zero_boundary), (_zero_boundary, _zero_boundary)]
        step_fn = douglas_adi_step(theta=0.5)
        result = fd_solvers.solve_backward(start_time=final_t, end_time=0, coord_grid=grid, values_grid=final_values, time_step=time_step, one_step_fn=step_fn, boundary_conditions=bound_cond, zeroth_order_coeff_fn=zeroth_order_coeff_fn, dtype=grid[0].dtype)
        self._assertClose(expected, result)

    def testNoTimeDependence(self):
        if False:
            for i in range(10):
                print('nop')
        'Test for the case where all terms (quadratic, linear, shift) are null.'
        grid = grids.uniform_grid(minimums=[-10, -20], maximums=[10, 20], sizes=[201, 301], dtype=tf.float32)
        ys = self.evaluate(grid[0])
        xs = self.evaluate(grid[1])
        time_step = 0.1
        final_t = 1
        variance = 1
        final_cond = np.outer(_gaussian(ys, variance), _gaussian(xs, variance))
        final_values = tf.expand_dims(tf.constant(final_cond, dtype=tf.float32), axis=0)
        bound_cond = [(_zero_boundary, _zero_boundary), (_zero_boundary, _zero_boundary)]
        step_fn = douglas_adi_step(theta=0.5)
        result = fd_solvers.solve_backward(start_time=final_t, end_time=0, coord_grid=grid, values_grid=final_values, time_step=time_step, one_step_fn=step_fn, boundary_conditions=bound_cond, dtype=grid[0].dtype)
        expected = final_cond
        self._assertClose(expected, result)

    def testAnisotropicDiffusion_WithDirichletBoundaries(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests solving 2d diffusion equation with Dirichlet boundary conditions.\n\n    The equation is `u_{t} + u_{xx} + 2 u_{yy} = 0`.\n    The final condition is `u(t=1, x, y) = e * sin(x/sqrt(2)) * cos(y / 2)`.\n    The following function satisfies this PDE and final condition:\n    `u(t, x, y) = exp(t) * sin(x / sqrt(2)) * cos(y / 2)`.\n    We impose Dirichlet boundary conditions using this function:\n    `u(t, x_min, y) = exp(t) * sin(x_min / sqrt(2)) * cos(y / 2)`, etc.\n    The other tests below are similar, but with other types of boundary\n    conditions.\n    '
        time_step = 0.01
        final_t = 1
        x_min = -20
        x_max = 20
        y_min = -10
        y_max = 10
        grid = grids.uniform_grid(minimums=[y_min, x_min], maximums=[y_max, x_max], sizes=[201, 301], dtype=tf.float32)
        ys = self.evaluate(grid[0])
        xs = self.evaluate(grid[1])

        def second_order_coeff_fn(t, location_grid):
            if False:
                print('Hello World!')
            del t, location_grid
            return [[2, None], [None, 1]]

        @dirichlet
        def lower_bound_x(t, location_grid):
            if False:
                return 10
            del location_grid
            return tf.exp(t) * np.sin(x_min / _SQRT2) * tf.sin(ys / 2)

        @dirichlet
        def upper_bound_x(t, location_grid):
            if False:
                for i in range(10):
                    print('nop')
            del location_grid
            return tf.exp(t) * np.sin(x_max / _SQRT2) * tf.sin(ys / 2)

        @dirichlet
        def lower_bound_y(t, location_grid):
            if False:
                while True:
                    i = 10
            del location_grid
            return tf.exp(t) * tf.sin(xs / _SQRT2) * np.sin(y_min / 2)

        @dirichlet
        def upper_bound_y(t, location_grid):
            if False:
                print('Hello World!')
            del location_grid
            return tf.exp(t) * tf.sin(xs / _SQRT2) * np.sin(y_max / 2)
        expected = np.outer(np.sin(ys / 2), np.sin(xs / _SQRT2))
        final_values = tf.expand_dims(tf.constant(np.outer(np.sin(ys / 2), np.sin(xs / _SQRT2)) * np.exp(final_t), dtype=tf.float32), axis=0)
        bound_cond = [(lower_bound_y, upper_bound_y), (lower_bound_x, upper_bound_x)]
        step_fn = douglas_adi_step(theta=0.5)
        result = fd_solvers.solve_backward(start_time=final_t, end_time=0, coord_grid=grid, values_grid=final_values, time_step=time_step, one_step_fn=step_fn, boundary_conditions=bound_cond, second_order_coeff_fn=second_order_coeff_fn, dtype=grid[0].dtype)
        self._assertClose(expected, result)

    def testAnisotropicDiffusion_WithNeumannBoundaries(self):
        if False:
            while True:
                i = 10
        'Tests solving 2d diffusion equation with Neumann boundary conditions.'
        time_step = 0.01
        final_t = 1
        x_min = -20
        x_max = 20
        y_min = -10
        y_max = 10
        grid = grids.uniform_grid(minimums=[y_min, x_min], maximums=[y_max, x_max], sizes=[201, 301], dtype=tf.float32)
        ys = self.evaluate(grid[0])
        xs = self.evaluate(grid[1])

        def second_order_coeff_fn(t, location_grid):
            if False:
                while True:
                    i = 10
            del t, location_grid
            return [[2, None], [None, 1]]

        @neumann
        def lower_bound_x(t, location_grid):
            if False:
                print('Hello World!')
            del location_grid
            return -tf.exp(t) * np.cos(x_min / _SQRT2) * tf.sin(ys / 2) / _SQRT2

        @neumann
        def upper_bound_x(t, location_grid):
            if False:
                while True:
                    i = 10
            del location_grid
            return tf.exp(t) * np.cos(x_max / _SQRT2) * tf.sin(ys / 2) / _SQRT2

        @neumann
        def lower_bound_y(t, location_grid):
            if False:
                return 10
            del location_grid
            return -tf.exp(t) * tf.sin(xs / _SQRT2) * np.cos(y_min / 2) / 2

        @neumann
        def upper_bound_y(t, location_grid):
            if False:
                for i in range(10):
                    print('nop')
            del location_grid
            return tf.exp(t) * tf.sin(xs / _SQRT2) * np.cos(y_max / 2) / 2
        expected = np.outer(np.sin(ys / 2), np.sin(xs / _SQRT2))
        final_values = tf.expand_dims(tf.constant(np.outer(np.sin(ys / 2), np.sin(xs / _SQRT2)) * np.exp(final_t), dtype=tf.float32), axis=0)
        bound_cond = [(lower_bound_y, upper_bound_y), (lower_bound_x, upper_bound_x)]
        step_fn = douglas_adi_step(theta=0.5)
        result = fd_solvers.solve_backward(start_time=final_t, end_time=0, coord_grid=grid, values_grid=final_values, time_step=time_step, one_step_fn=step_fn, boundary_conditions=bound_cond, second_order_coeff_fn=second_order_coeff_fn, dtype=grid[0].dtype)
        self._assertClose(expected, result)

    def testAnisotropicDiffusion_WithMixedBoundaries(self):
        if False:
            while True:
                i = 10
        'Tests solving 2d diffusion equation with mixed boundary conditions.'
        time_step = 0.01
        final_t = 1
        x_min = -20
        x_max = 20
        y_min = -10
        y_max = 10
        grid = grids.uniform_grid(minimums=[y_min, x_min], maximums=[y_max, x_max], sizes=[201, 301], dtype=tf.float32)
        ys = self.evaluate(grid[0])
        xs = self.evaluate(grid[1])

        def second_order_coeff_fn(t, location_grid):
            if False:
                return 10
            del t, location_grid
            return [[2, None], [None, 1]]

        @dirichlet
        def lower_bound_x(t, location_grid):
            if False:
                return 10
            del location_grid
            return tf.exp(t) * np.sin(x_min / _SQRT2) * tf.sin(ys / 2)

        @neumann
        def upper_bound_x(t, location_grid):
            if False:
                for i in range(10):
                    print('nop')
            del location_grid
            return tf.exp(t) * np.cos(x_max / _SQRT2) * tf.sin(ys / 2) / _SQRT2

        @neumann
        def lower_bound_y(t, location_grid):
            if False:
                for i in range(10):
                    print('nop')
            del location_grid
            return -tf.exp(t) * tf.sin(xs / _SQRT2) * np.cos(y_min / 2) / 2

        @dirichlet
        def upper_bound_y(t, location_grid):
            if False:
                print('Hello World!')
            del location_grid
            return tf.exp(t) * tf.sin(xs / _SQRT2) * np.sin(y_max / 2)
        expected = np.outer(np.sin(ys / 2), np.sin(xs / _SQRT2))
        final_values = tf.expand_dims(tf.constant(np.outer(np.sin(ys / 2), np.sin(xs / _SQRT2)) * np.exp(final_t), dtype=tf.float32), axis=0)
        bound_cond = [(lower_bound_y, upper_bound_y), (lower_bound_x, upper_bound_x)]
        step_fn = douglas_adi_step(theta=0.5)
        result = fd_solvers.solve_backward(start_time=final_t, end_time=0, coord_grid=grid, values_grid=final_values, time_step=time_step, one_step_fn=step_fn, boundary_conditions=bound_cond, second_order_coeff_fn=second_order_coeff_fn, dtype=grid[0].dtype)
        self._assertClose(expected, result)

    def testAnisotropicDiffusion_WithRobinBoundaries(self):
        if False:
            i = 10
            return i + 15
        'Tests solving 2d diffusion equation with Robin boundary conditions.'
        time_step = 0.01
        final_t = 1
        x_min = -20
        x_max = 20
        y_min = -10
        y_max = 10
        grid = grids.uniform_grid(minimums=[y_min, x_min], maximums=[y_max, x_max], sizes=[201, 301], dtype=tf.float32)
        ys = self.evaluate(grid[0])
        xs = self.evaluate(grid[1])

        def second_order_coeff_fn(t, location_grid):
            if False:
                print('Hello World!')
            del t, location_grid
            return [[2, None], [None, 1]]

        def lower_bound_x(t, location_grid):
            if False:
                for i in range(10):
                    print('nop')
            del location_grid
            f = tf.exp(t) * tf.sin(ys / 2) * (np.sin(x_min / _SQRT2) - np.cos(x_min / _SQRT2) / _SQRT2)
            return (1, 1, f)

        def upper_bound_x(t, location_grid):
            if False:
                i = 10
                return i + 15
            del location_grid
            f = tf.exp(t) * tf.sin(ys / 2) * (np.sin(x_max / _SQRT2) + 2 * np.cos(x_max / _SQRT2) / _SQRT2)
            return (1, 2, f)

        def lower_bound_y(t, location_grid):
            if False:
                while True:
                    i = 10
            del location_grid
            f = tf.exp(t) * tf.sin(xs / _SQRT2) * (np.sin(y_min / 2) - 3 * np.cos(y_min / 2) / 2)
            return (1, 3, f)

        def upper_bound_y(t, location_grid):
            if False:
                while True:
                    i = 10
            del location_grid
            f = tf.exp(t) * tf.sin(xs / _SQRT2) * (2 * np.sin(y_max / 2) + 3 * np.cos(y_max / 2) / 2)
            return (2, 3, f)
        expected = np.outer(np.sin(ys / 2), np.sin(xs / _SQRT2))
        final_values = tf.expand_dims(tf.constant(np.outer(np.sin(ys / 2), np.sin(xs / _SQRT2)) * np.exp(final_t), dtype=tf.float32), axis=0)
        bound_cond = [(lower_bound_y, upper_bound_y), (lower_bound_x, upper_bound_x)]
        step_fn = douglas_adi_step(theta=0.5)
        result = fd_solvers.solve_backward(start_time=final_t, end_time=0, coord_grid=grid, values_grid=final_values, time_step=time_step, one_step_fn=step_fn, boundary_conditions=bound_cond, second_order_coeff_fn=second_order_coeff_fn, dtype=grid[0].dtype)
        self._assertClose(expected, result)

    def _assertClose(self, expected, stepper_result):
        if False:
            return 10
        actual = self.evaluate(stepper_result[0])
        self.assertLess(np.max(np.abs(actual - expected)) / np.max(expected), 0.01)

    def testAnisotropicDiffusion_InForwardDirection(self):
        if False:
            i = 10
            return i + 15
        'Tests solving 2d diffusion equation in forward direction.\n\n    The equation is `u_{t} - Dx u_{xx} - Dy u_{yy} = 0`.\n    The initial condition is a gaussian centered at (0, 0) with variance sigma.\n    The variance along each dimension should evolve as `sigma + 2 Dx (t - t_0)`\n    and `sigma + 2 Dy (t - t_0)`.\n    '
        grid = grids.uniform_grid(minimums=[-10, -20], maximums=[10, 20], sizes=[201, 301], dtype=tf.float32)
        ys = self.evaluate(grid[0])
        xs = self.evaluate(grid[1])
        diff_coeff_x = 0.4
        diff_coeff_y = 0.25
        time_step = 0.1
        final_t = 1.0
        initial_variance = 1

        def quadratic_coeff_fn(t, location_grid):
            if False:
                for i in range(10):
                    print('nop')
            del t, location_grid
            u_xx = -diff_coeff_x
            u_yy = -diff_coeff_y
            u_xy = None
            return [[u_yy, u_xy], [u_xy, u_xx]]
        final_values = tf.expand_dims(tf.constant(np.outer(_gaussian(ys, initial_variance), _gaussian(xs, initial_variance)), dtype=tf.float32), axis=0)
        bound_cond = [(_zero_boundary, _zero_boundary), (_zero_boundary, _zero_boundary)]
        step_fn = douglas_adi_step(theta=0.5)
        result = fd_solvers.solve_forward(start_time=0.0, end_time=final_t, coord_grid=grid, values_grid=final_values, time_step=time_step, one_step_fn=step_fn, boundary_conditions=bound_cond, second_order_coeff_fn=quadratic_coeff_fn, dtype=grid[0].dtype)
        variance_x = initial_variance + 2 * diff_coeff_x * final_t
        variance_y = initial_variance + 2 * diff_coeff_y * final_t
        expected = np.outer(_gaussian(ys, variance_y), _gaussian(xs, variance_x))
        self._assertClose(expected, result)

    def testReferenceEquation(self):
        if False:
            return 10
        'Tests the equation used as reference for a few further tests.\n\n    We solve the heat equation `u_t = u_xx + u_yy` on x = [0...1], y = [0...1]\n    with boundary conditions `u(x, y, t=0) = (1/2 - |x-1/2|)(1/2-|y-1/2|), and\n    zero Dirichlet on all spatial boundaries.\n\n    The exact solution of the diffusion equation with zero-Dirichlet rectangular\n    boundaries is `u(x, y, t) = u(x, t) * u(y, t)`,\n    `u(z, t) = sum_{n=1..inf} b_n sin(pi n z) exp(-n^2 pi^2 t)`,\n    `b_n = 2 integral_{0..1} sin(pi n z) u(z, t=0) dz.`\n\n    The initial conditions are taken so that the integral easily calculates, and\n    the sum can be approximated by a few first terms (given large enough `t`).\n    See the result in _reference_heat_equation_solution.\n\n    Using this solution helps to simplify the tests, as we don\'t have to\n    maintain complicated boundary conditions in each test or tweak the\n    parameters to keep the "support" of the function far from boundaries.\n    '
        grid = grids.uniform_grid(minimums=[0, 0], maximums=[1, 1], sizes=[201, 301], dtype=tf.float32)
        (ys, xs) = grid
        final_t = 0.1
        time_step = 0.002

        def second_order_coeff_fn(t, coord_grid):
            if False:
                print('Hello World!')
            del t, coord_grid
            return [[-1, None], [None, -1]]
        initial = _reference_2d_pde_initial_cond(xs, ys)
        expected = _reference_2d_pde_solution(xs, ys, final_t)
        actual = fd_solvers.solve_forward(start_time=0, end_time=final_t, coord_grid=grid, values_grid=initial, time_step=time_step, second_order_coeff_fn=second_order_coeff_fn)[0]
        self.assertAllClose(expected, actual, atol=0.001, rtol=0.001)

    def testReference_WithExponentMultiplier(self):
        if False:
            i = 10
            return i + 15
        'Tests solving diffusion equation with an exponent multiplier.\n\n    Take the heat equation `v_{t} - v_{xx} - v_{yy} = 0` and substitute\n    `v = exp(x + 2y) u`.\n    This yields `u_{t} - u_{xx} - u_{yy} - 2u_{x} - 4u_{y} - 5u = 0`. The test\n    compares numerical solution of this equation to the exact one, which is the\n    diffusion equation solution times `exp(-x-2y)`.\n    '
        grid = grids.uniform_grid(minimums=[0, 0], maximums=[1, 1], sizes=[201, 301], dtype=tf.float32)
        (ys, xs) = grid
        final_t = 0.1
        time_step = 0.002

        def second_order_coeff_fn(t, coord_grid):
            if False:
                i = 10
                return i + 15
            del t, coord_grid
            return [[-1, None], [None, -1]]

        def first_order_coeff_fn(t, coord_grid):
            if False:
                for i in range(10):
                    print('nop')
            del t, coord_grid
            return [-4, -2]

        def zeroth_order_coeff_fn(t, coord_grid):
            if False:
                return 10
            del t, coord_grid
            return -5
        exp = _dir_prod(tf.exp(-2 * ys), tf.exp(-xs))
        initial = exp * _reference_2d_pde_initial_cond(xs, ys)
        expected = exp * _reference_2d_pde_solution(xs, ys, final_t)
        actual = fd_solvers.solve_forward(start_time=0, end_time=final_t, coord_grid=grid, values_grid=initial, time_step=time_step, second_order_coeff_fn=second_order_coeff_fn, first_order_coeff_fn=first_order_coeff_fn, zeroth_order_coeff_fn=zeroth_order_coeff_fn)[0]
        self.assertAllClose(expected, actual, atol=0.001, rtol=0.001)

    def testInnerSecondOrderCoeff(self):
        if False:
            print('Hello World!')
        'Tests handling inner_second_order_coeff.\n\n    As in previous test, take the diffusion equation\n    `v_{t} - v_{xx} - v_{yy} = 0` and substitute `v = exp(x + 2y) u`, but this\n    time keep exponent under the derivative:\n    `u_{t} - exp(-x)[exp(x)u]_{xx} - exp(-2y)[exp(2y)u]_{yy} = 0`.\n    Expect the same solution as in previous test.\n    '
        grid = grids.uniform_grid(minimums=[0, 0], maximums=[1, 1], sizes=[201, 251], dtype=tf.float32)
        (ys, xs) = grid
        final_t = 0.1
        time_step = 0.002

        def second_order_coeff_fn(t, coord_grid):
            if False:
                i = 10
                return i + 15
            del t
            (y, x) = tf.meshgrid(*coord_grid, indexing='ij')
            return [[-tf.exp(-2 * y), None], [None, -tf.exp(-x)]]

        def inner_second_order_coeff_fn(t, coord_grid):
            if False:
                for i in range(10):
                    print('nop')
            del t
            (y, x) = tf.meshgrid(*coord_grid, indexing='ij')
            return [[tf.exp(2 * y), None], [None, tf.exp(x)]]
        exp = _dir_prod(tf.exp(-2 * ys), tf.exp(-xs))
        initial = exp * _reference_2d_pde_initial_cond(xs, ys)
        expected = exp * _reference_2d_pde_solution(xs, ys, final_t)
        actual = fd_solvers.solve_forward(start_time=0, end_time=final_t, coord_grid=grid, values_grid=initial, time_step=time_step, second_order_coeff_fn=second_order_coeff_fn, inner_second_order_coeff_fn=inner_second_order_coeff_fn)[0]
        self.assertAllClose(expected, actual, atol=0.001, rtol=0.001)

    def testInnerFirstAndSecondOrderCoeff(self):
        if False:
            print('Hello World!')
        'Tests handling both inner_first_order_coeff and inner_second_order_coeff.\n\n    We saw previously that the solution of\n    `u_{t} - u_{xx} - u_{yy} - 2u_{x} - 4u_{y} - 5u = 0` is\n    `u = exp(-x-2y) v`, where `v` solves the diffusion equation. Substitute now\n    `u = exp(-x-2y) v` without expanding the derivatives:\n    `v_{t} - exp(x)[exp(-x)v]_{xx} - exp(2y)[exp(-2y)v]_{yy} -\n      2exp(x)[exp(-x)v]_{x} - 4exp(2y)[exp(-2y)v]_{y} - 5v = 0`.\n    Solve this equation and expect the solution of the diffusion equation.\n    '
        grid = grids.uniform_grid(minimums=[0, 0], maximums=[1, 1], sizes=[201, 251], dtype=tf.float32)
        (ys, xs) = grid
        final_t = 0.1
        time_step = 0.002

        def second_order_coeff_fn(t, coord_grid):
            if False:
                for i in range(10):
                    print('nop')
            del t
            (y, x) = tf.meshgrid(*coord_grid, indexing='ij')
            return [[-tf.exp(2 * y), None], [None, -tf.exp(x)]]

        def inner_second_order_coeff_fn(t, coord_grid):
            if False:
                for i in range(10):
                    print('nop')
            del t
            (y, x) = tf.meshgrid(*coord_grid, indexing='ij')
            return [[tf.exp(-2 * y), None], [None, tf.exp(-x)]]

        def first_order_coeff_fn(t, coord_grid):
            if False:
                for i in range(10):
                    print('nop')
            del t
            (y, x) = tf.meshgrid(*coord_grid, indexing='ij')
            return [-4 * tf.exp(2 * y), -2 * tf.exp(x)]

        def inner_first_order_coeff_fn(t, coord_grid):
            if False:
                while True:
                    i = 10
            del t
            (y, x) = tf.meshgrid(*coord_grid, indexing='ij')
            return [tf.exp(-2 * y), tf.exp(-x)]

        def zeroth_order_coeff_fn(t, coord_grid):
            if False:
                for i in range(10):
                    print('nop')
            del t, coord_grid
            return -5
        initial = _reference_2d_pde_initial_cond(xs, ys)
        expected = _reference_2d_pde_solution(xs, ys, final_t)
        actual = fd_solvers.solve_forward(start_time=0, end_time=final_t, coord_grid=grid, values_grid=initial, time_step=time_step, second_order_coeff_fn=second_order_coeff_fn, first_order_coeff_fn=first_order_coeff_fn, zeroth_order_coeff_fn=zeroth_order_coeff_fn, inner_second_order_coeff_fn=inner_second_order_coeff_fn, inner_first_order_coeff_fn=inner_first_order_coeff_fn)[0]
        self.assertAllClose(expected, actual, atol=0.001, rtol=0.001)

    def testReferenceEquation_WithTransformationYieldingMixedTerm(self):
        if False:
            return 10
        'Tests an equation with mixed terms against exact solution.\n\n    Take the reference equation `v_{t} = v_{xx} + v_{yy}` and substitute\n    `v(x, y, t) = u(x, 2y - x, t)`. This yields\n    `u_{t} = u_{xx} + 5u_{zz} - 2u_{xz}`, where `z = 2y - x`.\n    Having `u(x, z, t) = v(x, (x+z)/2, t)` where `v(x, y, t)` is the known\n    solution of the reference equation, we derive the boundary conditions\n    and the expected solution for `u(x, y, t)`.\n    '
        grid = grids.uniform_grid(minimums=[0, 0], maximums=[1, 1], sizes=[201, 251], dtype=tf.float32)
        final_t = 0.1
        time_step = 0.002

        def second_order_coeff_fn(t, coord_grid):
            if False:
                return 10
            del t, coord_grid
            return [[-5, 1], [None, -1]]

        @dirichlet
        def boundary_lower_z(t, coord_grid):
            if False:
                while True:
                    i = 10
            x = coord_grid[1]
            return _reference_pde_solution(x, t) * _reference_pde_solution(x / 2, t)

        @dirichlet
        def boundary_upper_z(t, coord_grid):
            if False:
                i = 10
                return i + 15
            x = coord_grid[1]
            return _reference_pde_solution(x, t) * _reference_pde_solution((x + 1) / 2, t)
        (z_mesh, x_mesh) = tf.meshgrid(grid[0], grid[1], indexing='ij')
        initial = _reference_pde_initial_cond(x_mesh) * _reference_pde_initial_cond((x_mesh + z_mesh) / 2)
        expected = _reference_pde_solution(x_mesh, final_t) * _reference_pde_solution((x_mesh + z_mesh) / 2, final_t)
        actual = fd_solvers.solve_forward(start_time=0, end_time=final_t, coord_grid=grid, values_grid=initial, time_step=time_step, second_order_coeff_fn=second_order_coeff_fn, boundary_conditions=[(boundary_lower_z, boundary_upper_z), (_zero_boundary, _zero_boundary)])[0]
        self.assertAllClose(expected, actual, atol=0.001, rtol=0.001)

    def testInnerMixedSecondOrderCoeffs(self):
        if False:
            while True:
                i = 10
        'Tests handling coefficients under the mixed second derivative.\n\n    Take the equation from the previous test,\n    `u_{t} = u_{xx} + 5u_{zz} - 2u_{xz}` and substitute `u = exp(xz) w`,\n    leaving the exponent under the derivatives:\n    `w_{t} = exp(-xz) [exp(xz) u]_{xx} + 5 exp(-xz) [exp(xz) u]_{zz}\n    - 2 exp(-xz) [exp(xz) u]_{xz}`.\n    We now have a coefficient under the mixed derivative. Test that the solution\n    is `w = exp(-xz) u`, where u is from the previous test.\n    '
        grid = grids.uniform_grid(minimums=[0, 0], maximums=[1, 1], sizes=[201, 251], dtype=tf.float32)
        final_t = 0.1
        time_step = 0.002

        def second_order_coeff_fn(t, coord_grid):
            if False:
                for i in range(10):
                    print('nop')
            del t
            (z, x) = tf.meshgrid(*coord_grid, indexing='ij')
            exp = tf.math.exp(-z * x)
            return [[-5 * exp, exp], [None, -exp]]

        def inner_second_order_coeff_fn(t, coord_grid):
            if False:
                for i in range(10):
                    print('nop')
            del t
            (z, x) = tf.meshgrid(*coord_grid, indexing='ij')
            exp = tf.math.exp(z * x)
            return [[exp, exp], [None, exp]]

        @dirichlet
        def boundary_lower_z(t, coord_grid):
            if False:
                for i in range(10):
                    print('nop')
            x = coord_grid[1]
            return _reference_pde_solution(x, t) * _reference_pde_solution(x / 2, t)

        @dirichlet
        def boundary_upper_z(t, coord_grid):
            if False:
                return 10
            x = coord_grid[1]
            return tf.exp(-x) * _reference_pde_solution(x, t) * _reference_pde_solution((x + 1) / 2, t)
        (z, x) = tf.meshgrid(*grid, indexing='ij')
        exp = tf.math.exp(-z * x)
        initial = exp * (_reference_pde_initial_cond(x) * _reference_pde_initial_cond((x + z) / 2))
        expected = exp * (_reference_pde_solution(x, final_t) * _reference_pde_solution((x + z) / 2, final_t))
        actual = fd_solvers.solve_forward(start_time=0, end_time=final_t, coord_grid=grid, values_grid=initial, time_step=time_step, second_order_coeff_fn=second_order_coeff_fn, inner_second_order_coeff_fn=inner_second_order_coeff_fn, boundary_conditions=[(boundary_lower_z, boundary_upper_z), (_zero_boundary, _zero_boundary)])[0]
        self.assertAllClose(expected, actual, atol=0.001, rtol=0.001)

    def testCompareExpandedAndNotExpandedPdes(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests comparing PDEs with expanded derivatives and without.\n\n    The equation is\n    `u_{t} + [x u]_{x} + [y^2 u]_{y} - [sin(x) u]_{xx} - [cos(y) u]_yy\n     + [x^3 y^2 u]_{xy} = 0`.\n    Solve the equation, expand the derivatives and solve the equation again.\n    Expect the results to be equal.\n    '
        grid = grids.uniform_grid(minimums=[0, 0], maximums=[1, 1], sizes=[201, 251], dtype=tf.float32)
        final_t = 0.1
        time_step = 0.002
        (y, x) = grid
        initial = _reference_2d_pde_initial_cond(x, y)

        def inner_second_order_coeff_fn(t, coord_grid):
            if False:
                i = 10
                return i + 15
            del t
            (y, x) = tf.meshgrid(*coord_grid, indexing='ij')
            return [[-tf.math.cos(y), x ** 3 * y ** 2 / 2], [None, -tf.math.sin(x)]]

        def inner_first_order_coeff_fn(t, coord_grid):
            if False:
                while True:
                    i = 10
            del t
            (y, x) = tf.meshgrid(*coord_grid, indexing='ij')
            return [y ** 2, x]
        result_not_expanded = fd_solvers.solve_forward(start_time=0, end_time=final_t, coord_grid=grid, values_grid=initial, time_step=time_step, inner_second_order_coeff_fn=inner_second_order_coeff_fn, inner_first_order_coeff_fn=inner_first_order_coeff_fn)[0]

        def second_order_coeff_fn(t, coord_grid):
            if False:
                for i in range(10):
                    print('nop')
            del t
            (y, x) = tf.meshgrid(*coord_grid, indexing='ij')
            return [[-tf.math.cos(y), x ** 3 * y ** 2 / 2], [None, -tf.math.sin(x)]]

        def first_order_coeff_fn(t, coord_grid):
            if False:
                for i in range(10):
                    print('nop')
            del t
            (y, x) = tf.meshgrid(*coord_grid, indexing='ij')
            return [y ** 2 * (1 + 3 * x ** 2) + 2 * tf.math.sin(y), x * (1 + 2 * x ** 2 * y) - 2 * tf.math.cos(x)]

        def zeroth_order_coeff_fn(t, coord_grid):
            if False:
                i = 10
                return i + 15
            del t
            (y, x) = tf.meshgrid(*coord_grid, indexing='ij')
            return 1 + 2 * y + tf.math.sin(x) + tf.math.cos(x) + 6 * x ** 2 * y
        result_expanded = fd_solvers.solve_forward(start_time=0, end_time=final_t, coord_grid=grid, values_grid=initial, time_step=time_step, second_order_coeff_fn=second_order_coeff_fn, first_order_coeff_fn=first_order_coeff_fn, zeroth_order_coeff_fn=zeroth_order_coeff_fn)[0]
        self.assertAllClose(result_not_expanded, result_expanded, atol=0.001, rtol=0.001)

    @parameterized.named_parameters({'testcase_name': 'WithDefault', 'include_defualt_bc': True}, {'testcase_name': 'WithoutDefault', 'include_defualt_bc': False})
    def testMixedTermsWithMixedBoundary(self, include_defualt_bc):
        if False:
            return 10
        'Tests solving a batch of PDEs with mixed terms and mixed boundaries.\n\n    The equation are\n    `u_{t} + u_{xx}  + u_{yy} + u_{zz}\n    + sin(x) * cos(y) * u_{xy} + cos(y) * cos(z) * u_{yz}\n    +  (2 + cos(x) * sin(y) - sin(y) * sin(z)) * u = 0\n     with initial condition `u(0.1, x, y) = exp(0.1) * sin(x) * cos(y) * cos(z)`\n     and  boundary conditions implied by the solution\n     `u(t, x, y) = sin(x) * cos(y) * cos(z)`.\n    '
        dtype = tf.float64
        grid = grids.uniform_grid(minimums=[0, 0, 0], maximums=[3 * np.pi / 2, 2.5, 2.75], sizes=[51, 61, 71], dtype=dtype)
        zs = grid[0]
        ys = grid[1]
        xs = grid[2]
        time_step = 0.01
        final_t = tf.constant(0.1, dtype=dtype)

        def second_order_coeff_fn(t, location_grid):
            if False:
                while True:
                    i = 10
            del t
            (z, y, x) = tf.meshgrid(*location_grid, indexing='ij')
            u_zz = 1
            u_xx = 1
            u_yy = 1
            u_xy = tf.math.sin(x) * tf.math.cos(y) / 2
            u_yz = tf.math.cos(y) * tf.math.cos(z) / 2
            return [[u_zz, u_yz, None], [u_yz, u_yy, u_xy], [None, u_xy, u_xx]]

        def zeroth_order_coeff_fn(t, location_grid):
            if False:
                i = 10
                return i + 15
            del t
            (z, y, x) = tf.meshgrid(*location_grid, indexing='ij')
            return 2 + tf.math.sin(y) * tf.math.cos(x) - tf.math.sin(y) * tf.math.sin(z)
        init_values = tf.expand_dims(tf.math.exp(final_t) * tf.reshape(tf.math.cos(zs), [-1, 1, 1]) * tf.reshape(tf.math.cos(ys), [1, -1, 1]) * tf.reshape(tf.math.sin(xs), [1, 1, -1]), axis=0)

        @neumann
        def lower_boundary_x_fn(t, location_grid):
            if False:
                print('Hello World!')
            del location_grid
            return -tf.math.exp(t) * tf.math.cos(xs[0]) * tf.expand_dims(tf.math.cos(ys), 0) * tf.expand_dims(tf.math.cos(zs), -1)

        @neumann
        def upper_boundary_x_fn(t, x):
            if False:
                print('Hello World!')
            del x
            return tf.math.exp(t) * tf.math.cos(xs[-1]) * tf.expand_dims(tf.math.cos(ys), 0) * tf.expand_dims(tf.math.cos(zs), -1)

        @dirichlet
        def lower_boundary_y_fn(t, location_grid):
            if False:
                return 10
            del location_grid
            return tf.math.exp(t) * tf.expand_dims(tf.math.sin(xs), 0) * tf.math.cos(ys[0]) * tf.expand_dims(tf.math.cos(zs), -1)

        @dirichlet
        def upper_boundary_y_fn(t, x):
            if False:
                i = 10
                return i + 15
            del x
            return tf.math.exp(t) * tf.expand_dims(tf.math.sin(xs), 0) * tf.math.cos(ys[-1]) * tf.expand_dims(tf.math.cos(zs), -1)

        @dirichlet
        def lower_boundary_z_fn(t, location_grid):
            if False:
                for i in range(10):
                    print('nop')
            del location_grid
            return tf.math.exp(t) * tf.expand_dims(tf.math.sin(xs), 0) * tf.expand_dims(tf.math.cos(ys), -1) * tf.math.cos(zs[0])

        @neumann
        def upper_boundary_z_fn(t, x):
            if False:
                for i in range(10):
                    print('nop')
            del x
            return -tf.math.exp(t) * tf.expand_dims(tf.math.sin(xs), 0) * tf.expand_dims(tf.math.cos(ys), -1) * tf.math.sin(zs[-1])
        if include_defualt_bc:
            bound_cond = [(lower_boundary_z_fn, None), (lower_boundary_y_fn, upper_boundary_y_fn), (None, upper_boundary_x_fn)]
        else:
            bound_cond = [(lower_boundary_z_fn, upper_boundary_z_fn), (lower_boundary_y_fn, upper_boundary_y_fn), (lower_boundary_x_fn, upper_boundary_x_fn)]
        step_fn = douglas_adi_step(theta=0.5)
        result = fd_solvers.solve_backward(start_time=final_t, end_time=0, coord_grid=grid, values_grid=init_values, time_step=time_step, one_step_fn=step_fn, boundary_conditions=bound_cond, second_order_coeff_fn=second_order_coeff_fn, zeroth_order_coeff_fn=zeroth_order_coeff_fn, dtype=grid[0].dtype)
        expected = tf.reshape(tf.math.cos(zs), [-1, 1, 1]) * tf.reshape(tf.math.cos(ys), [1, -1, 1]) * tf.reshape(tf.math.sin(xs), [1, 1, -1]) + tf.zeros_like(result[0])
        with self.subTest(name='CorrectShape'):
            self.assertAllEqual(result[0].shape.as_list(), [1, 51, 61, 71])
        with self.subTest(name='CorrectSolution'):
            self.assertAllClose(expected, result[0], atol=0.01, rtol=0.01)

    @parameterized.named_parameters({'testcase_name': 'WithDefault', 'include_default_bc': True}, {'testcase_name': 'WithoutDefault', 'include_default_bc': False})
    def testMixedTermsWithMixedBoundaryBatchGrid(self, include_default_bc):
        if False:
            return 10
        'Tests solving a batch of PDEs with batch grid.\n\n    The equation are\n    `u_{t} + u_{xx}  + u_{yy} + u_{zz}\n    + sin(x) * cos(y) * u_{xy} + cos(y) * cos(z) * u_{yz}\n    +  (2 + cos(x) * sin(y) - sin(y) * sin(z)) * u = 0\n     with initial condition `u(0.1, x, y) = exp(0.1) * sin(x) * cos(y) * cos(z)`\n     and  boundary conditions implied by the solution\n     `u(t, x, y) = sin(x) * cos(y) * cos(z)`.\n    '
        dtype = np.float64
        grid1 = grids.uniform_grid(minimums=[0, 0, 0], maximums=[3 * np.pi / 2, 2.5, 2.75], sizes=[51, 61, 71], dtype=dtype)
        grid2 = grids.uniform_grid(minimums=[0, 0, 0], maximums=[3 * np.pi / 2, 2.0, 2.75], sizes=[51, 61, 71], dtype=dtype)
        grid = [grid1[0], tf.stack([grid1[1], grid2[1]]), grid2[2]]
        zs = grid[0]
        ys = grid[1]
        xs = grid[2]
        time_step = 0.01
        final_t = tf.constant(0.1, dtype=dtype)

        def meshgrid_fn(args):
            if False:
                print('Hello World!')
            return tf.meshgrid(*args, indexing='ij')

        def vectorized_meshgrid(grid):
            if False:
                i = 10
                return i + 15
            return tf.vectorized_map(meshgrid_fn, grid, fallback_to_while_loop=False)

        def second_order_coeff_fn(t, location_grid):
            if False:
                print('Hello World!')
            del t
            (z, y, x) = vectorized_meshgrid(location_grid)
            u_zz = 1
            u_xx = 1
            u_yy = 1
            u_xy = tf.math.sin(x) * tf.math.cos(y) / 2
            u_yz = tf.math.cos(y) * tf.math.cos(z) / 2
            return [[u_zz, u_yz, None], [u_yz, u_yy, u_xy], [None, u_xy, u_xx]]

        def zeroth_order_coeff_fn(t, location_grid):
            if False:
                while True:
                    i = 10
            del t
            (z, y, x) = vectorized_meshgrid(location_grid)
            res = 2 + tf.math.sin(y) * tf.math.cos(x) - tf.math.sin(y) * tf.math.sin(z)
            return res
        init_values = tf.math.exp(final_t) * tf.math.cos(zs)[..., tf.newaxis, tf.newaxis] * tf.math.cos(ys)[..., tf.newaxis, :, tf.newaxis] * tf.math.sin(xs)[..., tf.newaxis, tf.newaxis, :]

        @neumann
        def lower_boundary_x_fn(t, location_grid):
            if False:
                while True:
                    i = 10
            del location_grid
            return -tf.math.exp(t) * tf.math.cos(xs[..., 0])[..., tf.newaxis, tf.newaxis] * tf.expand_dims(tf.math.cos(ys), -2) * tf.expand_dims(tf.math.cos(zs), -1)

        @neumann
        def upper_boundary_x_fn(t, x):
            if False:
                i = 10
                return i + 15
            del x
            return tf.math.exp(t) * tf.math.cos(xs[..., -1])[..., tf.newaxis, tf.newaxis] * tf.expand_dims(tf.math.cos(ys), -2) * tf.expand_dims(tf.math.cos(zs), -1)

        @dirichlet
        def lower_boundary_y_fn(t, location_grid):
            if False:
                print('Hello World!')
            del location_grid
            return tf.math.exp(t) * tf.expand_dims(tf.math.sin(xs), -2) * tf.math.cos(ys[..., 0])[..., tf.newaxis, tf.newaxis] * tf.expand_dims(tf.math.cos(zs), -1)

        @dirichlet
        def upper_boundary_y_fn(t, x):
            if False:
                while True:
                    i = 10
            del x
            return tf.math.exp(t) * tf.expand_dims(tf.math.sin(xs), -2) * tf.math.cos(ys[..., -1])[..., tf.newaxis, tf.newaxis] * tf.expand_dims(tf.math.cos(zs), -1)

        @dirichlet
        def lower_boundary_z_fn(t, location_grid):
            if False:
                i = 10
                return i + 15
            del location_grid
            return tf.math.exp(t) * tf.expand_dims(tf.math.sin(xs), -2) * tf.expand_dims(tf.math.cos(ys), -1) * tf.math.cos(zs[..., 0])[..., tf.newaxis, tf.newaxis]

        @neumann
        def upper_boundary_z_fn(t, x):
            if False:
                while True:
                    i = 10
            del x
            return -tf.math.exp(t) * tf.expand_dims(tf.math.sin(xs), -2) * tf.expand_dims(tf.math.cos(ys), -1) * tf.math.sin(zs[..., -1])[..., tf.newaxis, tf.newaxis]
        if include_default_bc:
            bound_cond = [(lower_boundary_z_fn, None), (lower_boundary_y_fn, upper_boundary_y_fn), (None, upper_boundary_x_fn)]
        else:
            bound_cond = [(lower_boundary_z_fn, upper_boundary_z_fn), (lower_boundary_y_fn, upper_boundary_y_fn), (lower_boundary_x_fn, upper_boundary_x_fn)]
        step_fn = douglas_adi_step(theta=0.5)
        result = fd_solvers.solve_backward(start_time=final_t, end_time=0, coord_grid=grid, values_grid=init_values, time_step=time_step, one_step_fn=step_fn, boundary_conditions=bound_cond, second_order_coeff_fn=second_order_coeff_fn, zeroth_order_coeff_fn=zeroth_order_coeff_fn, dtype=grid[0].dtype)
        expected = tf.math.cos(zs)[..., tf.newaxis, tf.newaxis] * tf.math.cos(ys)[..., tf.newaxis, :, tf.newaxis] * tf.math.sin(xs)[..., tf.newaxis, tf.newaxis, :] + tf.zeros_like(result[0])
        with self.subTest(name='CorrectShape'):
            self.assertAllEqual(result[0].shape.as_list(), [2, 51, 61, 71])
        with self.subTest(name='CorrectSolution'):
            self.assertAllClose(expected, result[0], atol=0.01, rtol=0.01)

def _gaussian(xs, variance):
    if False:
        i = 10
        return i + 15
    return np.exp(-np.square(xs) / (2 * variance)) / np.sqrt(2 * np.pi * variance)

@dirichlet
def _zero_boundary(t, locations):
    if False:
        return 10
    del t, locations
    return 0

@neumann
def _zero_grad_boundary(t, locations):
    if False:
        while True:
            i = 10
    del t, locations
    return 0

def _reference_2d_pde_initial_cond(xs, ys):
    if False:
        return 10
    'Initial conditions for the reference 2d diffusion equation.'
    return _dir_prod(_reference_pde_initial_cond(ys), _reference_pde_initial_cond(xs))

def _reference_2d_pde_solution(xs, ys, t, num_terms=5):
    if False:
        print('Hello World!')
    return _dir_prod(_reference_pde_solution(ys, t, num_terms), _reference_pde_solution(xs, t, num_terms))

def _reference_pde_initial_cond(xs):
    if False:
        i = 10
        return i + 15
    'Initial conditions for the reference diffusion equation.'
    return -tf.math.abs(xs - 0.5) + 0.5

def _reference_pde_solution(xs, t, num_terms=5):
    if False:
        print('Hello World!')
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

def _dir_prod(a, b):
    if False:
        print('Hello World!')
    'Calculates the direct product of two Tensors.'
    return tf.tensordot(a, b, ([], []))
if __name__ == '__main__':
    tf.test.main()