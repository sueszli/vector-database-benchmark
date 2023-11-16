"""Weighted implicit-explicit time marching scheme for parabolic PDEs."""
import tensorflow.compat.v2 as tf
from tf_quant_finance.math.pde.steppers.parabolic_equation_stepper import parabolic_equation_step

def weighted_implicit_explicit_step(theta):
    if False:
        while True:
            i = 10
    'Creates a stepper function with weighted implicit-explicit scheme.\n\n  Given a space-discretized equation\n\n  ```\n  du/dt = A(t) u(t) + b(t)\n  ```\n  (here `u` is a value vector, `A` and `b` are the matrix and the vector defined\n  by the PDE), the scheme approximates the right-hand side as a weighted average\n  of values taken before and after a time step:\n\n  ```\n  (u(t2) - u(t1)) / (t2 - t1) = theta * (A(t1) u(t1) + b(t1))\n     + (1 - theta) (A(t2) u(t2) + b(t2)).\n  ```\n\n  Includes as particular cases the implicit (`theta = 0`), explicit\n  (`theta = 1`), and Crank-Nicolson (`theta = 0.5`) schemes.\n\n  The scheme is stable for `theta >= 0.5`, is second order accurate if\n  `theta = 0.5` (i.e. in Crank-Nicolson case), and first order accurate\n  otherwise.\n\n  More details can be found in `weighted_implicit_explicit_scheme` below.\n\n  Args:\n    theta: A float in range `[0, 1]`. A parameter used to mix implicit and\n      explicit schemes together. Value of `0.0` corresponds to the fully\n      implicit scheme, `1.0` to the fully explicit, and `0.5` to the\n      Crank-Nicolson scheme.\n\n  Returns:\n    Callable to be used in finite-difference PDE solvers (see fd_solvers.py).\n  '
    scheme = weighted_implicit_explicit_scheme(theta)

    def step_fn(time, next_time, coord_grid, value_grid, boundary_conditions, second_order_coeff_fn, first_order_coeff_fn, zeroth_order_coeff_fn, inner_second_order_coeff_fn, inner_first_order_coeff_fn, num_steps_performed, dtype=None, name=None):
        if False:
            for i in range(10):
                print('nop')
        'Performs the step.'
        del num_steps_performed
        name = name or 'weighted_implicit_explicit_scheme'
        return parabolic_equation_step(time, next_time, coord_grid, value_grid, boundary_conditions, second_order_coeff_fn, first_order_coeff_fn, zeroth_order_coeff_fn, inner_second_order_coeff_fn, inner_first_order_coeff_fn, time_marching_scheme=scheme, dtype=dtype, name=name)
    return step_fn

def weighted_implicit_explicit_scheme(theta):
    if False:
        while True:
            i = 10
    "Constructs weighted implicit-explicit scheme.\n\n  Approximates the space-discretized equation of `du/dt = A(t) u(t) + b(t)` as\n  ```\n  (u(t2) - u(t1)) / (t2 - t1) = theta * (A u(t1) + b)\n     + (1 - theta) (A u(t2) + b),\n  ```\n  where `A = A((t1 + t2)/2)`, `b = b((t1 + t2)/2)`, and `theta` is a float\n  between `0` and `1`.\n\n  Note that typically `A` and `b` are evaluated at `t1` and `t2` in\n  the explicit and implicit terms respectively (the two terms of the right-hand\n  side). Instead, we evaluate them at the midpoint `(t1 + t2)/2`, which saves\n  some computation. One can check that evaluating at midpoint doesn't change the\n  order of accuracy of the scheme: it is still second order accurate in\n  `t2 - t1` if `theta = 0.5` and first order accurate otherwise.\n\n  The solution is the following:\n  `u(t2) = (1 - (1 - theta) dt A)^(-1) * (1 + theta dt A) u(t1) + dt b`.\n\n  The main bottleneck here is inverting the matrix `(1 - (1 - theta) dt A)`.\n  This matrix is tridiagonal (each point is influenced by the two neighbouring\n  points), and thus the inversion can be efficiently performed using\n  `tf.linalg.tridiagonal_solve`.\n\n  #### References:\n  [1] I.V. Puzynin, A.V. Selin, S.I. Vinitsky, A high-order accuracy method for\n  numerical solving of the time-dependent Schrodinger equation, Comput. Phys.\n  Commun. 123 (1999), 1.\n  https://www.sciencedirect.com/science/article/pii/S0010465599002246\n\n  Args:\n    theta: A float in range `[0, 1]`. A parameter used to mix implicit and\n      explicit schemes together. Value of `0.0` corresponds to the fully\n      implicit scheme, `1.0` to the fully explicit, and `0.5` to the\n      Crank-Nicolson scheme.\n\n  Returns:\n    A callable that consumes the following arguments by keyword:\n      1. value_grid: Grid of values at time `t1`, i.e. `u(t1)`.\n      2. t1: Time before the step.\n      3. t2: Time after the step.\n      4. equation_params_fn: A callable that takes a scalar `Tensor` argument\n        representing time, and constructs the tridiagonal matrix `A`\n        (a tuple of three `Tensor`s, main, upper, and lower diagonals)\n        and the inhomogeneous term `b`. All of the `Tensor`s are of the same\n        `dtype` as `value_grid` and of the shape broadcastable with the\n        shape of `value_grid`.\n    The callable returns a `Tensor` of the same shape and `dtype` as\n    `value_grid` and represents an approximate solution `u(t2)`.\n  "
    if theta < 0 or theta > 1:
        raise ValueError('`theta` should be in [0, 1]. Supplied: {}'.format(theta))

    def _marching_scheme(value_grid, t1, t2, equation_params_fn):
        if False:
            return 10
        'Constructs the time marching scheme.'
        ((diag, superdiag, subdiag), inhomog_term) = equation_params_fn((t1 + t2) / 2)
        if theta == 0:
            rhs = value_grid
        else:
            rhs = _weighted_scheme_explicit_part(value_grid, diag, superdiag, subdiag, theta, t1, t2)
        if inhomog_term is not None:
            rhs += inhomog_term * (t2 - t1)
        if theta < 1:
            return _weighted_scheme_implicit_part(rhs, diag, superdiag, subdiag, theta, t1, t2)
        return rhs
    return _marching_scheme

def _weighted_scheme_explicit_part(vec, diag, upper, lower, theta, t1, t2):
    if False:
        return 10
    'Explicit step of the weighted implicit-explicit scheme.\n\n  Args:\n    vec: A real dtype `Tensor` of shape `[num_equations, num_grid_points - 2]`.\n      Represents the multiplied vector. "- 2" accounts for the boundary points,\n      which the time-marching schemes do not touch.\n    diag: A real dtype `Tensor` of the shape\n      `[num_equations, num_grid_points - 2]`. Represents the main diagonal of\n      a 3-diagonal matrix of the discretized PDE.\n    upper: A real dtype `Tensor` of the shape\n      `[num_equations, num_grid_points - 2]`. Represents the upper diagonal of\n      a 3-diagonal matrix of the discretized PDE.\n    lower:  A real dtype `Tensor` of the shape\n      `[num_equations, num_grid_points - 2]`. Represents the lower diagonal of\n      a 3-diagonal matrix of the discretized PDE.\n    theta: A Python float between 0 and 1.\n    t1: Smaller of the two times defining the step.\n    t2: Greater of the two times defining the step.\n\n  Returns:\n    A tensor of the same shape and dtype as `vec`.\n  '
    multiplier = theta * (t2 - t1)
    diag = 1 + multiplier * diag
    upper = multiplier * upper
    lower = multiplier * lower
    diag_part = diag * vec
    zeros = tf.zeros_like(lower[..., :1])
    lower_part = tf.concat((zeros, lower[..., 1:] * vec[..., :-1]), axis=-1)
    upper_part = tf.concat((upper[..., :-1] * vec[..., 1:], zeros), axis=-1)
    return lower_part + diag_part + upper_part

def _weighted_scheme_implicit_part(vec, diag, upper, lower, theta, t1, t2):
    if False:
        return 10
    'Implicit step of the weighted implicit-explicit scheme.\n\n  Args:\n    vec: A real dtype `Tensor` of shape `[num_equations, num_grid_points - 2]`.\n      Represents the multiplied vector. "- 2" accounts for the boundary points,\n      which the time-marching schemes do not touch.\n    diag: A real dtype `Tensor` of the shape\n      `[num_equations, num_grid_points - 2]`. Represents the main diagonal of\n      a 3-diagonal matrix of the discretized PDE.\n    upper: A real dtype `Tensor` of the shape\n      `[num_equations, num_grid_points - 2]`. Represents the upper diagonal of\n      a 3-diagonal matrix of the discretized PDE.\n    lower:  A real dtype `Tensor` of the shape\n      `[num_equations, num_grid_points - 2]`. Represents the lower diagonal of\n      a 3-diagonal matrix of the discretized PDE.\n    theta: A Python float between 0 and 1.\n    t1: Smaller of the two times defining the step.\n    t2: Greater of the two times defining the step.\n\n  Returns:\n    A tensor of the same shape and dtype as `vec`.\n  '
    multiplier = (1 - theta) * (t1 - t2)
    diag = 1 + multiplier * diag
    upper = multiplier * upper
    lower = multiplier * lower
    return tf.linalg.tridiagonal_solve([upper, diag, lower], vec, diagonals_format='sequence', transpose_rhs=True, partial_pivoting=False)
__all__ = ['weighted_implicit_explicit_scheme', 'weighted_implicit_explicit_step']