"""Extrapolation time marching scheme for parabolic PDEs."""
from tf_quant_finance.math.pde.steppers.implicit import implicit_scheme
from tf_quant_finance.math.pde.steppers.parabolic_equation_stepper import parabolic_equation_step

def extrapolation_step():
    if False:
        return 10
    'Creates a stepper function with Extrapolation time marching scheme.\n\n  Extrapolation scheme combines two half-steps and the full time step to obtain\n  desirable properties. See more details below in `extrapolation_scheme`.\n\n  It is slower than Crank-Nicolson scheme, but deals better with value grids\n  that have discontinuities. Consider also `oscillation_damped_crank_nicolson`,\n  an efficient combination of Crank-Nicolson and Extrapolation schemes.\n\n  Returns:\n    Callable to be used in finite-difference PDE solvers (see fd_solvers.py).\n  '

    def step_fn(time, next_time, coord_grid, value_grid, boundary_conditions, second_order_coeff_fn, first_order_coeff_fn, zeroth_order_coeff_fn, inner_second_order_coeff_fn, inner_first_order_coeff_fn, num_steps_performed, dtype=None, name=None):
        if False:
            print('Hello World!')
        'Performs the step.'
        del num_steps_performed
        name = name or 'extrapolation_step'
        return parabolic_equation_step(time, next_time, coord_grid, value_grid, boundary_conditions, second_order_coeff_fn, first_order_coeff_fn, zeroth_order_coeff_fn, inner_second_order_coeff_fn, inner_first_order_coeff_fn, time_marching_scheme=extrapolation_scheme, dtype=dtype, name=name)
    return step_fn

def extrapolation_scheme(value_grid, t1, t2, equation_params_fn):
    if False:
        for i in range(10):
            print('nop')
    'Constructs extrapolation implicit-explicit scheme.\n\n  Performs two implicit half-steps, one full implicit step, and combines them\n  with such coefficients that ensure second-order errors. More computationally\n  expensive than Crank-Nicolson scheme, but provides a better approximation for\n  high-wavenumber components, which results in absence of oscillations typical\n  for Crank-Nicolson scheme in case of non-smooth initial conditions. See [1]\n  for details.\n\n  #### References:\n  [1]: D. Lawson, J & Ll Morris, J. The Extrapolation of First Order Methods\n  for Parabolic Partial Differential Equations. I. 1978\n  SIAM Journal on Numerical Analysis. 15. 1212-1224.\n  https://epubs.siam.org/doi/abs/10.1137/0715082\n\n  Args:\n    value_grid: A `Tensor` of real dtype. Grid of solution values at the current\n      time.\n    t1: Time before the step.\n    t2: Time after the step.\n    equation_params_fn: A callable that takes a scalar `Tensor` argument\n      representing time and constructs the tridiagonal matrix `A`\n      (a tuple of three `Tensor`s, main, upper, and lower diagonals)\n      and the inhomogeneous term `b`. All of the `Tensor`s are of the same\n      `dtype` as `inner_value_grid` and of the shape broadcastable with the\n      shape of `inner_value_grid`.\n\n  Returns:\n    A `Tensor` of the same shape and `dtype` a\n    `values_grid` and represents an approximate solution `u(t2)`.\n  '
    first_half_step = implicit_scheme(value_grid, t1, (t1 + t2) / 2, equation_params_fn)
    two_half_steps = implicit_scheme(first_half_step, (t1 + t2) / 2, t2, equation_params_fn)
    full_step = implicit_scheme(value_grid, t1, t2, equation_params_fn)
    return 2 * two_half_steps - full_step
__all__ = ['extrapolation_scheme', 'extrapolation_step']