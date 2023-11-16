"""Composition of two time marching schemes."""
import tensorflow.compat.v2 as tf
from tf_quant_finance.math.pde.steppers.parabolic_equation_stepper import parabolic_equation_step

def composite_scheme_step(first_scheme_steps, first_scheme, second_scheme):
    if False:
        while True:
            i = 10
    'Composes two time marching schemes.\n\n  Applies a step of parabolic PDE solver using `first_scheme` if number of\n  performed steps is less than `first_scheme_steps`, and using `second_scheme`\n  otherwise.\n\n  Args:\n    first_scheme_steps: A Python integer. Number of steps to apply\n      `first_scheme` on.\n    first_scheme: First time marching scheme (see `time_marching_scheme`\n      argument of `parabolic_equation_step`).\n    second_scheme: Second time marching scheme (see `time_marching_scheme`\n      argument of `parabolic_equation_step`).\n\n  Returns:\n     Callable to be used in finite-difference PDE solvers (see fd_solvers.py).\n  '

    def step_fn(time, next_time, coord_grid, value_grid, boundary_conditions, second_order_coeff_fn, first_order_coeff_fn, zeroth_order_coeff_fn, inner_second_order_coeff_fn, inner_first_order_coeff_fn, num_steps_performed, dtype=None, name=None):
        if False:
            i = 10
            return i + 15
        'Performs the step.'
        name = name or 'composite_scheme_step'

        def scheme(*args, **kwargs):
            if False:
                return 10
            return tf.cond(num_steps_performed < first_scheme_steps, lambda : first_scheme(*args, **kwargs), lambda : second_scheme(*args, **kwargs))
        return parabolic_equation_step(time, next_time, coord_grid, value_grid, boundary_conditions, second_order_coeff_fn, first_order_coeff_fn, zeroth_order_coeff_fn, inner_second_order_coeff_fn, inner_first_order_coeff_fn, time_marching_scheme=scheme, dtype=dtype, name=name)
    return step_fn
__all__ = ['composite_scheme_step']