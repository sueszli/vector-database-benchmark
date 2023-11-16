"""Crank-Nicolson time marching scheme for parabolic PDEs."""
from tf_quant_finance.math.pde.steppers.parabolic_equation_stepper import parabolic_equation_step
from tf_quant_finance.math.pde.steppers.weighted_implicit_explicit import weighted_implicit_explicit_scheme

def crank_nicolson_step():
    if False:
        return 10
    'Creates a stepper function with Crank-Nicolson time marching scheme.\n\n  Crank-Nicolson time marching scheme is one of the the most widely used schemes\n  for 1D PDEs. Given a space-discretized equation\n\n  ```\n  du/dt = A(t) u(t) + b(t)\n  ```\n  (here `u` is a value vector, `A` and `b` are the matrix and the vector defined\n  by the PDE), it approximates the right-hand side as an average of values taken\n  before and after the time step:\n\n  ```\n  (u(t2) - u(t1)) / (t2 - t1) = (A(t1) u(t1) + b(t1) + A(t2) u(t2) + b(t2)) / 2.\n  ```\n\n  Crank-Nicolson has second order accuracy and is stable.\n\n  More details can be found in `weighted_implicit_explicit.py` describing the\n  weighted implicit-explicit scheme - Crank-Nicolson scheme is a special case\n  with `theta = 0.5`.\n\n  Returns:\n    Callable to be used in finite-difference PDE solvers (see fd_solvers.py).\n  '

    def step_fn(time, next_time, coord_grid, value_grid, boundary_conditions, second_order_coeff_fn, first_order_coeff_fn, zeroth_order_coeff_fn, inner_second_order_coeff_fn, inner_first_order_coeff_fn, num_steps_performed, dtype=None, name=None):
        if False:
            for i in range(10):
                print('nop')
        'Performs the step.'
        del num_steps_performed
        name = name or 'crank_nicolson_step'
        return parabolic_equation_step(time, next_time, coord_grid, value_grid, boundary_conditions, second_order_coeff_fn, first_order_coeff_fn, zeroth_order_coeff_fn, inner_second_order_coeff_fn, inner_first_order_coeff_fn, time_marching_scheme=crank_nicolson_scheme, dtype=dtype, name=name)
    return step_fn
crank_nicolson_scheme = weighted_implicit_explicit_scheme(theta=0.5)
__all__ = ['crank_nicolson_step', 'crank_nicolson_scheme']