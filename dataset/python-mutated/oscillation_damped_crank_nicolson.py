"""Crank-Nicolson with oscillation damping time marching scheme."""
from tf_quant_finance.math.pde.steppers.composite_stepper import composite_scheme_step
from tf_quant_finance.math.pde.steppers.crank_nicolson import crank_nicolson_scheme
from tf_quant_finance.math.pde.steppers.extrapolation import extrapolation_scheme

def oscillation_damped_crank_nicolson_step(extrapolation_steps=1):
    if False:
        for i in range(10):
            print('nop')
    'Scheme similar to Crank-Nicolson, but ensuring damping of oscillations.\n\n  Performs first (or first few) steps with Extrapolation scheme, then proceeds\n  with Crank-Nicolson scheme. This combines absence of oscillations by virtue\n  of Extrapolation scheme with lower computational cost of Crank-Nicolson\n  scheme.\n\n  See [1], [2] ([2] mostly discusses using fully implicit scheme on the first\n  step, but mentions using extrapolation scheme for better accuracy in the end).\n\n  #### References:\n  [1]: D. Lawson, J & Ll Morris, J. The Extrapolation of First Order Methods for\n    Parabolic Partial Differential Equations. I. 1978 SIAM Journal on Numerical\n    Analysis. 15. 1212-1224.\n    https://epubs.siam.org/doi/abs/10.1137/0715082\n  [2]: B. Giles, Michael & Carter, Rebecca. Convergence analysis of\n    Crank-Nicolson and Rannacher time-marching. J. Comput. Finance. 9. 2005.\n    https://core.ac.uk/download/pdf/1633712.pdf\n\n  Args:\n    extrapolation_steps: number of first steps to which to apply the\n      Extrapolation scheme. Defaults to `1`.\n\n  Returns:\n    Callable to use as `one_step_fn` in fd_solvers.\n  '
    return composite_scheme_step(extrapolation_steps, extrapolation_scheme, crank_nicolson_scheme)
__all__ = ['oscillation_damped_crank_nicolson_step']