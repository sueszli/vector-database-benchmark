import itertools
import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import ode

def _band_count(a):
    if False:
        while True:
            i = 10
    'Returns ml and mu, the lower and upper band sizes of a.'
    (nrows, ncols) = a.shape
    ml = 0
    for k in range(-nrows + 1, 0):
        if np.diag(a, k).any():
            ml = -k
            break
    mu = 0
    for k in range(nrows - 1, 0, -1):
        if np.diag(a, k).any():
            mu = k
            break
    return (ml, mu)

def _linear_func(t, y, a):
    if False:
        for i in range(10):
            print('nop')
    'Linear system dy/dt = a * y'
    return a.dot(y)

def _linear_jac(t, y, a):
    if False:
        for i in range(10):
            print('nop')
    'Jacobian of a * y is a.'
    return a

def _linear_banded_jac(t, y, a):
    if False:
        return 10
    'Banded Jacobian.'
    (ml, mu) = _band_count(a)
    bjac = [np.r_[[0] * k, np.diag(a, k)] for k in range(mu, 0, -1)]
    bjac.append(np.diag(a))
    for k in range(-1, -ml - 1, -1):
        bjac.append(np.r_[np.diag(a, k), [0] * -k])
    return bjac

def _solve_linear_sys(a, y0, tend=1, dt=0.1, solver=None, method='bdf', use_jac=True, with_jacobian=False, banded=False):
    if False:
        i = 10
        return i + 15
    'Use scipy.integrate.ode to solve a linear system of ODEs.\n\n    a : square ndarray\n        Matrix of the linear system to be solved.\n    y0 : ndarray\n        Initial condition\n    tend : float\n        Stop time.\n    dt : float\n        Step size of the output.\n    solver : str\n        If not None, this must be "vode", "lsoda" or "zvode".\n    method : str\n        Either "bdf" or "adams".\n    use_jac : bool\n        Determines if the jacobian function is passed to ode().\n    with_jacobian : bool\n        Passed to ode.set_integrator().\n    banded : bool\n        Determines whether a banded or full jacobian is used.\n        If `banded` is True, `lband` and `uband` are determined by the\n        values in `a`.\n    '
    if banded:
        (lband, uband) = _band_count(a)
    else:
        lband = None
        uband = None
    if use_jac:
        if banded:
            r = ode(_linear_func, _linear_banded_jac)
        else:
            r = ode(_linear_func, _linear_jac)
    else:
        r = ode(_linear_func)
    if solver is None:
        if np.iscomplexobj(a):
            solver = 'zvode'
        else:
            solver = 'vode'
    r.set_integrator(solver, with_jacobian=with_jacobian, method=method, lband=lband, uband=uband, rtol=1e-09, atol=1e-10)
    t0 = 0
    r.set_initial_value(y0, t0)
    r.set_f_params(a)
    r.set_jac_params(a)
    t = [t0]
    y = [y0]
    while r.successful() and r.t < tend:
        r.integrate(r.t + dt)
        t.append(r.t)
        y.append(r.y)
    t = np.array(t)
    y = np.array(y)
    return (t, y)

def _analytical_solution(a, y0, t):
    if False:
        while True:
            i = 10
    '\n    Analytical solution to the linear differential equations dy/dt = a*y.\n\n    The solution is only valid if `a` is diagonalizable.\n\n    Returns a 2-D array with shape (len(t), len(y0)).\n    '
    (lam, v) = np.linalg.eig(a)
    c = np.linalg.solve(v, y0)
    e = c * np.exp(lam * t.reshape(-1, 1))
    sol = e.dot(v.T)
    return sol

def test_banded_ode_solvers():
    if False:
        i = 10
        return i + 15
    t_exact = np.linspace(0, 1.0, 5)
    a_real = np.array([[-0.6, 0.1, 0.0, 0.0, 0.0], [0.2, -0.5, 0.9, 0.0, 0.0], [0.1, 0.1, -0.4, 0.1, 0.0], [0.0, 0.3, -0.1, -0.9, -0.3], [0.0, 0.0, 0.1, 0.1, -0.7]])
    a_real_upper = np.triu(a_real)
    a_real_lower = np.tril(a_real)
    a_real_diag = np.triu(a_real_lower)
    real_matrices = [a_real, a_real_upper, a_real_lower, a_real_diag]
    real_solutions = []
    for a in real_matrices:
        y0 = np.arange(1, a.shape[0] + 1)
        y_exact = _analytical_solution(a, y0, t_exact)
        real_solutions.append((y0, t_exact, y_exact))

    def check_real(idx, solver, meth, use_jac, with_jac, banded):
        if False:
            print('Hello World!')
        a = real_matrices[idx]
        (y0, t_exact, y_exact) = real_solutions[idx]
        (t, y) = _solve_linear_sys(a, y0, tend=t_exact[-1], dt=t_exact[1] - t_exact[0], solver=solver, method=meth, use_jac=use_jac, with_jacobian=with_jac, banded=banded)
        assert_allclose(t, t_exact)
        assert_allclose(y, y_exact)
    for idx in range(len(real_matrices)):
        p = [['vode', 'lsoda'], ['bdf', 'adams'], [False, True], [False, True], [False, True]]
        for (solver, meth, use_jac, with_jac, banded) in itertools.product(*p):
            check_real(idx, solver, meth, use_jac, with_jac, banded)
    a_complex = a_real - 0.5j * a_real
    a_complex_diag = np.diag(np.diag(a_complex))
    complex_matrices = [a_complex, a_complex_diag]
    complex_solutions = []
    for a in complex_matrices:
        y0 = np.arange(1, a.shape[0] + 1) + 1j
        y_exact = _analytical_solution(a, y0, t_exact)
        complex_solutions.append((y0, t_exact, y_exact))

    def check_complex(idx, solver, meth, use_jac, with_jac, banded):
        if False:
            print('Hello World!')
        a = complex_matrices[idx]
        (y0, t_exact, y_exact) = complex_solutions[idx]
        (t, y) = _solve_linear_sys(a, y0, tend=t_exact[-1], dt=t_exact[1] - t_exact[0], solver=solver, method=meth, use_jac=use_jac, with_jacobian=with_jac, banded=banded)
        assert_allclose(t, t_exact)
        assert_allclose(y, y_exact)
    for idx in range(len(complex_matrices)):
        p = [['bdf', 'adams'], [False, True], [False, True], [False, True]]
        for (meth, use_jac, with_jac, banded) in itertools.product(*p):
            check_complex(idx, 'zvode', meth, use_jac, with_jac, banded)