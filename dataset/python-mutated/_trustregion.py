"""Trust-region optimization."""
import math
import warnings
import numpy as np
import scipy.linalg
from ._optimize import _check_unknown_options, _status_message, OptimizeResult, _prepare_scalar_function, _call_callback_maybe_halt
from scipy.optimize._hessian_update_strategy import HessianUpdateStrategy
from scipy.optimize._differentiable_functions import FD_METHODS
__all__ = []

def _wrap_function(function, args):
    if False:
        for i in range(10):
            print('nop')
    ncalls = [0]
    if function is None:
        return (ncalls, None)

    def function_wrapper(x, *wrapper_args):
        if False:
            for i in range(10):
                print('nop')
        ncalls[0] += 1
        return function(np.copy(x), *wrapper_args + args)
    return (ncalls, function_wrapper)

class BaseQuadraticSubproblem:
    """
    Base/abstract class defining the quadratic model for trust-region
    minimization. Child classes must implement the ``solve`` method.

    Values of the objective function, Jacobian and Hessian (if provided) at
    the current iterate ``x`` are evaluated on demand and then stored as
    attributes ``fun``, ``jac``, ``hess``.
    """

    def __init__(self, x, fun, jac, hess=None, hessp=None):
        if False:
            for i in range(10):
                print('nop')
        self._x = x
        self._f = None
        self._g = None
        self._h = None
        self._g_mag = None
        self._cauchy_point = None
        self._newton_point = None
        self._fun = fun
        self._jac = jac
        self._hess = hess
        self._hessp = hessp

    def __call__(self, p):
        if False:
            while True:
                i = 10
        return self.fun + np.dot(self.jac, p) + 0.5 * np.dot(p, self.hessp(p))

    @property
    def fun(self):
        if False:
            print('Hello World!')
        'Value of objective function at current iteration.'
        if self._f is None:
            self._f = self._fun(self._x)
        return self._f

    @property
    def jac(self):
        if False:
            for i in range(10):
                print('nop')
        'Value of Jacobian of objective function at current iteration.'
        if self._g is None:
            self._g = self._jac(self._x)
        return self._g

    @property
    def hess(self):
        if False:
            return 10
        'Value of Hessian of objective function at current iteration.'
        if self._h is None:
            self._h = self._hess(self._x)
        return self._h

    def hessp(self, p):
        if False:
            while True:
                i = 10
        if self._hessp is not None:
            return self._hessp(self._x, p)
        else:
            return np.dot(self.hess, p)

    @property
    def jac_mag(self):
        if False:
            for i in range(10):
                print('nop')
        'Magnitude of jacobian of objective function at current iteration.'
        if self._g_mag is None:
            self._g_mag = scipy.linalg.norm(self.jac)
        return self._g_mag

    def get_boundaries_intersections(self, z, d, trust_radius):
        if False:
            i = 10
            return i + 15
        '\n        Solve the scalar quadratic equation ||z + t d|| == trust_radius.\n        This is like a line-sphere intersection.\n        Return the two values of t, sorted from low to high.\n        '
        a = np.dot(d, d)
        b = 2 * np.dot(z, d)
        c = np.dot(z, z) - trust_radius ** 2
        sqrt_discriminant = math.sqrt(b * b - 4 * a * c)
        aux = b + math.copysign(sqrt_discriminant, b)
        ta = -aux / (2 * a)
        tb = -2 * c / aux
        return sorted([ta, tb])

    def solve(self, trust_radius):
        if False:
            print('Hello World!')
        raise NotImplementedError('The solve method should be implemented by the child class')

def _minimize_trust_region(fun, x0, args=(), jac=None, hess=None, hessp=None, subproblem=None, initial_trust_radius=1.0, max_trust_radius=1000.0, eta=0.15, gtol=0.0001, maxiter=None, disp=False, return_all=False, callback=None, inexact=True, **unknown_options):
    if False:
        for i in range(10):
            print('nop')
    '\n    Minimization of scalar function of one or more variables using a\n    trust-region algorithm.\n\n    Options for the trust-region algorithm are:\n        initial_trust_radius : float\n            Initial trust radius.\n        max_trust_radius : float\n            Never propose steps that are longer than this value.\n        eta : float\n            Trust region related acceptance stringency for proposed steps.\n        gtol : float\n            Gradient norm must be less than `gtol`\n            before successful termination.\n        maxiter : int\n            Maximum number of iterations to perform.\n        disp : bool\n            If True, print convergence message.\n        inexact : bool\n            Accuracy to solve subproblems. If True requires less nonlinear\n            iterations, but more vector products. Only effective for method\n            trust-krylov.\n\n    This function is called by the `minimize` function.\n    It is not supposed to be called directly.\n    '
    _check_unknown_options(unknown_options)
    if jac is None:
        raise ValueError('Jacobian is currently required for trust-region methods')
    if hess is None and hessp is None:
        raise ValueError('Either the Hessian or the Hessian-vector product is currently required for trust-region methods')
    if subproblem is None:
        raise ValueError('A subproblem solving strategy is required for trust-region methods')
    if not 0 <= eta < 0.25:
        raise Exception('invalid acceptance stringency')
    if max_trust_radius <= 0:
        raise Exception('the max trust radius must be positive')
    if initial_trust_radius <= 0:
        raise ValueError('the initial trust radius must be positive')
    if initial_trust_radius >= max_trust_radius:
        raise ValueError('the initial trust radius must be less than the max trust radius')
    x0 = np.asarray(x0).flatten()
    sf = _prepare_scalar_function(fun, x0, jac=jac, hess=hess, args=args)
    fun = sf.fun
    jac = sf.grad
    if callable(hess):
        hess = sf.hess
    elif callable(hessp):
        pass
    elif hess in FD_METHODS or isinstance(hess, HessianUpdateStrategy):
        hess = None

        def hessp(x, p, *args):
            if False:
                return 10
            return sf.hess(x).dot(p)
    else:
        raise ValueError('Either the Hessian or the Hessian-vector product is currently required for trust-region methods')
    (nhessp, hessp) = _wrap_function(hessp, args)
    if maxiter is None:
        maxiter = len(x0) * 200
    warnflag = 0
    trust_radius = initial_trust_radius
    x = x0
    if return_all:
        allvecs = [x]
    m = subproblem(x, fun, jac, hess, hessp)
    k = 0
    while m.jac_mag >= gtol:
        try:
            (p, hits_boundary) = m.solve(trust_radius)
        except np.linalg.LinAlgError:
            warnflag = 3
            break
        predicted_value = m(p)
        x_proposed = x + p
        m_proposed = subproblem(x_proposed, fun, jac, hess, hessp)
        actual_reduction = m.fun - m_proposed.fun
        predicted_reduction = m.fun - predicted_value
        if predicted_reduction <= 0:
            warnflag = 2
            break
        rho = actual_reduction / predicted_reduction
        if rho < 0.25:
            trust_radius *= 0.25
        elif rho > 0.75 and hits_boundary:
            trust_radius = min(2 * trust_radius, max_trust_radius)
        if rho > eta:
            x = x_proposed
            m = m_proposed
        if return_all:
            allvecs.append(np.copy(x))
        k += 1
        intermediate_result = OptimizeResult(x=x, fun=m.fun)
        if _call_callback_maybe_halt(callback, intermediate_result):
            break
        if m.jac_mag < gtol:
            warnflag = 0
            break
        if k >= maxiter:
            warnflag = 1
            break
    status_messages = (_status_message['success'], _status_message['maxiter'], 'A bad approximation caused failure to predict improvement.', 'A linalg error occurred, such as a non-psd Hessian.')
    if disp:
        if warnflag == 0:
            print(status_messages[warnflag])
        else:
            warnings.warn(status_messages[warnflag], RuntimeWarning, 3)
        print('         Current function value: %f' % m.fun)
        print('         Iterations: %d' % k)
        print('         Function evaluations: %d' % sf.nfev)
        print('         Gradient evaluations: %d' % sf.ngev)
        print('         Hessian evaluations: %d' % (sf.nhev + nhessp[0]))
    result = OptimizeResult(x=x, success=warnflag == 0, status=warnflag, fun=m.fun, jac=m.jac, nfev=sf.nfev, njev=sf.ngev, nhev=sf.nhev + nhessp[0], nit=k, message=status_messages[warnflag])
    if hess is not None:
        result['hess'] = m.hess
    if return_all:
        result['allvecs'] = allvecs
    return result