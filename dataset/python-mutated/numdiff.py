"""numerical differentiation function, gradient, Jacobian, and Hessian

Author : josef-pkt
License : BSD

Notes
-----
These are simple forward differentiation, so that we have them available
without dependencies.

* Jacobian should be faster than numdifftools because it does not use loop over
  observations.
* numerical precision will vary and depend on the choice of stepsizes
"""
import numpy as np
from statsmodels.compat.pandas import Appender, Substitution
EPS = np.finfo(float).eps
_hessian_docs = '\n    Calculate Hessian with finite difference derivative approximation\n\n    Parameters\n    ----------\n    x : array_like\n       value at which function derivative is evaluated\n    f : function\n       function of one array f(x, `*args`, `**kwargs`)\n    epsilon : float or array_like, optional\n       Stepsize used, if None, then stepsize is automatically chosen\n       according to EPS**(1/%(scale)s)*x.\n    args : tuple\n        Arguments for function `f`.\n    kwargs : dict\n        Keyword arguments for function `f`.\n    %(extra_params)s\n\n    Returns\n    -------\n    hess : ndarray\n       array of partial second derivatives, Hessian\n    %(extra_returns)s\n\n    Notes\n    -----\n    Equation (%(equation_number)s) in Ridout. Computes the Hessian as::\n\n      %(equation)s\n\n    where e[j] is a vector with element j == 1 and the rest are zero and\n    d[i] is epsilon[i].\n\n    References\n    ----------:\n\n    Ridout, M.S. (2009) Statistical applications of the complex-step method\n        of numerical differentiation. The American Statistician, 63, 66-74\n'

def _get_epsilon(x, s, epsilon, n):
    if False:
        for i in range(10):
            print('nop')
    if epsilon is None:
        h = EPS ** (1.0 / s) * np.maximum(np.abs(x), 0.1)
    elif np.isscalar(epsilon):
        h = np.empty(n)
        h.fill(epsilon)
    else:
        h = np.asarray(epsilon)
        if h.shape != x.shape:
            raise ValueError('If h is not a scalar it must have the same shape as x.')
    return np.asarray(h)

def approx_fprime(x, f, epsilon=None, args=(), kwargs={}, centered=False):
    if False:
        return 10
    '\n    Gradient of function, or Jacobian if function f returns 1d array\n\n    Parameters\n    ----------\n    x : ndarray\n        parameters at which the derivative is evaluated\n    f : function\n        `f(*((x,)+args), **kwargs)` returning either one value or 1d array\n    epsilon : float, optional\n        Stepsize, if None, optimal stepsize is used. This is EPS**(1/2)*x for\n        `centered` == False and EPS**(1/3)*x for `centered` == True.\n    args : tuple\n        Tuple of additional arguments for function `f`.\n    kwargs : dict\n        Dictionary of additional keyword arguments for function `f`.\n    centered : bool\n        Whether central difference should be returned. If not, does forward\n        differencing.\n\n    Returns\n    -------\n    grad : ndarray\n        gradient or Jacobian\n\n    Notes\n    -----\n    If f returns a 1d array, it returns a Jacobian. If a 2d array is returned\n    by f (e.g., with a value for each observation), it returns a 3d array\n    with the Jacobian of each observation with shape xk x nobs x xk. I.e.,\n    the Jacobian of the first observation would be [:, 0, :]\n    '
    n = len(x)
    f0 = f(*(x,) + args, **kwargs)
    dim = np.atleast_1d(f0).shape
    grad = np.zeros((n,) + dim, np.promote_types(float, x.dtype))
    ei = np.zeros((n,), float)
    if not centered:
        epsilon = _get_epsilon(x, 2, epsilon, n)
        for k in range(n):
            ei[k] = epsilon[k]
            grad[k, :] = (f(*(x + ei,) + args, **kwargs) - f0) / epsilon[k]
            ei[k] = 0.0
    else:
        epsilon = _get_epsilon(x, 3, epsilon, n) / 2.0
        for k in range(n):
            ei[k] = epsilon[k]
            grad[k, :] = (f(*(x + ei,) + args, **kwargs) - f(*(x - ei,) + args, **kwargs)) / (2 * epsilon[k])
            ei[k] = 0.0
    if n == 1:
        return grad.T
    else:
        return grad.squeeze().T

def _approx_fprime_scalar(x, f, epsilon=None, args=(), kwargs={}, centered=False):
    if False:
        return 10
    '\n    Gradient of function vectorized for scalar parameter.\n\n    This assumes that the function ``f`` is vectorized for a scalar parameter.\n    The function value ``f(x)`` has then the same shape as the input ``x``.\n    The derivative returned by this function also has the same shape as ``x``.\n\n    Parameters\n    ----------\n    x : ndarray\n        Parameters at which the derivative is evaluated.\n    f : function\n        `f(*((x,)+args), **kwargs)` returning either one value or 1d array\n    epsilon : float, optional\n        Stepsize, if None, optimal stepsize is used. This is EPS**(1/2)*x for\n        `centered` == False and EPS**(1/3)*x for `centered` == True.\n    args : tuple\n        Tuple of additional arguments for function `f`.\n    kwargs : dict\n        Dictionary of additional keyword arguments for function `f`.\n    centered : bool\n        Whether central difference should be returned. If not, does forward\n        differencing.\n\n    Returns\n    -------\n    grad : ndarray\n        Array of derivatives, gradient evaluated at parameters ``x``.\n    '
    x = np.asarray(x)
    n = 1
    f0 = f(*(x,) + args, **kwargs)
    if not centered:
        eps = _get_epsilon(x, 2, epsilon, n)
        grad = (f(*(x + eps,) + args, **kwargs) - f0) / eps
    else:
        eps = _get_epsilon(x, 3, epsilon, n) / 2.0
        grad = (f(*(x + eps,) + args, **kwargs) - f(*(x - eps,) + args, **kwargs)) / (2 * eps)
    return grad

def approx_fprime_cs(x, f, epsilon=None, args=(), kwargs={}):
    if False:
        print('Hello World!')
    '\n    Calculate gradient or Jacobian with complex step derivative approximation\n\n    Parameters\n    ----------\n    x : ndarray\n        parameters at which the derivative is evaluated\n    f : function\n        `f(*((x,)+args), **kwargs)` returning either one value or 1d array\n    epsilon : float, optional\n        Stepsize, if None, optimal stepsize is used. Optimal step-size is\n        EPS*x. See note.\n    args : tuple\n        Tuple of additional arguments for function `f`.\n    kwargs : dict\n        Dictionary of additional keyword arguments for function `f`.\n\n    Returns\n    -------\n    partials : ndarray\n       array of partial derivatives, Gradient or Jacobian\n\n    Notes\n    -----\n    The complex-step derivative has truncation error O(epsilon**2), so\n    truncation error can be eliminated by choosing epsilon to be very small.\n    The complex-step derivative avoids the problem of round-off error with\n    small epsilon because there is no subtraction.\n    '
    n = len(x)
    epsilon = _get_epsilon(x, 1, epsilon, n)
    increments = np.identity(n) * 1j * epsilon
    partials = [f(x + ih, *args, **kwargs).imag / epsilon[i] for (i, ih) in enumerate(increments)]
    return np.array(partials).T

def _approx_fprime_cs_scalar(x, f, epsilon=None, args=(), kwargs={}):
    if False:
        return 10
    '\n    Calculate gradient for scalar parameter with complex step derivatives.\n\n    This assumes that the function ``f`` is vectorized for a scalar parameter.\n    The function value ``f(x)`` has then the same shape as the input ``x``.\n    The derivative returned by this function also has the same shape as ``x``.\n\n    Parameters\n    ----------\n    x : ndarray\n        Parameters at which the derivative is evaluated.\n    f : function\n        `f(*((x,)+args), **kwargs)` returning either one value or 1d array.\n    epsilon : float, optional\n        Stepsize, if None, optimal stepsize is used. Optimal step-size is\n        EPS*x. See note.\n    args : tuple\n        Tuple of additional arguments for function `f`.\n    kwargs : dict\n        Dictionary of additional keyword arguments for function `f`.\n\n    Returns\n    -------\n    partials : ndarray\n       Array of derivatives, gradient evaluated for parameters ``x``.\n\n    Notes\n    -----\n    The complex-step derivative has truncation error O(epsilon**2), so\n    truncation error can be eliminated by choosing epsilon to be very small.\n    The complex-step derivative avoids the problem of round-off error with\n    small epsilon because there is no subtraction.\n    '
    x = np.asarray(x)
    n = x.shape[-1]
    epsilon = _get_epsilon(x, 1, epsilon, n)
    eps = 1j * epsilon
    partials = f(x + eps, *args, **kwargs).imag / epsilon
    return np.array(partials)

def approx_hess_cs(x, f, epsilon=None, args=(), kwargs={}):
    if False:
        for i in range(10):
            print('nop')
    'Calculate Hessian with complex-step derivative approximation\n\n    Parameters\n    ----------\n    x : array_like\n       value at which function derivative is evaluated\n    f : function\n       function of one array f(x)\n    epsilon : float\n       stepsize, if None, then stepsize is automatically chosen\n\n    Returns\n    -------\n    hess : ndarray\n       array of partial second derivatives, Hessian\n\n    Notes\n    -----\n    based on equation 10 in\n    M. S. RIDOUT: Statistical Applications of the Complex-step Method\n    of Numerical Differentiation, University of Kent, Canterbury, Kent, U.K.\n\n    The stepsize is the same for the complex and the finite difference part.\n    '
    n = len(x)
    h = _get_epsilon(x, 3, epsilon, n)
    ee = np.diag(h)
    hess = np.outer(h, h)
    n = len(x)
    for i in range(n):
        for j in range(i, n):
            hess[i, j] = np.squeeze((f(*(x + 1j * ee[i, :] + ee[j, :],) + args, **kwargs) - f(*(x + 1j * ee[i, :] - ee[j, :],) + args, **kwargs)).imag / 2.0 / hess[i, j])
            hess[j, i] = hess[i, j]
    return hess

@Substitution(scale='3', extra_params='return_grad : bool\n        Whether or not to also return the gradient\n', extra_returns='grad : nparray\n        Gradient if return_grad == True\n', equation_number='7', equation='1/(d_j*d_k) * ((f(x + d[j]*e[j] + d[k]*e[k]) - f(x + d[j]*e[j])))\n')
@Appender(_hessian_docs)
def approx_hess1(x, f, epsilon=None, args=(), kwargs={}, return_grad=False):
    if False:
        for i in range(10):
            print('nop')
    n = len(x)
    h = _get_epsilon(x, 3, epsilon, n)
    ee = np.diag(h)
    f0 = f(*(x,) + args, **kwargs)
    g = np.zeros(n)
    for i in range(n):
        g[i] = f(*(x + ee[i, :],) + args, **kwargs)
    hess = np.outer(h, h)
    for i in range(n):
        for j in range(i, n):
            hess[i, j] = (f(*(x + ee[i, :] + ee[j, :],) + args, **kwargs) - g[i] - g[j] + f0) / hess[i, j]
            hess[j, i] = hess[i, j]
    if return_grad:
        grad = (g - f0) / h
        return (hess, grad)
    else:
        return hess

@Substitution(scale='3', extra_params='return_grad : bool\n        Whether or not to also return the gradient\n', extra_returns='grad : ndarray\n        Gradient if return_grad == True\n', equation_number='8', equation='1/(2*d_j*d_k) * ((f(x + d[j]*e[j] + d[k]*e[k]) - f(x + d[j]*e[j])) -\n                 (f(x + d[k]*e[k]) - f(x)) +\n                 (f(x - d[j]*e[j] - d[k]*e[k]) - f(x + d[j]*e[j])) -\n                 (f(x - d[k]*e[k]) - f(x)))\n')
@Appender(_hessian_docs)
def approx_hess2(x, f, epsilon=None, args=(), kwargs={}, return_grad=False):
    if False:
        return 10
    n = len(x)
    h = _get_epsilon(x, 3, epsilon, n)
    ee = np.diag(h)
    f0 = f(*(x,) + args, **kwargs)
    g = np.zeros(n)
    gg = np.zeros(n)
    for i in range(n):
        g[i] = f(*(x + ee[i, :],) + args, **kwargs)
        gg[i] = f(*(x - ee[i, :],) + args, **kwargs)
    hess = np.outer(h, h)
    for i in range(n):
        for j in range(i, n):
            hess[i, j] = (f(*(x + ee[i, :] + ee[j, :],) + args, **kwargs) - g[i] - g[j] + f0 + f(*(x - ee[i, :] - ee[j, :],) + args, **kwargs) - gg[i] - gg[j] + f0) / (2 * hess[i, j])
            hess[j, i] = hess[i, j]
    if return_grad:
        grad = (g - f0) / h
        return (hess, grad)
    else:
        return hess

@Substitution(scale='4', extra_params='', extra_returns='', equation_number='9', equation='1/(4*d_j*d_k) * ((f(x + d[j]*e[j] + d[k]*e[k]) - f(x + d[j]*e[j]\n                                                     - d[k]*e[k])) -\n                 (f(x - d[j]*e[j] + d[k]*e[k]) - f(x - d[j]*e[j]\n                                                     - d[k]*e[k]))')
@Appender(_hessian_docs)
def approx_hess3(x, f, epsilon=None, args=(), kwargs={}):
    if False:
        while True:
            i = 10
    n = len(x)
    h = _get_epsilon(x, 4, epsilon, n)
    ee = np.diag(h)
    hess = np.outer(h, h)
    for i in range(n):
        for j in range(i, n):
            hess[i, j] = np.squeeze((f(*(x + ee[i, :] + ee[j, :],) + args, **kwargs) - f(*(x + ee[i, :] - ee[j, :],) + args, **kwargs) - (f(*(x - ee[i, :] + ee[j, :],) + args, **kwargs) - f(*(x - ee[i, :] - ee[j, :],) + args, **kwargs))) / (4.0 * hess[i, j]))
            hess[j, i] = hess[i, j]
    return hess
approx_hess = approx_hess3
approx_hess.__doc__ += '\n    This is an alias for approx_hess3'