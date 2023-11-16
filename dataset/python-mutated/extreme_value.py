""" Extreme Value Copulas
Created on Fri Jan 29 19:19:45 2021

Author: Josef Perktold
License: BSD-3

"""
import numpy as np
from .copulas import Copula

def copula_bv_ev(u, transform, args=()):
    if False:
        while True:
            i = 10
    'generic bivariate extreme value copula\n    '
    (u, v) = u
    return np.exp(np.log(u * v) * transform(np.log(u) / np.log(u * v), *args))

class ExtremeValueCopula(Copula):
    """Extreme value copula constructed from Pickand's dependence function.

    Currently only bivariate copulas are available.

    Parameters
    ----------
    transform: instance of transformation class
        Pickand's dependence function with required methods including first
        and second derivatives
    args : tuple
        Optional copula parameters. Copula parameters can be either provided
        when creating the instance or as arguments when calling methods.
    k_dim : int
        Currently only bivariate extreme value copulas are supported.

    Notes
    -----
    currently the following dependence function and copulas are available

    - AsymLogistic
    - AsymNegLogistic
    - AsymMixed
    - HR

    TEV and AsymBiLogistic currently do not have required derivatives for pdf.

    See Also
    --------
    dep_func_ev

    """

    def __init__(self, transform, args=(), k_dim=2):
        if False:
            i = 10
            return i + 15
        super().__init__(k_dim=k_dim)
        self.transform = transform
        self.k_args = transform.k_args
        self.args = args
        if k_dim != 2:
            raise ValueError('Only bivariate EV copulas are available.')

    def _handle_args(self, args):
        if False:
            while True:
                i = 10
        if isinstance(args, np.ndarray):
            args = tuple(args)
        if args == () or args is None:
            args = self.args
        if not isinstance(args, tuple):
            args = (args,)
        return args

    def cdf(self, u, args=()):
        if False:
            for i in range(10):
                print('nop')
        'Evaluate cdf of bivariate extreme value copula.\n\n        Parameters\n        ----------\n        u : array_like\n            Values of random bivariate random variable, each defined on [0, 1],\n            for which cdf is computed.\n            Can be two dimensional with multivariate components in columns and\n            observation in rows.\n        args : tuple\n            Required parameters for the copula. The meaning and number of\n            parameters in the tuple depends on the specific copula.\n\n        Returns\n        -------\n        CDF values at evaluation points.\n        '
        (u, v) = np.asarray(u).T
        args = self._handle_args(args)
        cdfv = np.exp(np.log(u * v) * self.transform(np.log(u) / np.log(u * v), *args))
        return cdfv

    def pdf(self, u, args=()):
        if False:
            return 10
        'Evaluate pdf of bivariate extreme value copula.\n\n        Parameters\n        ----------\n        u : array_like\n            Values of random bivariate random variable, each defined on [0, 1],\n            for which cdf is computed.\n            Can be two dimensional with multivariate components in columns and\n            observation in rows.\n        args : tuple\n            Required parameters for the copula. The meaning and number of\n            parameters in the tuple depends on the specific copula.\n\n        Returns\n        -------\n        PDF values at evaluation points.\n        '
        tr = self.transform
        (u1, u2) = np.asarray(u).T
        args = self._handle_args(args)
        log_u12 = np.log(u1 * u2)
        t = np.log(u1) / log_u12
        cdf = self.cdf(u, args)
        dep = tr(t, *args)
        d1 = tr.deriv(t, *args)
        d2 = tr.deriv2(t, *args)
        pdf_ = cdf / (u1 * u2) * ((dep + (1 - t) * d1) * (dep - t * d1) - d2 * (1 - t) * t / log_u12)
        return pdf_

    def logpdf(self, u, args=()):
        if False:
            i = 10
            return i + 15
        'Evaluate log-pdf of bivariate extreme value copula.\n\n        Parameters\n        ----------\n        u : array_like\n            Values of random bivariate random variable, each defined on [0, 1],\n            for which cdf is computed.\n            Can be two dimensional with multivariate components in columns and\n            observation in rows.\n        args : tuple\n            Required parameters for the copula. The meaning and number of\n            parameters in the tuple depends on the specific copula.\n\n        Returns\n        -------\n        Log-pdf values at evaluation points.\n        '
        return np.log(self.pdf(u, args=args))

    def conditional_2g1(self, u, args=()):
        if False:
            i = 10
            return i + 15
        "conditional distribution\n\n        not yet implemented\n\n        C2|1(u2|u1) := ∂C(u1, u2) / ∂u1 = C(u1, u2) / u1 * (A(t) − t A'(t))\n\n        where t = np.log(v)/np.log(u*v)\n        "
        raise NotImplementedError

    def fit_corr_param(self, data):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError