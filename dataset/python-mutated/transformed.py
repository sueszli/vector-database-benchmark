""" A class for the distribution of a non-linear monotonic transformation of a continuous random variable

simplest usage:
example: create log-gamma distribution, i.e. y = log(x),
            where x is gamma distributed (also available in scipy.stats)
    loggammaexpg = Transf_gen(stats.gamma, np.log, np.exp)

example: what is the distribution of the discount factor y=1/(1+x)
            where interest rate x is normally distributed with N(mux,stdx**2)')?
            (just to come up with a story that implies a nice transformation)
    invnormalg = Transf_gen(stats.norm, inversew, inversew_inv, decr=True, a=-np.inf)

This class does not work well for distributions with difficult shapes,
    e.g. 1/x where x is standard normal, because of the singularity and jump at zero.

Note: I'm working from my version of scipy.stats.distribution.
      But this script runs under scipy 0.6.0 (checked with numpy: 1.2.0rc2 and python 2.4)

This is not yet thoroughly tested, polished or optimized

TODO:
  * numargs handling is not yet working properly, numargs needs to be specified (default = 0 or 1)
  * feeding args and kwargs to underlying distribution is untested and incomplete
  * distinguish args and kwargs for the transformed and the underlying distribution
    - currently all args and no kwargs are transmitted to underlying distribution
    - loc and scale only work for transformed, but not for underlying distribution
    - possible to separate args for transformation and underlying distribution parameters

  * add _rvs as method, will be faster in many cases


Created on Tuesday, October 28, 2008, 12:40:37 PM
Author: josef-pktd
License: BSD

"""
from scipy import stats
from scipy.stats import distributions
import numpy as np

def get_u_argskwargs(**kwargs):
    if False:
        i = 10
        return i + 15
    u_kwargs = dict(((k.replace('u_', '', 1), v) for (k, v) in kwargs.items() if k.startswith('u_')))
    u_args = u_kwargs.pop('u_args', None)
    return (u_args, u_kwargs)

class Transf_gen(distributions.rv_continuous):
    """a class for non-linear monotonic transformation of a continuous random variable

    """

    def __init__(self, kls, func, funcinv, *args, **kwargs):
        if False:
            print('Hello World!')
        self.func = func
        self.funcinv = funcinv
        self.numargs = kwargs.pop('numargs', 0)
        name = kwargs.pop('name', 'transfdist')
        longname = kwargs.pop('longname', 'Non-linear transformed distribution')
        extradoc = kwargs.pop('extradoc', None)
        a = kwargs.pop('a', -np.inf)
        b = kwargs.pop('b', np.inf)
        self.decr = kwargs.pop('decr', False)
        (self.u_args, self.u_kwargs) = get_u_argskwargs(**kwargs)
        self.kls = kls
        super(Transf_gen, self).__init__(a=a, b=b, name=name, shapes=kls.shapes, longname=longname)

    def _cdf(self, x, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if not self.decr:
            return self.kls._cdf(self.funcinv(x), *args, **kwargs)
        else:
            return 1.0 - self.kls._cdf(self.funcinv(x), *args, **kwargs)

    def _ppf(self, q, *args, **kwargs):
        if False:
            print('Hello World!')
        if not self.decr:
            return self.func(self.kls._ppf(q, *args, **kwargs))
        else:
            return self.func(self.kls._ppf(1 - q, *args, **kwargs))

def inverse(x):
    if False:
        for i in range(10):
            print('nop')
    return np.divide(1.0, x)
(mux, stdx) = (0.05, 0.1)
(mux, stdx) = (9.0, 1.0)

def inversew(x):
    if False:
        return 10
    return 1.0 / (1 + mux + x * stdx)

def inversew_inv(x):
    if False:
        while True:
            i = 10
    return (1.0 / x - 1.0 - mux) / stdx

def identit(x):
    if False:
        for i in range(10):
            print('nop')
    return x
invdnormalg = Transf_gen(stats.norm, inversew, inversew_inv, decr=True, numargs=0, name='discf', longname='normal-based discount factor')
lognormalg = Transf_gen(stats.norm, np.exp, np.log, numargs=2, a=0, name='lnnorm', longname='Exp transformed normal')
loggammaexpg = Transf_gen(stats.gamma, np.log, np.exp, numargs=1)
'univariate distribution of a non-linear monotonic transformation of a\nrandom variable\n\n'

class ExpTransf_gen(distributions.rv_continuous):
    """Distribution based on log/exp transformation

    the constructor can be called with a distribution class
    and generates the distribution of the transformed random variable

    """

    def __init__(self, kls, *args, **kwargs):
        if False:
            return 10
        if 'numargs' in kwargs:
            self.numargs = kwargs['numargs']
        else:
            self.numargs = 1
        if 'name' in kwargs:
            name = kwargs['name']
        else:
            name = 'Log transformed distribution'
        if 'a' in kwargs:
            a = kwargs['a']
        else:
            a = 0
        super(ExpTransf_gen, self).__init__(a=a, name=name)
        self.kls = kls

    def _cdf(self, x, *args):
        if False:
            for i in range(10):
                print('nop')
        return self.kls._cdf(np.log(x), *args)

    def _ppf(self, q, *args):
        if False:
            print('Hello World!')
        return np.exp(self.kls._ppf(q, *args))

class LogTransf_gen(distributions.rv_continuous):
    """Distribution based on log/exp transformation

    the constructor can be called with a distribution class
    and generates the distribution of the transformed random variable

    """

    def __init__(self, kls, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if 'numargs' in kwargs:
            self.numargs = kwargs['numargs']
        else:
            self.numargs = 1
        if 'name' in kwargs:
            name = kwargs['name']
        else:
            name = 'Log transformed distribution'
        if 'a' in kwargs:
            a = kwargs['a']
        else:
            a = 0
        super(LogTransf_gen, self).__init__(a=a, name=name)
        self.kls = kls

    def _cdf(self, x, *args):
        if False:
            print('Hello World!')
        return self.kls._cdf(np.exp(x), *args)

    def _ppf(self, q, *args):
        if False:
            i = 10
            return i + 15
        return np.log(self.kls._ppf(q, *args))

def examples_transf():
    if False:
        while True:
            i = 10
    print('Results for lognormal')
    lognormalg = ExpTransf_gen(stats.norm, a=0, name='Log transformed normal general')
    print(lognormalg.cdf(1))
    print(stats.lognorm.cdf(1, 1))
    print(lognormalg.stats())
    print(stats.lognorm.stats(1))
    print(lognormalg.rvs(size=5))
    print('Results for expgamma')
    loggammaexpg = LogTransf_gen(stats.gamma)
    print(loggammaexpg._cdf(1, 10))
    print(stats.loggamma.cdf(1, 10))
    print(loggammaexpg._cdf(2, 15))
    print(stats.loggamma.cdf(2, 15))
    print('Results for loglaplace')
    loglaplaceg = LogTransf_gen(stats.laplace)
    print(loglaplaceg._cdf(2, 10))
    print(stats.loglaplace.cdf(2, 10))
    loglaplaceexpg = ExpTransf_gen(stats.laplace)
    print(loglaplaceexpg._cdf(2, 10))
'\nCreated on Apr 28, 2009\n\n@author: Josef Perktold\n'
' A class for the distribution of a non-linear u-shaped or hump shaped transformation of a\ncontinuous random variable\n\nThis is a companion to the distributions of non-linear monotonic transformation to the case\nwhen the inverse mapping is a 2-valued correspondence, for example for absolute value or square\n\nsimplest usage:\nexample: create squared distribution, i.e. y = x**2,\n            where x is normal or t distributed\n\n\nThis class does not work well for distributions with difficult shapes,\n    e.g. 1/x where x is standard normal, because of the singularity and jump at zero.\n\n\nThis verifies for normal - chi2, normal - halfnorm, foldnorm, and t - F\n\nTODO:\n  * numargs handling is not yet working properly,\n    numargs needs to be specified (default = 0 or 1)\n  * feeding args and kwargs to underlying distribution works in t distribution example\n  * distinguish args and kwargs for the transformed and the underlying distribution\n    - currently all args and no kwargs are transmitted to underlying distribution\n    - loc and scale only work for transformed, but not for underlying distribution\n    - possible to separate args for transformation and underlying distribution parameters\n\n  * add _rvs as method, will be faster in many cases\n\n'

class TransfTwo_gen(distributions.rv_continuous):
    """Distribution based on a non-monotonic (u- or hump-shaped transformation)

    the constructor can be called with a distribution class, and functions
    that define the non-linear transformation.
    and generates the distribution of the transformed random variable

    Note: the transformation, it's inverse and derivatives need to be fully
    specified: func, funcinvplus, funcinvminus, derivplus,  derivminus.
    Currently no numerical derivatives or inverse are calculated

    This can be used to generate distribution instances similar to the
    distributions in scipy.stats.

    """

    def __init__(self, kls, func, funcinvplus, funcinvminus, derivplus, derivminus, *args, **kwargs):
        if False:
            return 10
        self.func = func
        self.funcinvplus = funcinvplus
        self.funcinvminus = funcinvminus
        self.derivplus = derivplus
        self.derivminus = derivminus
        self.numargs = kwargs.pop('numargs', 0)
        name = kwargs.pop('name', 'transfdist')
        longname = kwargs.pop('longname', 'Non-linear transformed distribution')
        extradoc = kwargs.pop('extradoc', None)
        a = kwargs.pop('a', -np.inf)
        b = kwargs.pop('b', np.inf)
        self.shape = kwargs.pop('shape', False)
        (self.u_args, self.u_kwargs) = get_u_argskwargs(**kwargs)
        self.kls = kls
        super(TransfTwo_gen, self).__init__(a=a, b=b, name=name, shapes=kls.shapes, longname=longname)

    def _rvs(self, *args):
        if False:
            return 10
        self.kls._size = self._size
        return self.func(self.kls._rvs(*args))

    def _pdf(self, x, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if self.shape == 'u':
            signpdf = 1
        elif self.shape == 'hump':
            signpdf = -1
        else:
            raise ValueError('shape can only be `u` or `hump`')
        return signpdf * (self.derivplus(x) * self.kls._pdf(self.funcinvplus(x), *args, **kwargs) - self.derivminus(x) * self.kls._pdf(self.funcinvminus(x), *args, **kwargs))

    def _cdf(self, x, *args, **kwargs):
        if False:
            while True:
                i = 10
        if self.shape == 'u':
            return self.kls._cdf(self.funcinvplus(x), *args, **kwargs) - self.kls._cdf(self.funcinvminus(x), *args, **kwargs)
        else:
            return 1.0 - self._sf(x, *args, **kwargs)

    def _sf(self, x, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if self.shape == 'hump':
            return self.kls._cdf(self.funcinvplus(x), *args, **kwargs) - self.kls._cdf(self.funcinvminus(x), *args, **kwargs)
        else:
            return 1.0 - self._cdf(x, *args, **kwargs)

    def _munp(self, n, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self._mom0_sc(n, *args)

class SquareFunc:
    """class to hold quadratic function with inverse function and derivative

    using instance methods instead of class methods, if we want extension
    to parametrized function
    """

    def inverseplus(self, x):
        if False:
            for i in range(10):
                print('nop')
        return np.sqrt(x)

    def inverseminus(self, x):
        if False:
            return 10
        return 0.0 - np.sqrt(x)

    def derivplus(self, x):
        if False:
            while True:
                i = 10
        return 0.5 / np.sqrt(x)

    def derivminus(self, x):
        if False:
            while True:
                i = 10
        return 0.0 - 0.5 / np.sqrt(x)

    def squarefunc(self, x):
        if False:
            return 10
        return np.power(x, 2)
sqfunc = SquareFunc()
squarenormalg = TransfTwo_gen(stats.norm, sqfunc.squarefunc, sqfunc.inverseplus, sqfunc.inverseminus, sqfunc.derivplus, sqfunc.derivminus, shape='u', a=0.0, b=np.inf, numargs=0, name='squarenorm', longname='squared normal distribution')
squaretg = TransfTwo_gen(stats.t, sqfunc.squarefunc, sqfunc.inverseplus, sqfunc.inverseminus, sqfunc.derivplus, sqfunc.derivminus, shape='u', a=0.0, b=np.inf, numargs=1, name='squarenorm', longname='squared t distribution')

def inverseplus(x):
    if False:
        i = 10
        return i + 15
    return np.sqrt(-x)

def inverseminus(x):
    if False:
        while True:
            i = 10
    return 0.0 - np.sqrt(-x)

def derivplus(x):
    if False:
        for i in range(10):
            print('nop')
    return 0.0 - 0.5 / np.sqrt(-x)

def derivminus(x):
    if False:
        return 10
    return 0.5 / np.sqrt(-x)

def negsquarefunc(x):
    if False:
        while True:
            i = 10
    return -np.power(x, 2)
negsquarenormalg = TransfTwo_gen(stats.norm, negsquarefunc, inverseplus, inverseminus, derivplus, derivminus, shape='hump', a=-np.inf, b=0.0, numargs=0, name='negsquarenorm', longname='negative squared normal distribution')

def inverseplus(x):
    if False:
        while True:
            i = 10
    return x

def inverseminus(x):
    if False:
        print('Hello World!')
    return 0.0 - x

def derivplus(x):
    if False:
        i = 10
        return i + 15
    return 1.0

def derivminus(x):
    if False:
        return 10
    return 0.0 - 1.0

def absfunc(x):
    if False:
        while True:
            i = 10
    return np.abs(x)
absnormalg = TransfTwo_gen(stats.norm, np.abs, inverseplus, inverseminus, derivplus, derivminus, shape='u', a=0.0, b=np.inf, numargs=0, name='absnorm', longname='absolute of normal distribution')