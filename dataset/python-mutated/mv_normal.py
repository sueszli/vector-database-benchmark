"""Multivariate Normal and t distributions



Created on Sat May 28 15:38:23 2011

@author: Josef Perktold

TODO:
* renaming,
    - after adding t distribution, cov does not make sense for Sigma    DONE
    - should mean also be renamed to mu, if there will be distributions
      with mean != mu
* not sure about corner cases
    - behavior with (almost) singular sigma or transforms
    - df <= 2, is everything correct if variance is not finite or defined ?
* check to return possibly univariate distribution for marginals or conditional
    distributions, does univariate special case work? seems ok for conditional
* are all the extra transformation methods useful outside of testing ?
  - looks like I have some mixup in definitions of standardize, normalize
* new methods marginal, conditional, ... just added, typos ?
  - largely tested for MVNormal, not yet for MVT   DONE
* conditional: reusing, vectorizing, should we reuse a projection matrix or
  allow for a vectorized, conditional_mean similar to OLS.predict
* add additional things similar to LikelihoodModelResults? quadratic forms,
  F distribution, and others ???
* add Delta method for nonlinear functions here, current function is hidden
  somewhere in miscmodels
* raise ValueErrors for wrong input shapes, currently only partially checked

* quantile method (ppf for equal bounds for multiple testing) is missing
  http://svitsrv25.epfl.ch/R-doc/library/mvtnorm/html/qmvt.html seems to use
  just a root finder for inversion of cdf

* normalize has ambiguous definition, and mixing it up in different versions
  std from sigma or std from cov ?
  I would like to get what I need for mvt-cdf, or not
  univariate standard t distribution has scale=1 but std>1
  FIXED: add std_sigma, and normalize uses std_sigma

* more work: bivariate distributions,
  inherit from multivariate but overwrite some methods for better efficiency,
  e.g. cdf and expect

I kept the original MVNormal0 class as reference, can be deleted


See Also
--------
sandbox/examples/ex_mvelliptical.py

Examples
--------

Note, several parts of these examples are random and the numbers will not be
(exactly) the same.

>>> import numpy as np
>>> import statsmodels.sandbox.distributions.mv_normal as mvd
>>>
>>> from numpy.testing import assert_array_almost_equal
>>>
>>> cov3 = np.array([[ 1.  ,  0.5 ,  0.75],
...                    [ 0.5 ,  1.5 ,  0.6 ],
...                    [ 0.75,  0.6 ,  2.  ]])

>>> mu = np.array([-1, 0.0, 2.0])

multivariate normal distribution
--------------------------------

>>> mvn3 = mvd.MVNormal(mu, cov3)
>>> mvn3.rvs(size=3)
array([[-0.08559948, -1.0319881 ,  1.76073533],
       [ 0.30079522,  0.55859618,  4.16538667],
       [-1.36540091, -1.50152847,  3.87571161]])

>>> mvn3.std
array([ 1.        ,  1.22474487,  1.41421356])
>>> a = [0.0, 1.0, 1.5]
>>> mvn3.pdf(a)
0.013867410439318712
>>> mvn3.cdf(a)
0.31163181123730122

Monte Carlo integration

>>> mvn3.expect_mc(lambda x: (x<a).all(-1), size=100000)
0.30958999999999998
>>> mvn3.expect_mc(lambda x: (x<a).all(-1), size=1000000)
0.31197399999999997

multivariate t distribution
---------------------------

>>> mvt3 = mvd.MVT(mu, cov3, 4)
>>> mvt3.rvs(size=4)
array([[-0.94185437,  0.3933273 ,  2.40005487],
       [ 0.07563648,  0.06655433,  7.90752238],
       [ 1.06596474,  0.32701158,  2.03482886],
       [ 3.80529746,  7.0192967 ,  8.41899229]])

>>> mvt3.pdf(a)
0.010402959362646937
>>> mvt3.cdf(a)
0.30269483623249821
>>> mvt3.expect_mc(lambda x: (x<a).all(-1), size=1000000)
0.30271199999999998

>>> mvt3.cov
array([[ 2. ,  1. ,  1.5],
       [ 1. ,  3. ,  1.2],
       [ 1.5,  1.2,  4. ]])
>>> mvt3.corr
array([[ 1.        ,  0.40824829,  0.53033009],
       [ 0.40824829,  1.        ,  0.34641016],
       [ 0.53033009,  0.34641016,  1.        ]])

get normalized distribution

>>> mvt3n = mvt3.normalized()
>>> mvt3n.sigma
array([[ 1.        ,  0.40824829,  0.53033009],
       [ 0.40824829,  1.        ,  0.34641016],
       [ 0.53033009,  0.34641016,  1.        ]])
>>> mvt3n.cov
array([[ 2.        ,  0.81649658,  1.06066017],
       [ 0.81649658,  2.        ,  0.69282032],
       [ 1.06066017,  0.69282032,  2.        ]])

What's currently there?

>>> [i for i in dir(mvn3) if not i[0]=='_']
['affine_transformed', 'cdf', 'cholsigmainv', 'conditional', 'corr', 'cov',
'expect_mc', 'extra_args', 'logdetsigma', 'logpdf', 'marginal', 'mean',
'normalize', 'normalized', 'normalized2', 'nvars', 'pdf', 'rvs', 'sigma',
'sigmainv', 'standardize', 'standardized', 'std', 'std_sigma', 'whiten']

>>> [i for i in dir(mvt3) if not i[0]=='_']
['affine_transformed', 'cdf', 'cholsigmainv', 'corr', 'cov', 'df', 'expect_mc',
'extra_args', 'logdetsigma', 'logpdf', 'marginal', 'mean', 'normalize',
'normalized', 'normalized2', 'nvars', 'pdf', 'rvs', 'sigma', 'sigmainv',
'standardize', 'standardized', 'std', 'std_sigma', 'whiten']

"""
import numpy as np
from scipy import special
from statsmodels.sandbox.distributions.multivariate import mvstdtprob
from .extras import mvnormcdf

def expect_mc(dist, func=lambda x: 1, size=50000):
    if False:
        print('Hello World!')
    'calculate expected value of function by Monte Carlo integration\n\n    Parameters\n    ----------\n    dist : distribution instance\n        needs to have rvs defined as a method for drawing random numbers\n    func : callable\n        function for which expectation is calculated, this function needs to\n        be vectorized, integration is over axis=0\n    size : int\n        number of random samples to use in the Monte Carlo integration,\n\n\n    Notes\n    -----\n    this does not batch\n\n    Returns\n    -------\n    expected value : ndarray\n        return of function func integrated over axis=0 by MonteCarlo, this will\n        have the same shape as the return of func without axis=0\n\n    Examples\n    --------\n\n    integrate probability that both observations are negative\n\n    >>> mvn = mve.MVNormal([0,0],2.)\n    >>> mve.expect_mc(mvn, lambda x: (x<np.array([0,0])).all(-1), size=100000)\n    0.25306000000000001\n\n    get tail probabilities of marginal distribution (should be 0.1)\n\n    >>> c = stats.norm.isf(0.05, scale=np.sqrt(2.))\n    >>> expect_mc(mvn, lambda x: (np.abs(x)>np.array([c, c])), size=100000)\n    array([ 0.09969,  0.0986 ])\n\n    or calling the method\n\n    >>> mvn.expect_mc(lambda x: (np.abs(x)>np.array([c, c])), size=100000)\n    array([ 0.09937,  0.10075])\n\n\n    '

    def fun(x):
        if False:
            print('Hello World!')
        return func(x)
    rvs = dist.rvs(size=size)
    return fun(rvs).mean(0)

def expect_mc_bounds(dist, func=lambda x: 1, size=50000, lower=None, upper=None, conditional=False, overfact=1.2):
    if False:
        i = 10
        return i + 15
    'calculate expected value of function by Monte Carlo integration\n\n    Parameters\n    ----------\n    dist : distribution instance\n        needs to have rvs defined as a method for drawing random numbers\n    func : callable\n        function for which expectation is calculated, this function needs to\n        be vectorized, integration is over axis=0\n    size : int\n        minimum number of random samples to use in the Monte Carlo integration,\n        the actual number used can be larger because of oversampling.\n    lower : None or array_like\n        lower integration bounds, if None, then it is set to -inf\n    upper : None or array_like\n        upper integration bounds, if None, then it is set to +inf\n    conditional : bool\n        If true, then the expectation is conditional on being in within\n        [lower, upper] bounds, otherwise it is unconditional\n    overfact : float\n        oversampling factor, the actual number of random variables drawn in\n        each attempt are overfact * remaining draws. Extra draws are also\n        used in the integration.\n\n\n    Notes\n    -----\n    this does not batch\n\n    Returns\n    -------\n    expected value : ndarray\n        return of function func integrated over axis=0 by MonteCarlo, this will\n        have the same shape as the return of func without axis=0\n\n    Examples\n    --------\n    >>> mvn = mve.MVNormal([0,0],2.)\n    >>> mve.expect_mc_bounds(mvn, lambda x: np.ones(x.shape[0]),\n                                lower=[-10,-10],upper=[0,0])\n    0.24990416666666668\n\n    get 3 marginal moments with one integration\n\n    >>> mvn = mve.MVNormal([0,0],1.)\n    >>> mve.expect_mc_bounds(mvn, lambda x: np.dstack([x, x**2, x**3, x**4]),\n        lower=[-np.inf,-np.inf], upper=[np.inf,np.inf])\n    array([[  2.88629497e-03,   9.96706297e-01,  -2.51005344e-03,\n              2.95240921e+00],\n           [ -5.48020088e-03,   9.96004409e-01,  -2.23803072e-02,\n              2.96289203e+00]])\n    >>> from scipy import stats\n    >>> [stats.norm.moment(i) for i in [1,2,3,4]]\n    [0.0, 1.0, 0.0, 3.0]\n\n\n    '
    rvsdim = dist.rvs(size=1).shape[-1]
    if lower is None:
        lower = -np.inf * np.ones(rvsdim)
    else:
        lower = np.asarray(lower)
    if upper is None:
        upper = np.inf * np.ones(rvsdim)
    else:
        upper = np.asarray(upper)

    def fun(x):
        if False:
            return 10
        return func(x)
    rvsli = []
    used = 0
    total = 0
    while True:
        remain = size - used
        rvs = dist.rvs(size=int(remain * overfact))
        total += int(size * overfact)
        rvsok = rvs[((rvs >= lower) & (rvs <= upper)).all(-1)]
        rvsok = np.atleast_2d(rvsok)
        used += rvsok.shape[0]
        rvsli.append(rvsok)
        print(used)
        if used >= size:
            break
    rvs = np.vstack(rvsli)
    print(rvs.shape)
    assert used == rvs.shape[0]
    mean_conditional = fun(rvs).mean(0)
    if conditional:
        return mean_conditional
    else:
        return mean_conditional * (used * 1.0 / total)

def bivariate_normal(x, mu, cov):
    if False:
        print('Hello World!')
    '\n    Bivariate Gaussian distribution for equal shape *X*, *Y*.\n\n    See `bivariate normal\n    <http://mathworld.wolfram.com/BivariateNormalDistribution.html>`_\n    at mathworld.\n    '
    (X, Y) = np.transpose(x)
    (mux, muy) = mu
    (sigmax, sigmaxy, tmp, sigmay) = np.ravel(cov)
    (sigmax, sigmay) = (np.sqrt(sigmax), np.sqrt(sigmay))
    Xmu = X - mux
    Ymu = Y - muy
    rho = sigmaxy / (sigmax * sigmay)
    z = Xmu ** 2 / sigmax ** 2 + Ymu ** 2 / sigmay ** 2 - 2 * rho * Xmu * Ymu / (sigmax * sigmay)
    denom = 2 * np.pi * sigmax * sigmay * np.sqrt(1 - rho ** 2)
    return np.exp(-z / (2 * (1 - rho ** 2))) / denom

class BivariateNormal:

    def __init__(self, mean, cov):
        if False:
            while True:
                i = 10
        self.mean = mu
        self.cov = cov
        (self.sigmax, self.sigmaxy, tmp, self.sigmay) = np.ravel(cov)
        self.nvars = 2

    def rvs(self, size=1):
        if False:
            while True:
                i = 10
        return np.random.multivariate_normal(self.mean, self.cov, size=size)

    def pdf(self, x):
        if False:
            for i in range(10):
                print('nop')
        return bivariate_normal(x, self.mean, self.cov)

    def logpdf(self, x):
        if False:
            for i in range(10):
                print('nop')
        return np.log(self.pdf(x))

    def cdf(self, x):
        if False:
            i = 10
            return i + 15
        return self.expect(upper=x)

    def expect(self, func=lambda x: 1, lower=(-10, -10), upper=(10, 10)):
        if False:
            return 10

        def fun(x, y):
            if False:
                return 10
            x = np.column_stack((x, y))
            return func(x) * self.pdf(x)
        from scipy.integrate import dblquad
        return dblquad(fun, lower[0], upper[0], lambda y: lower[1], lambda y: upper[1])

    def kl(self, other):
        if False:
            return 10
        'Kullback-Leibler divergence between this and another distribution\n\n        int f(x) (log f(x) - log g(x)) dx\n\n        where f is the pdf of self, and g is the pdf of other\n\n        uses double integration with scipy.integrate.dblquad\n\n        limits currently hardcoded\n\n        '
        fun = lambda x: self.logpdf(x) - other.logpdf(x)
        return self.expect(fun)

    def kl_mc(self, other, size=500000):
        if False:
            print('Hello World!')
        fun = lambda x: self.logpdf(x) - other.logpdf(x)
        rvs = self.rvs(size=size)
        return fun(rvs).mean()

class MVElliptical:
    """Base Class for multivariate elliptical distributions, normal and t

    contains common initialization, and some common methods
    subclass needs to implement at least rvs and logpdf methods

    """

    def __init__(self, mean, sigma, *args, **kwds):
        if False:
            for i in range(10):
                print('nop')
        'initialize instance\n\n        Parameters\n        ----------\n        mean : array_like\n            parameter mu (might be renamed), for symmetric distributions this\n            is the mean\n        sigma : array_like, 2d\n            dispersion matrix, covariance matrix in normal distribution, but\n            only proportional to covariance matrix in t distribution\n        args : list\n            distribution specific arguments, e.g. df for t distribution\n        kwds : dict\n            currently not used\n\n        '
        self.extra_args = []
        self.mean = np.asarray(mean)
        self.sigma = sigma = np.asarray(sigma)
        sigma = np.squeeze(sigma)
        self.nvars = nvars = len(mean)
        if sigma.shape == ():
            self.sigma = np.eye(nvars) * sigma
            self.sigmainv = np.eye(nvars) / sigma
            self.cholsigmainv = np.eye(nvars) / np.sqrt(sigma)
        elif sigma.ndim == 1 and len(sigma) == nvars:
            self.sigma = np.diag(sigma)
            self.sigmainv = np.diag(1.0 / sigma)
            self.cholsigmainv = np.diag(1.0 / np.sqrt(sigma))
        elif sigma.shape == (nvars, nvars):
            self.sigmainv = np.linalg.pinv(sigma)
            self.cholsigmainv = np.linalg.cholesky(self.sigmainv).T
        else:
            raise ValueError('sigma has invalid shape')
        self.logdetsigma = np.log(np.linalg.det(self.sigma))

    def rvs(self, size=1):
        if False:
            i = 10
            return i + 15
        'random variable\n\n        Parameters\n        ----------\n        size : int or tuple\n            the number and shape of random variables to draw.\n\n        Returns\n        -------\n        rvs : ndarray\n            the returned random variables with shape given by size and the\n            dimension of the multivariate random vector as additional last\n            dimension\n\n\n        '
        raise NotImplementedError

    def logpdf(self, x):
        if False:
            print('Hello World!')
        'logarithm of probability density function\n\n        Parameters\n        ----------\n        x : array_like\n            can be 1d or 2d, if 2d, then each row is taken as independent\n            multivariate random vector\n\n        Returns\n        -------\n        logpdf : float or array\n            probability density value of each random vector\n\n\n        this should be made to work with 2d x,\n        with multivariate normal vector in each row and iid across rows\n        does not work now because of dot in whiten\n\n        '
        raise NotImplementedError

    def cdf(self, x, **kwds):
        if False:
            i = 10
            return i + 15
        'cumulative distribution function\n\n        Parameters\n        ----------\n        x : array_like\n            can be 1d or 2d, if 2d, then each row is taken as independent\n            multivariate random vector\n        kwds : dict\n            contains options for the numerical calculation of the cdf\n\n        Returns\n        -------\n        cdf : float or array\n            probability density value of each random vector\n\n        '
        raise NotImplementedError

    def affine_transformed(self, shift, scale_matrix):
        if False:
            for i in range(10):
                print('nop')
        'affine transformation define in subclass because of distribution\n        specific restrictions'
        raise NotImplementedError

    def whiten(self, x):
        if False:
            return 10
        '\n        whiten the data by linear transformation\n\n        Parameters\n        ----------\n        x : array_like, 1d or 2d\n            Data to be whitened, if 2d then each row contains an independent\n            sample of the multivariate random vector\n\n        Returns\n        -------\n        np.dot(x, self.cholsigmainv.T)\n\n        Notes\n        -----\n        This only does rescaling, it does not subtract the mean, use standardize\n        for this instead\n\n        See Also\n        --------\n        standardize : subtract mean and rescale to standardized random variable.\n        '
        x = np.asarray(x)
        return np.dot(x, self.cholsigmainv.T)

    def pdf(self, x):
        if False:
            i = 10
            return i + 15
        'probability density function\n\n        Parameters\n        ----------\n        x : array_like\n            can be 1d or 2d, if 2d, then each row is taken as independent\n            multivariate random vector\n\n        Returns\n        -------\n        pdf : float or array\n            probability density value of each random vector\n\n        '
        return np.exp(self.logpdf(x))

    def standardize(self, x):
        if False:
            print('Hello World!')
        'standardize the random variable, i.e. subtract mean and whiten\n\n        Parameters\n        ----------\n        x : array_like, 1d or 2d\n            Data to be whitened, if 2d then each row contains an independent\n            sample of the multivariate random vector\n\n        Returns\n        -------\n        np.dot(x - self.mean, self.cholsigmainv.T)\n\n        Notes\n        -----\n\n\n        See Also\n        --------\n        whiten : rescale random variable, standardize without subtracting mean.\n\n\n        '
        return self.whiten(x - self.mean)

    def standardized(self):
        if False:
            for i in range(10):
                print('nop')
        'return new standardized MVNormal instance\n        '
        return self.affine_transformed(-self.mean, self.cholsigmainv)

    def normalize(self, x):
        if False:
            print('Hello World!')
        'normalize the random variable, i.e. subtract mean and rescale\n\n        The distribution will have zero mean and sigma equal to correlation\n\n        Parameters\n        ----------\n        x : array_like, 1d or 2d\n            Data to be whitened, if 2d then each row contains an independent\n            sample of the multivariate random vector\n\n        Returns\n        -------\n        (x - self.mean)/std_sigma\n\n        Notes\n        -----\n\n\n        See Also\n        --------\n        whiten : rescale random variable, standardize without subtracting mean.\n\n\n        '
        std_ = np.atleast_2d(self.std_sigma)
        return (x - self.mean) / std_

    def normalized(self, demeaned=True):
        if False:
            print('Hello World!')
        'return a normalized distribution where sigma=corr\n\n        if demeaned is True, then mean will be set to zero\n\n        '
        if demeaned:
            mean_new = np.zeros_like(self.mean)
        else:
            mean_new = self.mean / self.std_sigma
        sigma_new = self.corr
        args = [getattr(self, ea) for ea in self.extra_args]
        return self.__class__(mean_new, sigma_new, *args)

    def normalized2(self, demeaned=True):
        if False:
            i = 10
            return i + 15
        'return a normalized distribution where sigma=corr\n\n\n\n        second implementation for testing affine transformation\n        '
        if demeaned:
            shift = -self.mean
        else:
            shift = self.mean * (1.0 / self.std_sigma - 1.0)
        return self.affine_transformed(shift, np.diag(1.0 / self.std_sigma))

    @property
    def std(self):
        if False:
            i = 10
            return i + 15
        'standard deviation, square root of diagonal elements of cov\n        '
        return np.sqrt(np.diag(self.cov))

    @property
    def std_sigma(self):
        if False:
            for i in range(10):
                print('nop')
        'standard deviation, square root of diagonal elements of sigma\n        '
        return np.sqrt(np.diag(self.sigma))

    @property
    def corr(self):
        if False:
            i = 10
            return i + 15
        'correlation matrix'
        return self.cov / np.outer(self.std, self.std)
    expect_mc = expect_mc

    def marginal(self, indices):
        if False:
            for i in range(10):
                print('nop')
        'return marginal distribution for variables given by indices\n\n        this should be correct for normal and t distribution\n\n        Parameters\n        ----------\n        indices : array_like, int\n            list of indices of variables in the marginal distribution\n\n        Returns\n        -------\n        mvdist : instance\n            new instance of the same multivariate distribution class that\n            contains the marginal distribution of the variables given in\n            indices\n\n        '
        indices = np.asarray(indices)
        mean_new = self.mean[indices]
        sigma_new = self.sigma[indices[:, None], indices]
        args = [getattr(self, ea) for ea in self.extra_args]
        return self.__class__(mean_new, sigma_new, *args)

class MVNormal0:
    """Class for Multivariate Normal Distribution

    original full version, kept for testing, new version inherits from
    MVElliptical

    uses Cholesky decomposition of covariance matrix for the transformation
    of the data

    """

    def __init__(self, mean, cov):
        if False:
            while True:
                i = 10
        self.mean = mean
        self.cov = cov = np.asarray(cov)
        cov = np.squeeze(cov)
        self.nvars = nvars = len(mean)
        if cov.shape == ():
            self.cov = np.eye(nvars) * cov
            self.covinv = np.eye(nvars) / cov
            self.cholcovinv = np.eye(nvars) / np.sqrt(cov)
        elif cov.ndim == 1 and len(cov) == nvars:
            self.cov = np.diag(cov)
            self.covinv = np.diag(1.0 / cov)
            self.cholcovinv = np.diag(1.0 / np.sqrt(cov))
        elif cov.shape == (nvars, nvars):
            self.covinv = np.linalg.pinv(cov)
            self.cholcovinv = np.linalg.cholesky(self.covinv).T
        else:
            raise ValueError('cov has invalid shape')
        self.logdetcov = np.log(np.linalg.det(self.cov))

    def whiten(self, x):
        if False:
            for i in range(10):
                print('nop')
        '\n        whiten the data by linear transformation\n\n        Parameters\n        ----------\n        X : array_like, 1d or 2d\n            Data to be whitened, if 2d then each row contains an independent\n            sample of the multivariate random vector\n\n        Returns\n        -------\n        np.dot(x, self.cholcovinv.T)\n\n        Notes\n        -----\n        This only does rescaling, it does not subtract the mean, use standardize\n        for this instead\n\n        See Also\n        --------\n        standardize : subtract mean and rescale to standardized random variable.\n        '
        x = np.asarray(x)
        if np.any(self.cov):
            return np.dot(x, self.cholcovinv.T)
        else:
            return x

    def rvs(self, size=1):
        if False:
            i = 10
            return i + 15
        'random variable\n\n        Parameters\n        ----------\n        size : int or tuple\n            the number and shape of random variables to draw.\n\n        Returns\n        -------\n        rvs : ndarray\n            the returned random variables with shape given by size and the\n            dimension of the multivariate random vector as additional last\n            dimension\n\n        Notes\n        -----\n        uses numpy.random.multivariate_normal directly\n\n        '
        return np.random.multivariate_normal(self.mean, self.cov, size=size)

    def pdf(self, x):
        if False:
            return 10
        'probability density function\n\n        Parameters\n        ----------\n        x : array_like\n            can be 1d or 2d, if 2d, then each row is taken as independent\n            multivariate random vector\n\n        Returns\n        -------\n        pdf : float or array\n            probability density value of each random vector\n\n        '
        return np.exp(self.logpdf(x))

    def logpdf(self, x):
        if False:
            for i in range(10):
                print('nop')
        'logarithm of probability density function\n\n        Parameters\n        ----------\n        x : array_like\n            can be 1d or 2d, if 2d, then each row is taken as independent\n            multivariate random vector\n\n        Returns\n        -------\n        logpdf : float or array\n            probability density value of each random vector\n\n\n        this should be made to work with 2d x,\n        with multivariate normal vector in each row and iid across rows\n        does not work now because of dot in whiten\n\n        '
        x = np.asarray(x)
        x_whitened = self.whiten(x - self.mean)
        SSR = np.sum(x_whitened ** 2, -1)
        llf = -SSR
        llf -= self.nvars * np.log(2.0 * np.pi)
        llf -= self.logdetcov
        llf *= 0.5
        return llf
    expect_mc = expect_mc

class MVNormal(MVElliptical):
    """Class for Multivariate Normal Distribution

    uses Cholesky decomposition of covariance matrix for the transformation
    of the data

    """
    __name__ == 'Multivariate Normal Distribution'

    def rvs(self, size=1):
        if False:
            print('Hello World!')
        'random variable\n\n        Parameters\n        ----------\n        size : int or tuple\n            the number and shape of random variables to draw.\n\n        Returns\n        -------\n        rvs : ndarray\n            the returned random variables with shape given by size and the\n            dimension of the multivariate random vector as additional last\n            dimension\n\n        Notes\n        -----\n        uses numpy.random.multivariate_normal directly\n\n        '
        return np.random.multivariate_normal(self.mean, self.sigma, size=size)

    def logpdf(self, x):
        if False:
            i = 10
            return i + 15
        'logarithm of probability density function\n\n        Parameters\n        ----------\n        x : array_like\n            can be 1d or 2d, if 2d, then each row is taken as independent\n            multivariate random vector\n\n        Returns\n        -------\n        logpdf : float or array\n            probability density value of each random vector\n\n\n        this should be made to work with 2d x,\n        with multivariate normal vector in each row and iid across rows\n        does not work now because of dot in whiten\n\n        '
        x = np.asarray(x)
        x_whitened = self.whiten(x - self.mean)
        SSR = np.sum(x_whitened ** 2, -1)
        llf = -SSR
        llf -= self.nvars * np.log(2.0 * np.pi)
        llf -= self.logdetsigma
        llf *= 0.5
        return llf

    def cdf(self, x, **kwds):
        if False:
            return 10
        'cumulative distribution function\n\n        Parameters\n        ----------\n        x : array_like\n            can be 1d or 2d, if 2d, then each row is taken as independent\n            multivariate random vector\n        kwds : dict\n            contains options for the numerical calculation of the cdf\n\n        Returns\n        -------\n        cdf : float or array\n            probability density value of each random vector\n\n        '
        return mvnormcdf(x, self.mean, self.cov, **kwds)

    @property
    def cov(self):
        if False:
            return 10
        'covariance matrix'
        return self.sigma

    def affine_transformed(self, shift, scale_matrix):
        if False:
            return 10
        "return distribution of an affine transform\n\n        for full rank scale_matrix only\n\n        Parameters\n        ----------\n        shift : array_like\n            shift of mean\n        scale_matrix : array_like\n            linear transformation matrix\n\n        Returns\n        -------\n        mvt : instance of MVNormal\n            instance of multivariate normal distribution given by affine\n            transformation\n\n        Notes\n        -----\n        the affine transformation is defined by\n        y = a + B x\n\n        where a is shift,\n        B is a scale matrix for the linear transformation\n\n        Notes\n        -----\n        This should also work to select marginal distributions, but not\n        tested for this case yet.\n\n        currently only tested because it's called by standardized\n\n        "
        B = scale_matrix
        mean_new = np.dot(B, self.mean) + shift
        sigma_new = np.dot(np.dot(B, self.sigma), B.T)
        return MVNormal(mean_new, sigma_new)

    def conditional(self, indices, values):
        if False:
            for i in range(10):
                print('nop')
        'return conditional distribution\n\n        indices are the variables to keep, the complement is the conditioning\n        set\n        values are the values of the conditioning variables\n\n        \\bar{\\mu} = \\mu_1 + \\Sigma_{12} \\Sigma_{22}^{-1} \\left( a - \\mu_2 \\right)\n\n        and covariance matrix\n\n        \\overline{\\Sigma} = \\Sigma_{11} - \\Sigma_{12} \\Sigma_{22}^{-1} \\Sigma_{21}.T\n\n        Parameters\n        ----------\n        indices : array_like, int\n            list of indices of variables in the marginal distribution\n        given : array_like\n            values of the conditioning variables\n\n        Returns\n        -------\n        mvn : instance of MVNormal\n            new instance of the MVNormal class that contains the conditional\n            distribution of the variables given in indices for given\n             values of the excluded variables.\n\n\n        '
        keep = np.asarray(indices)
        given = np.asarray([i for i in range(self.nvars) if i not in keep])
        sigmakk = self.sigma[keep[:, None], keep]
        sigmagg = self.sigma[given[:, None], given]
        sigmakg = self.sigma[keep[:, None], given]
        sigmagk = self.sigma[given[:, None], keep]
        sigma_new = sigmakk - np.dot(sigmakg, np.linalg.solve(sigmagg, sigmagk))
        mean_new = self.mean[keep] + np.dot(sigmakg, np.linalg.solve(sigmagg, values - self.mean[given]))
        return MVNormal(mean_new, sigma_new)
np_log = np.log
np_pi = np.pi
sps_gamln = special.gammaln

class MVT(MVElliptical):
    __name__ == 'Multivariate Student T Distribution'

    def __init__(self, mean, sigma, df):
        if False:
            print('Hello World!')
        'initialize instance\n\n        Parameters\n        ----------\n        mean : array_like\n            parameter mu (might be renamed), for symmetric distributions this\n            is the mean\n        sigma : array_like, 2d\n            dispersion matrix, covariance matrix in normal distribution, but\n            only proportional to covariance matrix in t distribution\n        args : list\n            distribution specific arguments, e.g. df for t distribution\n        kwds : dict\n            currently not used\n\n        '
        super(MVT, self).__init__(mean, sigma)
        self.extra_args = ['df']
        self.df = df

    def rvs(self, size=1):
        if False:
            i = 10
            return i + 15
        'random variables with Student T distribution\n\n        Parameters\n        ----------\n        size : int or tuple\n            the number and shape of random variables to draw.\n\n        Returns\n        -------\n        rvs : ndarray\n            the returned random variables with shape given by size and the\n            dimension of the multivariate random vector as additional last\n            dimension\n            - TODO: Not sure if this works for size tuples with len>1.\n\n        Notes\n        -----\n        generated as a chi-square mixture of multivariate normal random\n        variables.\n        does this require df>2 ?\n\n\n        '
        from .multivariate import multivariate_t_rvs
        return multivariate_t_rvs(self.mean, self.sigma, df=self.df, n=size)

    def logpdf(self, x):
        if False:
            for i in range(10):
                print('nop')
        'logarithm of probability density function\n\n        Parameters\n        ----------\n        x : array_like\n            can be 1d or 2d, if 2d, then each row is taken as independent\n            multivariate random vector\n\n        Returns\n        -------\n        logpdf : float or array\n            probability density value of each random vector\n\n        '
        x = np.asarray(x)
        df = self.df
        nvars = self.nvars
        x_whitened = self.whiten(x - self.mean)
        llf = -nvars * np_log(df * np_pi)
        llf -= self.logdetsigma
        llf -= (df + nvars) * np_log(1 + np.sum(x_whitened ** 2, -1) / df)
        llf *= 0.5
        llf += sps_gamln((df + nvars) / 2.0) - sps_gamln(df / 2.0)
        return llf

    def cdf(self, x, **kwds):
        if False:
            return 10
        'cumulative distribution function\n\n        Parameters\n        ----------\n        x : array_like\n            can be 1d or 2d, if 2d, then each row is taken as independent\n            multivariate random vector\n        kwds : dict\n            contains options for the numerical calculation of the cdf\n\n        Returns\n        -------\n        cdf : float or array\n            probability density value of each random vector\n\n        '
        lower = -np.inf * np.ones_like(x)
        upper = (x - self.mean) / self.std_sigma
        return mvstdtprob(lower, upper, self.corr, self.df, **kwds)

    @property
    def cov(self):
        if False:
            i = 10
            return i + 15
        'covariance matrix\n\n        The covariance matrix for the t distribution does not exist for df<=2,\n        and is equal to sigma * df/(df-2) for df>2\n\n        '
        if self.df <= 2:
            return np.nan * np.ones_like(self.sigma)
        else:
            return self.df / (self.df - 2.0) * self.sigma

    def affine_transformed(self, shift, scale_matrix):
        if False:
            for i in range(10):
                print('nop')
        "return distribution of a full rank affine transform\n\n        for full rank scale_matrix only\n\n        Parameters\n        ----------\n        shift : array_like\n            shift of mean\n        scale_matrix : array_like\n            linear transformation matrix\n\n        Returns\n        -------\n        mvt : instance of MVT\n            instance of multivariate t distribution given by affine\n            transformation\n\n\n        Notes\n        -----\n\n        This checks for eigvals<=0, so there are possible problems for cases\n        with positive eigenvalues close to zero.\n\n        see: http://www.statlect.com/mcdstu1.htm\n\n        I'm not sure about general case, non-full rank transformation are not\n        multivariate t distributed.\n\n        y = a + B x\n\n        where a is shift,\n        B is full rank scale matrix with same dimension as sigma\n\n        "
        B = scale_matrix
        if not B.shape == (self.nvars, self.nvars):
            if (np.linalg.eigvals(B) <= 0).any():
                raise ValueError('affine transform has to be full rank')
        mean_new = np.dot(B, self.mean) + shift
        sigma_new = np.dot(np.dot(B, self.sigma), B.T)
        return MVT(mean_new, sigma_new, self.df)

def quad2d(func=lambda x: 1, lower=(-10, -10), upper=(10, 10)):
    if False:
        while True:
            i = 10

    def fun(x, y):
        if False:
            return 10
        x = np.column_stack((x, y))
        return func(x)
    from scipy.integrate import dblquad
    return dblquad(fun, lower[0], upper[0], lambda y: lower[1], lambda y: upper[1])
if __name__ == '__main__':
    from numpy.testing import assert_almost_equal, assert_array_almost_equal
    examples = ['mvn']
    mu = (0, 0)
    covx = np.array([[1.0, 0.5], [0.5, 1.0]])
    mu3 = [-1, 0.0, 2.0]
    cov3 = np.array([[1.0, 0.5, 0.75], [0.5, 1.5, 0.6], [0.75, 0.6, 2.0]])
    if 'mvn' in examples:
        bvn = BivariateNormal(mu, covx)
        rvs = bvn.rvs(size=1000)
        print(rvs.mean(0))
        print(np.cov(rvs, rowvar=0))
        print(bvn.expect())
        print(bvn.cdf([0, 0]))
        bvn1 = BivariateNormal(mu, np.eye(2))
        bvn2 = BivariateNormal(mu, 4 * np.eye(2))
        fun = lambda x: np.log(bvn1.pdf(x)) - np.log(bvn.pdf(x))
        print(bvn1.expect(fun))
        print(bvn1.kl(bvn2), bvn1.kl_mc(bvn2))
        print(bvn2.kl(bvn1), bvn2.kl_mc(bvn1))
        print(bvn1.kl(bvn), bvn1.kl_mc(bvn))
        mvn = MVNormal(mu, covx)
        mvn.pdf([0, 0])
        mvn.pdf(np.zeros((2, 2)))
        cov3 = np.array([[1.0, 0.5, 0.75], [0.5, 1.5, 0.6], [0.75, 0.6, 2.0]])
        mu3 = [-1, 0.0, 2.0]
        mvn3 = MVNormal(mu3, cov3)
        mvn3.pdf((0.0, 2.0, 3.0))
        mvn3.logpdf((0.0, 2.0, 3.0))
        r_val = [-7.667977543898155, -6.917977543898155, -5.167977543898155]
        assert_array_almost_equal(mvn3.logpdf(cov3), r_val, decimal=14)
        r_val = [0.000467562492721686, 0.000989829804859273, 0.005696077243833402]
        assert_array_almost_equal(mvn3.pdf(cov3), r_val, decimal=17)
        mvn3c = MVNormal(np.array([0, 0, 0]), cov3)
        r_val = [0.02914269740502042, 0.02269635555984291, 0.01767593948287269]
        assert_array_almost_equal(mvn3c.pdf(cov3), r_val, decimal=16)
        mvn3b = MVNormal((0, 0, 0), 1)
        fun = lambda x: np.log(mvn3.pdf(x)) - np.log(mvn3b.pdf(x))
        print(mvn3.expect_mc(fun))
        print(mvn3.expect_mc(fun, size=200000))
    mvt = MVT((0, 0), 1, 5)
    assert_almost_equal(mvt.logpdf(np.array([0.0, 0.0])), -1.837877066409345, decimal=15)
    assert_almost_equal(mvt.pdf(np.array([0.0, 0.0])), 0.1591549430918953, decimal=15)
    mvt.logpdf(np.array([1.0, 1.0])) - -3.01552989458359
    mvt1 = MVT((0, 0), 1, 1)
    mvt1.logpdf(np.array([1.0, 1.0])) - -3.48579549941151
    rvs = mvt.rvs(100000)
    assert_almost_equal(np.cov(rvs, rowvar=0), mvt.cov, decimal=1)
    mvt31 = MVT(mu3, cov3, 1)
    assert_almost_equal(mvt31.pdf(cov3), [0.0007276818698165781, 0.0009980625182293658, 0.0027661422056214652], decimal=18)
    mvt = MVT(mu3, cov3, 3)
    assert_almost_equal(mvt.pdf(cov3), [0.00086377742424741, 0.001277510788307594, 0.004156314279452241], decimal=17)