import math
import numpy as np
from scipy import linalg, stats, special
from .linalg_decomp_1 import SvdArray
sqrt2pi = math.sqrt(2 * np.pi)
logsqrt2pi = math.log(sqrt2pi)

class StandardNormal:
    """Distribution of vector x, with independent distribution N(0,1)

    this is the same as univariate normal for pdf and logpdf

    other methods not checked/adjusted yet

    """

    def rvs(self, size):
        if False:
            return 10
        return np.random.standard_normal(size)

    def pdf(self, x):
        if False:
            return 10
        return np.exp(-x ** 2 * 0.5) / sqrt2pi

    def logpdf(self, x):
        if False:
            return 10
        return -x ** 2 * 0.5 - logsqrt2pi

    def _cdf(self, x):
        if False:
            print('Hello World!')
        return special.ndtr(x)

    def _logcdf(self, x):
        if False:
            print('Hello World!')
        return np.log(special.ndtr(x))

    def _ppf(self, q):
        if False:
            while True:
                i = 10
        return special.ndtri(q)

class AffineTransform:
    """affine full rank transformation of a multivariate distribution

    no dimension checking, assumes everything broadcasts correctly
    first version without bound support

    provides distribution of y given distribution of x
    y = const + tmat * x

    """

    def __init__(self, const, tmat, dist):
        if False:
            i = 10
            return i + 15
        self.const = const
        self.tmat = tmat
        self.dist = dist
        self.nrv = len(const)
        if not np.equal(self.nrv, tmat.shape).all():
            raise ValueError('dimension of const and tmat do not agree')
        self.tmatinv = linalg.inv(tmat)
        self.absdet = np.abs(np.linalg.det(self.tmat))
        self.logabsdet = np.log(np.abs(np.linalg.det(self.tmat)))
        self.dist

    def rvs(self, size):
        if False:
            print('Hello World!')
        print((size,) + (self.nrv,))
        return self.transform(self.dist.rvs(size=(size,) + (self.nrv,)))

    def transform(self, x):
        if False:
            while True:
                i = 10
        return np.dot(x, self.tmat) + self.const

    def invtransform(self, y):
        if False:
            return 10
        return np.dot(self.tmatinv, y - self.const)

    def pdf(self, x):
        if False:
            for i in range(10):
                print('nop')
        return 1.0 / self.absdet * self.dist.pdf(self.invtransform(x))

    def logpdf(self, x):
        if False:
            i = 10
            return i + 15
        return -self.logabsdet + self.dist.logpdf(self.invtransform(x))

class MultivariateNormalChol:
    """multivariate normal distribution with cholesky decomposition of sigma

    ignoring mean at the beginning, maybe

    needs testing for broadcasting to contemporaneously but not intertemporaly
    correlated random variable, which axis?,
    maybe swapaxis or rollaxis if x.ndim != mean.ndim == (sigma.ndim - 1)

    initially 1d is ok, 2d should work with iid in axis 0 and mvn in axis 1

    """

    def __init__(self, mean, sigma):
        if False:
            return 10
        self.mean = mean
        self.sigma = sigma
        self.sigmainv = sigmainv
        self.cholsigma = linalg.cholesky(sigma)
        self.cholsigmainv = linalg.cholesky(sigmainv)[::-1, ::-1]

    def whiten(self, x):
        if False:
            return 10
        return np.dot(cholsigmainv, x)

    def logpdf_obs(self, x):
        if False:
            i = 10
            return i + 15
        x = x - self.mean
        x_whitened = self.whiten(x)
        logdetsigma = np.log(np.linalg.det(sigma))
        sigma2 = 1.0
        llike = 0.5 * (np.log(sigma2) - 2.0 * np.log(np.diagonal(self.cholsigmainv)) + x_whitened ** 2 / sigma2 + np.log(2 * np.pi))
        return llike

    def logpdf(self, x):
        if False:
            while True:
                i = 10
        return self.logpdf_obs(x).sum(-1)

    def pdf(self, x):
        if False:
            print('Hello World!')
        return np.exp(self.logpdf(x))

class MultivariateNormal:

    def __init__(self, mean, sigma):
        if False:
            for i in range(10):
                print('nop')
        self.mean = mean
        self.sigma = SvdArray(sigma)

def loglike_ar1(x, rho):
    if False:
        return 10
    'loglikelihood of AR(1) process, as a test case\n\n    sigma_u partially hard coded\n\n    Greene chapter 12 eq. (12-31)\n    '
    x = np.asarray(x)
    u = np.r_[x[0], x[1:] - rho * x[:-1]]
    sigma_u2 = 2 * (1 - rho ** 2)
    loglik = 0.5 * (-(u ** 2).sum(0) / sigma_u2 + np.log(1 - rho ** 2) - x.shape[0] * (np.log(2 * np.pi) + np.log(sigma_u2)))
    return loglik

def ar2transform(x, arcoefs):
    if False:
        print('Hello World!')
    '\n\n    (Greene eq 12-30)\n    '
    (a1, a2) = arcoefs
    y = np.zeros_like(x)
    y[0] = np.sqrt((1 + a2) * ((1 - a2) ** 2 - a1 ** 2) / (1 - a2)) * x[0]
    y[1] = np.sqrt(1 - a2 ** 2) * x[2] - a1 * np.sqrt(1 - a1 ** 2) / (1 - a2) * x[1]
    y[2:] = x[2:] - a1 * x[1:-1] - a2 * x[:-2]
    return y

def mvn_loglike(x, sigma):
    if False:
        return 10
    'loglike multivariate normal\n\n    assumes x is 1d, (nobs,) and sigma is 2d (nobs, nobs)\n\n    brute force from formula\n    no checking of correct inputs\n    use of inv and log-det should be replace with something more efficient\n    '
    sigmainv = linalg.inv(sigma)
    logdetsigma = np.log(np.linalg.det(sigma))
    nobs = len(x)
    llf = -np.dot(x, np.dot(sigmainv, x))
    llf -= nobs * np.log(2 * np.pi)
    llf -= logdetsigma
    llf *= 0.5
    return llf

def mvn_nloglike_obs(x, sigma):
    if False:
        for i in range(10):
            print('nop')
    'loglike multivariate normal\n\n    assumes x is 1d, (nobs,) and sigma is 2d (nobs, nobs)\n\n    brute force from formula\n    no checking of correct inputs\n    use of inv and log-det should be replace with something more efficient\n    '
    sigmainv = linalg.inv(sigma)
    cholsigmainv = linalg.cholesky(sigmainv)
    x_whitened = np.dot(cholsigmainv, x)
    logdetsigma = np.log(np.linalg.det(sigma))
    sigma2 = 1.0
    llike = 0.5 * (np.log(sigma2) - 2.0 * np.log(np.diagonal(cholsigmainv)) + x_whitened ** 2 / sigma2 + np.log(2 * np.pi))
    return (llike, x_whitened ** 2)
nobs = 10
x = np.arange(nobs)
autocov = 2 * 0.8 ** np.arange(nobs)
sigma = linalg.toeplitz(autocov)
cholsigma = linalg.cholesky(sigma).T
sigmainv = linalg.inv(sigma)
cholsigmainv = linalg.cholesky(sigmainv)
x_whitened = np.dot(cholsigmainv, x)
logdetsigma = np.log(np.linalg.det(sigma))
sigma2 = 1.0
llike = 0.5 * (np.log(sigma2) - 2.0 * np.log(np.diagonal(cholsigmainv)) + x_whitened ** 2 / sigma2 + np.log(2 * np.pi))
(ll, ls) = mvn_nloglike_obs(x, sigma)
print(ll.sum(), 'll.sum()')
print(llike.sum(), 'llike.sum()')
print(np.log(stats.norm._pdf(x_whitened)).sum() - 0.5 * logdetsigma)
print('stats whitened')
print(np.log(stats.norm.pdf(x, scale=np.sqrt(np.diag(sigma)))).sum())
print('stats scaled')
print(0.5 * (np.dot(linalg.cho_solve((linalg.cho_factor(sigma, lower=False)[0].T, False), x.T), x) + nobs * np.log(2 * np.pi) - 2.0 * np.log(np.diagonal(cholsigmainv)).sum()))
print(0.5 * (np.dot(linalg.cho_solve((linalg.cho_factor(sigma)[0].T, False), x.T), x) + nobs * np.log(2 * np.pi) - 2.0 * np.log(np.diagonal(cholsigmainv)).sum()))
print(0.5 * (np.dot(linalg.cho_solve(linalg.cho_factor(sigma), x.T), x) + nobs * np.log(2 * np.pi) - 2.0 * np.log(np.diagonal(cholsigmainv)).sum()))
print(mvn_loglike(x, sigma))
normtransf = AffineTransform(np.zeros(nobs), cholsigma, StandardNormal())
print(normtransf.logpdf(x_whitened).sum())
print(loglike_ar1(x, 0.8))
mch = MultivariateNormalChol(np.zeros(nobs), sigma)
print(mch.logpdf(x))
xw = mch.whiten(x)
print('xSigmax', np.dot(xw, xw))
print('xSigmax', np.dot(x, linalg.cho_solve(linalg.cho_factor(mch.sigma), x)))
print('xSigmax', np.dot(x, linalg.cho_solve((mch.cholsigma, False), x)))