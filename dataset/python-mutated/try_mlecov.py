"""Multivariate Normal Model with full covariance matrix

toeplitz structure is not exploited, need cholesky or inv for toeplitz

Author: josef-pktd
"""
import numpy as np
from scipy import linalg
from scipy.linalg import toeplitz
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.datasets import sunspots
from statsmodels.tsa.arima_process import ArmaProcess, arma_acovf, arma_generate_sample

def mvn_loglike_sum(x, sigma):
    if False:
        i = 10
        return i + 15
    'loglike multivariate normal\n\n    copied from GLS and adjusted names\n    not sure why this differes from mvn_loglike\n    '
    nobs = len(x)
    nobs2 = nobs / 2.0
    SSR = (x ** 2).sum()
    llf = -np.log(SSR) * nobs2
    llf -= (1 + np.log(np.pi / nobs2)) * nobs2
    if np.any(sigma) and sigma.ndim == 2:
        llf -= 0.5 * np.log(np.linalg.det(sigma))
    return llf

def mvn_loglike(x, sigma):
    if False:
        i = 10
        return i + 15
    'loglike multivariate normal\n\n    assumes x is 1d, (nobs,) and sigma is 2d (nobs, nobs)\n\n    brute force from formula\n    no checking of correct inputs\n    use of inv and log-det should be replace with something more efficient\n    '
    sigmainv = linalg.inv(sigma)
    logdetsigma = np.log(np.linalg.det(sigma))
    nobs = len(x)
    llf = -np.dot(x, np.dot(sigmainv, x))
    llf -= nobs * np.log(2 * np.pi)
    llf -= logdetsigma
    llf *= 0.5
    return llf

def mvn_loglike_chol(x, sigma):
    if False:
        while True:
            i = 10
    'loglike multivariate normal\n\n    assumes x is 1d, (nobs,) and sigma is 2d (nobs, nobs)\n\n    brute force from formula\n    no checking of correct inputs\n    use of inv and log-det should be replace with something more efficient\n    '
    sigmainv = np.linalg.inv(sigma)
    cholsigmainv = np.linalg.cholesky(sigmainv).T
    x_whitened = np.dot(cholsigmainv, x)
    logdetsigma = np.log(np.linalg.det(sigma))
    nobs = len(x)
    from scipy import stats
    print('scipy.stats')
    print(np.log(stats.norm.pdf(x_whitened)).sum())
    llf = -np.dot(x_whitened.T, x_whitened)
    llf -= nobs * np.log(2 * np.pi)
    llf -= logdetsigma
    llf *= 0.5
    return (llf, logdetsigma, 2 * np.sum(np.log(np.diagonal(cholsigmainv))))

def mvn_nloglike_obs(x, sigma):
    if False:
        while True:
            i = 10
    'loglike multivariate normal\n\n    assumes x is 1d, (nobs,) and sigma is 2d (nobs, nobs)\n\n    brute force from formula\n    no checking of correct inputs\n    use of inv and log-det should be replace with something more efficient\n    '
    sigmainv = np.linalg.inv(sigma)
    cholsigmainv = np.linalg.cholesky(sigmainv).T
    x_whitened = np.dot(cholsigmainv, x)
    logdetsigma = np.log(np.linalg.det(sigma))
    sigma2 = 1.0
    llike = 0.5 * (np.log(sigma2) - 2.0 * np.log(np.diagonal(cholsigmainv)) + x_whitened ** 2 / sigma2 + np.log(2 * np.pi))
    return llike

def invertibleroots(ma):
    if False:
        print('Hello World!')
    proc = ArmaProcess(ma=ma)
    return proc.invertroots(retnew=False)

def getpoly(self, params):
    if False:
        while True:
            i = 10
    ar = np.r_[[1], -params[:self.nar]]
    ma = np.r_[[1], params[-self.nma:]]
    import numpy.polynomial as poly
    return (poly.Polynomial(ar), poly.Polynomial(ma))

class MLEGLS(GenericLikelihoodModel):
    """ARMA model with exact loglikelhood for short time series

    Inverts (nobs, nobs) matrix, use only for nobs <= 200 or so.

    This class is a pattern for small sample GLS-like models. Intended use
    for loglikelihood of initial observations for ARMA.



    TODO:
    This might be missing the error variance. Does it assume error is
       distributed N(0,1)
    Maybe extend to mean handling, or assume it is already removed.
    """

    def _params2cov(self, params, nobs):
        if False:
            i = 10
            return i + 15
        'get autocovariance matrix from ARMA regression parameter\n\n        ar parameters are assumed to have rhs parameterization\n\n        '
        ar = np.r_[[1], -params[:self.nar]]
        ma = np.r_[[1], params[-self.nma:]]
        autocov = arma_acovf(ar, ma, nobs=nobs)
        autocov = autocov[:nobs]
        sigma = toeplitz(autocov)
        return sigma

    def loglike(self, params):
        if False:
            while True:
                i = 10
        sig = self._params2cov(params[:-1], self.nobs)
        sig = sig * params[-1] ** 2
        loglik = mvn_loglike(self.endog, sig)
        return loglik

    def fit_invertible(self, *args, **kwds):
        if False:
            print('Hello World!')
        res = self.fit(*args, **kwds)
        ma = np.r_[[1], res.params[self.nar:self.nar + self.nma]]
        (mainv, wasinvertible) = invertibleroots(ma)
        if not wasinvertible:
            start_params = res.params.copy()
            start_params[self.nar:self.nar + self.nma] = mainv[1:]
            res = self.fit(start_params=start_params)
        return res
if __name__ == '__main__':
    nobs = 50
    ar = [1.0, -0.8, 0.1]
    ma = [1.0, 0.1, 0.2]
    np.random.seed(9875789)
    y = arma_generate_sample(ar, ma, nobs, 2)
    y -= y.mean()
    mod = MLEGLS(y)
    (mod.nar, mod.nma) = (2, 2)
    mod.nobs = len(y)
    res = mod.fit(start_params=[0.1, -0.8, 0.2, 0.1, 1.0])
    print('DGP', ar, ma)
    print(res.params)
    from statsmodels.regression import yule_walker
    print(yule_walker(y, 2))
    (arpoly, mapoly) = getpoly(mod, res.params[:-1])
    data = sunspots.load()
    sigma = mod._params2cov(res.params[:-1], nobs) * res.params[-1] ** 2
    print(mvn_loglike(y, sigma))
    llo = mvn_nloglike_obs(y, sigma)
    print(llo.sum(), llo.shape)
    print(mvn_loglike_chol(y, sigma))
    print(mvn_loglike_sum(y, sigma))