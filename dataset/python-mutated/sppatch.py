"""patching scipy to fit distributions and expect method

This adds new methods to estimate continuous distribution parameters with some
fixed/frozen parameters. It also contains functions that calculate the expected
value of a function for any continuous or discrete distribution

It temporarily also contains Bootstrap and Monte Carlo function for testing the
distribution fit, but these are neither general nor verified.

Author: josef-pktd
License: Simplified BSD
"""
from statsmodels.compat.python import lmap
import numpy as np
from scipy import stats, optimize, integrate
stats.distributions.vonmises.a = -np.pi
stats.distributions.vonmises.b = np.pi

def _fitstart(self, x):
    if False:
        i = 10
        return i + 15
    'example method, method of moment estimator as starting values\n\n    Parameters\n    ----------\n    x : ndarray\n        data for which the parameters are estimated\n\n    Returns\n    -------\n    est : tuple\n        preliminary estimates used as starting value for fitting, not\n        necessarily a consistent estimator\n\n    Notes\n    -----\n    This needs to be written and attached to each individual distribution\n\n    This example was written for the gamma distribution, but not verified\n    with literature\n\n    '
    loc = np.min([x.min(), 0])
    a = 4 / stats.skew(x) ** 2
    scale = np.std(x) / np.sqrt(a)
    return (a, loc, scale)

def _fitstart_beta(self, x, fixed=None):
    if False:
        for i in range(10):
            print('nop')
    'method of moment estimator as starting values for beta distribution\n\n    Parameters\n    ----------\n    x : ndarray\n        data for which the parameters are estimated\n    fixed : None or array_like\n        sequence of numbers and np.nan to indicate fixed parameters and parameters\n        to estimate\n\n    Returns\n    -------\n    est : tuple\n        preliminary estimates used as starting value for fitting, not\n        necessarily a consistent estimator\n\n    Notes\n    -----\n    This needs to be written and attached to each individual distribution\n\n    References\n    ----------\n    for method of moment estimator for known loc and scale\n    https://en.wikipedia.org/wiki/Beta_distribution#Parameter_estimation\n    http://www.itl.nist.gov/div898/handbook/eda/section3/eda366h.htm\n    NIST reference also includes reference to MLE in\n    Johnson, Kotz, and Balakrishan, Volume II, pages 221-235\n\n    '
    (a, b) = (x.min(), x.max())
    eps = (a - b) * 0.01
    if fixed is None:
        loc = a - eps
        scale = (a - b) * (1 + 2 * eps)
    else:
        if np.isnan(fixed[-2]):
            loc = a - eps
        else:
            loc = fixed[-2]
        if np.isnan(fixed[-1]):
            scale = b + eps - loc
        else:
            scale = fixed[-1]
    scale = float(scale)
    xtrans = (x - loc) / scale
    xm = xtrans.mean()
    xv = xtrans.var()
    tmp = xm * (1 - xm) / xv - 1
    p = xm * tmp
    q = (1 - xm) * tmp
    return (p, q, loc, scale)

def _fitstart_poisson(self, x, fixed=None):
    if False:
        print('Hello World!')
    'maximum likelihood estimator as starting values for Poisson distribution\n\n    Parameters\n    ----------\n    x : ndarray\n        data for which the parameters are estimated\n    fixed : None or array_like\n        sequence of numbers and np.nan to indicate fixed parameters and parameters\n        to estimate\n\n    Returns\n    -------\n    est : tuple\n        preliminary estimates used as starting value for fitting, not\n        necessarily a consistent estimator\n\n    Notes\n    -----\n    This needs to be written and attached to each individual distribution\n\n    References\n    ----------\n    MLE :\n    https://en.wikipedia.org/wiki/Poisson_distribution#Maximum_likelihood\n\n    '
    a = x.min()
    eps = 0
    if fixed is None:
        loc = a - eps
    elif np.isnan(fixed[-1]):
        loc = a - eps
    else:
        loc = fixed[-1]
    xtrans = x - loc
    lambd = xtrans.mean()
    return (lambd, loc)

def nnlf_fr(self, thetash, x, frmask):
    if False:
        return 10
    try:
        if frmask is not None:
            theta = frmask.copy()
            theta[np.isnan(frmask)] = thetash
        else:
            theta = thetash
        loc = theta[-2]
        scale = theta[-1]
        args = tuple(theta[:-2])
    except IndexError:
        raise ValueError('Not enough input arguments.')
    if not self._argcheck(*args) or scale <= 0:
        return np.inf
    x = np.array((x - loc) / scale)
    cond0 = (x <= self.a) | (x >= self.b)
    if np.any(cond0):
        return np.inf
    else:
        N = len(x)
        return self._nnlf(x, *args) + N * np.log(scale)

def fit_fr(self, data, *args, **kwds):
    if False:
        print('Hello World!')
    "estimate distribution parameters by MLE taking some parameters as fixed\n\n    Parameters\n    ----------\n    data : ndarray, 1d\n        data for which the distribution parameters are estimated,\n    args : list ? check\n        starting values for optimization\n    kwds :\n\n      - 'frozen' : array_like\n           values for frozen distribution parameters and, for elements with\n           np.nan, the corresponding parameter will be estimated\n\n    Returns\n    -------\n    argest : ndarray\n        estimated parameters\n\n\n    Examples\n    --------\n    generate random sample\n    >>> np.random.seed(12345)\n    >>> x = stats.gamma.rvs(2.5, loc=0, scale=1.2, size=200)\n\n    estimate all parameters\n    >>> stats.gamma.fit(x)\n    array([ 2.0243194 ,  0.20395655,  1.44411371])\n    >>> stats.gamma.fit_fr(x, frozen=[np.nan, np.nan, np.nan])\n    array([ 2.0243194 ,  0.20395655,  1.44411371])\n\n    keep loc fixed, estimate shape and scale parameters\n    >>> stats.gamma.fit_fr(x, frozen=[np.nan, 0.0, np.nan])\n    array([ 2.45603985,  1.27333105])\n\n    keep loc and scale fixed, estimate shape parameter\n    >>> stats.gamma.fit_fr(x, frozen=[np.nan, 0.0, 1.0])\n    array([ 3.00048828])\n    >>> stats.gamma.fit_fr(x, frozen=[np.nan, 0.0, 1.2])\n    array([ 2.57792969])\n\n    estimate only scale parameter for fixed shape and loc\n    >>> stats.gamma.fit_fr(x, frozen=[2.5, 0.0, np.nan])\n    array([ 1.25087891])\n\n    Notes\n    -----\n    self is an instance of a distribution class. This can be attached to\n    scipy.stats.distributions.rv_continuous\n\n    *Todo*\n\n    * check if docstring is correct\n    * more input checking, args is list ? might also apply to current fit method\n\n    "
    (loc0, scale0) = lmap(kwds.get, ['loc', 'scale'], [0.0, 1.0])
    Narg = len(args)
    if Narg == 0 and hasattr(self, '_fitstart'):
        x0 = self._fitstart(data)
    elif Narg > self.numargs:
        raise ValueError('Too many input arguments.')
    else:
        args += (1.0,) * (self.numargs - Narg)
        x0 = args + (loc0, scale0)
    if 'frozen' in kwds:
        frmask = np.array(kwds['frozen'])
        if len(frmask) != self.numargs + 2:
            raise ValueError('Incorrect number of frozen arguments.')
        else:
            for n in range(len(frmask)):
                if isinstance(frmask[n], np.ndarray) and frmask[n].size == 1:
                    frmask[n] = frmask[n].item()
            frmask = frmask.astype(np.float64)
            x0 = np.array(x0)[np.isnan(frmask)]
    else:
        frmask = None
    return optimize.fmin(self.nnlf_fr, x0, args=(np.ravel(data), frmask), disp=0)

def expect(self, fn=None, args=(), loc=0, scale=1, lb=None, ub=None, conditional=False):
    if False:
        return 10
    "calculate expected value of a function with respect to the distribution\n\n    location and scale only tested on a few examples\n\n    Parameters\n    ----------\n        all parameters are keyword parameters\n        fn : function (default: identity mapping)\n           Function for which integral is calculated. Takes only one argument.\n        args : tuple\n           argument (parameters) of the distribution\n        lb, ub : numbers\n           lower and upper bound for integration, default is set to the support\n           of the distribution\n        conditional : bool (False)\n           If true then the integral is corrected by the conditional probability\n           of the integration interval. The return value is the expectation\n           of the function, conditional on being in the given interval.\n\n    Returns\n    -------\n        expected value : float\n\n    Notes\n    -----\n    This function has not been checked for it's behavior when the integral is\n    not finite. The integration behavior is inherited from scipy.integrate.quad.\n\n    "
    if fn is None:

        def fun(x, *args):
            if False:
                print('Hello World!')
            return x * self.pdf(x, *args, loc=loc, scale=scale)
    else:

        def fun(x, *args):
            if False:
                return 10
            return fn(x) * self.pdf(x, *args, loc=loc, scale=scale)
    if lb is None:
        lb = loc + self.a * scale
    if ub is None:
        ub = loc + self.b * scale
    if conditional:
        invfac = self.sf(lb, *args, loc=loc, scale=scale) - self.sf(ub, *args, loc=loc, scale=scale)
    else:
        invfac = 1.0
    return integrate.quad(fun, lb, ub, args=args)[0] / invfac

def expect_v2(self, fn=None, args=(), loc=0, scale=1, lb=None, ub=None, conditional=False):
    if False:
        print('Hello World!')
    "calculate expected value of a function with respect to the distribution\n\n    location and scale only tested on a few examples\n\n    Parameters\n    ----------\n        all parameters are keyword parameters\n        fn : function (default: identity mapping)\n           Function for which integral is calculated. Takes only one argument.\n        args : tuple\n           argument (parameters) of the distribution\n        lb, ub : numbers\n           lower and upper bound for integration, default is set using\n           quantiles of the distribution, see Notes\n        conditional : bool (False)\n           If true then the integral is corrected by the conditional probability\n           of the integration interval. The return value is the expectation\n           of the function, conditional on being in the given interval.\n\n    Returns\n    -------\n        expected value : float\n\n    Notes\n    -----\n    This function has not been checked for it's behavior when the integral is\n    not finite. The integration behavior is inherited from scipy.integrate.quad.\n\n    The default limits are lb = self.ppf(1e-9, *args), ub = self.ppf(1-1e-9, *args)\n\n    For some heavy tailed distributions, 'alpha', 'cauchy', 'halfcauchy',\n    'levy', 'levy_l', and for 'ncf', the default limits are not set correctly\n    even  when the expectation of the function is finite. In this case, the\n    integration limits, lb and ub, should be chosen by the user. For example,\n    for the ncf distribution, ub=1000 works in the examples.\n\n    There are also problems with numerical integration in some other cases,\n    for example if the distribution is very concentrated and the default limits\n    are too large.\n\n    "
    if fn is None:

        def fun(x, *args):
            if False:
                i = 10
                return i + 15
            return (loc + x * scale) * self._pdf(x, *args)
    else:

        def fun(x, *args):
            if False:
                while True:
                    i = 10
            return fn(loc + x * scale) * self._pdf(x, *args)
    if lb is None:
        try:
            lb = self.ppf(1e-09, *args)
        except ValueError:
            lb = self.a
    else:
        lb = max(self.a, (lb - loc) / (1.0 * scale))
    if ub is None:
        try:
            ub = self.ppf(1 - 1e-09, *args)
        except ValueError:
            ub = self.b
    else:
        ub = min(self.b, (ub - loc) / (1.0 * scale))
    if conditional:
        invfac = self._sf(lb, *args) - self._sf(ub, *args)
    else:
        invfac = 1.0
    return integrate.quad(fun, lb, ub, args=args, limit=500)[0] / invfac

def expect_discrete(self, fn=None, args=(), loc=0, lb=None, ub=None, conditional=False):
    if False:
        i = 10
        return i + 15
    'calculate expected value of a function with respect to the distribution\n    for discrete distribution\n\n    Parameters\n    ----------\n        (self : distribution instance as defined in scipy stats)\n        fn : function (default: identity mapping)\n           Function for which integral is calculated. Takes only one argument.\n        args : tuple\n           argument (parameters) of the distribution\n        optional keyword parameters\n        lb, ub : numbers\n           lower and upper bound for integration, default is set to the support\n           of the distribution, lb and ub are inclusive (ul<=k<=ub)\n        conditional : bool (False)\n           If true then the expectation is corrected by the conditional\n           probability of the integration interval. The return value is the\n           expectation of the function, conditional on being in the given\n           interval (k such that ul<=k<=ub).\n\n    Returns\n    -------\n        expected value : float\n\n    Notes\n    -----\n    * function is not vectorized\n    * accuracy: uses self.moment_tol as stopping criterium\n        for heavy tailed distribution e.g. zipf(4), accuracy for\n        mean, variance in example is only 1e-5,\n        increasing precision (moment_tol) makes zipf very slow\n    * suppnmin=100 internal parameter for minimum number of points to evaluate\n        could be added as keyword parameter, to evaluate functions with\n        non-monotonic shapes, points include integers in (-suppnmin, suppnmin)\n    * uses maxcount=1000 limits the number of points that are evaluated\n        to break loop for infinite sums\n        (a maximum of suppnmin+1000 positive plus suppnmin+1000 negative integers\n        are evaluated)\n\n\n    '
    maxcount = 1000
    suppnmin = 100
    if fn is None:

        def fun(x):
            if False:
                while True:
                    i = 10
            return (x + loc) * self._pmf(x, *args)
    else:

        def fun(x):
            if False:
                return 10
            return fn(x + loc) * self._pmf(x, *args)
    self._argcheck(*args)
    if lb is None:
        lb = self.a
    else:
        lb = lb - loc
    if ub is None:
        ub = self.b
    else:
        ub = ub - loc
    if conditional:
        invfac = self.sf(lb, *args) - self.sf(ub + 1, *args)
    else:
        invfac = 1.0
    tot = 0.0
    (low, upp) = (self._ppf(0.001, *args), self._ppf(0.999, *args))
    low = max(min(-suppnmin, low), lb)
    upp = min(max(suppnmin, upp), ub)
    supp = np.arange(low, upp + 1, self.inc)
    tot = np.sum(fun(supp))
    diff = 1e+100
    pos = upp + self.inc
    count = 0
    while pos <= ub and diff > self.moment_tol and (count <= maxcount):
        diff = fun(pos)
        tot += diff
        pos += self.inc
        count += 1
    if self.a < 0:
        diff = 1e+100
        pos = low - self.inc
        while pos >= lb and diff > self.moment_tol and (count <= maxcount):
            diff = fun(pos)
            tot += diff
            pos -= self.inc
            count += 1
    if count > maxcount:
        print('sum did not converge')
    return tot / invfac
stats.distributions.rv_continuous.fit_fr = fit_fr
stats.distributions.rv_continuous.nnlf_fr = nnlf_fr
stats.distributions.rv_continuous.expect = expect
stats.distributions.rv_discrete.expect = expect_discrete
stats.distributions.beta_gen._fitstart = _fitstart_beta
stats.distributions.poisson_gen._fitstart = _fitstart_poisson

def distfitbootstrap(sample, distr, nrepl=100):
    if False:
        while True:
            i = 10
    'run bootstrap for estimation of distribution parameters\n\n    hard coded: only one shape parameter is allowed and estimated,\n        loc=0 and scale=1 are fixed in the estimation\n\n    Parameters\n    ----------\n    sample : ndarray\n        original sample data for bootstrap\n    distr : distribution instance with fit_fr method\n    nrepl : int\n        number of bootstrap replications\n\n    Returns\n    -------\n    res : array (nrepl,)\n        parameter estimates for all bootstrap replications\n\n    '
    nobs = len(sample)
    res = np.zeros(nrepl)
    for ii in range(nrepl):
        rvsind = np.random.randint(nobs, size=nobs)
        x = sample[rvsind]
        res[ii] = distr.fit_fr(x, frozen=[np.nan, 0.0, 1.0])
    return res

def distfitmc(sample, distr, nrepl=100, distkwds={}):
    if False:
        return 10
    'run Monte Carlo for estimation of distribution parameters\n\n    hard coded: only one shape parameter is allowed and estimated,\n        loc=0 and scale=1 are fixed in the estimation\n\n    Parameters\n    ----------\n    sample : ndarray\n        original sample data, in Monte Carlo only used to get nobs,\n    distr : distribution instance with fit_fr method\n    nrepl : int\n        number of Monte Carlo replications\n\n    Returns\n    -------\n    res : array (nrepl,)\n        parameter estimates for all Monte Carlo replications\n\n    '
    arg = distkwds.pop('arg')
    nobs = len(sample)
    res = np.zeros(nrepl)
    for ii in range(nrepl):
        x = distr.rvs(arg, size=nobs, **distkwds)
        res[ii] = distr.fit_fr(x, frozen=[np.nan, 0.0, 1.0])
    return res

def printresults(sample, arg, bres, kind='bootstrap'):
    if False:
        while True:
            i = 10
    "calculate and print(Bootstrap or Monte Carlo result\n\n    Parameters\n    ----------\n    sample : ndarray\n        original sample data\n    arg : float   (for general case will be array)\n    bres : ndarray\n        parameter estimates from Bootstrap or Monte Carlo run\n    kind : {'bootstrap', 'montecarlo'}\n        output is printed for Mootstrap (default) or Monte Carlo\n\n    Returns\n    -------\n    None, currently only printing\n\n    Notes\n    -----\n    still a bit a mess because it is used for both Bootstrap and Monte Carlo\n\n    made correction:\n        reference point for bootstrap is estimated parameter\n\n    not clear:\n        I'm not doing any ddof adjustment in estimation of variance, do we\n        need ddof>0 ?\n\n    todo: return results and string instead of printing\n\n    "
    print('true parameter value')
    print(arg)
    print('MLE estimate of parameters using sample (nobs=%d)' % nobs)
    argest = distr.fit_fr(sample, frozen=[np.nan, 0.0, 1.0])
    print(argest)
    if kind == 'bootstrap':
        argorig = arg
        arg = argest
    print('%s distribution of parameter estimate (nrepl=%d)' % (kind, nrepl))
    print('mean = %f, bias=%f' % (bres.mean(0), bres.mean(0) - arg))
    print('median', np.median(bres, axis=0))
    print('var and std', bres.var(0), np.sqrt(bres.var(0)))
    bmse = ((bres - arg) ** 2).mean(0)
    print('mse, rmse', bmse, np.sqrt(bmse))
    bressorted = np.sort(bres)
    print('%s confidence interval (90%% coverage)' % kind)
    print(bressorted[np.floor(nrepl * 0.05)], bressorted[np.floor(nrepl * 0.95)])
    print('%s confidence interval (90%% coverage) normal approximation' % kind)
    print(stats.norm.ppf(0.05, loc=bres.mean(), scale=bres.std()))
    print(stats.norm.isf(0.05, loc=bres.mean(), scale=bres.std()))
    print('Kolmogorov-Smirnov test for normality of %s distribution' % kind)
    print(' - estimated parameters, p-values not really correct')
    print(stats.kstest(bres, 'norm', (bres.mean(), bres.std())))
if __name__ == '__main__':
    examplecases = ['largenumber', 'bootstrap', 'montecarlo'][:]
    if 'largenumber' in examplecases:
        print('\nDistribution: vonmises')
        for nobs in [200]:
            x = stats.vonmises.rvs(1.23, loc=0, scale=1, size=nobs)
            print('\nnobs:', nobs)
            print('true parameter')
            print('1.23, loc=0, scale=1')
            print('unconstrained')
            print(stats.vonmises.fit(x))
            print(stats.vonmises.fit_fr(x, frozen=[np.nan, np.nan, np.nan]))
            print('with fixed loc and scale')
            print(stats.vonmises.fit_fr(x, frozen=[np.nan, 0.0, 1.0]))
        print('\nDistribution: gamma')
        distr = stats.gamma
        (arg, loc, scale) = (2.5, 0.0, 20.0)
        for nobs in [200]:
            x = distr.rvs(arg, loc=loc, scale=scale, size=nobs)
            print('\nnobs:', nobs)
            print('true parameter')
            print('%f, loc=%f, scale=%f' % (arg, loc, scale))
            print('unconstrained')
            print(distr.fit(x))
            print(distr.fit_fr(x, frozen=[np.nan, np.nan, np.nan]))
            print('with fixed loc and scale')
            print(distr.fit_fr(x, frozen=[np.nan, 0.0, 1.0]))
            print('with fixed loc')
            print(distr.fit_fr(x, frozen=[np.nan, 0.0, np.nan]))
    ex = ['gamma', 'vonmises'][0]
    if ex == 'gamma':
        distr = stats.gamma
        (arg, loc, scale) = (2.5, 0.0, 1)
    elif ex == 'vonmises':
        distr = stats.vonmises
        (arg, loc, scale) = (1.5, 0.0, 1)
    else:
        raise ValueError('wrong example')
    nobs = 100
    nrepl = 1000
    sample = distr.rvs(arg, loc=loc, scale=scale, size=nobs)
    print('\nDistribution:', distr)
    if 'bootstrap' in examplecases:
        print('\nBootstrap')
        bres = distfitbootstrap(sample, distr, nrepl=nrepl)
        printresults(sample, arg, bres)
    if 'montecarlo' in examplecases:
        print('\nMonteCarlo')
        mcres = distfitmc(sample, distr, nrepl=nrepl, distkwds=dict(arg=arg, loc=loc, scale=scale))
        printresults(sample, arg, mcres, kind='montecarlo')