"""estimate distribution parameters by various methods
method of moments or matching quantiles, and Maximum Likelihood estimation
based on binned data and Maximum Product-of-Spacings

Warning: I'm still finding cut-and-paste and refactoring errors, e.g.
    hardcoded variables from outer scope in functions
    some results do not seem to make sense for Pareto case,
    looks better now after correcting some name errors

initially loosely based on a paper and blog for quantile matching
  by John D. Cook
  formula for gamma quantile (ppf) matching by him (from paper)
  http://www.codeproject.com/KB/recipes/ParameterPercentile.aspx
  http://www.johndcook.com/blog/2010/01/31/parameters-from-percentiles/
  this is what I actually used (in parts):
  http://www.bepress.com/mdandersonbiostat/paper55/

quantile based estimator
^^^^^^^^^^^^^^^^^^^^^^^^
only special cases for number or parameters so far
Is there a literature for GMM estimation of distribution parameters? check
    found one: Wu/Perloff 2007


binned estimator
^^^^^^^^^^^^^^^^
* I added this also
* use it for chisquare tests with estimation distribution parameters
* move this to distribution_extras (next to gof tests powerdiscrepancy and
  continuous) or add to distribution_patch


example: t-distribution
* works with quantiles if they contain tail quantiles
* results with momentcondquant do not look as good as mle estimate

TODOs
* rearange and make sure I do not use module globals (as I did initially) DONE
  make two version exactly identified method of moments with fsolve
  and GMM (?) version with fmin
  and maybe the special cases of JD Cook
  update: maybe exact (MM) version is not so interesting compared to GMM
* add semifrozen version of moment and quantile based estimators,
  e.g. for beta (both loc and scale fixed), or gamma (loc fixed)
* add beta example to the semifrozen MLE, fitfr, code
  -> added method of moment estimator to _fitstart for beta
* start a list of how well different estimators, especially current mle work
  for the different distributions
* need general GMM code (with optimal weights ?), looks like a good example
  for it
* get example for binned data estimation, mailing list a while ago
* any idea when these are better than mle ?
* check language: I use quantile to mean the value of the random variable, not
  quantile between 0 and 1.
* for GMM: move moment conditions to separate function, so that they can be
  used for further analysis, e.g. covariance matrix of parameter estimates
* question: Are GMM properties different for matching quantiles with cdf or
  ppf? Estimate should be the same, but derivatives of moment conditions
  differ.
* add maximum spacings estimator, Wikipedia, Per Brodtkorb -> basic version Done
* add parameter estimation based on empirical characteristic function
  (Carrasco/Florens), especially for stable distribution
* provide a model class based on estimating all distributions, and collect
  all distribution specific information


References
----------

Ximing Wu, Jeffrey M. Perloff, GMM estimation of a maximum entropy
distribution with interval data, Journal of Econometrics, Volume 138,
Issue 2, 'Information and Entropy Econometrics' - A Volume in Honor of
Arnold Zellner, June 2007, Pages 532-546, ISSN 0304-4076,
DOI: 10.1016/j.jeconom.2006.05.008.
http://www.sciencedirect.com/science/article/B6VC0-4K606TK-4/2/78bc07c6245546374490f777a6bdbbcc
http://escholarship.org/uc/item/7jf5w1ht  (working paper)

Johnson, Kotz, Balakrishnan: Volume 2


Author : josef-pktd
License : BSD
created : 2010-04-20

changes:
added Maximum Product-of-Spacings 2010-05-12

"""
import numpy as np
from scipy import stats, optimize, special
cache = {}

def gammamomentcond(distfn, params, mom2, quantile=None):
    if False:
        print('Hello World!')
    'estimate distribution parameters based method of moments (mean,\n    variance) for distributions with 1 shape parameter and fixed loc=0.\n\n    Returns\n    -------\n    cond : function\n\n    Notes\n    -----\n    first test version, quantile argument not used\n\n    '

    def cond(params):
        if False:
            return 10
        (alpha, scale) = params
        mom2s = distfn.stats(alpha, 0.0, scale)
        return np.array(mom2) - mom2s
    return cond

def gammamomentcond2(distfn, params, mom2, quantile=None):
    if False:
        while True:
            i = 10
    'estimate distribution parameters based method of moments (mean,\n    variance) for distributions with 1 shape parameter and fixed loc=0.\n\n    Returns\n    -------\n    difference : ndarray\n        difference between theoretical and empirical moments\n\n    Notes\n    -----\n    first test version, quantile argument not used\n\n    The only difference to previous function is return type.\n\n    '
    (alpha, scale) = params
    mom2s = distfn.stats(alpha, 0.0, scale)
    return np.array(mom2) - mom2s

def momentcondunbound(distfn, params, mom2, quantile=None):
    if False:
        return 10
    'moment conditions for estimating distribution parameters using method\n    of moments, uses mean, variance and one quantile for distributions\n    with 1 shape parameter.\n\n    Returns\n    -------\n    difference : ndarray\n        difference between theoretical and empirical moments and quantiles\n\n    '
    (shape, loc, scale) = params
    mom2diff = np.array(distfn.stats(shape, loc, scale)) - mom2
    if quantile is not None:
        (pq, xq) = quantile
        cdfdiff = distfn.cdf(xq, shape, loc, scale) - pq
        return np.concatenate([mom2diff, cdfdiff[:1]])
    return mom2diff

def momentcondunboundls(distfn, params, mom2, quantile=None, shape=None):
    if False:
        return 10
    'moment conditions for estimating loc and scale of a distribution\n    with method of moments using either 2 quantiles or 2 moments (not both).\n\n    Returns\n    -------\n    difference : ndarray\n        difference between theoretical and empirical moments or quantiles\n\n    '
    (loc, scale) = params
    mom2diff = np.array(distfn.stats(shape, loc, scale)) - mom2
    if quantile is not None:
        (pq, xq) = quantile
        cdfdiff = distfn.cdf(xq, shape, loc, scale) - pq
        return cdfdiff
    return mom2diff

def momentcondquant(distfn, params, mom2, quantile=None, shape=None):
    if False:
        for i in range(10):
            print('nop')
    'moment conditions for estimating distribution parameters by matching\n    quantiles, defines as many moment conditions as quantiles.\n\n    Returns\n    -------\n    difference : ndarray\n        difference between theoretical and empirical quantiles\n\n    Notes\n    -----\n    This can be used for method of moments or for generalized method of\n    moments.\n\n    '
    if len(params) == 2:
        (loc, scale) = params
    elif len(params) == 3:
        (shape, loc, scale) = params
    else:
        pass
    (pq, xq) = quantile
    cdfdiff = distfn.cdf(xq, *params) - pq
    return cdfdiff

def fitquantilesgmm(distfn, x, start=None, pquant=None, frozen=None):
    if False:
        while True:
            i = 10
    if pquant is None:
        pquant = np.array([0.01, 0.05, 0.1, 0.4, 0.6, 0.9, 0.95, 0.99])
    if start is None:
        if hasattr(distfn, '_fitstart'):
            start = distfn._fitstart(x)
        else:
            start = [1] * distfn.numargs + [0.0, 1.0]
    xqs = [stats.scoreatpercentile(x, p) for p in pquant * 100]
    mom2s = None
    parest = optimize.fmin(lambda params: np.sum(momentcondquant(distfn, params, mom2s, (pquant, xqs), shape=None) ** 2), start)
    return parest

def fitbinned(distfn, freq, binedges, start, fixed=None):
    if False:
        print('Hello World!')
    'estimate parameters of distribution function for binned data using MLE\n\n    Parameters\n    ----------\n    distfn : distribution instance\n        needs to have cdf method, as in scipy.stats\n    freq : ndarray, 1d\n        frequency count, e.g. obtained by histogram\n    binedges : ndarray, 1d\n        binedges including lower and upper bound\n    start : tuple or array_like ?\n        starting values, needs to have correct length\n\n    Returns\n    -------\n    paramest : ndarray\n        estimated parameters\n\n    Notes\n    -----\n    todo: add fixed parameter option\n\n    added factorial\n\n    '
    if fixed is not None:
        raise NotImplementedError
    nobs = np.sum(freq)
    lnnobsfact = special.gammaln(nobs + 1)

    def nloglike(params):
        if False:
            return 10
        'negative loglikelihood function of binned data\n\n        corresponds to multinomial\n        '
        prob = np.diff(distfn.cdf(binedges, *params))
        return -(lnnobsfact + np.sum(freq * np.log(prob) - special.gammaln(freq + 1)))
    return optimize.fmin(nloglike, start)

def fitbinnedgmm(distfn, freq, binedges, start, fixed=None, weightsoptimal=True):
    if False:
        print('Hello World!')
    'estimate parameters of distribution function for binned data using GMM\n\n    Parameters\n    ----------\n    distfn : distribution instance\n        needs to have cdf method, as in scipy.stats\n    freq : ndarray, 1d\n        frequency count, e.g. obtained by histogram\n    binedges : ndarray, 1d\n        binedges including lower and upper bound\n    start : tuple or array_like ?\n        starting values, needs to have correct length\n    fixed : None\n        not used yet\n    weightsoptimal : bool\n        If true, then the optimal weighting matrix for GMM is used. If false,\n        then the identity matrix is used\n\n    Returns\n    -------\n    paramest : ndarray\n        estimated parameters\n\n    Notes\n    -----\n    todo: add fixed parameter option\n\n    added factorial\n\n    '
    if fixed is not None:
        raise NotImplementedError
    nobs = np.sum(freq)
    if weightsoptimal:
        weights = freq / float(nobs)
    else:
        weights = np.ones(len(freq))
    freqnormed = freq / float(nobs)

    def gmmobjective(params):
        if False:
            while True:
                i = 10
        'negative loglikelihood function of binned data\n\n        corresponds to multinomial\n        '
        prob = np.diff(distfn.cdf(binedges, *params))
        momcond = freqnormed - prob
        return np.dot(momcond * weights, momcond)
    return optimize.fmin(gmmobjective, start)
'Estimating Parameters of Log-Normal Distribution with Maximum\nLikelihood and Maximum Product-of-Spacings\n\nMPS definiton from JKB page 233\n\nCreated on Tue May 11 13:52:50 2010\nAuthor: josef-pktd\nLicense: BSD\n'

def hess_ndt(fun, pars, args, options):
    if False:
        i = 10
        return i + 15
    import numdifftools as ndt
    if not ('stepMax' in options or 'stepFix' in options):
        options['stepMax'] = 1e-05
    f = lambda params: fun(params, *args)
    h = ndt.Hessian(f, **options)
    return (h(pars), h)

def logmps(params, xsorted, dist):
    if False:
        print('Hello World!')
    'calculate negative log of Product-of-Spacings\n\n    Parameters\n    ----------\n    params : array_like, tuple ?\n        parameters of the distribution funciton\n    xsorted : array_like\n        data that is already sorted\n    dist : instance of a distribution class\n        only cdf method is used\n\n    Returns\n    -------\n    mps : float\n        negative log of Product-of-Spacings\n\n\n    Notes\n    -----\n    MPS definiton from JKB page 233\n    '
    xcdf = np.r_[0.0, dist.cdf(xsorted, *params), 1.0]
    D = np.diff(xcdf)
    return -np.log(D).mean()

def getstartparams(dist, data):
    if False:
        for i in range(10):
            print('nop')
    'get starting values for estimation of distribution parameters\n\n    Parameters\n    ----------\n    dist : distribution instance\n        the distribution instance needs to have either a method fitstart\n        or an attribute numargs\n    data : ndarray\n        data for which preliminary estimator or starting value for\n        parameter estimation is desired\n\n    Returns\n    -------\n    x0 : ndarray\n        preliminary estimate or starting value for the parameters of\n        the distribution given the data, including loc and scale\n\n    '
    if hasattr(dist, 'fitstart'):
        x0 = dist.fitstart(data)
    elif np.isfinite(dist.a):
        x0 = np.r_[[1.0] * dist.numargs, data.min() - 1, 1.0]
    else:
        x0 = np.r_[[1.0] * dist.numargs, data.mean() - 1, 1.0]
    return x0

def fit_mps(dist, data, x0=None):
    if False:
        i = 10
        return i + 15
    'Estimate distribution parameters with Maximum Product-of-Spacings\n\n    Parameters\n    ----------\n    params : array_like, tuple ?\n        parameters of the distribution funciton\n    xsorted : array_like\n        data that is already sorted\n    dist : instance of a distribution class\n        only cdf method is used\n\n    Returns\n    -------\n    x : ndarray\n        estimates for the parameters of the distribution given the data,\n        including loc and scale\n\n\n    '
    xsorted = np.sort(data)
    if x0 is None:
        x0 = getstartparams(dist, xsorted)
    args = (xsorted, dist)
    print(x0)
    return optimize.fmin(logmps, x0, args=args)
if __name__ == '__main__':
    print('\n\nExample: gamma Distribution')
    print('---------------------------')
    alpha = 2
    xq = [0.5, 4]
    pq = [0.1, 0.9]
    print(stats.gamma.ppf(pq, alpha))
    xq = stats.gamma.ppf(pq, alpha)
    print(np.diff(stats.gamma.ppf(pq, np.linspace(0.01, 4, 10)[:, None]) * xq[::-1]))
    print(optimize.fsolve(lambda alpha: np.diff(stats.gamma.ppf(pq, alpha) * xq[::-1]), 3.0))
    distfn = stats.gamma
    mcond = gammamomentcond(distfn, [5.0, 10], mom2=stats.gamma.stats(alpha, 0.0, 1.0), quantile=None)
    print(optimize.fsolve(mcond, [1.0, 2.0]))
    mom2 = stats.gamma.stats(alpha, 0.0, 1.0)
    print(optimize.fsolve(lambda params: gammamomentcond2(distfn, params, mom2), [1.0, 2.0]))
    grvs = stats.gamma.rvs(alpha, 0.0, 2.0, size=1000)
    mom2 = np.array([grvs.mean(), grvs.var()])
    alphaestq = optimize.fsolve(lambda params: gammamomentcond2(distfn, params, mom2), [1.0, 3.0])
    print(alphaestq)
    print('scale = ', xq / stats.gamma.ppf(pq, alphaestq))
    print('\n\nExample: beta Distribution')
    print('--------------------------')
    stats.distributions.beta_gen._fitstart = lambda self, data: (5, 5, 0, 1)
    pq = np.array([0.01, 0.05, 0.1, 0.4, 0.6, 0.9, 0.95, 0.99])
    rvsb = stats.beta.rvs(10, 15, size=2000)
    print('true params', 10, 15, 0, 1)
    print(stats.beta.fit(rvsb))
    xqsb = [stats.scoreatpercentile(rvsb, p) for p in pq * 100]
    mom2s = np.array([rvsb.mean(), rvsb.var()])
    betaparest_gmmquantile = optimize.fmin(lambda params: np.sum(momentcondquant(stats.beta, params, mom2s, (pq, xqsb), shape=None) ** 2), [10, 10, 0.0, 1.0], maxiter=2000)
    print('betaparest_gmmquantile', betaparest_gmmquantile)
    print('\n\nExample: t Distribution')
    print('-----------------------')
    nobs = 1000
    distfn = stats.t
    pq = np.array([0.1, 0.9])
    paramsdgp = (5, 0, 1)
    trvs = distfn.rvs(5, 0, 1, size=nobs)
    xqs = [stats.scoreatpercentile(trvs, p) for p in pq * 100]
    mom2th = distfn.stats(*paramsdgp)
    mom2s = np.array([trvs.mean(), trvs.var()])
    tparest_gmm3quantilefsolve = optimize.fsolve(lambda params: momentcondunbound(distfn, params, mom2s, (pq, xqs)), [10, 1.0, 2.0])
    print('tparest_gmm3quantilefsolve', tparest_gmm3quantilefsolve)
    tparest_gmm3quantile = optimize.fmin(lambda params: np.sum(momentcondunbound(distfn, params, mom2s, (pq, xqs)) ** 2), [10, 1.0, 2.0])
    print('tparest_gmm3quantile', tparest_gmm3quantile)
    print(distfn.fit(trvs))
    print(optimize.fsolve(lambda params: momentcondunboundls(distfn, params, mom2s, shape=5), [1.0, 2.0]))
    print(optimize.fmin(lambda params: np.sum(momentcondunboundls(distfn, params, mom2s, shape=5) ** 2), [1.0, 2.0]))
    print(distfn.fit(trvs))
    print(optimize.fsolve(lambda params: momentcondunboundls(distfn, params, mom2s, (pq, xqs), shape=5), [1.0, 2.0]))
    pq = np.array([0.01, 0.05, 0.1, 0.4, 0.6, 0.9, 0.95, 0.99])
    xqs = [stats.scoreatpercentile(trvs, p) for p in pq * 100]
    tparest_gmmquantile = optimize.fmin(lambda params: np.sum(momentcondquant(distfn, params, mom2s, (pq, xqs), shape=None) ** 2), [10, 1.0, 2.0])
    print('tparest_gmmquantile', tparest_gmmquantile)
    tparest_gmmquantile2 = fitquantilesgmm(distfn, trvs, start=[10, 1.0, 2.0], pquant=None, frozen=None)
    print('tparest_gmmquantile2', tparest_gmmquantile2)
    bt = stats.t.ppf(np.linspace(0, 1, 21), 5)
    (ft, bt) = np.histogram(trvs, bins=bt)
    print('fitbinned t-distribution')
    tparest_mlebinew = fitbinned(stats.t, ft, bt, [10, 0, 1])
    tparest_gmmbinewidentity = fitbinnedgmm(stats.t, ft, bt, [10, 0, 1])
    tparest_gmmbinewoptimal = fitbinnedgmm(stats.t, ft, bt, [10, 0, 1], weightsoptimal=False)
    print(paramsdgp)
    (ft2, bt2) = np.histogram(trvs, bins=50)
    'fitbinned t-distribution'
    tparest_mlebinel = fitbinned(stats.t, ft2, bt2, [10, 0, 1])
    tparest_gmmbinelidentity = fitbinnedgmm(stats.t, ft2, bt2, [10, 0, 1])
    tparest_gmmbineloptimal = fitbinnedgmm(stats.t, ft2, bt2, [10, 0, 1], weightsoptimal=False)
    tparest_mle = stats.t.fit(trvs)
    np.set_printoptions(precision=6)
    print('sample size', nobs)
    print('true (df, loc, scale)      ', paramsdgp)
    print('parest_mle                 ', tparest_mle)
    print
    print('tparest_mlebinel           ', tparest_mlebinel)
    print('tparest_gmmbinelidentity   ', tparest_gmmbinelidentity)
    print('tparest_gmmbineloptimal    ', tparest_gmmbineloptimal)
    print
    print('tparest_mlebinew           ', tparest_mlebinew)
    print('tparest_gmmbinewidentity   ', tparest_gmmbinewidentity)
    print('tparest_gmmbinewoptimal    ', tparest_gmmbinewoptimal)
    print
    print('tparest_gmmquantileidentity', tparest_gmmquantile)
    print('tparest_gmm3quantilefsolve ', tparest_gmm3quantilefsolve)
    print('tparest_gmm3quantile       ', tparest_gmm3quantile)
    ' example results:\n    standard error for df estimate looks large\n    note: iI do not impose that df is an integer, (b/c not necessary)\n    need Monte Carlo to check variance of estimators\n\n\n    sample size 1000\n    true (df, loc, scale)       (5, 0, 1)\n    parest_mle                  [ 4.571405 -0.021493  1.028584]\n\n    tparest_mlebinel            [ 4.534069 -0.022605  1.02962 ]\n    tparest_gmmbinelidentity    [ 2.653056  0.012807  0.896958]\n    tparest_gmmbineloptimal     [ 2.437261 -0.020491  0.923308]\n\n    tparest_mlebinew            [ 2.999124 -0.0199    0.948811]\n    tparest_gmmbinewidentity    [ 2.900939 -0.020159  0.93481 ]\n    tparest_gmmbinewoptimal     [ 2.977764 -0.024925  0.946487]\n\n    tparest_gmmquantileidentity [ 3.940797 -0.046469  1.002001]\n    tparest_gmm3quantilefsolve  [ 10.   1.   2.]\n    tparest_gmm3quantile        [ 6.376101 -0.029322  1.112403]\n    '
    print('\n\nExample: Lognormal Distribution')
    print('-------------------------------')
    sh = np.exp(10)
    sh = 0.01
    print(sh)
    x = stats.lognorm.rvs(sh, loc=100, scale=10, size=200)
    print(x.min())
    print(stats.lognorm.fit(x, 1.0, loc=x.min() - 1, scale=1))
    xsorted = np.sort(x)
    x0 = [1.0, x.min() - 1, 1]
    args = (xsorted, stats.lognorm)
    print(optimize.fmin(logmps, x0, args=args))
    print('\n\nExample: Lomax, Pareto, Generalized Pareto Distributions')
    print('--------------------------------------------------------')
    p2rvs = stats.genpareto.rvs(2, size=500)
    p2rvssorted = np.sort(p2rvs)
    argsp = (p2rvssorted, stats.pareto)
    x0p = [1.0, p2rvs.min() - 5, 1]
    print(optimize.fmin(logmps, x0p, args=argsp))
    print(stats.pareto.fit(p2rvs, 0.5, loc=-20, scale=0.5))
    print('gpdparest_ mle', stats.genpareto.fit(p2rvs))
    parsgpd = fit_mps(stats.genpareto, p2rvs)
    print('gpdparest_ mps', parsgpd)
    argsgpd = (p2rvssorted, stats.genpareto)
    options = dict(stepFix=1e-07)
    (he, h) = hess_ndt(logmps, parsgpd, argsgpd, options)
    print(np.linalg.eigh(he)[0])
    f = lambda params: logmps(params, *argsgpd)
    print(f(parsgpd))
    (fp2, bp2) = np.histogram(p2rvs, bins=50)
    'fitbinned t-distribution'
    gpdparest_mlebinel = fitbinned(stats.genpareto, fp2, bp2, x0p)
    gpdparest_gmmbinelidentity = fitbinnedgmm(stats.genpareto, fp2, bp2, x0p)
    print('gpdparest_mlebinel', gpdparest_mlebinel)
    print('gpdparest_gmmbinelidentity', gpdparest_gmmbinelidentity)
    gpdparest_gmmquantile2 = fitquantilesgmm(stats.genpareto, p2rvs, start=x0p, pquant=None, frozen=None)
    print('gpdparest_gmmquantile2', gpdparest_gmmquantile2)
    print(fitquantilesgmm(stats.genpareto, p2rvs, start=x0p, pquant=np.linspace(0.01, 0.99, 10), frozen=None))
    (fp2, bp2) = np.histogram(p2rvs, bins=stats.genpareto(2).ppf(np.linspace(0, 0.99, 10)))
    print('fitbinnedgmm equal weight bins')
    print(fitbinnedgmm(stats.genpareto, fp2, bp2, x0p))