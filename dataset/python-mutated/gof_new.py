"""More Goodness of fit tests

contains

GOF : 1 sample gof tests based on Stephens 1970, plus AD A^2
bootstrap : vectorized bootstrap p-values for gof test with fitted parameters


Created : 2011-05-21
Author : Josef Perktold

parts based on ks_2samp and kstest from scipy.stats
(license: Scipy BSD, but were completely rewritten by Josef Perktold)


References
----------

"""
from statsmodels.compat.python import lmap
import numpy as np
from scipy.stats import distributions
from statsmodels.tools.decorators import cache_readonly
from scipy.special import kolmogorov as ksprob

def ks_2samp(data1, data2):
    if False:
        return 10
    '\n    Computes the Kolmogorov-Smirnof statistic on 2 samples.\n\n    This is a two-sided test for the null hypothesis that 2 independent samples\n    are drawn from the same continuous distribution.\n\n    Parameters\n    ----------\n    a, b : sequence of 1-D ndarrays\n        two arrays of sample observations assumed to be drawn from a continuous\n        distribution, sample sizes can be different\n\n\n    Returns\n    -------\n    D : float\n        KS statistic\n    p-value : float\n        two-tailed p-value\n\n\n    Notes\n    -----\n\n    This tests whether 2 samples are drawn from the same distribution. Note\n    that, like in the case of the one-sample K-S test, the distribution is\n    assumed to be continuous.\n\n    This is the two-sided test, one-sided tests are not implemented.\n    The test uses the two-sided asymptotic Kolmogorov-Smirnov distribution.\n\n    If the K-S statistic is small or the p-value is high, then we cannot\n    reject the hypothesis that the distributions of the two samples\n    are the same.\n\n    Examples\n    --------\n\n    >>> from scipy import stats\n    >>> import numpy as np\n    >>> from scipy.stats import ks_2samp\n\n    >>> #fix random seed to get the same result\n    >>> np.random.seed(12345678)\n\n    >>> n1 = 200  # size of first sample\n    >>> n2 = 300  # size of second sample\n\n    different distribution\n    we can reject the null hypothesis since the pvalue is below 1%\n\n    >>> rvs1 = stats.norm.rvs(size=n1,loc=0.,scale=1)\n    >>> rvs2 = stats.norm.rvs(size=n2,loc=0.5,scale=1.5)\n    >>> ks_2samp(rvs1,rvs2)\n    (0.20833333333333337, 4.6674975515806989e-005)\n\n    slightly different distribution\n    we cannot reject the null hypothesis at a 10% or lower alpha since\n    the pvalue at 0.144 is higher than 10%\n\n    >>> rvs3 = stats.norm.rvs(size=n2,loc=0.01,scale=1.0)\n    >>> ks_2samp(rvs1,rvs3)\n    (0.10333333333333333, 0.14498781825751686)\n\n    identical distribution\n    we cannot reject the null hypothesis since the pvalue is high, 41%\n\n    >>> rvs4 = stats.norm.rvs(size=n2,loc=0.0,scale=1.0)\n    >>> ks_2samp(rvs1,rvs4)\n    (0.07999999999999996, 0.41126949729859719)\n    '
    (data1, data2) = lmap(np.asarray, (data1, data2))
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    n1 = len(data1)
    n2 = len(data2)
    data1 = np.sort(data1)
    data2 = np.sort(data2)
    data_all = np.concatenate([data1, data2])
    cdf1 = np.searchsorted(data1, data_all, side='right') / (1.0 * n1)
    cdf2 = np.searchsorted(data2, data_all, side='right') / (1.0 * n2)
    d = np.max(np.absolute(cdf1 - cdf2))
    en = np.sqrt(n1 * n2 / float(n1 + n2))
    try:
        prob = ksprob((en + 0.12 + 0.11 / en) * d)
    except:
        prob = 1.0
    return (d, prob)

def kstest(rvs, cdf, args=(), N=20, alternative='two_sided', mode='approx', **kwds):
    if False:
        while True:
            i = 10
    '\n    Perform the Kolmogorov-Smirnov test for goodness of fit\n\n    This performs a test of the distribution G(x) of an observed\n    random variable against a given distribution F(x). Under the null\n    hypothesis the two distributions are identical, G(x)=F(x). The\n    alternative hypothesis can be either \'two_sided\' (default), \'less\'\n    or \'greater\'. The KS test is only valid for continuous distributions.\n\n    Parameters\n    ----------\n    rvs : str or array or callable\n        string: name of a distribution in scipy.stats\n\n        array: 1-D observations of random variables\n\n        callable: function to generate random variables, requires keyword\n        argument `size`\n\n    cdf : str or callable\n        string: name of a distribution in scipy.stats, if rvs is a string then\n        cdf can evaluate to `False` or be the same as rvs\n        callable: function to evaluate cdf\n\n    args : tuple, sequence\n        distribution parameters, used if rvs or cdf are strings\n    N : int\n        sample size if rvs is string or callable\n    alternative : \'two_sided\' (default), \'less\' or \'greater\'\n        defines the alternative hypothesis (see explanation)\n\n    mode : \'approx\' (default) or \'asymp\'\n        defines the distribution used for calculating p-value\n\n        \'approx\' : use approximation to exact distribution of test statistic\n\n        \'asymp\' : use asymptotic distribution of test statistic\n\n\n    Returns\n    -------\n    D : float\n        KS test statistic, either D, D+ or D-\n    p-value :  float\n        one-tailed or two-tailed p-value\n\n    Notes\n    -----\n\n    In the one-sided test, the alternative is that the empirical\n    cumulative distribution function of the random variable is "less"\n    or "greater" than the cumulative distribution function F(x) of the\n    hypothesis, G(x)<=F(x), resp. G(x)>=F(x).\n\n    Examples\n    --------\n\n    >>> from scipy import stats\n    >>> import numpy as np\n    >>> from scipy.stats import kstest\n\n    >>> x = np.linspace(-15,15,9)\n    >>> kstest(x,\'norm\')\n    (0.44435602715924361, 0.038850142705171065)\n\n    >>> np.random.seed(987654321) # set random seed to get the same result\n    >>> kstest(\'norm\',\'\',N=100)\n    (0.058352892479417884, 0.88531190944151261)\n\n    is equivalent to this\n\n    >>> np.random.seed(987654321)\n    >>> kstest(stats.norm.rvs(size=100),\'norm\')\n    (0.058352892479417884, 0.88531190944151261)\n\n    Test against one-sided alternative hypothesis:\n\n    >>> np.random.seed(987654321)\n\n    Shift distribution to larger values, so that cdf_dgp(x)< norm.cdf(x):\n\n    >>> x = stats.norm.rvs(loc=0.2, size=100)\n    >>> kstest(x,\'norm\', alternative = \'less\')\n    (0.12464329735846891, 0.040989164077641749)\n\n    Reject equal distribution against alternative hypothesis: less\n\n    >>> kstest(x,\'norm\', alternative = \'greater\')\n    (0.0072115233216311081, 0.98531158590396395)\n\n    Do not reject equal distribution against alternative hypothesis: greater\n\n    >>> kstest(x,\'norm\', mode=\'asymp\')\n    (0.12464329735846891, 0.08944488871182088)\n\n\n    Testing t distributed random variables against normal distribution:\n\n    With 100 degrees of freedom the t distribution looks close to the normal\n    distribution, and the kstest does not reject the hypothesis that the sample\n    came from the normal distribution\n\n    >>> np.random.seed(987654321)\n    >>> stats.kstest(stats.t.rvs(100,size=100),\'norm\')\n    (0.072018929165471257, 0.67630062862479168)\n\n    With 3 degrees of freedom the t distribution looks sufficiently different\n    from the normal distribution, that we can reject the hypothesis that the\n    sample came from the normal distribution at a alpha=10% level\n\n    >>> np.random.seed(987654321)\n    >>> stats.kstest(stats.t.rvs(3,size=100),\'norm\')\n    (0.131016895759829, 0.058826222555312224)\n    '
    if isinstance(rvs, str):
        if not cdf or cdf == rvs:
            cdf = getattr(distributions, rvs).cdf
            rvs = getattr(distributions, rvs).rvs
        else:
            raise AttributeError('if rvs is string, cdf has to be the same distribution')
    if isinstance(cdf, str):
        cdf = getattr(distributions, cdf).cdf
    if callable(rvs):
        kwds = {'size': N}
        vals = np.sort(rvs(*args, **kwds))
    else:
        vals = np.sort(rvs)
        N = len(vals)
    cdfvals = cdf(vals, *args)
    if alternative in ['two_sided', 'greater']:
        Dplus = (np.arange(1.0, N + 1) / N - cdfvals).max()
        if alternative == 'greater':
            return (Dplus, distributions.ksone.sf(Dplus, N))
    if alternative in ['two_sided', 'less']:
        Dmin = (cdfvals - np.arange(0.0, N) / N).max()
        if alternative == 'less':
            return (Dmin, distributions.ksone.sf(Dmin, N))
    if alternative == 'two_sided':
        D = np.max([Dplus, Dmin])
        if mode == 'asymp':
            return (D, distributions.kstwobign.sf(D * np.sqrt(N)))
        if mode == 'approx':
            pval_two = distributions.kstwobign.sf(D * np.sqrt(N))
            if N > 2666 or pval_two > 0.8 - N * 0.3 / 1000.0:
                return (D, distributions.kstwobign.sf(D * np.sqrt(N)))
            else:
                return (D, distributions.ksone.sf(D, N) * 2)

def dplus_st70_upp(stat, nobs):
    if False:
        return 10
    mod_factor = np.sqrt(nobs) + 0.12 + 0.11 / np.sqrt(nobs)
    stat_modified = stat * mod_factor
    pval = np.exp(-2 * stat_modified ** 2)
    digits = np.sum(stat > np.array([0.82, 0.82, 1.0]))
    return (stat_modified, pval, digits)
dminus_st70_upp = dplus_st70_upp

def d_st70_upp(stat, nobs):
    if False:
        for i in range(10):
            print('nop')
    mod_factor = np.sqrt(nobs) + 0.12 + 0.11 / np.sqrt(nobs)
    stat_modified = stat * mod_factor
    pval = 2 * np.exp(-2 * stat_modified ** 2)
    digits = np.sum(stat > np.array([0.91, 0.91, 1.08]))
    return (stat_modified, pval, digits)

def v_st70_upp(stat, nobs):
    if False:
        for i in range(10):
            print('nop')
    mod_factor = np.sqrt(nobs) + 0.155 + 0.24 / np.sqrt(nobs)
    stat_modified = stat * mod_factor
    zsqu = stat_modified ** 2
    pval = (8 * zsqu - 2) * np.exp(-2 * zsqu)
    digits = np.sum(stat > np.array([1.06, 1.06, 1.26]))
    return (stat_modified, pval, digits)

def wsqu_st70_upp(stat, nobs):
    if False:
        i = 10
        return i + 15
    nobsinv = 1.0 / nobs
    stat_modified = (stat - 0.4 * nobsinv + 0.6 * nobsinv ** 2) * (1 + nobsinv)
    pval = 0.05 * np.exp(2.79 - 6 * stat_modified)
    digits = np.nan
    return (stat_modified, pval, digits)

def usqu_st70_upp(stat, nobs):
    if False:
        i = 10
        return i + 15
    nobsinv = 1.0 / nobs
    stat_modified = stat - 0.1 * nobsinv + 0.1 * nobsinv ** 2
    stat_modified *= 1 + 0.8 * nobsinv
    pval = 2 * np.exp(-2 * stat_modified * np.pi ** 2)
    digits = np.sum(stat > np.array([0.29, 0.29, 0.34]))
    return (stat_modified, pval, digits)

def a_st70_upp(stat, nobs):
    if False:
        return 10
    nobsinv = 1.0 / nobs
    stat_modified = stat - 0.7 * nobsinv + 0.9 * nobsinv ** 2
    stat_modified *= 1 + 1.23 * nobsinv
    pval = 1.273 * np.exp(-2 * stat_modified / 2.0 * np.pi ** 2)
    digits = np.sum(stat > np.array([0.11, 0.11, 0.452]))
    return (stat_modified, pval, digits)
gof_pvals = {}
gof_pvals['stephens70upp'] = {'d_plus': dplus_st70_upp, 'd_minus': dplus_st70_upp, 'd': d_st70_upp, 'v': v_st70_upp, 'wsqu': wsqu_st70_upp, 'usqu': usqu_st70_upp, 'a': a_st70_upp}

def pval_kstest_approx(D, N):
    if False:
        i = 10
        return i + 15
    pval_two = distributions.kstwobign.sf(D * np.sqrt(N))
    if N > 2666 or pval_two > 0.8 - N * 0.3 / 1000.0:
        return (D, distributions.kstwobign.sf(D * np.sqrt(N)), np.nan)
    else:
        return (D, distributions.ksone.sf(D, N) * 2, np.nan)
gof_pvals['scipy'] = {'d_plus': lambda Dplus, N: (Dplus, distributions.ksone.sf(Dplus, N), np.nan), 'd_minus': lambda Dmin, N: (Dmin, distributions.ksone.sf(Dmin, N), np.nan), 'd': lambda D, N: (D, distributions.kstwobign.sf(D * np.sqrt(N)), np.nan)}
gof_pvals['scipy_approx'] = {'d': pval_kstest_approx}

class GOF:
    """One Sample Goodness of Fit tests

    includes Kolmogorov-Smirnov D, D+, D-, Kuiper V, Cramer-von Mises W^2, U^2 and
    Anderson-Darling A, A^2. The p-values for all tests except for A^2 are based on
    the approximatiom given in Stephens 1970. A^2 has currently no p-values. For
    the Kolmogorov-Smirnov test the tests as given in scipy.stats are also available
    as options.




    design: I might want to retest with different distributions, to calculate
    data summary statistics only once, or add separate class that holds
    summary statistics and data (sounds good).




    """

    def __init__(self, rvs, cdf, args=(), N=20):
        if False:
            return 10
        if isinstance(rvs, str):
            if not cdf or cdf == rvs:
                cdf = getattr(distributions, rvs).cdf
                rvs = getattr(distributions, rvs).rvs
            else:
                raise AttributeError('if rvs is string, cdf has to be the same distribution')
        if isinstance(cdf, str):
            cdf = getattr(distributions, cdf).cdf
        if callable(rvs):
            kwds = {'size': N}
            vals = np.sort(rvs(*args, **kwds))
        else:
            vals = np.sort(rvs)
            N = len(vals)
        cdfvals = cdf(vals, *args)
        self.nobs = N
        self.vals_sorted = vals
        self.cdfvals = cdfvals

    @cache_readonly
    def d_plus(self):
        if False:
            i = 10
            return i + 15
        nobs = self.nobs
        cdfvals = self.cdfvals
        return (np.arange(1.0, nobs + 1) / nobs - cdfvals).max()

    @cache_readonly
    def d_minus(self):
        if False:
            for i in range(10):
                print('nop')
        nobs = self.nobs
        cdfvals = self.cdfvals
        return (cdfvals - np.arange(0.0, nobs) / nobs).max()

    @cache_readonly
    def d(self):
        if False:
            for i in range(10):
                print('nop')
        return np.max([self.d_plus, self.d_minus])

    @cache_readonly
    def v(self):
        if False:
            return 10
        'Kuiper'
        return self.d_plus + self.d_minus

    @cache_readonly
    def wsqu(self):
        if False:
            i = 10
            return i + 15
        'Cramer von Mises'
        nobs = self.nobs
        cdfvals = self.cdfvals
        wsqu = ((cdfvals - (2.0 * np.arange(1.0, nobs + 1) - 1) / nobs / 2.0) ** 2).sum() + 1.0 / nobs / 12.0
        return wsqu

    @cache_readonly
    def usqu(self):
        if False:
            return 10
        nobs = self.nobs
        cdfvals = self.cdfvals
        usqu = self.wsqu - nobs * (cdfvals.mean() - 0.5) ** 2
        return usqu

    @cache_readonly
    def a(self):
        if False:
            i = 10
            return i + 15
        nobs = self.nobs
        cdfvals = self.cdfvals
        msum = 0
        for j in range(1, nobs):
            mj = cdfvals[j] - cdfvals[:j]
            mask = mj > 0.5
            mj[mask] = 1 - mj[mask]
            msum += mj.sum()
        a = nobs / 4.0 - 2.0 / nobs * msum
        return a

    @cache_readonly
    def asqu(self):
        if False:
            for i in range(10):
                print('nop')
        'Stephens 1974, does not have p-value formula for A^2'
        nobs = self.nobs
        cdfvals = self.cdfvals
        asqu = -((2.0 * np.arange(1.0, nobs + 1) - 1) * (np.log(cdfvals) + np.log(1 - cdfvals[::-1]))).sum() / nobs - nobs
        return asqu

    def get_test(self, testid='d', pvals='stephens70upp'):
        if False:
            while True:
                i = 10
        '\n\n        '
        stat = getattr(self, testid)
        if pvals == 'stephens70upp':
            return (gof_pvals[pvals][testid](stat, self.nobs), stat)
        else:
            return gof_pvals[pvals][testid](stat, self.nobs)

def gof_mc(randfn, distr, nobs=100):
    if False:
        i = 10
        return i + 15
    from collections import defaultdict
    results = defaultdict(list)
    for i in range(1000):
        rvs = randfn(nobs)
        goft = GOF(rvs, distr)
        for ti in all_gofs:
            results[ti].append(goft.get_test(ti, 'stephens70upp')[0][1])
    resarr = np.array([results[ti] for ti in all_gofs])
    print('         ', '      '.join(all_gofs))
    print('at 0.01:', (resarr < 0.01).mean(1))
    print('at 0.05:', (resarr < 0.05).mean(1))
    print('at 0.10:', (resarr < 0.1).mean(1))

def asquare(cdfvals, axis=0):
    if False:
        i = 10
        return i + 15
    'vectorized Anderson Darling A^2, Stephens 1974'
    ndim = len(cdfvals.shape)
    nobs = cdfvals.shape[axis]
    slice_reverse = [slice(None)] * ndim
    islice = [None] * ndim
    islice[axis] = slice(None)
    slice_reverse[axis] = slice(None, None, -1)
    asqu = -((2.0 * np.arange(1.0, nobs + 1)[tuple(islice)] - 1) * (np.log(cdfvals) + np.log(1 - cdfvals[tuple(slice_reverse)])) / nobs).sum(axis) - nobs
    return asqu

def bootstrap(distr, args=(), nobs=200, nrep=100, value=None, batch_size=None):
    if False:
        for i in range(10):
            print('nop')
    'Monte Carlo (or parametric bootstrap) p-values for gof\n\n    currently hardcoded for A^2 only\n\n    assumes vectorized fit_vec method,\n    builds and analyses (nobs, nrep) sample in one step\n\n    rename function to less generic\n\n    this works also with nrep=1\n\n    '
    if batch_size is not None:
        if value is None:
            raise ValueError('using batching requires a value')
        n_batch = int(np.ceil(nrep / float(batch_size)))
        count = 0
        for irep in range(n_batch):
            rvs = distr.rvs(args, **{'size': (batch_size, nobs)})
            params = distr.fit_vec(rvs, axis=1)
            params = lmap(lambda x: np.expand_dims(x, 1), params)
            cdfvals = np.sort(distr.cdf(rvs, params), axis=1)
            stat = asquare(cdfvals, axis=1)
            count += (stat >= value).sum()
        return count / float(n_batch * batch_size)
    else:
        rvs = distr.rvs(args, **{'size': (nrep, nobs)})
        params = distr.fit_vec(rvs, axis=1)
        params = lmap(lambda x: np.expand_dims(x, 1), params)
        cdfvals = np.sort(distr.cdf(rvs, params), axis=1)
        stat = asquare(cdfvals, axis=1)
        if value is None:
            stat_sorted = np.sort(stat)
            return stat_sorted
        else:
            return (stat >= value).mean()

def bootstrap2(value, distr, args=(), nobs=200, nrep=100):
    if False:
        while True:
            i = 10
    'Monte Carlo (or parametric bootstrap) p-values for gof\n\n    currently hardcoded for A^2 only\n\n    non vectorized, loops over all parametric bootstrap replications and calculates\n    and returns specific p-value,\n\n    rename function to less generic\n\n    '
    count = 0
    for irep in range(nrep):
        rvs = distr.rvs(args, **{'size': nobs})
        params = distr.fit_vec(rvs)
        cdfvals = np.sort(distr.cdf(rvs, params))
        stat = asquare(cdfvals, axis=0)
        count += stat >= value
    return count * 1.0 / nrep

class NewNorm:
    """just a holder for modified distributions
    """

    def fit_vec(self, x, axis=0):
        if False:
            return 10
        return (x.mean(axis), x.std(axis))

    def cdf(self, x, args):
        if False:
            return 10
        return distributions.norm.cdf(x, loc=args[0], scale=args[1])

    def rvs(self, args, size):
        if False:
            for i in range(10):
                print('nop')
        loc = args[0]
        scale = args[1]
        return loc + scale * distributions.norm.rvs(size=size)
if __name__ == '__main__':
    from scipy import stats
    rvs = stats.t.rvs(3, size=200)
    print('scipy kstest')
    print(kstest(rvs, 'norm'))
    goft = GOF(rvs, 'norm')
    print(goft.get_test())
    all_gofs = ['d', 'd_plus', 'd_minus', 'v', 'wsqu', 'usqu', 'a']
    for ti in all_gofs:
        print(ti, goft.get_test(ti, 'stephens70upp'))
    print('\nIs it correctly sized?')
    from collections import defaultdict
    results = defaultdict(list)
    nobs = 200
    for i in range(100):
        rvs = np.random.randn(nobs)
        goft = GOF(rvs, 'norm')
        for ti in all_gofs:
            results[ti].append(goft.get_test(ti, 'stephens70upp')[0][1])
    resarr = np.array([results[ti] for ti in all_gofs])
    print('         ', '      '.join(all_gofs))
    print('at 0.01:', (resarr < 0.01).mean(1))
    print('at 0.05:', (resarr < 0.05).mean(1))
    print('at 0.10:', (resarr < 0.1).mean(1))
    gof_mc(lambda nobs: stats.t.rvs(3, size=nobs), 'norm', nobs=200)
    nobs = 200
    nrep = 100
    bt = bootstrap(NewNorm(), args=(0, 1), nobs=nobs, nrep=nrep, value=None)
    quantindex = np.floor(nrep * np.array([0.99, 0.95, 0.9])).astype(int)
    print(bt[quantindex])
    '\n    >>> np.array([15.0, 10.0, 5.0, 2.5, 1.0])/100.  #Stephens\n    array([ 0.15 ,  0.1  ,  0.05 ,  0.025,  0.01 ])\n    >>> nobs = 100\n    >>> [bootstrap(NewNorm(), args=(0,1), nobs=nobs, nrep=10000, value=c/ (1 + 4./nobs - 25./nobs**2)) for c in [0.576, 0.656, 0.787, 0.918, 1.092]]\n    [0.1545, 0.10009999999999999, 0.049000000000000002, 0.023, 0.0104]\n    >>>\n    '