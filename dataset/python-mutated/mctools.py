"""Helper class for Monte Carlo Studies for (currently) statistical tests

Most of it should also be usable for Bootstrap, and for MC for estimators.
Takes the sample generator, dgb, and the statistical results, statistic,
as functions in the argument.


Author: Josef Perktold (josef-pktd)
License: BSD-3


TODOs, Design
-------------
If we only care about univariate analysis, i.e. marginal if statistics returns
more than one value, the we only need to store the sorted mcres not the
original res. Do we want to extend to multivariate analysis?

Use distribution function to keep track of MC results, ECDF, non-paramatric?
Large parts are similar to a 2d array of independent multivariate random
variables. Joint distribution is not used (yet).

I guess this is currently only for one sided test statistics, e.g. for
two-sided tests basend on t or normal distribution use the absolute value.

"""
from statsmodels.compat.python import lrange
import numpy as np
from statsmodels.iolib.table import SimpleTable

class StatTestMC:
    """class to run Monte Carlo study on a statistical test'''

    TODO
    print(summary, for quantiles and for histogram
    draft in trying out script log

    Parameters
    ----------
    dgp : callable
        Function that generates the data to be used in Monte Carlo that should
        return a new sample with each call
    statistic : callable
        Function that calculates the test statistic, which can return either
        a single statistic or a 1d array_like (tuple, list, ndarray).
        see also statindices in description of run

    Attributes
    ----------
    many methods store intermediate results

    self.mcres : ndarray (nrepl, nreturns) or (nrepl, len(statindices))
        Monte Carlo results stored by run


    Notes
    -----

    .. Warning::
       This is (currently) designed for a single call to run. If run is
       called a second time with different arguments, then some attributes might
       not be updated, and, therefore, not correspond to the same run.

    .. Warning::
       Under Construction, do not expect stability in Api or implementation


    Examples
    --------

    Define a function that defines our test statistic:

    def lb(x):
        s,p = acorr_ljungbox(x, lags=4)
        return np.r_[s, p]

    Note lb returns eight values.

    Define a random sample generator, for example 500 independently, normal
    distributed observations in a sample:


    def normalnoisesim(nobs=500, loc=0.0):
        return (loc+np.random.randn(nobs))

    Create instance and run Monte Carlo. Using statindices=list(range(4)) means that
    only the first for values of the return of the statistic (lb) are stored
    in the Monte Carlo results.

    mc1 = StatTestMC(normalnoisesim, lb)
    mc1.run(5000, statindices=list(range(4)))

    Most of the other methods take an idx which indicates for which columns
    the results should be presented, e.g.

    print(mc1.cdf(crit, [1,2,3])[1]
    """

    def __init__(self, dgp, statistic):
        if False:
            return 10
        self.dgp = dgp
        self.statistic = statistic

    def run(self, nrepl, statindices=None, dgpargs=[], statsargs=[]):
        if False:
            return 10
        'run the actual Monte Carlo and save results\n\n        Parameters\n        ----------\n        nrepl : int\n            number of Monte Carlo repetitions\n        statindices : None or list of integers\n           determines which values of the return of the statistic\n           functions are stored in the Monte Carlo. Default None\n           means the entire return. If statindices is a list of\n           integers, then it will be used as index into the return.\n        dgpargs : tuple\n           optional parameters for the DGP\n        statsargs : tuple\n           optional parameters for the statistics function\n\n        Returns\n        -------\n        None, all results are attached\n\n\n        '
        self.nrepl = nrepl
        self.statindices = statindices
        self.dgpargs = dgpargs
        self.statsargs = statsargs
        dgp = self.dgp
        statfun = self.statistic
        mcres0 = statfun(dgp(*dgpargs), *statsargs)
        self.nreturn = nreturns = len(np.ravel(mcres0))
        if statindices is None:
            mcres = np.zeros(nrepl)
            mcres[0] = mcres0
            for ii in range(1, nrepl - 1, nreturns):
                x = dgp(*dgpargs)
                mcres[ii] = statfun(x, *statsargs)
        else:
            self.nreturn = nreturns = len(statindices)
            self.mcres = mcres = np.zeros((nrepl, nreturns))
            mcres[0] = [mcres0[i] for i in statindices]
            for ii in range(1, nrepl - 1):
                x = dgp(*dgpargs)
                ret = statfun(x, *statsargs)
                mcres[ii] = [ret[i] for i in statindices]
        self.mcres = mcres

    def histogram(self, idx=None, critval=None):
        if False:
            i = 10
            return i + 15
        'calculate histogram values\n\n        does not do any plotting\n\n        I do not remember what I wanted here, looks similar to the new cdf\n        method, but this also does a binned pdf (self.histo)\n\n\n        '
        if self.mcres.ndim == 2:
            if idx is not None:
                mcres = self.mcres[:, idx]
            else:
                raise ValueError('currently only 1 statistic at a time')
        else:
            mcres = self.mcres
        if critval is None:
            histo = np.histogram(mcres, bins=10)
        else:
            if not critval[0] == -np.inf:
                bins = np.r_[-np.inf, critval, np.inf]
            if not critval[0] == -np.inf:
                bins = np.r_[bins, np.inf]
            histo = np.histogram(mcres, bins=np.r_[-np.inf, critval, np.inf])
        self.histo = histo
        self.cumhisto = np.cumsum(histo[0]) * 1.0 / self.nrepl
        self.cumhistoreversed = np.cumsum(histo[0][::-1])[::-1] * 1.0 / self.nrepl
        return (histo, self.cumhisto, self.cumhistoreversed)

    def get_mc_sorted(self):
        if False:
            i = 10
            return i + 15
        if not hasattr(self, 'mcressort'):
            self.mcressort = np.sort(self.mcres, axis=0)
        return self.mcressort

    def quantiles(self, idx=None, frac=[0.01, 0.025, 0.05, 0.1, 0.975]):
        if False:
            for i in range(10):
                print('nop')
        'calculate quantiles of Monte Carlo results\n\n        similar to ppf\n\n        Parameters\n        ----------\n        idx : None or list of integers\n            List of indices into the Monte Carlo results (columns) that should\n            be used in the calculation\n        frac : array_like, float\n            Defines which quantiles should be calculated. For example a frac\n            of 0.1 finds the 10% quantile, x such that cdf(x)=0.1\n\n        Returns\n        -------\n        frac : ndarray\n            same values as input, TODO: I should drop this again ?\n        quantiles : ndarray, (len(frac), len(idx))\n            the quantiles with frac in rows and idx variables in columns\n\n        Notes\n        -----\n\n        rename to ppf ? make frac required\n        change sequence idx, frac\n\n\n        '
        if self.mcres.ndim == 2:
            if idx is not None:
                mcres = self.mcres[:, idx]
            else:
                raise ValueError('currently only 1 statistic at a time')
        else:
            mcres = self.mcres
        self.frac = frac = np.asarray(frac)
        mc_sorted = self.get_mc_sorted()[:, idx]
        return (frac, mc_sorted[(self.nrepl * frac).astype(int)])

    def cdf(self, x, idx=None):
        if False:
            i = 10
            return i + 15
        'calculate cumulative probabilities of Monte Carlo results\n\n        Parameters\n        ----------\n        idx : None or list of integers\n            List of indices into the Monte Carlo results (columns) that should\n            be used in the calculation\n        frac : array_like, float\n            Defines which quantiles should be calculated. For example a frac\n            of 0.1 finds the 10% quantile, x such that cdf(x)=0.1\n\n        Returns\n        -------\n        x : ndarray\n            same as input, TODO: I should drop this again ?\n        probs : ndarray, (len(x), len(idx))\n            the quantiles with frac in rows and idx variables in columns\n\n\n\n        '
        idx = np.atleast_1d(idx).tolist()
        mc_sorted = self.get_mc_sorted()
        x = np.asarray(x)
        if x.ndim > 1 and x.shape[1] == len(idx):
            use_xi = True
        else:
            use_xi = False
        x_ = x
        probs = []
        for (i, ix) in enumerate(idx):
            if use_xi:
                x_ = x[:, i]
            probs.append(np.searchsorted(mc_sorted[:, ix], x_) / float(self.nrepl))
        probs = np.asarray(probs).T
        return (x, probs)

    def plot_hist(self, idx, distpdf=None, bins=50, ax=None, kwds=None):
        if False:
            i = 10
            return i + 15
        'plot the histogram against a reference distribution\n\n        Parameters\n        ----------\n        idx : None or list of integers\n            List of indices into the Monte Carlo results (columns) that should\n            be used in the calculation\n        distpdf : callable\n            probability density function of reference distribution\n        bins : {int, array_like}\n            used unchanged for matplotlibs hist call\n        ax : TODO: not implemented yet\n        kwds : None or tuple of dicts\n            extra keyword options to the calls to the matplotlib functions,\n            first dictionary is for his, second dictionary for plot of the\n            reference distribution\n\n        Returns\n        -------\n        None\n\n\n        '
        if kwds is None:
            kwds = ({}, {})
        if self.mcres.ndim == 2:
            if idx is not None:
                mcres = self.mcres[:, idx]
            else:
                raise ValueError('currently only 1 statistic at a time')
        else:
            mcres = self.mcres
        lsp = np.linspace(mcres.min(), mcres.max(), 100)
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.hist(mcres, bins=bins, normed=True, **kwds[0])
        plt.plot(lsp, distpdf(lsp), 'r', **kwds[1])

    def summary_quantiles(self, idx, distppf, frac=[0.01, 0.025, 0.05, 0.1, 0.975], varnames=None, title=None):
        if False:
            while True:
                i = 10
        'summary table for quantiles (critical values)\n\n        Parameters\n        ----------\n        idx : None or list of integers\n            List of indices into the Monte Carlo results (columns) that should\n            be used in the calculation\n        distppf : callable\n            probability density function of reference distribution\n            TODO: use `crit` values instead or additional, see summary_cdf\n        frac : array_like, float\n            probabilities for which\n        varnames : None, or list of strings\n            optional list of variable names, same length as idx\n\n        Returns\n        -------\n        table : instance of SimpleTable\n            use `print(table` to see results\n\n        '
        idx = np.atleast_1d(idx)
        (quant, mcq) = self.quantiles(idx, frac=frac)
        crit = distppf(np.atleast_2d(quant).T)
        mml = []
        for (i, ix) in enumerate(idx):
            mml.extend([mcq[:, i], crit[:, i]])
        mmlar = np.column_stack([quant] + mml)
        if title:
            title = title + ' Quantiles (critical values)'
        else:
            title = 'Quantiles (critical values)'
        if varnames is None:
            varnames = ['var%d' % i for i in range(mmlar.shape[1] // 2)]
        headers = ['\nprob'] + ['%s\n%s' % (i, t) for i in varnames for t in ['mc', 'dist']]
        return SimpleTable(mmlar, txt_fmt={'data_fmts': ['%#6.3f'] + ['%#10.4f'] * (mmlar.shape[1] - 1)}, title=title, headers=headers)

    def summary_cdf(self, idx, frac, crit, varnames=None, title=None):
        if False:
            print('Hello World!')
        'summary table for cumulative density function\n\n\n        Parameters\n        ----------\n        idx : None or list of integers\n            List of indices into the Monte Carlo results (columns) that should\n            be used in the calculation\n        frac : array_like, float\n            probabilities for which\n        crit : array_like\n            values for which cdf is calculated\n        varnames : None, or list of strings\n            optional list of variable names, same length as idx\n\n        Returns\n        -------\n        table : instance of SimpleTable\n            use `print(table` to see results\n\n\n        '
        idx = np.atleast_1d(idx)
        mml = []
        for i in range(len(idx)):
            mml.append(self.cdf(crit[:, i], [idx[i]])[1].ravel())
        mmlar = np.column_stack([frac] + mml)
        if title:
            title = title + ' Probabilites'
        else:
            title = 'Probabilities'
        if varnames is None:
            varnames = ['var%d' % i for i in range(mmlar.shape[1] - 1)]
        headers = ['prob'] + varnames
        return SimpleTable(mmlar, txt_fmt={'data_fmts': ['%#6.3f'] + ['%#10.4f'] * (np.array(mml).shape[1] - 1)}, title=title, headers=headers)
if __name__ == '__main__':
    from scipy import stats
    from statsmodels.stats.diagnostic import acorr_ljungbox

    def randwalksim(nobs=100, drift=0.0):
        if False:
            while True:
                i = 10
        return (drift + np.random.randn(nobs)).cumsum()

    def normalnoisesim(nobs=500, loc=0.0):
        if False:
            for i in range(10):
                print('nop')
        return loc + np.random.randn(nobs)
    print('\nLjung Box')

    def lb4(x):
        if False:
            i = 10
            return i + 15
        (s, p) = acorr_ljungbox(x, lags=4, return_df=True)
        return (s[-1], p[-1])

    def lb1(x):
        if False:
            for i in range(10):
                print('nop')
        (s, p) = acorr_ljungbox(x, lags=1, return_df=True)
        return (s[0], p[0])

    def lb(x):
        if False:
            return 10
        (s, p) = acorr_ljungbox(x, lags=4, return_df=True)
        return np.r_[s, p]
    print('Results with MC class')
    mc1 = StatTestMC(normalnoisesim, lb)
    mc1.run(10000, statindices=lrange(8))
    print(mc1.histogram(1, critval=[0.01, 0.025, 0.05, 0.1, 0.975]))
    print(mc1.quantiles(1))
    print(mc1.quantiles(0))
    print(mc1.histogram(0))
    print(mc1.summary_quantiles([1, 2, 3], stats.chi2([2, 3, 4]).ppf, varnames=['lag 1', 'lag 2', 'lag 3'], title='acorr_ljungbox'))
    print(mc1.cdf(0.1026, 1))
    print(mc1.cdf(0.7278, 3))
    print(mc1.cdf(0.7278, [1, 2, 3]))
    frac = [0.01, 0.025, 0.05, 0.1, 0.975]
    crit = stats.chi2([2, 4]).ppf(np.atleast_2d(frac).T)
    print(mc1.summary_cdf([1, 3], frac, crit, title='acorr_ljungbox'))
    crit = stats.chi2([2, 3, 4]).ppf(np.atleast_2d(frac).T)
    print(mc1.summary_cdf([1, 2, 3], frac, crit, varnames=['lag 1', 'lag 2', 'lag 3'], title='acorr_ljungbox'))
    print(mc1.cdf(crit, [1, 2, 3])[1].shape)
    '\n    >>> mc1.cdf(crit[:,0], [1])[1].shape\n    (5, 1)\n    >>> mc1.cdf(crit[:,0], [1,3])[1].shape\n    (5, 2)\n    >>> mc1.cdf(crit[:,:], [1,3])[1].shape\n    (2, 5, 2)\n    '
    doplot = 0
    if doplot:
        import matplotlib.pyplot as plt
        mc1.plot_hist(0, stats.chi2(2).pdf)
        plt.show()