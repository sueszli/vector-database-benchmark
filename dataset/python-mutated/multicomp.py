"""

from pystatsmodels mailinglist 20100524

Notes:
 - unfinished, unverified, but most parts seem to work in MonteCarlo
 - one example taken from lecture notes looks ok
 - needs cases with non-monotonic inequality for test to see difference between
   one-step, step-up and step-down procedures
 - FDR does not look really better then Bonferoni in the MC examples that I tried
update:
 - now tested against R, stats and multtest,
   I have all of their methods for p-value correction
 - getting Hommel was impossible until I found reference for pvalue correction
 - now, since I have p-values correction, some of the original tests (rej/norej)
   implementation is not really needed anymore. I think I keep it for reference.
   Test procedure for Hommel in development session log
 - I have not updated other functions and classes in here.
   - multtest has some good helper function according to docs
 - still need to update references, the real papers
 - fdr with estimated true hypothesis still missing
 - multiple comparison procedures incomplete or missing
 - I will get multiple comparison for now only for independent case, which might
   be conservative in correlated case (?).


some References:

Gibbons, Jean Dickinson and Chakraborti Subhabrata, 2003, Nonparametric Statistical
Inference, Fourth Edition, Marcel Dekker
    p.363: 10.4 THE KRUSKAL-WALLIS ONE-WAY ANOVA TEST AND MULTIPLE COMPARISONS
    p.367: multiple comparison for kruskal formula used in multicomp.kruskal

Sheskin, David J., 2004, Handbook of Parametric and Nonparametric Statistical
Procedures, 3rd ed., Chapman&Hall/CRC
    Test 21: The Single-Factor Between-Subjects Analysis of Variance
    Test 22: The Kruskal-Wallis One-Way Analysis of Variance by Ranks Test

Zwillinger, Daniel and Stephen Kokoska, 2000, CRC standard probability and
statistics tables and formulae, Chapman&Hall/CRC
    14.9 WILCOXON RANKSUM (MANN WHITNEY) TEST


S. Paul Wright, Adjusted P-Values for Simultaneous Inference, Biometrics
    Vol. 48, No. 4 (Dec., 1992), pp. 1005-1013, International Biometric Society
    Stable URL: http://www.jstor.org/stable/2532694
 (p-value correction for Hommel in appendix)

for multicomparison

new book "multiple comparison in R"
Hsu is a good reference but I do not have it.


Author: Josef Pktd and example from H Raja and rewrite from Vincent Davis


TODO
----
* name of function multipletests, rename to something like pvalue_correction?


"""
from collections import namedtuple
from statsmodels.compat.python import lzip, lrange
import copy
import math
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from scipy import stats, interpolate
from statsmodels.iolib.table import SimpleTable
from statsmodels.stats.multitest import multipletests, _ecdf as ecdf, fdrcorrection as fdrcorrection0, fdrcorrection_twostage
from statsmodels.graphics import utils
from statsmodels.tools.sm_exceptions import ValueWarning
try:
    from scipy.stats import studentized_range
except ImportError:
    from statsmodels.stats.libqsturng import qsturng, psturng
    studentized_range_tuple = namedtuple('studentized_range', ['ppf', 'sf'])
    studentized_range = studentized_range_tuple(ppf=qsturng, sf=psturng)
qcrit = '\n  2     3     4     5     6     7     8     9     10\n5   3.64 5.70   4.60 6.98   5.22 7.80   5.67 8.42   6.03 8.91   6.33 9.32   6.58 9.67   6.80 9.97   6.99 10.24\n6   3.46 5.24   4.34 6.33   4.90 7.03   5.30 7.56   5.63 7.97   5.90 8.32   6.12 8.61   6.32 8.87   6.49 9.10\n7   3.34 4.95   4.16 5.92   4.68 6.54   5.06 7.01   5.36 7.37   5.61 7.68   5.82 7.94   6.00 8.17   6.16 8.37\n8   3.26 4.75   4.04 5.64   4.53 6.20   4.89 6.62   5.17 6.96   5.40 7.24       5.60 7.47   5.77 7.68   5.92 7.86\n9   3.20 4.60   3.95 5.43   4.41 5.96   4.76 6.35   5.02 6.66   5.24 6.91       5.43 7.13   5.59 7.33   5.74 7.49\n10  3.15 4.48   3.88 5.27   4.33 5.77   4.65 6.14   4.91 6.43   5.12 6.67       5.30 6.87   5.46 7.05   5.60 7.21\n11  3.11 4.39   3.82 5.15   4.26 5.62   4.57 5.97   4.82 6.25   5.03 6.48 5.20 6.67   5.35 6.84   5.49 6.99\n12  3.08 4.32   3.77 5.05   4.20 5.50   4.51 5.84   4.75 6.10   4.95 6.32 5.12 6.51   5.27 6.67   5.39 6.81\n13  3.06 4.26   3.73 4.96   4.15 5.40   4.45 5.73   4.69 5.98   4.88 6.19 5.05 6.37   5.19 6.53   5.32 6.67\n14  3.03 4.21   3.70 4.89   4.11 5.32   4.41 5.63   4.64 5.88   4.83 6.08 4.99 6.26   5.13 6.41   5.25 6.54\n15  3.01 4.17   3.67 4.84   4.08 5.25   4.37 5.56   4.59 5.80   4.78 5.99 4.94 6.16   5.08 6.31   5.20 6.44\n16  3.00 4.13   3.65 4.79   4.05 5.19   4.33 5.49   4.56 5.72   4.74 5.92 4.90 6.08   5.03 6.22   5.15 6.35\n17  2.98 4.10   3.63 4.74   4.02 5.14   4.30 5.43   4.52 5.66   4.70 5.85 4.86 6.01   4.99 6.15   5.11 6.27\n18  2.97 4.07   3.61 4.70   4.00 5.09   4.28 5.38   4.49 5.60   4.67 5.79 4.82 5.94   4.96 6.08   5.07 6.20\n19  2.96 4.05   3.59 4.67   3.98 5.05   4.25 5.33   4.47 5.55   4.65 5.73 4.79 5.89   4.92 6.02   5.04 6.14\n20  2.95 4.02   3.58 4.64   3.96 5.02   4.23 5.29   4.45 5.51   4.62 5.69 4.77 5.84   4.90 5.97   5.01 6.09\n24  2.92 3.96   3.53 4.55   3.90 4.91   4.17 5.17   4.37 5.37   4.54 5.54 4.68 5.69   4.81 5.81   4.92 5.92\n30  2.89 3.89   3.49 4.45   3.85 4.80   4.10 5.05   4.30 5.24   4.46 5.40 4.60 5.54   4.72 5.65   4.82 5.76\n40  2.86 3.82   3.44 4.37   3.79 4.70   4.04 4.93   4.23 5.11   4.39 5.26 4.52 5.39   4.63 5.50   4.73 5.60\n60  2.83 3.76   3.40 4.28   3.74 4.59   3.98 4.82   4.16 4.99   4.31 5.13 4.44 5.25   4.55 5.36   4.65 5.45\n120   2.80 3.70   3.36 4.20   3.68 4.50   3.92 4.71   4.10 4.87   4.24 5.01 4.36 5.12   4.47 5.21   4.56 5.30\ninfinity  2.77 3.64   3.31 4.12   3.63 4.40   3.86 4.60   4.03 4.76   4.17 4.88   4.29 4.99   4.39 5.08   4.47 5.16\n'
res = [line.split() for line in qcrit.replace('infinity', '9999').split('\n')]
c = np.array(res[2:-1]).astype(float)
ccols = np.arange(2, 11)
crows = c[:, 0]
cv005 = c[:, 1::2]
cv001 = c[:, 2::2]

def get_tukeyQcrit(k, df, alpha=0.05):
    if False:
        i = 10
        return i + 15
    "\n    return critical values for Tukey's HSD (Q)\n\n    Parameters\n    ----------\n    k : int in {2, ..., 10}\n        number of tests\n    df : int\n        degrees of freedom of error term\n    alpha : {0.05, 0.01}\n        type 1 error, 1-confidence level\n\n\n\n    not enough error checking for limitations\n    "
    if alpha == 0.05:
        intp = interpolate.interp1d(crows, cv005[:, k - 2])
    elif alpha == 0.01:
        intp = interpolate.interp1d(crows, cv001[:, k - 2])
    else:
        raise ValueError('only implemented for alpha equal to 0.01 and 0.05')
    return intp(df)

def get_tukeyQcrit2(k, df, alpha=0.05):
    if False:
        print('Hello World!')
    "\n    return critical values for Tukey's HSD (Q)\n\n    Parameters\n    ----------\n    k : int in {2, ..., 10}\n        number of tests\n    df : int\n        degrees of freedom of error term\n    alpha : {0.05, 0.01}\n        type 1 error, 1-confidence level\n\n\n\n    not enough error checking for limitations\n    "
    return studentized_range.ppf(1 - alpha, k, df)

def get_tukey_pvalue(k, df, q):
    if False:
        print('Hello World!')
    "\n    return adjusted p-values for Tukey's HSD\n\n    Parameters\n    ----------\n    k : int in {2, ..., 10}\n        number of tests\n    df : int\n        degrees of freedom of error term\n    q : scalar, array_like; q >= 0\n        quantile value of Studentized Range\n\n    "
    return studentized_range.sf(q, k, df)

def Tukeythreegene(first, second, third):
    if False:
        return 10
    firstmean = np.mean(first)
    secondmean = np.mean(second)
    thirdmean = np.mean(third)
    firststd = np.std(first)
    secondstd = np.std(second)
    thirdstd = np.std(third)
    firsts2 = math.pow(firststd, 2)
    seconds2 = math.pow(secondstd, 2)
    thirds2 = math.pow(thirdstd, 2)
    mserrornum = firsts2 * 2 + seconds2 * 2 + thirds2 * 2
    mserrorden = len(first) + len(second) + len(third) - 3
    mserror = mserrornum / mserrorden
    standarderror = math.sqrt(mserror / len(first))
    dftotal = len(first) + len(second) + len(third) - 1
    dfgroups = 2
    dferror = dftotal - dfgroups
    qcrit = 0.5
    qcrit = get_tukeyQcrit(3, dftotal, alpha=0.05)
    qtest3to1 = math.fabs(thirdmean - firstmean) / standarderror
    qtest3to2 = math.fabs(thirdmean - secondmean) / standarderror
    qtest2to1 = math.fabs(secondmean - firstmean) / standarderror
    conclusion = []
    print(qtest3to1)
    print(qtest3to2)
    print(qtest2to1)
    if qtest3to1 > qcrit:
        conclusion.append('3to1null')
    else:
        conclusion.append('3to1alt')
    if qtest3to2 > qcrit:
        conclusion.append('3to2null')
    else:
        conclusion.append('3to2alt')
    if qtest2to1 > qcrit:
        conclusion.append('2to1null')
    else:
        conclusion.append('2to1alt')
    return conclusion

def Tukeythreegene2(genes):
    if False:
        for i in range(10):
            print('nop')
    'gend is a list, ie [first, second, third]'
    means = []
    stds = []
    for gene in genes:
        means.append(np.mean(gene))
        std.append(np.std(gene))
    stds2 = []
    for std in stds:
        stds2.append(math.pow(std, 2))
    mserrornum = sum(stds2) * 2
    mserrorden = len(genes[0]) + len(genes[1]) + len(genes[2]) - 3
    mserror = mserrornum / mserrorden

def catstack(args):
    if False:
        i = 10
        return i + 15
    x = np.hstack(args)
    labels = np.hstack([k * np.ones(len(arr)) for (k, arr) in enumerate(args)])
    return (x, labels)

def maxzero(x):
    if False:
        return 10
    'find all up zero crossings and return the index of the highest\n\n    Not used anymore\n\n\n    >>> np.random.seed(12345)\n    >>> x = np.random.randn(8)\n    >>> x\n    array([-0.20470766,  0.47894334, -0.51943872, -0.5557303 ,  1.96578057,\n            1.39340583,  0.09290788,  0.28174615])\n    >>> maxzero(x)\n    (4, array([1, 4]))\n\n\n    no up-zero-crossing at end\n\n    >>> np.random.seed(0)\n    >>> x = np.random.randn(8)\n    >>> x\n    array([ 1.76405235,  0.40015721,  0.97873798,  2.2408932 ,  1.86755799,\n           -0.97727788,  0.95008842, -0.15135721])\n    >>> maxzero(x)\n    (None, array([6]))\n    '
    x = np.asarray(x)
    cond1 = x[:-1] < 0
    cond2 = x[1:] > 0
    allzeros = np.nonzero(cond1 & cond2 | (x[1:] == 0))[0] + 1
    if x[-1] >= 0:
        maxz = max(allzeros)
    else:
        maxz = None
    return (maxz, allzeros)

def maxzerodown(x):
    if False:
        return 10
    'find all up zero crossings and return the index of the highest\n\n    Not used anymore\n\n    >>> np.random.seed(12345)\n    >>> x = np.random.randn(8)\n    >>> x\n    array([-0.20470766,  0.47894334, -0.51943872, -0.5557303 ,  1.96578057,\n            1.39340583,  0.09290788,  0.28174615])\n    >>> maxzero(x)\n    (4, array([1, 4]))\n\n\n    no up-zero-crossing at end\n\n    >>> np.random.seed(0)\n    >>> x = np.random.randn(8)\n    >>> x\n    array([ 1.76405235,  0.40015721,  0.97873798,  2.2408932 ,  1.86755799,\n           -0.97727788,  0.95008842, -0.15135721])\n    >>> maxzero(x)\n    (None, array([6]))\n'
    x = np.asarray(x)
    cond1 = x[:-1] > 0
    cond2 = x[1:] < 0
    allzeros = np.nonzero(cond1 & cond2 | (x[1:] == 0))[0] + 1
    if x[-1] <= 0:
        maxz = max(allzeros)
    else:
        maxz = None
    return (maxz, allzeros)

def rejectionline(n, alpha=0.5):
    if False:
        while True:
            i = 10
    'reference line for rejection in multiple tests\n\n    Not used anymore\n\n    from: section 3.2, page 60\n    '
    t = np.arange(n) / float(n)
    frej = t / (t * (1 - alpha) + alpha)
    return frej

def fdrcorrection_bak(pvals, alpha=0.05, method='indep'):
    if False:
        while True:
            i = 10
    'Reject False discovery rate correction for pvalues\n\n    Old version, to be deleted\n\n\n    missing: methods that estimate fraction of true hypotheses\n\n    '
    pvals = np.asarray(pvals)
    pvals_sortind = np.argsort(pvals)
    pvals_sorted = pvals[pvals_sortind]
    pecdf = ecdf(pvals_sorted)
    if method in ['i', 'indep', 'p', 'poscorr']:
        rline = pvals_sorted / alpha
    elif method in ['n', 'negcorr']:
        cm = np.sum(1.0 / np.arange(1, len(pvals)))
        rline = pvals_sorted / alpha * cm
    elif method in ['g', 'onegcorr']:
        rline = pvals_sorted / (pvals_sorted * (1 - alpha) + alpha)
    elif method in ['oth', 'o2negcorr']:
        cm = np.sum(np.arange(len(pvals)))
        rline = pvals_sorted / alpha / cm
    else:
        raise ValueError('method not available')
    reject = pecdf >= rline
    if reject.any():
        rejectmax = max(np.nonzero(reject)[0])
    else:
        rejectmax = 0
    reject[:rejectmax] = True
    return reject[pvals_sortind.argsort()]

def mcfdr(nrepl=100, nobs=50, ntests=10, ntrue=6, mu=0.5, alpha=0.05, rho=0.0):
    if False:
        return 10
    'MonteCarlo to test fdrcorrection\n    '
    nfalse = ntests - ntrue
    locs = np.array([0.0] * ntrue + [mu] * (ntests - ntrue))
    results = []
    for i in range(nrepl):
        rvs = locs + randmvn(rho, size=(nobs, ntests))
        (tt, tpval) = stats.ttest_1samp(rvs, 0)
        res = fdrcorrection_bak(np.abs(tpval), alpha=alpha, method='i')
        res0 = fdrcorrection0(np.abs(tpval), alpha=alpha)
        results.append([np.sum(res[:ntrue]), np.sum(res[ntrue:])] + [np.sum(res0[:ntrue]), np.sum(res0[ntrue:])] + res.tolist() + np.sort(tpval).tolist() + [np.sum(tpval[:ntrue] < alpha), np.sum(tpval[ntrue:] < alpha)] + [np.sum(tpval[:ntrue] < alpha / ntests), np.sum(tpval[ntrue:] < alpha / ntests)])
    return np.array(results)

def randmvn(rho, size=(1, 2), standardize=False):
    if False:
        for i in range(10):
            print('nop')
    'create random draws from equi-correlated multivariate normal distribution\n\n    Parameters\n    ----------\n    rho : float\n        correlation coefficient\n    size : tuple of int\n        size is interpreted (nobs, nvars) where each row\n\n    Returns\n    -------\n    rvs : ndarray\n        nobs by nvars where each row is a independent random draw of nvars-\n        dimensional correlated rvs\n\n    '
    (nobs, nvars) = size
    if 0 < rho and rho < 1:
        rvs = np.random.randn(nobs, nvars + 1)
        rvs2 = rvs[:, :-1] * np.sqrt(1 - rho) + rvs[:, -1:] * np.sqrt(rho)
    elif rho == 0:
        rvs2 = np.random.randn(nobs, nvars)
    elif rho < 0:
        if rho < -1.0 / (nvars - 1):
            raise ValueError('rho has to be larger than -1./(nvars-1)')
        elif rho == -1.0 / (nvars - 1):
            rho = -1.0 / (nvars - 1 + 1e-10)
        A = rho * np.ones((nvars, nvars)) + (1 - rho) * np.eye(nvars)
        rvs2 = np.dot(np.random.randn(nobs, nvars), np.linalg.cholesky(A).T)
    if standardize:
        rvs2 = stats.zscore(rvs2)
    return rvs2

def tiecorrect(xranks):
    if False:
        for i in range(10):
            print('nop')
    '\n\n    should be equivalent of scipy.stats.tiecorrect\n\n    '
    rankbincount = np.bincount(np.asarray(xranks, dtype=int))
    nties = rankbincount[rankbincount > 1]
    ntot = float(len(xranks))
    tiecorrection = 1 - (nties ** 3 - nties).sum() / (ntot ** 3 - ntot)
    return tiecorrection

class GroupsStats:
    """
    statistics by groups (another version)

    groupstats as a class with lazy evaluation (not yet - decorators are still
    missing)

    written this time as equivalent of scipy.stats.rankdata
    gs = GroupsStats(X, useranks=True)
    assert_almost_equal(gs.groupmeanfilter, stats.rankdata(X[:,0]), 15)

    TODO: incomplete doc strings

    """

    def __init__(self, x, useranks=False, uni=None, intlab=None):
        if False:
            print('Hello World!')
        'descriptive statistics by groups\n\n        Parameters\n        ----------\n        x : ndarray, 2d\n            first column data, second column group labels\n        useranks : bool\n            if true, then use ranks as data corresponding to the\n            scipy.stats.rankdata definition (start at 1, ties get mean)\n        uni, intlab : arrays (optional)\n            to avoid call to unique, these can be given as inputs\n\n\n        '
        self.x = np.asarray(x)
        if intlab is None:
            (uni, intlab) = np.unique(x[:, 1], return_inverse=True)
        elif uni is None:
            uni = np.unique(x[:, 1])
        self.useranks = useranks
        self.uni = uni
        self.intlab = intlab
        self.groupnobs = groupnobs = np.bincount(intlab)
        self.runbasic(useranks=useranks)

    def runbasic_old(self, useranks=False):
        if False:
            return 10
        'runbasic_old'
        x = self.x
        if useranks:
            self.xx = x[:, 1].argsort().argsort() + 1
        else:
            self.xx = x[:, 0]
        self.groupsum = groupranksum = np.bincount(self.intlab, weights=self.xx)
        self.groupmean = grouprankmean = groupranksum * 1.0 / self.groupnobs
        self.groupmeanfilter = grouprankmean[self.intlab]

    def runbasic(self, useranks=False):
        if False:
            while True:
                i = 10
        'runbasic'
        x = self.x
        if useranks:
            (xuni, xintlab) = np.unique(x[:, 0], return_inverse=True)
            ranksraw = x[:, 0].argsort().argsort() + 1
            self.xx = GroupsStats(np.column_stack([ranksraw, xintlab]), useranks=False).groupmeanfilter
        else:
            self.xx = x[:, 0]
        self.groupsum = groupranksum = np.bincount(self.intlab, weights=self.xx)
        self.groupmean = grouprankmean = groupranksum * 1.0 / self.groupnobs
        self.groupmeanfilter = grouprankmean[self.intlab]

    def groupdemean(self):
        if False:
            return 10
        'groupdemean'
        return self.xx - self.groupmeanfilter

    def groupsswithin(self):
        if False:
            while True:
                i = 10
        'groupsswithin'
        xtmp = self.groupdemean()
        return np.bincount(self.intlab, weights=xtmp ** 2)

    def groupvarwithin(self):
        if False:
            print('Hello World!')
        'groupvarwithin'
        return self.groupsswithin() / (self.groupnobs - 1)

class TukeyHSDResults:
    """Results from Tukey HSD test, with additional plot methods

    Can also compute and plot additional post-hoc evaluations using this
    results class.

    Attributes
    ----------
    reject : array of boolean, True if we reject Null for group pair
    meandiffs : pairwise mean differences
    confint : confidence interval for pairwise mean differences
    std_pairs : standard deviation of pairwise mean differences
    q_crit : critical value of studentized range statistic at given alpha
    halfwidths : half widths of simultaneous confidence interval
    pvalues : adjusted p-values from the HSD test

    Notes
    -----
    halfwidths is only available after call to `plot_simultaneous`.

    Other attributes contain information about the data from the
    MultiComparison instance: data, df_total, groups, groupsunique, variance.
    """

    def __init__(self, mc_object, results_table, q_crit, reject=None, meandiffs=None, std_pairs=None, confint=None, df_total=None, reject2=None, variance=None, pvalues=None):
        if False:
            return 10
        self._multicomp = mc_object
        self._results_table = results_table
        self.q_crit = q_crit
        self.reject = reject
        self.meandiffs = meandiffs
        self.std_pairs = std_pairs
        self.confint = confint
        self.df_total = df_total
        self.reject2 = reject2
        self.variance = variance
        self.pvalues = pvalues
        self.data = self._multicomp.data
        self.groups = self._multicomp.groups
        self.groupsunique = self._multicomp.groupsunique

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return str(self._results_table)

    def summary(self):
        if False:
            return 10
        'Summary table that can be printed\n        '
        return self._results_table

    def _simultaneous_ci(self):
        if False:
            while True:
                i = 10
        'Compute simultaneous confidence intervals for comparison of means.\n        '
        self.halfwidths = simultaneous_ci(self.q_crit, self.variance, self._multicomp.groupstats.groupnobs, self._multicomp.pairindices)

    def plot_simultaneous(self, comparison_name=None, ax=None, figsize=(10, 6), xlabel=None, ylabel=None):
        if False:
            return 10
        'Plot a universal confidence interval of each group mean\n\n        Visualize significant differences in a plot with one confidence\n        interval per group instead of all pairwise confidence intervals.\n\n        Parameters\n        ----------\n        comparison_name : str, optional\n            if provided, plot_intervals will color code all groups that are\n            significantly different from the comparison_name red, and will\n            color code insignificant groups gray. Otherwise, all intervals will\n            just be plotted in black.\n        ax : matplotlib axis, optional\n            An axis handle on which to attach the plot.\n        figsize : tuple, optional\n            tuple for the size of the figure generated\n        xlabel : str, optional\n            Name to be displayed on x axis\n        ylabel : str, optional\n            Name to be displayed on y axis\n\n        Returns\n        -------\n        Figure\n            handle to figure object containing interval plots\n\n        Notes\n        -----\n        Multiple comparison tests are nice, but lack a good way to be\n        visualized. If you have, say, 6 groups, showing a graph of the means\n        between each group will require 15 confidence intervals.\n        Instead, we can visualize inter-group differences with a single\n        interval for each group mean. Hochberg et al. [1] first proposed this\n        idea and used Tukey\'s Q critical value to compute the interval widths.\n        Unlike plotting the differences in the means and their respective\n        confidence intervals, any two pairs can be compared for significance\n        by looking for overlap.\n\n        References\n        ----------\n        .. [*] Hochberg, Y., and A. C. Tamhane. Multiple Comparison Procedures.\n               Hoboken, NJ: John Wiley & Sons, 1987.\n\n        Examples\n        --------\n        >>> from statsmodels.examples.try_tukey_hsd import cylinders, cyl_labels\n        >>> from statsmodels.stats.multicomp import MultiComparison\n        >>> cardata = MultiComparison(cylinders, cyl_labels)\n        >>> results = cardata.tukeyhsd()\n        >>> results.plot_simultaneous()\n        <matplotlib.figure.Figure at 0x...>\n\n        This example shows an example plot comparing significant differences\n        in group means. Significant differences at the alpha=0.05 level can be\n        identified by intervals that do not overlap (i.e. USA vs Japan,\n        USA vs Germany).\n\n        >>> results.plot_simultaneous(comparison_name="USA")\n        <matplotlib.figure.Figure at 0x...>\n\n        Optionally provide one of the group names to color code the plot to\n        highlight group means different from comparison_name.\n        '
        (fig, ax1) = utils.create_mpl_ax(ax)
        if figsize is not None:
            fig.set_size_inches(figsize)
        if getattr(self, 'halfwidths', None) is None:
            self._simultaneous_ci()
        means = self._multicomp.groupstats.groupmean
        sigidx = []
        nsigidx = []
        minrange = [means[i] - self.halfwidths[i] for i in range(len(means))]
        maxrange = [means[i] + self.halfwidths[i] for i in range(len(means))]
        if comparison_name is None:
            ax1.errorbar(means, lrange(len(means)), xerr=self.halfwidths, marker='o', linestyle='None', color='k', ecolor='k')
        else:
            if comparison_name not in self.groupsunique:
                raise ValueError('comparison_name not found in group names.')
            midx = np.where(self.groupsunique == comparison_name)[0][0]
            for i in range(len(means)):
                if self.groupsunique[i] == comparison_name:
                    continue
                if min(maxrange[i], maxrange[midx]) - max(minrange[i], minrange[midx]) < 0:
                    sigidx.append(i)
                else:
                    nsigidx.append(i)
            ax1.errorbar(means[midx], midx, xerr=self.halfwidths[midx], marker='o', linestyle='None', color='b', ecolor='b')
            ax1.plot([minrange[midx]] * 2, [-1, self._multicomp.ngroups], linestyle='--', color='0.7')
            ax1.plot([maxrange[midx]] * 2, [-1, self._multicomp.ngroups], linestyle='--', color='0.7')
            if len(sigidx) > 0:
                ax1.errorbar(means[sigidx], sigidx, xerr=self.halfwidths[sigidx], marker='o', linestyle='None', color='r', ecolor='r')
            if len(nsigidx) > 0:
                ax1.errorbar(means[nsigidx], nsigidx, xerr=self.halfwidths[nsigidx], marker='o', linestyle='None', color='0.5', ecolor='0.5')
        ax1.set_title('Multiple Comparisons Between All Pairs (Tukey)')
        r = np.max(maxrange) - np.min(minrange)
        ax1.set_ylim([-1, self._multicomp.ngroups])
        ax1.set_xlim([np.min(minrange) - r / 10.0, np.max(maxrange) + r / 10.0])
        ylbls = [''] + self.groupsunique.astype(str).tolist() + ['']
        ax1.set_yticks(np.arange(-1, len(means) + 1))
        ax1.set_yticklabels(ylbls)
        ax1.set_xlabel(xlabel if xlabel is not None else '')
        ax1.set_ylabel(ylabel if ylabel is not None else '')
        return fig

class MultiComparison:
    """Tests for multiple comparisons

    Parameters
    ----------
    data : ndarray
        independent data samples
    groups : ndarray
        group labels corresponding to each data point
    group_order : list[str], optional
        the desired order for the group mean results to be reported in. If
        not specified, results are reported in increasing order.
        If group_order does not contain all labels that are in groups, then
        only those observations are kept that have a label in group_order.

    """

    def __init__(self, data, groups, group_order=None):
        if False:
            for i in range(10):
                print('nop')
        if len(data) != len(groups):
            raise ValueError('data has %d elements and groups has %d' % (len(data), len(groups)))
        self.data = np.asarray(data)
        self.groups = groups = np.asarray(groups)
        if group_order is None:
            (self.groupsunique, self.groupintlab) = np.unique(groups, return_inverse=True)
        else:
            for grp in group_order:
                if grp not in groups:
                    raise ValueError("group_order value '%s' not found in groups" % grp)
            self.groupsunique = np.array(group_order)
            self.groupintlab = np.empty(len(data), int)
            self.groupintlab.fill(-999)
            count = 0
            for name in self.groupsunique:
                idx = np.where(self.groups == name)[0]
                count += len(idx)
                self.groupintlab[idx] = np.where(self.groupsunique == name)[0]
            if count != self.data.shape[0]:
                import warnings
                warnings.warn('group_order does not contain all groups:' + ' dropping observations', ValueWarning)
                mask_keep = self.groupintlab != -999
                self.groupintlab = self.groupintlab[mask_keep]
                self.data = self.data[mask_keep]
                self.groups = self.groups[mask_keep]
        if len(self.groupsunique) < 2:
            raise ValueError('2 or more groups required for multiple comparisons')
        self.datali = [self.data[self.groups == k] for k in self.groupsunique]
        self.pairindices = np.triu_indices(len(self.groupsunique), 1)
        self.nobs = self.data.shape[0]
        self.ngroups = len(self.groupsunique)

    def getranks(self):
        if False:
            while True:
                i = 10
        'convert data to rankdata and attach\n\n\n        This creates rankdata as it is used for non-parametric tests, where\n        in the case of ties the average rank is assigned.\n\n\n        '
        self.ranks = GroupsStats(np.column_stack([self.data, self.groupintlab]), useranks=True)
        self.rankdata = self.ranks.groupmeanfilter

    def kruskal(self, pairs=None, multimethod='T'):
        if False:
            print('Hello World!')
        '\n        pairwise comparison for kruskal-wallis test\n\n        This is just a reimplementation of scipy.stats.kruskal and does\n        not yet use a multiple comparison correction.\n\n        '
        self.getranks()
        tot = self.nobs
        meanranks = self.ranks.groupmean
        groupnobs = self.ranks.groupnobs
        f = tot * (tot + 1.0) / 12.0 / stats.tiecorrect(self.rankdata)
        print('MultiComparison.kruskal')
        for (i, j) in zip(*self.pairindices):
            pdiff = np.abs(meanranks[i] - meanranks[j])
            se = np.sqrt(f * np.sum(1.0 / groupnobs[[i, j]]))
            Q = pdiff / se
            print(i, j, pdiff, se, pdiff / se, pdiff / se > 2.631)
            print(stats.norm.sf(Q) * 2)
            return stats.norm.sf(Q) * 2

    def allpairtest(self, testfunc, alpha=0.05, method='bonf', pvalidx=1):
        if False:
            print('Hello World!')
        "run a pairwise test on all pairs with multiple test correction\n\n        The statistical test given in testfunc is calculated for all pairs\n        and the p-values are adjusted by methods in multipletests. The p-value\n        correction is generic and based only on the p-values, and does not\n        take any special structure of the hypotheses into account.\n\n        Parameters\n        ----------\n        testfunc : function\n            A test function for two (independent) samples. It is assumed that\n            the return value on position pvalidx is the p-value.\n        alpha : float\n            familywise error rate\n        method : str\n            This specifies the method for the p-value correction. Any method\n            of multipletests is possible.\n        pvalidx : int (default: 1)\n            position of the p-value in the return of testfunc\n\n        Returns\n        -------\n        sumtab : SimpleTable instance\n            summary table for printing\n\n        errors:  TODO: check if this is still wrong, I think it's fixed.\n        results from multipletests are in different order\n        pval_corrected can be larger than 1 ???\n        "
        res = []
        for (i, j) in zip(*self.pairindices):
            res.append(testfunc(self.datali[i], self.datali[j]))
        res = np.array(res)
        (reject, pvals_corrected, alphacSidak, alphacBonf) = multipletests(res[:, pvalidx], alpha=alpha, method=method)
        (i1, i2) = self.pairindices
        if pvals_corrected is None:
            resarr = np.array(lzip(self.groupsunique[i1], self.groupsunique[i2], np.round(res[:, 0], 4), np.round(res[:, 1], 4), reject), dtype=[('group1', object), ('group2', object), ('stat', float), ('pval', float), ('reject', np.bool_)])
        else:
            resarr = np.array(lzip(self.groupsunique[i1], self.groupsunique[i2], np.round(res[:, 0], 4), np.round(res[:, 1], 4), np.round(pvals_corrected, 4), reject), dtype=[('group1', object), ('group2', object), ('stat', float), ('pval', float), ('pval_corr', float), ('reject', np.bool_)])
        results_table = SimpleTable(resarr, headers=resarr.dtype.names)
        results_table.title = 'Test Multiple Comparison %s \n%s%4.2f method=%s' % (testfunc.__name__, 'FWER=', alpha, method) + '\nalphacSidak=%4.2f, alphacBonf=%5.3f' % (alphacSidak, alphacBonf)
        return (results_table, (res, reject, pvals_corrected, alphacSidak, alphacBonf), resarr)

    def tukeyhsd(self, alpha=0.05):
        if False:
            i = 10
            return i + 15
        "\n        Tukey's range test to compare means of all pairs of groups\n\n        Parameters\n        ----------\n        alpha : float, optional\n            Value of FWER at which to calculate HSD.\n\n        Returns\n        -------\n        results : TukeyHSDResults instance\n            A results class containing relevant data and some post-hoc\n            calculations\n        "
        self.groupstats = GroupsStats(np.column_stack([self.data, self.groupintlab]), useranks=False)
        gmeans = self.groupstats.groupmean
        gnobs = self.groupstats.groupnobs
        var_ = np.var(self.groupstats.groupdemean(), ddof=len(gmeans))
        res = tukeyhsd(gmeans, gnobs, var_, df=None, alpha=alpha, q_crit=None)
        resarr = np.array(lzip(self.groupsunique[res[0][0]], self.groupsunique[res[0][1]], np.round(res[2], 4), np.round(res[8], 4), np.round(res[4][:, 0], 4), np.round(res[4][:, 1], 4), res[1]), dtype=[('group1', object), ('group2', object), ('meandiff', float), ('p-adj', float), ('lower', float), ('upper', float), ('reject', np.bool_)])
        results_table = SimpleTable(resarr, headers=resarr.dtype.names)
        results_table.title = 'Multiple Comparison of Means - Tukey HSD, ' + 'FWER=%4.2f' % alpha
        return TukeyHSDResults(self, results_table, res[5], res[1], res[2], res[3], res[4], res[6], res[7], var_, res[8])

def rankdata(x):
    if False:
        while True:
            i = 10
    'rankdata, equivalent to scipy.stats.rankdata\n\n    just a different implementation, I have not yet compared speed\n\n    '
    (uni, intlab) = np.unique(x[:, 0], return_inverse=True)
    groupnobs = np.bincount(intlab)
    groupxsum = np.bincount(intlab, weights=X[:, 0])
    groupxmean = groupxsum * 1.0 / groupnobs
    rankraw = x[:, 0].argsort().argsort()
    groupranksum = np.bincount(intlab, weights=rankraw)
    grouprankmean = groupranksum * 1.0 / groupnobs + 1
    return grouprankmean[intlab]

def compare_ordered(vals, alpha):
    if False:
        i = 10
        return i + 15
    'simple ordered sequential comparison of means\n\n    vals : array_like\n        means or rankmeans for independent groups\n\n    incomplete, no return, not used yet\n    '
    vals = np.asarray(vals)
    alphaf = alpha
    sortind = np.argsort(vals)
    pvals = vals[sortind]
    sortrevind = sortind.argsort()
    ntests = len(vals)
    (v1, v2) = np.triu_indices(ntests, 1)
    for i in range(4):
        for j in range(4, i, -1):
            print(i, j)

def varcorrection_unbalanced(nobs_all, srange=False):
    if False:
        while True:
            i = 10
    'correction factor for variance with unequal sample sizes\n\n    this is just a harmonic mean\n\n    Parameters\n    ----------\n    nobs_all : array_like\n        The number of observations for each sample\n    srange : bool\n        if true, then the correction is divided by the number of samples\n        for the variance of the studentized range statistic\n\n    Returns\n    -------\n    correction : float\n        Correction factor for variance.\n\n\n    Notes\n    -----\n\n    variance correction factor is\n\n    1/k * sum_i 1/n_i\n\n    where k is the number of samples and summation is over i=0,...,k-1.\n    If all n_i are the same, then the correction factor is 1.\n\n    This needs to be multiplied by the joint variance estimate, means square\n    error, MSE. To obtain the correction factor for the standard deviation,\n    square root needs to be taken.\n\n    '
    nobs_all = np.asarray(nobs_all)
    if not srange:
        return (1.0 / nobs_all).sum()
    else:
        return (1.0 / nobs_all).sum() / len(nobs_all)

def varcorrection_pairs_unbalanced(nobs_all, srange=False):
    if False:
        return 10
    'correction factor for variance with unequal sample sizes for all pairs\n\n    this is just a harmonic mean\n\n    Parameters\n    ----------\n    nobs_all : array_like\n        The number of observations for each sample\n    srange : bool\n        if true, then the correction is divided by 2 for the variance of\n        the studentized range statistic\n\n    Returns\n    -------\n    correction : ndarray\n        Correction factor for variance.\n\n\n    Notes\n    -----\n\n    variance correction factor is\n\n    1/k * sum_i 1/n_i\n\n    where k is the number of samples and summation is over i=0,...,k-1.\n    If all n_i are the same, then the correction factor is 1.\n\n    This needs to be multiplies by the joint variance estimate, means square\n    error, MSE. To obtain the correction factor for the standard deviation,\n    square root needs to be taken.\n\n    For the studentized range statistic, the resulting factor has to be\n    divided by 2.\n\n    '
    (n1, n2) = np.meshgrid(nobs_all, nobs_all)
    if not srange:
        return 1.0 / n1 + 1.0 / n2
    else:
        return (1.0 / n1 + 1.0 / n2) / 2.0

def varcorrection_unequal(var_all, nobs_all, df_all):
    if False:
        while True:
            i = 10
    "return joint variance from samples with unequal variances and unequal\n    sample sizes\n\n    something is wrong\n\n    Parameters\n    ----------\n    var_all : array_like\n        The variance for each sample\n    nobs_all : array_like\n        The number of observations for each sample\n    df_all : array_like\n        degrees of freedom for each sample\n\n    Returns\n    -------\n    varjoint : float\n        joint variance.\n    dfjoint : float\n        joint Satterthwait's degrees of freedom\n\n\n    Notes\n    -----\n    (copy, paste not correct)\n    variance is\n\n    1/k * sum_i 1/n_i\n\n    where k is the number of samples and summation is over i=0,...,k-1.\n    If all n_i are the same, then the correction factor is 1/n.\n\n    This needs to be multiplies by the joint variance estimate, means square\n    error, MSE. To obtain the correction factor for the standard deviation,\n    square root needs to be taken.\n\n    This is for variance of mean difference not of studentized range.\n    "
    var_all = np.asarray(var_all)
    var_over_n = var_all * 1.0 / nobs_all
    varjoint = var_over_n.sum()
    dfjoint = varjoint ** 2 / (var_over_n ** 2 * df_all).sum()
    return (varjoint, dfjoint)

def varcorrection_pairs_unequal(var_all, nobs_all, df_all):
    if False:
        for i in range(10):
            print('nop')
    "return joint variance from samples with unequal variances and unequal\n    sample sizes for all pairs\n\n    something is wrong\n\n    Parameters\n    ----------\n    var_all : array_like\n        The variance for each sample\n    nobs_all : array_like\n        The number of observations for each sample\n    df_all : array_like\n        degrees of freedom for each sample\n\n    Returns\n    -------\n    varjoint : ndarray\n        joint variance.\n    dfjoint : ndarray\n        joint Satterthwait's degrees of freedom\n\n\n    Notes\n    -----\n\n    (copy, paste not correct)\n    variance is\n\n    1/k * sum_i 1/n_i\n\n    where k is the number of samples and summation is over i=0,...,k-1.\n    If all n_i are the same, then the correction factor is 1.\n\n    This needs to be multiplies by the joint variance estimate, means square\n    error, MSE. To obtain the correction factor for the standard deviation,\n    square root needs to be taken.\n\n    TODO: something looks wrong with dfjoint, is formula from SPSS\n    "
    (v1, v2) = np.meshgrid(var_all, var_all)
    (n1, n2) = np.meshgrid(nobs_all, nobs_all)
    (df1, df2) = np.meshgrid(df_all, df_all)
    varjoint = v1 / n1 + v2 / n2
    dfjoint = varjoint ** 2 / (df1 * (v1 / n1) ** 2 + df2 * (v2 / n2) ** 2)
    return (varjoint, dfjoint)

def tukeyhsd(mean_all, nobs_all, var_all, df=None, alpha=0.05, q_crit=None):
    if False:
        for i in range(10):
            print('nop')
    "simultaneous Tukey HSD\n\n\n    check: instead of sorting, I use absolute value of pairwise differences\n    in means. That's irrelevant for the test, but maybe reporting actual\n    differences would be better.\n    CHANGED: meandiffs are with sign, studentized range uses abs\n\n    q_crit added for testing\n\n    TODO: error in variance calculation when nobs_all is scalar, missing 1/n\n\n    "
    mean_all = np.asarray(mean_all)
    n_means = len(mean_all)
    if df is None:
        df = nobs_all - 1
    if np.size(df) == 1:
        df_total = n_means * df
        df = np.ones(n_means) * df
    else:
        df_total = np.sum(df)
    if np.size(nobs_all) == 1 and np.size(var_all) == 1:
        var_pairs = 1.0 * var_all / nobs_all * np.ones((n_means, n_means))
    elif np.size(var_all) == 1:
        var_pairs = var_all * varcorrection_pairs_unbalanced(nobs_all, srange=True)
    elif np.size(var_all) > 1:
        (var_pairs, df_sum) = varcorrection_pairs_unequal(nobs_all, var_all, df)
        var_pairs /= 2.0
    else:
        raise ValueError('not supposed to be here')
    meandiffs_ = mean_all - mean_all[:, None]
    std_pairs_ = np.sqrt(var_pairs)
    (idx1, idx2) = np.triu_indices(n_means, 1)
    meandiffs = meandiffs_[idx1, idx2]
    std_pairs = std_pairs_[idx1, idx2]
    st_range = np.abs(meandiffs) / std_pairs
    df_total_ = max(df_total, 5)
    if q_crit is None:
        q_crit = get_tukeyQcrit2(n_means, df_total, alpha=alpha)
    pvalues = get_tukey_pvalue(n_means, df_total, st_range)
    pvalues = np.atleast_1d(pvalues)
    reject = st_range > q_crit
    crit_int = std_pairs * q_crit
    reject2 = np.abs(meandiffs) > crit_int
    confint = np.column_stack((meandiffs - crit_int, meandiffs + crit_int))
    return ((idx1, idx2), reject, meandiffs, std_pairs, confint, q_crit, df_total, reject2, pvalues)

def simultaneous_ci(q_crit, var, groupnobs, pairindices=None):
    if False:
        while True:
            i = 10
    "Compute simultaneous confidence intervals for comparison of means.\n\n    q_crit value is generated from tukey hsd test. Variance is considered\n    across all groups. Returned halfwidths can be thought of as uncertainty\n    intervals around each group mean. They allow for simultaneous\n    comparison of pairwise significance among any pairs (by checking for\n    overlap)\n\n    Parameters\n    ----------\n    q_crit : float\n        The Q critical value studentized range statistic from Tukey's HSD\n    var : float\n        The group variance\n    groupnobs : array_like object\n        Number of observations contained in each group.\n    pairindices : tuple of lists, optional\n        Indices corresponding to the upper triangle of matrix. Computed\n        here if not supplied\n\n    Returns\n    -------\n    halfwidths : ndarray\n        Half the width of each confidence interval for each group given in\n        groupnobs\n\n    See Also\n    --------\n    MultiComparison : statistics class providing significance tests\n    tukeyhsd : among other things, computes q_crit value\n\n    References\n    ----------\n    .. [*] Hochberg, Y., and A. C. Tamhane. Multiple Comparison Procedures.\n           Hoboken, NJ: John Wiley & Sons, 1987.)\n    "
    ng = len(groupnobs)
    if pairindices is None:
        pairindices = np.triu_indices(ng, 1)
    gvar = var / groupnobs
    d12 = np.sqrt(gvar[pairindices[0]] + gvar[pairindices[1]])
    d = np.zeros((ng, ng))
    d[pairindices] = d12
    d = d + d.conj().T
    sum1 = np.sum(d12)
    sum2 = np.sum(d, axis=0)
    if ng > 2:
        w = ((ng - 1.0) * sum2 - sum1) / ((ng - 1.0) * (ng - 2.0))
    else:
        w = sum1 * np.ones((2, 1)) / 2.0
    return q_crit / np.sqrt(2) * w

def distance_st_range(mean_all, nobs_all, var_all, df=None, triu=False):
    if False:
        for i in range(10):
            print('nop')
    'pairwise distance matrix, outsourced from tukeyhsd\n\n\n\n    CHANGED: meandiffs are with sign, studentized range uses abs\n\n    q_crit added for testing\n\n    TODO: error in variance calculation when nobs_all is scalar, missing 1/n\n\n    '
    mean_all = np.asarray(mean_all)
    n_means = len(mean_all)
    if df is None:
        df = nobs_all - 1
    if np.size(df) == 1:
        df_total = n_means * df
    else:
        df_total = np.sum(df)
    if np.size(nobs_all) == 1 and np.size(var_all) == 1:
        var_pairs = 1.0 * var_all / nobs_all * np.ones((n_means, n_means))
    elif np.size(var_all) == 1:
        var_pairs = var_all * varcorrection_pairs_unbalanced(nobs_all, srange=True)
    elif np.size(var_all) > 1:
        (var_pairs, df_sum) = varcorrection_pairs_unequal(nobs_all, var_all, df)
        var_pairs /= 2.0
    else:
        raise ValueError('not supposed to be here')
    meandiffs = mean_all - mean_all[:, None]
    std_pairs = np.sqrt(var_pairs)
    (idx1, idx2) = np.triu_indices(n_means, 1)
    if triu:
        meandiffs = meandiffs_[idx1, idx2]
        std_pairs = std_pairs_[idx1, idx2]
    st_range = np.abs(meandiffs) / std_pairs
    return (st_range, meandiffs, std_pairs, (idx1, idx2))

def contrast_allpairs(nm):
    if False:
        return 10
    'contrast or restriction matrix for all pairs of nm variables\n\n    Parameters\n    ----------\n    nm : int\n\n    Returns\n    -------\n    contr : ndarray, 2d, (nm*(nm-1)/2, nm)\n       contrast matrix for all pairwise comparisons\n\n    '
    contr = []
    for i in range(nm):
        for j in range(i + 1, nm):
            contr_row = np.zeros(nm)
            contr_row[i] = 1
            contr_row[j] = -1
            contr.append(contr_row)
    return np.array(contr)

def contrast_all_one(nm):
    if False:
        i = 10
        return i + 15
    'contrast or restriction matrix for all against first comparison\n\n    Parameters\n    ----------\n    nm : int\n\n    Returns\n    -------\n    contr : ndarray, 2d, (nm-1, nm)\n       contrast matrix for all against first comparisons\n\n    '
    contr = np.column_stack((np.ones(nm - 1), -np.eye(nm - 1)))
    return contr

def contrast_diff_mean(nm):
    if False:
        print('Hello World!')
    'contrast or restriction matrix for all against mean comparison\n\n    Parameters\n    ----------\n    nm : int\n\n    Returns\n    -------\n    contr : ndarray, 2d, (nm-1, nm)\n       contrast matrix for all against mean comparisons\n\n    '
    return np.eye(nm) - np.ones((nm, nm)) / nm

def tukey_pvalues(std_range, nm, df):
    if False:
        for i in range(10):
            print('nop')
    contr = contrast_allpairs(nm)
    corr = np.dot(contr, contr.T) / 2.0
    tstat = std_range / np.sqrt(2) * np.ones(corr.shape[0])
    return multicontrast_pvalues(tstat, corr, df=df)

def multicontrast_pvalues(tstat, tcorr, df=None, dist='t', alternative='two-sided'):
    if False:
        i = 10
        return i + 15
    'pvalues for simultaneous tests\n\n    '
    from statsmodels.sandbox.distributions.multivariate import mvstdtprob
    if df is None and dist == 't':
        raise ValueError('df has to be specified for the t-distribution')
    tstat = np.asarray(tstat)
    ntests = len(tstat)
    cc = np.abs(tstat)
    pval_global = 1 - mvstdtprob(-cc, cc, tcorr, df)
    pvals = []
    for ti in cc:
        limits = ti * np.ones(ntests)
        pvals.append(1 - mvstdtprob(-cc, cc, tcorr, df))
    return (pval_global, np.asarray(pvals))

class StepDown:
    """a class for step down methods

    This is currently for simple tree subset descend, similar to homogeneous_subsets,
    but checks all leave-one-out subsets instead of assuming an ordered set.
    Comment in SAS manual:
    SAS only uses interval subsets of the sorted list, which is sufficient for range
    tests (maybe also equal variance and balanced sample sizes are required).
    For F-test based critical distances, the restriction to intervals is not sufficient.

    This version uses a single critical value of the studentized range distribution
    for all comparisons, and is therefore a step-down version of Tukey HSD.
    The class is written so it can be subclassed, where the get_distance_matrix and
    get_crit are overwritten to obtain other step-down procedures such as REGW.

    iter_subsets can be overwritten, to get a recursion as in the many to one comparison
    with a control such as in Dunnet's test.


    A one-sided right tail test is not covered because the direction of the inequality
    is hard coded in check_set.  Also Peritz's check of partitions is not possible, but
    I have not seen it mentioned in any more recent references.
    I have only partially read the step-down procedure for closed tests by Westfall.

    One change to make it more flexible, is to separate out the decision on a subset,
    also because the F-based tests, FREGW in SPSS, take information from all elements of
    a set and not just pairwise comparisons. I have not looked at the details of
    the F-based tests such as Sheffe yet. It looks like running an F-test on equality
    of means in each subset. This would also outsource how pairwise conditions are
    combined, any larger or max. This would also imply that the distance matrix cannot
    be calculated in advance for tests like the F-based ones.


    """

    def __init__(self, vals, nobs_all, var_all, df=None):
        if False:
            for i in range(10):
                print('nop')
        self.vals = vals
        self.n_vals = len(vals)
        self.nobs_all = nobs_all
        self.var_all = var_all
        self.df = df

    def get_crit(self, alpha):
        if False:
            return 10
        '\n        get_tukeyQcrit\n\n        currently tukey Q, add others\n        '
        q_crit = get_tukeyQcrit(self.n_vals, self.df, alpha=alpha)
        return q_crit * np.ones(self.n_vals)

    def get_distance_matrix(self):
        if False:
            while True:
                i = 10
        'studentized range statistic'
        dres = distance_st_range(self.vals, self.nobs_all, self.var_all, df=self.df)
        self.distance_matrix = dres[0]

    def iter_subsets(self, indices):
        if False:
            for i in range(10):
                print('nop')
        'Iterate substeps'
        for ii in range(len(indices)):
            idxsub = copy.copy(indices)
            idxsub.pop(ii)
            yield idxsub

    def check_set(self, indices):
        if False:
            for i in range(10):
                print('nop')
        'check whether pairwise distances of indices satisfy condition\n\n        '
        indtup = tuple(indices)
        if indtup in self.cache_result:
            return self.cache_result[indtup]
        else:
            set_distance_matrix = self.distance_matrix[np.asarray(indices)[:, None], indices]
            n_elements = len(indices)
            if np.any(set_distance_matrix > self.crit[n_elements - 1]):
                res = True
            else:
                res = False
            self.cache_result[indtup] = res
            return res

    def stepdown(self, indices):
        if False:
            print('Hello World!')
        'stepdown'
        print(indices)
        if self.check_set(indices):
            if len(indices) > 2:
                for subs in self.iter_subsets(indices):
                    self.stepdown(subs)
            else:
                self.rejected.append(tuple(indices))
        else:
            self.accepted.append(tuple(indices))
            return indices

    def run(self, alpha):
        if False:
            return 10
        'main function to run the test,\n\n        could be done in __call__ instead\n        this could have all the initialization code\n\n        '
        self.cache_result = {}
        self.crit = self.get_crit(alpha)
        self.accepted = []
        self.rejected = []
        self.get_distance_matrix()
        self.stepdown(lrange(self.n_vals))
        return (list(set(self.accepted)), list(set(sd.rejected)))

def homogeneous_subsets(vals, dcrit):
    if False:
        return 10
    'recursively check all pairs of vals for minimum distance\n\n    step down method as in Newman-Keuls and Ryan procedures. This is not a\n    closed procedure since not all partitions are checked.\n\n    Parameters\n    ----------\n    vals : array_like\n        values that are pairwise compared\n    dcrit : array_like or float\n        critical distance for rejecting, either float, or 2-dimensional array\n        with distances on the upper triangle.\n\n    Returns\n    -------\n    rejs : list of pairs\n        list of pair-indices with (strictly) larger than critical difference\n    nrejs : list of pairs\n        list of pair-indices with smaller than critical difference\n    lli : list of tuples\n        list of subsets with smaller than critical difference\n    res : tree\n        result of all comparisons (for checking)\n\n\n    this follows description in SPSS notes on Post-Hoc Tests\n\n    Because of the recursive structure, some comparisons are made several\n    times, but only unique pairs or sets are returned.\n\n    Examples\n    --------\n    >>> m = [0, 2, 2.5, 3, 6, 8, 9, 9.5,10 ]\n    >>> rej, nrej, ssli, res = homogeneous_subsets(m, 2)\n    >>> set_partition(ssli)\n    ([(5, 6, 7, 8), (1, 2, 3), (4,)], [0])\n    >>> [np.array(m)[list(pp)] for pp in set_partition(ssli)[0]]\n    [array([  8. ,   9. ,   9.5,  10. ]), array([ 2. ,  2.5,  3. ]), array([ 6.])]\n\n\n    '
    nvals = len(vals)
    indices_ = lrange(nvals)
    rejected = []
    subsetsli = []
    if np.size(dcrit) == 1:
        dcrit = dcrit * np.ones((nvals, nvals))

    def subsets(vals, indices_):
        if False:
            return 10
        'recursive function for constructing homogeneous subset\n\n        registers rejected and subsetli in outer scope\n        '
        (i, j) = (indices_[0], indices_[-1])
        if vals[-1] - vals[0] > dcrit[i, j]:
            rejected.append((indices_[0], indices_[-1]))
            return [subsets(vals[:-1], indices_[:-1]), subsets(vals[1:], indices_[1:]), (indices_[0], indices_[-1])]
        else:
            subsetsli.append(tuple(indices_))
            return indices_
    res = subsets(vals, indices_)
    all_pairs = [(i, j) for i in range(nvals) for j in range(nvals - 1, i, -1)]
    rejs = set(rejected)
    not_rejected = list(set(all_pairs) - rejs)
    return (list(rejs), not_rejected, list(set(subsetsli)), res)

def set_partition(ssli):
    if False:
        while True:
            i = 10
    'extract a partition from a list of tuples\n\n    this should be correctly called select largest disjoint sets.\n    Begun and Gabriel 1981 do not seem to be bothered by sets of accepted\n    hypothesis with joint elements,\n    e.g. maximal_accepted_sets = { {1,2,3}, {2,3,4} }\n\n    This creates a set partition from a list of sets given as tuples.\n    It tries to find the partition with the largest sets. That is, sets are\n    included after being sorted by length.\n\n    If the list does not include the singletons, then it will be only a\n    partial partition. Missing items are singletons (I think).\n\n    Examples\n    --------\n    >>> li\n    [(5, 6, 7, 8), (1, 2, 3), (4, 5), (0, 1)]\n    >>> set_partition(li)\n    ([(5, 6, 7, 8), (1, 2, 3)], [0, 4])\n\n    '
    part = []
    for s in sorted(list(set(ssli)), key=len)[::-1]:
        s_ = set(s).copy()
        if not any((set(s_).intersection(set(t)) for t in part)):
            part.append(s)
    missing = list(set((i for ll in ssli for i in ll)) - set((i for ll in part for i in ll)))
    return (part, missing)

def set_remove_subs(ssli):
    if False:
        return 10
    'remove sets that are subsets of another set from a list of tuples\n\n    Parameters\n    ----------\n    ssli : list of tuples\n        each tuple is considered as a set\n\n    Returns\n    -------\n    part : list of tuples\n        new list with subset tuples removed, it is sorted by set-length of tuples. The\n        list contains original tuples, duplicate elements are not removed.\n\n    Examples\n    --------\n    >>> set_remove_subs([(0, 1), (1, 2), (1, 2, 3), (0,)])\n    [(1, 2, 3), (0, 1)]\n    >>> set_remove_subs([(0, 1), (1, 2), (1,1, 1, 2, 3), (0,)])\n    [(1, 1, 1, 2, 3), (0, 1)]\n\n    '
    part = []
    for s in sorted(list(set(ssli)), key=lambda x: len(set(x)))[::-1]:
        if not any((set(s).issubset(set(t)) for t in part)):
            part.append(s)
    return part
if __name__ == '__main__':
    examples = ['tukey', 'tukeycrit', 'fdr', 'fdrmc', 'bonf', 'randmvn', 'multicompdev', 'None']
    if 'tukey' in examples:
        x = np.array([[0, 0, 1]]).T + np.random.randn(3, 20)
        print(Tukeythreegene(*x))
    if 'fdr' in examples or 'bonf' in examples:
        from .ex_multicomp import example_fdr_bonferroni
        example_fdr_bonferroni()
    if 'fdrmc' in examples:
        mcres = mcfdr(nobs=100, nrepl=1000, ntests=30, ntrue=30, mu=0.1, alpha=0.05, rho=0.3)
        mcmeans = np.array(mcres).mean(0)
        print(mcmeans)
        print(mcmeans[0] / 6.0, 1 - mcmeans[1] / 4.0)
        print(mcmeans[:4], mcmeans[-4:])
    if 'randmvn' in examples:
        rvsmvn = randmvn(0.8, (5000, 5))
        print(np.corrcoef(rvsmvn, rowvar=0))
        print(rvsmvn.var(0))
    if 'tukeycrit' in examples:
        print(get_tukeyQcrit(8, 8, alpha=0.05), 5.6)
        print(get_tukeyQcrit(8, 8, alpha=0.01), 7.47)
    if 'multicompdev' in examples:
        X = np.array([[7.68, 1], [7.69, 1], [7.7, 1], [7.7, 1], [7.72, 1], [7.73, 1], [7.73, 1], [7.76, 1], [7.71, 2], [7.73, 2], [7.74, 2], [7.74, 2], [7.78, 2], [7.78, 2], [7.8, 2], [7.81, 2], [7.74, 3], [7.75, 3], [7.77, 3], [7.78, 3], [7.8, 3], [7.81, 3], [7.84, 3], [7.71, 4], [7.71, 4], [7.74, 4], [7.79, 4], [7.81, 4], [7.85, 4], [7.87, 4], [7.91, 4]])
        xli = [X[X[:, 1] == k, 0] for k in range(1, 5)]
        xranks = stats.rankdata(X[:, 0])
        xranksli = [xranks[X[:, 1] == k] for k in range(1, 5)]
        xnobs = np.array([len(xval) for xval in xli])
        meanranks = [item.mean() for item in xranksli]
        sumranks = [item.sum() for item in xranksli]
        stats.norm.sf(0.6744897501960817)
        stats.norm.isf(0.25)
        mrs = np.sort(meanranks)
        (v1, v2) = np.triu_indices(4, 1)
        print('\nsorted rank differences')
        print(mrs[v2] - mrs[v1])
        diffidx = np.argsort(mrs[v2] - mrs[v1])[::-1]
        mrs[v2[diffidx]] - mrs[v1[diffidx]]
        print('\nkruskal for all pairs')
        for (i, j) in zip(v2[diffidx], v1[diffidx]):
            print(i, j, stats.kruskal(xli[i], xli[j]))
            (mwu, mwupval) = stats.mannwhitneyu(xli[i], xli[j], use_continuity=False)
            print(mwu, mwupval * 2, mwupval * 2 < 0.05 / 6.0, mwupval * 2 < 0.1 / 6.0)
        (uni, intlab) = np.unique(X[:, 0], return_inverse=True)
        groupnobs = np.bincount(intlab)
        groupxsum = np.bincount(intlab, weights=X[:, 0])
        groupxmean = groupxsum * 1.0 / groupnobs
        rankraw = X[:, 0].argsort().argsort()
        groupranksum = np.bincount(intlab, weights=rankraw)
        grouprankmean = groupranksum * 1.0 / groupnobs + 1
        assert_almost_equal(grouprankmean[intlab], stats.rankdata(X[:, 0]), 15)
        gs = GroupsStats(X, useranks=True)
        print('\ngroupmeanfilter and grouprankmeans')
        print(gs.groupmeanfilter)
        print(grouprankmean[intlab])
        (xuni, xintlab) = np.unique(X[:, 0], return_inverse=True)
        gs2 = GroupsStats(np.column_stack([X[:, 0], xintlab]), useranks=True)
        rankbincount = np.bincount(xranks.astype(int))
        nties = rankbincount[rankbincount > 1]
        ntot = float(len(xranks))
        tiecorrection = 1 - (nties ** 3 - nties).sum() / (ntot ** 3 - ntot)
        assert_almost_equal(tiecorrection, stats.tiecorrect(xranks), 15)
        print('\ntiecorrection for data and ranks')
        print(tiecorrection)
        print(tiecorrect(xranks))
        tot = X.shape[0]
        t = 500
        f = tot * (tot + 1.0) / 12.0 - t / (6.0 * (tot - 1.0))
        f = tot * (tot + 1.0) / 12.0 / stats.tiecorrect(xranks)
        print('\npairs of mean rank differences')
        for (i, j) in zip(v2[diffidx], v1[diffidx]):
            pdiff = np.abs(meanranks[i] - meanranks[j])
            se = np.sqrt(f * np.sum(1.0 / xnobs[[i, j]]))
            print(i, j, pdiff, se, pdiff / se, pdiff / se > 2.631)
        multicomp = MultiComparison(*X.T)
        multicomp.kruskal()
        gsr = GroupsStats(X, useranks=True)
        print('\nexamples for kruskal multicomparison')
        for i in range(10):
            (x1, x2) = (np.random.randn(30, 2) + np.array([0, 0.5])).T
            skw = stats.kruskal(x1, x2)
            mc2 = MultiComparison(np.r_[x1, x2], np.r_[np.zeros(len(x1)), np.ones(len(x2))])
            newskw = mc2.kruskal()
            print(skw, np.sqrt(skw[0]), skw[1] - newskw, (newskw / skw[1] - 1) * 100)
        (tablett, restt, arrtt) = multicomp.allpairtest(stats.ttest_ind)
        (tablemw, resmw, arrmw) = multicomp.allpairtest(stats.mannwhitneyu)
        print('')
        print(tablett)
        print('')
        print(tablemw)
        (tablemwhs, resmw, arrmw) = multicomp.allpairtest(stats.mannwhitneyu, method='hs')
        print('')
        print(tablemwhs)
    if 'last' in examples:
        xli = (np.random.randn(60, 4) + np.array([0, 0, 0.5, 0.5])).T
        (xrvs, xrvsgr) = catstack(xli)
        multicompr = MultiComparison(xrvs, xrvsgr)
        (tablett, restt, arrtt) = multicompr.allpairtest(stats.ttest_ind)
        print(tablett)
        xli = [[8, 10, 9, 10, 9], [7, 8, 5, 8, 5], [4, 8, 7, 5, 7]]
        (x, labels) = catstack(xli)
        gs4 = GroupsStats(np.column_stack([x, labels]))
        print(gs4.groupvarwithin())
    gmeans = np.array([7.71375, 7.76125, 7.78428571, 7.79875])
    gnobs = np.array([8, 8, 7, 8])
    sd = StepDown(gmeans, gnobs, 0.001, [27])
    pvals = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344, 0.0459, 0.324, 0.4262, 0.5719, 0.6528, 0.759, 1.0]
    print(fdrcorrection0(pvals, alpha=0.05, method='indep'))
    print(fdrcorrection_twostage(pvals, alpha=0.05, iter=False))
    res_tst = fdrcorrection_twostage(pvals, alpha=0.05, iter=False)
    assert_almost_equal([0.047619, 0.0649], res_tst[-1][:2], 3)
    assert_equal(8, res_tst[0].sum())
    print(fdrcorrection_twostage(pvals, alpha=0.05, iter=True))
    print('fdr_gbs', multipletests(pvals, alpha=0.05, method='fdr_gbs'))
    tukey_pvalues(3.649, 3, 16)