"""Inter Rater Agreement

contains
--------
fleiss_kappa
cohens_kappa

aggregate_raters:
    helper function to get data into fleiss_kappa format
to_table:
    helper function to create contingency table, can be used for cohens_kappa

Created on Thu Dec 06 22:57:56 2012
Author: Josef Perktold
License: BSD-3

References
----------
Wikipedia: kappa's initially based on these two pages
    https://en.wikipedia.org/wiki/Fleiss%27_kappa
    https://en.wikipedia.org/wiki/Cohen's_kappa
SAS-Manual : formulas for cohens_kappa, especially variances
see also R package irr

TODO
----
standard errors and hypothesis tests for fleiss_kappa
other statistics and tests,
   in R package irr, SAS has more
inconsistent internal naming, changed variable names as I added more
   functionality
convenience functions to create required data format from raw data
   DONE

"""
import numpy as np
from scipy import stats

class ResultsBunch(dict):
    template = '%r'

    def __init__(self, **kwds):
        if False:
            return 10
        dict.__init__(self, kwds)
        self.__dict__ = self
        self._initialize()

    def _initialize(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def __str__(self):
        if False:
            print('Hello World!')
        return self.template % self

def _int_ifclose(x, dec=1, width=4):
    if False:
        while True:
            i = 10
    "helper function for creating result string for int or float\n\n    only dec=1 and width=4 is implemented\n\n    Parameters\n    ----------\n    x : int or float\n        value to format\n    dec : 1\n        number of decimals to print if x is not an integer\n    width : 4\n        width of string\n\n    Returns\n    -------\n    xint : int or float\n        x is converted to int if it is within 1e-14 of an integer\n    x_string : str\n        x formatted as string, either '%4d' or '%4.1f'\n\n    "
    xint = int(round(x))
    if np.max(np.abs(xint - x)) < 1e-14:
        return (xint, '%4d' % xint)
    else:
        return (x, '%4.1f' % x)

def aggregate_raters(data, n_cat=None):
    if False:
        print('Hello World!')
    'convert raw data with shape (subject, rater) to (subject, cat_counts)\n\n    brings data into correct format for fleiss_kappa\n\n    bincount will raise exception if data cannot be converted to integer.\n\n    Parameters\n    ----------\n    data : array_like, 2-Dim\n        data containing category assignment with subjects in rows and raters\n        in columns.\n    n_cat : None or int\n        If None, then the data is converted to integer categories,\n        0,1,2,...,n_cat-1. Because of the relabeling only category levels\n        with non-zero counts are included.\n        If this is an integer, then the category levels in the data are already\n        assumed to be in integers, 0,1,2,...,n_cat-1. In this case, the\n        returned array may contain columns with zero count, if no subject\n        has been categorized with this level.\n\n    Returns\n    -------\n    arr : nd_array, (n_rows, n_cat)\n        Contains counts of raters that assigned a category level to individuals.\n        Subjects are in rows, category levels in columns.\n    categories : nd_array, (n_category_levels,)\n        Contains the category levels.\n\n    '
    data = np.asarray(data)
    n_rows = data.shape[0]
    if n_cat is None:
        (cat_uni, cat_int) = np.unique(data.ravel(), return_inverse=True)
        n_cat = len(cat_uni)
        data_ = cat_int.reshape(data.shape)
    else:
        cat_uni = np.arange(n_cat)
        data_ = data
    tt = np.zeros((n_rows, n_cat), int)
    for (idx, row) in enumerate(data_):
        ro = np.bincount(row)
        tt[idx, :len(ro)] = ro
    return (tt, cat_uni)

def to_table(data, bins=None):
    if False:
        for i in range(10):
            print('nop')
    'convert raw data with shape (subject, rater) to (rater1, rater2)\n\n    brings data into correct format for cohens_kappa\n\n    Parameters\n    ----------\n    data : array_like, 2-Dim\n        data containing category assignment with subjects in rows and raters\n        in columns.\n    bins : None, int or tuple of array_like\n        If None, then the data is converted to integer categories,\n        0,1,2,...,n_cat-1. Because of the relabeling only category levels\n        with non-zero counts are included.\n        If this is an integer, then the category levels in the data are already\n        assumed to be in integers, 0,1,2,...,n_cat-1. In this case, the\n        returned array may contain columns with zero count, if no subject\n        has been categorized with this level.\n        If bins are a tuple of two array_like, then the bins are directly used\n        by ``numpy.histogramdd``. This is useful if we want to merge categories.\n\n    Returns\n    -------\n    arr : nd_array, (n_cat, n_cat)\n        Contingency table that contains counts of category level with rater1\n        in rows and rater2 in columns.\n\n    Notes\n    -----\n    no NaN handling, delete rows with missing values\n\n    This works also for more than two raters. In that case the dimension of\n    the resulting contingency table is the same as the number of raters\n    instead of 2-dimensional.\n\n    '
    data = np.asarray(data)
    (n_rows, n_cols) = data.shape
    if bins is None:
        (cat_uni, cat_int) = np.unique(data.ravel(), return_inverse=True)
        n_cat = len(cat_uni)
        data_ = cat_int.reshape(data.shape)
        bins_ = np.arange(n_cat + 1) - 0.5
    elif np.isscalar(bins):
        bins_ = np.arange(bins + 1) - 0.5
        data_ = data
    else:
        bins_ = bins
        data_ = data
    tt = np.histogramdd(data_, (bins_,) * n_cols)
    return (tt[0], bins_)

def fleiss_kappa(table, method='fleiss'):
    if False:
        while True:
            i = 10
    'Fleiss\' and Randolph\'s kappa multi-rater agreement measure\n\n    Parameters\n    ----------\n    table : array_like, 2-D\n        assumes subjects in rows, and categories in columns. Convert raw data\n        into this format by using\n        :func:`statsmodels.stats.inter_rater.aggregate_raters`\n    method : str\n        Method \'fleiss\' returns Fleiss\' kappa which uses the sample margin\n        to define the chance outcome.\n        Method \'randolph\' or \'uniform\' (only first 4 letters are needed)\n        returns Randolph\'s (2005) multirater kappa which assumes a uniform\n        distribution of the categories to define the chance outcome.\n\n    Returns\n    -------\n    kappa : float\n        Fleiss\'s or Randolph\'s kappa statistic for inter rater agreement\n\n    Notes\n    -----\n    no variance or hypothesis tests yet\n\n    Interrater agreement measures like Fleiss\'s kappa measure agreement relative\n    to chance agreement. Different authors have proposed ways of defining\n    these chance agreements. Fleiss\' is based on the marginal sample distribution\n    of categories, while Randolph uses a uniform distribution of categories as\n    benchmark. Warrens (2010) showed that Randolph\'s kappa is always larger or\n    equal to Fleiss\' kappa. Under some commonly observed condition, Fleiss\' and\n    Randolph\'s kappa provide lower and upper bounds for two similar kappa_like\n    measures by Light (1971) and Hubert (1977).\n\n    References\n    ----------\n    Wikipedia https://en.wikipedia.org/wiki/Fleiss%27_kappa\n\n    Fleiss, Joseph L. 1971. "Measuring Nominal Scale Agreement among Many\n    Raters." Psychological Bulletin 76 (5): 378-82.\n    https://doi.org/10.1037/h0031619.\n\n    Randolph, Justus J. 2005 "Free-Marginal Multirater Kappa (multirater\n    K [free]): An Alternative to Fleiss\' Fixed-Marginal Multirater Kappa."\n    Presented at the Joensuu Learning and Instruction Symposium, vol. 2005\n    https://eric.ed.gov/?id=ED490661\n\n    Warrens, Matthijs J. 2010. "Inequalities between Multi-Rater Kappas."\n    Advances in Data Analysis and Classification 4 (4): 271-86.\n    https://doi.org/10.1007/s11634-010-0073-4.\n    '
    table = 1.0 * np.asarray(table)
    (n_sub, n_cat) = table.shape
    n_total = table.sum()
    n_rater = table.sum(1)
    n_rat = n_rater.max()
    assert n_total == n_sub * n_rat
    p_cat = table.sum(0) / n_total
    table2 = table * table
    p_rat = (table2.sum(1) - n_rat) / (n_rat * (n_rat - 1.0))
    p_mean = p_rat.mean()
    if method == 'fleiss':
        p_mean_exp = (p_cat * p_cat).sum()
    elif method.startswith('rand') or method.startswith('unif'):
        p_mean_exp = 1 / n_cat
    kappa = (p_mean - p_mean_exp) / (1 - p_mean_exp)
    return kappa

def cohens_kappa(table, weights=None, return_results=True, wt=None):
    if False:
        print('Hello World!')
    'Compute Cohen\'s kappa with variance and equal-zero test\n\n    Parameters\n    ----------\n    table : array_like, 2-Dim\n        square array with results of two raters, one rater in rows, second\n        rater in columns\n    weights : array_like\n        The interpretation of weights depends on the wt argument.\n        If both are None, then the simple kappa is computed.\n        see wt for the case when wt is not None\n        If weights is two dimensional, then it is directly used as a weight\n        matrix. For computing the variance of kappa, the maximum of the\n        weights is assumed to be smaller or equal to one.\n        TODO: fix conflicting definitions in the 2-Dim case for\n    wt : {None, str}\n        If wt and weights are None, then the simple kappa is computed.\n        If wt is given, but weights is None, then the weights are set to\n        be [0, 1, 2, ..., k].\n        If weights is a one-dimensional array, then it is used to construct\n        the weight matrix given the following options.\n\n        wt in [\'linear\', \'ca\' or None] : use linear weights, Cicchetti-Allison\n            actual weights are linear in the score "weights" difference\n        wt in [\'quadratic\', \'fc\'] : use linear weights, Fleiss-Cohen\n            actual weights are squared in the score "weights" difference\n        wt = \'toeplitz\' : weight matrix is constructed as a toeplitz matrix\n            from the one dimensional weights.\n\n    return_results : bool\n        If True (default), then an instance of KappaResults is returned.\n        If False, then only kappa is computed and returned.\n\n    Returns\n    -------\n    results or kappa\n        If return_results is True (default), then a results instance with all\n        statistics is returned\n        If return_results is False, then only kappa is calculated and returned.\n\n    Notes\n    -----\n    There are two conflicting definitions of the weight matrix, Wikipedia\n    versus SAS manual. However, the computation are invariant to rescaling\n    of the weights matrix, so there is no difference in the results.\n\n    Weights for \'linear\' and \'quadratic\' are interpreted as scores for the\n    categories, the weights in the computation are based on the pairwise\n    difference between the scores.\n    Weights for \'toeplitz\' are a interpreted as weighted distance. The distance\n    only depends on how many levels apart two entries in the table are but\n    not on the levels themselves.\n\n    example:\n\n    weights = \'0, 1, 2, 3\' and wt is either linear or toeplitz means that the\n    weighting only depends on the simple distance of levels.\n\n    weights = \'0, 0, 1, 1\' and wt = \'linear\' means that the first two levels\n    are zero distance apart and the same for the last two levels. This is\n    the sample as forming two aggregated levels by merging the first two and\n    the last two levels, respectively.\n\n    weights = [0, 1, 2, 3] and wt = \'quadratic\' is the same as squaring these\n    weights and using wt = \'toeplitz\'.\n\n    References\n    ----------\n    Wikipedia\n    SAS Manual\n\n    '
    table = np.asarray(table, float)
    agree = np.diag(table).sum()
    nobs = table.sum()
    probs = table / nobs
    freqs = probs
    probs_diag = np.diag(probs)
    freq_row = table.sum(1) / nobs
    freq_col = table.sum(0) / nobs
    prob_exp = freq_col * freq_row[:, None]
    assert np.allclose(prob_exp.sum(), 1)
    agree_exp = np.diag(prob_exp).sum()
    if weights is None and wt is None:
        kind = 'Simple'
        kappa = (agree / nobs - agree_exp) / (1 - agree_exp)
        if return_results:
            term_a = probs_diag * (1 - (freq_row + freq_col) * (1 - kappa)) ** 2
            term_a = term_a.sum()
            term_b = probs * (freq_col[:, None] + freq_row) ** 2
            d_idx = np.arange(table.shape[0])
            term_b[d_idx, d_idx] = 0
            term_b = (1 - kappa) ** 2 * term_b.sum()
            term_c = (kappa - agree_exp * (1 - kappa)) ** 2
            var_kappa = (term_a + term_b - term_c) / (1 - agree_exp) ** 2 / nobs
            term_c = freq_col * freq_row * (freq_col + freq_row)
            var_kappa0 = agree_exp + agree_exp ** 2 - term_c.sum()
            var_kappa0 /= (1 - agree_exp) ** 2 * nobs
    else:
        if weights is None:
            weights = np.arange(table.shape[0])
        kind = 'Weighted'
        weights = np.asarray(weights, float)
        if weights.ndim == 1:
            if wt in ['ca', 'linear', None]:
                weights = np.abs(weights[:, None] - weights) / (weights[-1] - weights[0])
            elif wt in ['fc', 'quadratic']:
                weights = (weights[:, None] - weights) ** 2 / (weights[-1] - weights[0]) ** 2
            elif wt == 'toeplitz':
                from scipy.linalg import toeplitz
                weights = toeplitz(weights)
            else:
                raise ValueError('wt option is not known')
        else:
            (rows, cols) = table.shape
            if table.shape != weights.shape:
                raise ValueError('weights are not square')
        kappa = 1 - (weights * table).sum() / nobs / (weights * prob_exp).sum()
        if return_results:
            var_kappa = np.nan
            var_kappa0 = np.nan
            w = 1.0 - weights
            w_row = (freq_col * w).sum(1)
            w_col = (freq_row[:, None] * w).sum(0)
            agree_wexp = (w * freq_col * freq_row[:, None]).sum()
            term_a = freqs * (w - (w_col + w_row[:, None]) * (1 - kappa)) ** 2
            fac = 1.0 / ((1 - agree_wexp) ** 2 * nobs)
            var_kappa = term_a.sum() - (kappa - agree_wexp * (1 - kappa)) ** 2
            var_kappa *= fac
            freqse = freq_col * freq_row[:, None]
            var_kappa0 = (freqse * (w - (w_col + w_row[:, None])) ** 2).sum()
            var_kappa0 -= agree_wexp ** 2
            var_kappa0 *= fac
    kappa_max = (np.minimum(freq_row, freq_col).sum() - agree_exp) / (1 - agree_exp)
    if return_results:
        res = KappaResults(kind=kind, kappa=kappa, kappa_max=kappa_max, weights=weights, var_kappa=var_kappa, var_kappa0=var_kappa0)
        return res
    else:
        return kappa
_kappa_template = '                  %(kind)s Kappa Coefficient\n              --------------------------------\n              Kappa                     %(kappa)6.4f\n              ASE                       %(std_kappa)6.4f\n            %(alpha_ci)s%% Lower Conf Limit      %(kappa_low)6.4f\n            %(alpha_ci)s%% Upper Conf Limit      %(kappa_upp)6.4f\n\n                 Test of H0: %(kind)s Kappa = 0\n\n              ASE under H0              %(std_kappa0)6.4f\n              Z                         %(z_value)6.4f\n              One-sided Pr >  Z         %(pvalue_one_sided)6.4f\n              Two-sided Pr > |Z|        %(pvalue_two_sided)6.4f\n'
'\n                   Weighted Kappa Coefficient\n              --------------------------------\n              Weighted Kappa            0.4701\n              ASE                       0.1457\n              95% Lower Conf Limit      0.1845\n              95% Upper Conf Limit      0.7558\n\n               Test of H0: Weighted Kappa = 0\n\n              ASE under H0              0.1426\n              Z                         3.2971\n              One-sided Pr >  Z         0.0005\n              Two-sided Pr > |Z|        0.0010\n'

class KappaResults(ResultsBunch):
    """Results for Cohen's kappa

    Attributes
    ----------
    kappa : cohen's kappa
    var_kappa : variance of kappa
    std_kappa : standard deviation of kappa
    alpha : one-sided probability for confidence interval
    kappa_low : lower (1-alpha) confidence limit
    kappa_upp : upper (1-alpha) confidence limit
    var_kappa0 : variance of kappa under H0: kappa=0
    std_kappa0 : standard deviation of kappa under H0: kappa=0
    z_value : test statistic for H0: kappa=0, is standard normal distributed
    pvalue_one_sided : one sided p-value for H0: kappa=0 and H1: kappa>0
    pvalue_two_sided : two sided p-value for H0: kappa=0 and H1: kappa!=0
    distribution_kappa : asymptotic normal distribution of kappa
    distribution_zero_null : asymptotic normal distribution of kappa under
        H0: kappa=0

    The confidence interval for kappa and the statistics for the test of
    H0: kappa=0 are based on the asymptotic normal distribution of kappa.

    """
    template = _kappa_template

    def _initialize(self):
        if False:
            print('Hello World!')
        if 'alpha' not in self:
            self['alpha'] = 0.025
            self['alpha_ci'] = _int_ifclose(100 - 0.025 * 200)[1]
        self['std_kappa'] = np.sqrt(self['var_kappa'])
        self['std_kappa0'] = np.sqrt(self['var_kappa0'])
        self['z_value'] = self['kappa'] / self['std_kappa0']
        self['pvalue_one_sided'] = stats.norm.sf(self['z_value'])
        self['pvalue_two_sided'] = stats.norm.sf(np.abs(self['z_value'])) * 2
        delta = stats.norm.isf(self['alpha']) * self['std_kappa']
        self['kappa_low'] = self['kappa'] - delta
        self['kappa_upp'] = self['kappa'] + delta
        self['distribution_kappa'] = stats.norm(loc=self['kappa'], scale=self['std_kappa'])
        self['distribution_zero_null'] = stats.norm(loc=0, scale=self['std_kappa0'])

    def __str__(self):
        if False:
            print('Hello World!')
        return self.template % self