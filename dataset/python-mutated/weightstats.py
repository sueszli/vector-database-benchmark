"""Tests and descriptive statistics with weights


Created on 2010-09-18

Author: josef-pktd
License: BSD (3-clause)


References
----------
SPSS manual
SAS manual

This follows in large parts the SPSS manual, which is largely the same as
the SAS manual with different, simpler notation.

Freq, Weight in SAS seems redundant since they always show up as product, SPSS
has only weights.

Notes
-----

This has potential problems with ddof, I started to follow numpy with ddof=0
by default and users can change it, but this might still mess up the t-tests,
since the estimates for the standard deviation will be based on the ddof that
the user chooses.
- fixed ddof for the meandiff ttest, now matches scipy.stats.ttest_ind

Note: scipy has now a separate, pooled variance option in ttest, but I have not
compared yet.

"""
import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly

class DescrStatsW:
    """
    Descriptive statistics and tests with weights for case weights

    Assumes that the data is 1d or 2d with (nobs, nvars) observations in rows,
    variables in columns, and that the same weight applies to each column.

    If degrees of freedom correction is used, then weights should add up to the
    number of observations. ttest also assumes that the sum of weights
    corresponds to the sample size.

    This is essentially the same as replicating each observations by its
    weight, if the weights are integers, often called case or frequency weights.

    Parameters
    ----------
    data : array_like, 1-D or 2-D
        dataset
    weights : None or 1-D ndarray
        weights for each observation, with same length as zero axis of data
    ddof : int
        default ddof=0, degrees of freedom correction used for second moments,
        var, std, cov, corrcoef.
        However, statistical tests are independent of `ddof`, based on the
        standard formulas.

    Examples
    --------

    >>> import numpy as np
    >>> np.random.seed(0)
    >>> x1_2d = 1.0 + np.random.randn(20, 3)
    >>> w1 = np.random.randint(1, 4, 20)
    >>> d1 = DescrStatsW(x1_2d, weights=w1)
    >>> d1.mean
    array([ 1.42739844,  1.23174284,  1.083753  ])
    >>> d1.var
    array([ 0.94855633,  0.52074626,  1.12309325])
    >>> d1.std_mean
    array([ 0.14682676,  0.10878944,  0.15976497])

    >>> tstat, pval, df = d1.ttest_mean(0)
    >>> tstat; pval; df
    array([  9.72165021,  11.32226471,   6.78342055])
    array([  1.58414212e-12,   1.26536887e-14,   2.37623126e-08])
    44.0

    >>> tstat, pval, df = d1.ttest_mean([0, 1, 1])
    >>> tstat; pval; df
    array([ 9.72165021,  2.13019609,  0.52422632])
    array([  1.58414212e-12,   3.87842808e-02,   6.02752170e-01])
    44.0

    # if weights are integers, then asrepeats can be used

    >>> x1r = d1.asrepeats()
    >>> x1r.shape
    ...
    >>> stats.ttest_1samp(x1r, [0, 1, 1])
    ...

    """

    def __init__(self, data, weights=None, ddof=0):
        if False:
            for i in range(10):
                print('nop')
        self.data = np.asarray(data)
        if weights is None:
            self.weights = np.ones(self.data.shape[0])
        else:
            self.weights = np.asarray(weights).astype(float)
            if len(self.weights.shape) > 1 and len(self.weights) > 1:
                self.weights = self.weights.squeeze()
        self.ddof = ddof

    @cache_readonly
    def sum_weights(self):
        if False:
            i = 10
            return i + 15
        'Sum of weights'
        return self.weights.sum(0)

    @cache_readonly
    def nobs(self):
        if False:
            for i in range(10):
                print('nop')
        'alias for number of observations/cases, equal to sum of weights\n        '
        return self.sum_weights

    @cache_readonly
    def sum(self):
        if False:
            for i in range(10):
                print('nop')
        'weighted sum of data'
        return np.dot(self.data.T, self.weights)

    @cache_readonly
    def mean(self):
        if False:
            while True:
                i = 10
        'weighted mean of data'
        return self.sum / self.sum_weights

    @cache_readonly
    def demeaned(self):
        if False:
            while True:
                i = 10
        'data with weighted mean subtracted'
        return self.data - self.mean

    @cache_readonly
    def sumsquares(self):
        if False:
            print('Hello World!')
        'weighted sum of squares of demeaned data'
        return np.dot((self.demeaned ** 2).T, self.weights)

    def var_ddof(self, ddof=0):
        if False:
            return 10
        'variance of data given ddof\n\n        Parameters\n        ----------\n        ddof : int, float\n            degrees of freedom correction, independent of attribute ddof\n\n        Returns\n        -------\n        var : float, ndarray\n            variance with denominator ``sum_weights - ddof``\n        '
        return self.sumsquares / (self.sum_weights - ddof)

    def std_ddof(self, ddof=0):
        if False:
            return 10
        'standard deviation of data with given ddof\n\n        Parameters\n        ----------\n        ddof : int, float\n            degrees of freedom correction, independent of attribute ddof\n\n        Returns\n        -------\n        std : float, ndarray\n            standard deviation with denominator ``sum_weights - ddof``\n        '
        return np.sqrt(self.var_ddof(ddof=ddof))

    @cache_readonly
    def var(self):
        if False:
            return 10
        'variance with default degrees of freedom correction\n        '
        return self.sumsquares / (self.sum_weights - self.ddof)

    @cache_readonly
    def _var(self):
        if False:
            print('Hello World!')
        'variance without degrees of freedom correction\n\n        used for statistical tests with controlled ddof\n        '
        return self.sumsquares / self.sum_weights

    @cache_readonly
    def std(self):
        if False:
            for i in range(10):
                print('nop')
        'standard deviation with default degrees of freedom correction\n        '
        return np.sqrt(self.var)

    @cache_readonly
    def cov(self):
        if False:
            for i in range(10):
                print('nop')
        'weighted covariance of data if data is 2 dimensional\n\n        assumes variables in columns and observations in rows\n        uses default ddof\n        '
        cov_ = np.dot(self.weights * self.demeaned.T, self.demeaned)
        cov_ /= self.sum_weights - self.ddof
        return cov_

    @cache_readonly
    def corrcoef(self):
        if False:
            print('Hello World!')
        'weighted correlation with default ddof\n\n        assumes variables in columns and observations in rows\n        '
        return self.cov / self.std / self.std[:, None]

    @cache_readonly
    def std_mean(self):
        if False:
            i = 10
            return i + 15
        'standard deviation of weighted mean\n        '
        std = self.std
        if self.ddof != 0:
            std = std * np.sqrt((self.sum_weights - self.ddof) / self.sum_weights)
        return std / np.sqrt(self.sum_weights - 1)

    def quantile(self, probs, return_pandas=True):
        if False:
            i = 10
            return i + 15
        '\n        Compute quantiles for a weighted sample.\n\n        Parameters\n        ----------\n        probs : array_like\n            A vector of probability points at which to calculate the\n            quantiles.  Each element of `probs` should fall in [0, 1].\n        return_pandas : bool\n            If True, return value is a Pandas DataFrame or Series.\n            Otherwise returns a ndarray.\n\n        Returns\n        -------\n        quantiles : Series, DataFrame, or ndarray\n            If `return_pandas` = True, returns one of the following:\n              * data are 1d, `return_pandas` = True: a Series indexed by\n                the probability points.\n              * data are 2d, `return_pandas` = True: a DataFrame with\n                the probability points as row index and the variables\n                as column index.\n\n            If `return_pandas` = False, returns an ndarray containing the\n            same values as the Series/DataFrame.\n\n        Notes\n        -----\n        To compute the quantiles, first, the weights are summed over\n        exact ties yielding distinct data values y_1 < y_2 < ..., and\n        corresponding weights w_1, w_2, ....  Let s_j denote the sum\n        of the first j weights, and let W denote the sum of all the\n        weights.  For a probability point p, if pW falls strictly\n        between s_j and s_{j+1} then the estimated quantile is\n        y_{j+1}.  If pW = s_j then the estimated quantile is (y_j +\n        y_{j+1})/2.  If pW < p_1 then the estimated quantile is y_1.\n\n        References\n        ----------\n        SAS documentation for weighted quantiles:\n\n        https://support.sas.com/documentation/cdl/en/procstat/63104/HTML/default/viewer.htm#procstat_univariate_sect028.htm\n        '
        import pandas as pd
        probs = np.asarray(probs)
        probs = np.atleast_1d(probs)
        if self.data.ndim == 1:
            rslt = self._quantile(self.data, probs)
            if return_pandas:
                rslt = pd.Series(rslt, index=probs)
        else:
            rslt = []
            for vec in self.data.T:
                rslt.append(self._quantile(vec, probs))
            rslt = np.column_stack(rslt)
            if return_pandas:
                columns = ['col%d' % (j + 1) for j in range(rslt.shape[1])]
                rslt = pd.DataFrame(data=rslt, columns=columns, index=probs)
        if return_pandas:
            rslt.index.name = 'p'
        return rslt

    def _quantile(self, vec, probs):
        if False:
            i = 10
            return i + 15
        import pandas as pd
        df = pd.DataFrame(index=np.arange(len(self.weights)))
        df['weights'] = self.weights
        df['vec'] = vec
        dfg = df.groupby('vec').agg('sum')
        weights = dfg.values[:, 0]
        values = np.asarray(dfg.index)
        cweights = np.cumsum(weights)
        totwt = cweights[-1]
        targets = probs * totwt
        ii = np.searchsorted(cweights, targets)
        rslt = values[ii]
        jj = np.flatnonzero(np.abs(targets - cweights[ii]) < 1e-10)
        jj = jj[ii[jj] < len(cweights) - 1]
        rslt[jj] = (values[ii[jj]] + values[ii[jj] + 1]) / 2
        return rslt

    def tconfint_mean(self, alpha=0.05, alternative='two-sided'):
        if False:
            for i in range(10):
                print('nop')
        "two-sided confidence interval for weighted mean of data\n\n        If the data is 2d, then these are separate confidence intervals\n        for each column.\n\n        Parameters\n        ----------\n        alpha : float\n            significance level for the confidence interval, coverage is\n            ``1-alpha``\n        alternative : str\n            This specifies the alternative hypothesis for the test that\n            corresponds to the confidence interval.\n            The alternative hypothesis, H1, has to be one of the following\n\n              'two-sided': H1: mean not equal to value (default)\n              'larger' :   H1: mean larger than value\n              'smaller' :  H1: mean smaller than value\n\n        Returns\n        -------\n        lower, upper : floats or ndarrays\n            lower and upper bound of confidence interval\n\n        Notes\n        -----\n        In a previous version, statsmodels 0.4, alpha was the confidence\n        level, e.g. 0.95\n        "
        dof = self.sum_weights - 1
        ci = _tconfint_generic(self.mean, self.std_mean, dof, alpha, alternative)
        return ci

    def zconfint_mean(self, alpha=0.05, alternative='two-sided'):
        if False:
            while True:
                i = 10
        "two-sided confidence interval for weighted mean of data\n\n        Confidence interval is based on normal distribution.\n        If the data is 2d, then these are separate confidence intervals\n        for each column.\n\n        Parameters\n        ----------\n        alpha : float\n            significance level for the confidence interval, coverage is\n            ``1-alpha``\n        alternative : str\n            This specifies the alternative hypothesis for the test that\n            corresponds to the confidence interval.\n            The alternative hypothesis, H1, has to be one of the following\n\n              'two-sided': H1: mean not equal to value (default)\n              'larger' :   H1: mean larger than value\n              'smaller' :  H1: mean smaller than value\n\n        Returns\n        -------\n        lower, upper : floats or ndarrays\n            lower and upper bound of confidence interval\n\n        Notes\n        -----\n        In a previous version, statsmodels 0.4, alpha was the confidence\n        level, e.g. 0.95\n        "
        return _zconfint_generic(self.mean, self.std_mean, alpha, alternative)

    def ttest_mean(self, value=0, alternative='two-sided'):
        if False:
            for i in range(10):
                print('nop')
        "ttest of Null hypothesis that mean is equal to value.\n\n        The alternative hypothesis H1 is defined by the following\n\n        - 'two-sided': H1: mean not equal to value\n        - 'larger' :   H1: mean larger than value\n        - 'smaller' :  H1: mean smaller than value\n\n        Parameters\n        ----------\n        value : float or array\n            the hypothesized value for the mean\n        alternative : str\n            The alternative hypothesis, H1, has to be one of the following:\n\n              - 'two-sided': H1: mean not equal to value (default)\n              - 'larger' :   H1: mean larger than value\n              - 'smaller' :  H1: mean smaller than value\n\n        Returns\n        -------\n        tstat : float\n            test statistic\n        pvalue : float\n            pvalue of the t-test\n        df : int or float\n\n        "
        tstat = (self.mean - value) / self.std_mean
        dof = self.sum_weights - 1
        if alternative == 'two-sided':
            pvalue = stats.t.sf(np.abs(tstat), dof) * 2
        elif alternative == 'larger':
            pvalue = stats.t.sf(tstat, dof)
        elif alternative == 'smaller':
            pvalue = stats.t.cdf(tstat, dof)
        else:
            raise ValueError('alternative not recognized')
        return (tstat, pvalue, dof)

    def ttost_mean(self, low, upp):
        if False:
            return 10
        'test of (non-)equivalence of one sample\n\n        TOST: two one-sided t tests\n\n        null hypothesis:  m < low or m > upp\n        alternative hypothesis:  low < m < upp\n\n        where m is the expected value of the sample (mean of the population).\n\n        If the pvalue is smaller than a threshold, say 0.05, then we reject the\n        hypothesis that the expected value of the sample (mean of the\n        population) is outside of the interval given by thresholds low and upp.\n\n        Parameters\n        ----------\n        low, upp : float\n            equivalence interval low < mean < upp\n\n        Returns\n        -------\n        pvalue : float\n            pvalue of the non-equivalence test\n        t1, pv1, df1 : tuple\n            test statistic, pvalue and degrees of freedom for lower threshold\n            test\n        t2, pv2, df2 : tuple\n            test statistic, pvalue and degrees of freedom for upper threshold\n            test\n\n        '
        (t1, pv1, df1) = self.ttest_mean(low, alternative='larger')
        (t2, pv2, df2) = self.ttest_mean(upp, alternative='smaller')
        return (np.maximum(pv1, pv2), (t1, pv1, df1), (t2, pv2, df2))

    def ztest_mean(self, value=0, alternative='two-sided'):
        if False:
            return 10
        "z-test of Null hypothesis that mean is equal to value.\n\n        The alternative hypothesis H1 is defined by the following\n        'two-sided': H1: mean not equal to value\n        'larger' :   H1: mean larger than value\n        'smaller' :  H1: mean smaller than value\n\n        Parameters\n        ----------\n        value : float or array\n            the hypothesized value for the mean\n        alternative : str\n            The alternative hypothesis, H1, has to be one of the following\n\n              'two-sided': H1: mean not equal to value (default)\n              'larger' :   H1: mean larger than value\n              'smaller' :  H1: mean smaller than value\n\n        Returns\n        -------\n        tstat : float\n            test statistic\n        pvalue : float\n            pvalue of the t-test\n\n        Notes\n        -----\n        This uses the same degrees of freedom correction as the t-test in the\n        calculation of the standard error of the mean, i.e it uses\n        `(sum_weights - 1)` instead of `sum_weights` in the denominator.\n        See Examples below for the difference.\n\n        Examples\n        --------\n\n        z-test on a proportion, with 20 observations, 15 of those are our event\n\n        >>> import statsmodels.api as sm\n        >>> x1 = [0, 1]\n        >>> w1 = [5, 15]\n        >>> d1 = sm.stats.DescrStatsW(x1, w1)\n        >>> d1.ztest_mean(0.5)\n        (2.5166114784235836, 0.011848940928347452)\n\n        This differs from the proportions_ztest because of the degrees of\n        freedom correction:\n        >>> sm.stats.proportions_ztest(15, 20.0, value=0.5)\n        (2.5819888974716112, 0.009823274507519247).\n\n        We can replicate the results from ``proportions_ztest`` if we increase\n        the weights to have artificially one more observation:\n\n        >>> sm.stats.DescrStatsW(x1, np.array(w1)*21./20).ztest_mean(0.5)\n        (2.5819888974716116, 0.0098232745075192366)\n        "
        tstat = (self.mean - value) / self.std_mean
        if alternative == 'two-sided':
            pvalue = stats.norm.sf(np.abs(tstat)) * 2
        elif alternative == 'larger':
            pvalue = stats.norm.sf(tstat)
        elif alternative == 'smaller':
            pvalue = stats.norm.cdf(tstat)
        return (tstat, pvalue)

    def ztost_mean(self, low, upp):
        if False:
            while True:
                i = 10
        'test of (non-)equivalence of one sample, based on z-test\n\n        TOST: two one-sided z-tests\n\n        null hypothesis:  m < low or m > upp\n        alternative hypothesis:  low < m < upp\n\n        where m is the expected value of the sample (mean of the population).\n\n        If the pvalue is smaller than a threshold, say 0.05, then we reject the\n        hypothesis that the expected value of the sample (mean of the\n        population) is outside of the interval given by thresholds low and upp.\n\n        Parameters\n        ----------\n        low, upp : float\n            equivalence interval low < mean < upp\n\n        Returns\n        -------\n        pvalue : float\n            pvalue of the non-equivalence test\n        t1, pv1 : tuple\n            test statistic and p-value for lower threshold test\n        t2, pv2 : tuple\n            test statistic and p-value for upper threshold test\n\n        '
        (t1, pv1) = self.ztest_mean(low, alternative='larger')
        (t2, pv2) = self.ztest_mean(upp, alternative='smaller')
        return (np.maximum(pv1, pv2), (t1, pv1), (t2, pv2))

    def get_compare(self, other, weights=None):
        if False:
            print('Hello World!')
        'return an instance of CompareMeans with self and other\n\n        Parameters\n        ----------\n        other : array_like or instance of DescrStatsW\n            If array_like then this creates an instance of DescrStatsW with\n            the given weights.\n        weights : None or array\n            weights are only used if other is not an instance of DescrStatsW\n\n        Returns\n        -------\n        cm : instance of CompareMeans\n            the instance has self attached as d1 and other as d2.\n\n        See Also\n        --------\n        CompareMeans\n\n        '
        if not isinstance(other, self.__class__):
            d2 = DescrStatsW(other, weights)
        else:
            d2 = other
        return CompareMeans(self, d2)

    def asrepeats(self):
        if False:
            i = 10
            return i + 15
        'get array that has repeats given by floor(weights)\n\n        observations with weight=0 are dropped\n\n        '
        w_int = np.floor(self.weights).astype(int)
        return np.repeat(self.data, w_int, axis=0)

def _tstat_generic(value1, value2, std_diff, dof, alternative, diff=0):
    if False:
        return 10
    "generic ttest based on summary statistic\n\n    The test statistic is :\n        tstat = (value1 - value2 - diff) / std_diff\n\n    and is assumed to be t-distributed with ``dof`` degrees of freedom.\n\n    Parameters\n    ----------\n    value1 : float or ndarray\n        Value, for example mean, of the first sample.\n    value2 : float or ndarray\n        Value, for example mean, of the second sample.\n    std_diff : float or ndarray\n        Standard error of the difference value1 - value2\n    dof : int or float\n        Degrees of freedom\n    alternative : str\n        The alternative hypothesis, H1, has to be one of the following\n\n           * 'two-sided' : H1: ``value1 - value2 - diff`` not equal to 0.\n           * 'larger' :   H1: ``value1 - value2 - diff > 0``\n           * 'smaller' :  H1: ``value1 - value2 - diff < 0``\n\n    diff : float\n        value of difference ``value1 - value2`` under the null hypothesis\n\n    Returns\n    -------\n    tstat : float or ndarray\n        Test statistic.\n    pvalue : float or ndarray\n        P-value of the hypothesis test assuming that the test statistic is\n        t-distributed with ``df`` degrees of freedom.\n    "
    tstat = (value1 - value2 - diff) / std_diff
    if alternative in ['two-sided', '2-sided', '2s']:
        pvalue = stats.t.sf(np.abs(tstat), dof) * 2
    elif alternative in ['larger', 'l']:
        pvalue = stats.t.sf(tstat, dof)
    elif alternative in ['smaller', 's']:
        pvalue = stats.t.cdf(tstat, dof)
    else:
        raise ValueError('invalid alternative')
    return (tstat, pvalue)

def _tconfint_generic(mean, std_mean, dof, alpha, alternative):
    if False:
        for i in range(10):
            print('nop')
    'generic t-confint based on summary statistic\n\n    Parameters\n    ----------\n    mean : float or ndarray\n        Value, for example mean, of the first sample.\n    std_mean : float or ndarray\n        Standard error of the difference value1 - value2\n    dof : int or float\n        Degrees of freedom\n    alpha : float\n        Significance level for the confidence interval, coverage is\n        ``1-alpha``.\n    alternative : str\n        The alternative hypothesis, H1, has to be one of the following\n\n           * \'two-sided\' : H1: ``value1 - value2 - diff`` not equal to 0.\n           * \'larger\' :   H1: ``value1 - value2 - diff > 0``\n           * \'smaller\' :  H1: ``value1 - value2 - diff < 0``\n\n    Returns\n    -------\n    lower : float or ndarray\n        Lower confidence limit. This is -inf for the one-sided alternative\n        "smaller".\n    upper : float or ndarray\n        Upper confidence limit. This is inf for the one-sided alternative\n        "larger".\n    '
    if alternative in ['two-sided', '2-sided', '2s']:
        tcrit = stats.t.ppf(1 - alpha / 2.0, dof)
        lower = mean - tcrit * std_mean
        upper = mean + tcrit * std_mean
    elif alternative in ['larger', 'l']:
        tcrit = stats.t.ppf(alpha, dof)
        lower = mean + tcrit * std_mean
        upper = np.inf
    elif alternative in ['smaller', 's']:
        tcrit = stats.t.ppf(1 - alpha, dof)
        lower = -np.inf
        upper = mean + tcrit * std_mean
    else:
        raise ValueError('invalid alternative')
    return (lower, upper)

def _zstat_generic(value1, value2, std_diff, alternative, diff=0):
    if False:
        i = 10
        return i + 15
    "generic (normal) z-test based on summary statistic\n\n    The test statistic is :\n        tstat = (value1 - value2 - diff) / std_diff\n\n    and is assumed to be normally distributed.\n\n    Parameters\n    ----------\n    value1 : float or ndarray\n        Value, for example mean, of the first sample.\n    value2 : float or ndarray\n        Value, for example mean, of the second sample.\n    std_diff : float or ndarray\n        Standard error of the difference value1 - value2\n    alternative : str\n        The alternative hypothesis, H1, has to be one of the following\n\n           * 'two-sided' : H1: ``value1 - value2 - diff`` not equal to 0.\n           * 'larger' :   H1: ``value1 - value2 - diff > 0``\n           * 'smaller' :  H1: ``value1 - value2 - diff < 0``\n\n    diff : float\n        value of difference ``value1 - value2`` under the null hypothesis\n\n    Returns\n    -------\n    tstat : float or ndarray\n        Test statistic.\n    pvalue : float or ndarray\n        P-value of the hypothesis test assuming that the test statistic is\n        t-distributed with ``df`` degrees of freedom.\n    "
    zstat = (value1 - value2 - diff) / std_diff
    if alternative in ['two-sided', '2-sided', '2s']:
        pvalue = stats.norm.sf(np.abs(zstat)) * 2
    elif alternative in ['larger', 'l']:
        pvalue = stats.norm.sf(zstat)
    elif alternative in ['smaller', 's']:
        pvalue = stats.norm.cdf(zstat)
    else:
        raise ValueError('invalid alternative')
    return (zstat, pvalue)

def _zstat_generic2(value, std, alternative):
    if False:
        i = 10
        return i + 15
    "generic (normal) z-test based on summary statistic\n\n    The test statistic is :\n        zstat = value / std\n\n    and is assumed to be normally distributed with standard deviation ``std``.\n\n    Parameters\n    ----------\n    value : float or ndarray\n        Value of a sample statistic, for example mean.\n    value2 : float or ndarray\n        Value, for example mean, of the second sample.\n    std : float or ndarray\n        Standard error of the sample statistic value.\n    alternative : str\n        The alternative hypothesis, H1, has to be one of the following\n\n           * 'two-sided' : H1: ``value1 - value2 - diff`` not equal to 0.\n           * 'larger' :   H1: ``value1 - value2 - diff > 0``\n           * 'smaller' :  H1: ``value1 - value2 - diff < 0``\n\n    Returns\n    -------\n    zstat : float or ndarray\n        Test statistic.\n    pvalue : float or ndarray\n        P-value of the hypothesis test assuming that the test statistic is\n        normally distributed.\n    "
    zstat = value / std
    if alternative in ['two-sided', '2-sided', '2s']:
        pvalue = stats.norm.sf(np.abs(zstat)) * 2
    elif alternative in ['larger', 'l']:
        pvalue = stats.norm.sf(zstat)
    elif alternative in ['smaller', 's']:
        pvalue = stats.norm.cdf(zstat)
    else:
        raise ValueError('invalid alternative')
    return (zstat, pvalue)

def _zconfint_generic(mean, std_mean, alpha, alternative):
    if False:
        i = 10
        return i + 15
    'generic normal-confint based on summary statistic\n\n    Parameters\n    ----------\n    mean : float or ndarray\n        Value, for example mean, of the first sample.\n    std_mean : float or ndarray\n        Standard error of the difference value1 - value2\n    alpha : float\n        Significance level for the confidence interval, coverage is\n        ``1-alpha``\n    alternative : str\n        The alternative hypothesis, H1, has to be one of the following\n\n           * \'two-sided\' : H1: ``value1 - value2 - diff`` not equal to 0.\n           * \'larger\' :   H1: ``value1 - value2 - diff > 0``\n           * \'smaller\' :  H1: ``value1 - value2 - diff < 0``\n\n    Returns\n    -------\n    lower : float or ndarray\n        Lower confidence limit. This is -inf for the one-sided alternative\n        "smaller".\n    upper : float or ndarray\n        Upper confidence limit. This is inf for the one-sided alternative\n        "larger".\n    '
    if alternative in ['two-sided', '2-sided', '2s']:
        zcrit = stats.norm.ppf(1 - alpha / 2.0)
        lower = mean - zcrit * std_mean
        upper = mean + zcrit * std_mean
    elif alternative in ['larger', 'l']:
        zcrit = stats.norm.ppf(alpha)
        lower = mean + zcrit * std_mean
        upper = np.inf
    elif alternative in ['smaller', 's']:
        zcrit = stats.norm.ppf(1 - alpha)
        lower = -np.inf
        upper = mean + zcrit * std_mean
    else:
        raise ValueError('invalid alternative')
    return (lower, upper)

class CompareMeans:
    """class for two sample comparison

    The tests and the confidence interval work for multi-endpoint comparison:
    If d1 and d2 have the same number of rows, then each column of the data
    in d1 is compared with the corresponding column in d2.

    Parameters
    ----------
    d1, d2 : instances of DescrStatsW

    Notes
    -----
    The result for the statistical tests and the confidence interval are
    independent of the user specified ddof.

    TODO: Extend to any number of groups or write a version that works in that
    case, like in SAS and SPSS.

    """

    def __init__(self, d1, d2):
        if False:
            i = 10
            return i + 15
        'assume d1, d2 hold the relevant attributes\n\n        '
        self.d1 = d1
        self.d2 = d2

    @classmethod
    def from_data(cls, data1, data2, weights1=None, weights2=None, ddof1=0, ddof2=0):
        if False:
            while True:
                i = 10
        'construct a CompareMeans object from data\n\n        Parameters\n        ----------\n        data1, data2 : array_like, 1-D or 2-D\n            compared datasets\n        weights1, weights2 : None or 1-D ndarray\n            weights for each observation of data1 and data2 respectively,\n            with same length as zero axis of corresponding dataset.\n        ddof1, ddof2 : int\n            default ddof1=0, ddof2=0, degrees of freedom for data1,\n            data2 respectively.\n\n        Returns\n        -------\n        A CompareMeans instance.\n\n        '
        return cls(DescrStatsW(data1, weights=weights1, ddof=ddof1), DescrStatsW(data2, weights=weights2, ddof=ddof2))

    def summary(self, use_t=True, alpha=0.05, usevar='pooled', value=0):
        if False:
            i = 10
            return i + 15
        "summarize the results of the hypothesis test\n\n        Parameters\n        ----------\n        use_t : bool, optional\n            if use_t is True, then t test results are returned\n            if use_t is False, then z test results are returned\n        alpha : float\n            significance level for the confidence interval, coverage is\n            ``1-alpha``\n        usevar : str, 'pooled' or 'unequal'\n            If ``pooled``, then the standard deviation of the samples is\n            assumed to be the same. If ``unequal``, then the variance of\n            Welch ttest will be used, and the degrees of freedom are those\n            of Satterthwaite if ``use_t`` is True.\n        value : float\n            difference between the means under the Null hypothesis.\n\n        Returns\n        -------\n        smry : SimpleTable\n\n        "
        d1 = self.d1
        d2 = self.d2
        confint_percents = 100 - alpha * 100
        if use_t:
            (tstat, pvalue, _) = self.ttest_ind(usevar=usevar, value=value)
            (lower, upper) = self.tconfint_diff(alpha=alpha, usevar=usevar)
        else:
            (tstat, pvalue) = self.ztest_ind(usevar=usevar, value=value)
            (lower, upper) = self.zconfint_diff(alpha=alpha, usevar=usevar)
        if usevar == 'pooled':
            std_err = self.std_meandiff_pooledvar
        else:
            std_err = self.std_meandiff_separatevar
        std_err = np.atleast_1d(std_err)
        tstat = np.atleast_1d(tstat)
        pvalue = np.atleast_1d(pvalue)
        lower = np.atleast_1d(lower)
        upper = np.atleast_1d(upper)
        conf_int = np.column_stack((lower, upper))
        params = np.atleast_1d(d1.mean - d2.mean - value)
        title = 'Test for equality of means'
        yname = 'y'
        xname = ['subset #%d' % (ii + 1) for ii in range(tstat.shape[0])]
        from statsmodels.iolib.summary import summary_params
        return summary_params((None, params, std_err, tstat, pvalue, conf_int), alpha=alpha, use_t=use_t, yname=yname, xname=xname, title=title)

    @cache_readonly
    def std_meandiff_separatevar(self):
        if False:
            return 10
        d1 = self.d1
        d2 = self.d2
        return np.sqrt(d1._var / (d1.nobs - 1) + d2._var / (d2.nobs - 1))

    @cache_readonly
    def std_meandiff_pooledvar(self):
        if False:
            while True:
                i = 10
        'variance assuming equal variance in both data sets\n\n        '
        d1 = self.d1
        d2 = self.d2
        var_pooled = (d1.sumsquares + d2.sumsquares) / (d1.nobs - 1 + d2.nobs - 1)
        return np.sqrt(var_pooled * (1.0 / d1.nobs + 1.0 / d2.nobs))

    def dof_satt(self):
        if False:
            while True:
                i = 10
        'degrees of freedom of Satterthwaite for unequal variance\n        '
        d1 = self.d1
        d2 = self.d2
        sem1 = d1._var / (d1.nobs - 1)
        sem2 = d2._var / (d2.nobs - 1)
        semsum = sem1 + sem2
        z1 = (sem1 / semsum) ** 2 / (d1.nobs - 1)
        z2 = (sem2 / semsum) ** 2 / (d2.nobs - 1)
        dof = 1.0 / (z1 + z2)
        return dof

    def ttest_ind(self, alternative='two-sided', usevar='pooled', value=0):
        if False:
            while True:
                i = 10
        "ttest for the null hypothesis of identical means\n\n        this should also be the same as onewaygls, except for ddof differences\n\n        Parameters\n        ----------\n        x1 : array_like, 1-D or 2-D\n            first of the two independent samples, see notes for 2-D case\n        x2 : array_like, 1-D or 2-D\n            second of the two independent samples, see notes for 2-D case\n        alternative : str\n            The alternative hypothesis, H1, has to be one of the following\n            'two-sided': H1: difference in means not equal to value (default)\n            'larger' :   H1: difference in means larger than value\n            'smaller' :  H1: difference in means smaller than value\n\n        usevar : str, 'pooled' or 'unequal'\n            If ``pooled``, then the standard deviation of the samples is assumed to be\n            the same. If ``unequal``, then Welch ttest with Satterthwait degrees\n            of freedom is used\n        value : float\n            difference between the means under the Null hypothesis.\n\n\n        Returns\n        -------\n        tstat : float\n            test statistic\n        pvalue : float\n            pvalue of the t-test\n        df : int or float\n            degrees of freedom used in the t-test\n\n        Notes\n        -----\n        The result is independent of the user specified ddof.\n\n        "
        d1 = self.d1
        d2 = self.d2
        if usevar == 'pooled':
            stdm = self.std_meandiff_pooledvar
            dof = d1.nobs - 1 + d2.nobs - 1
        elif usevar == 'unequal':
            stdm = self.std_meandiff_separatevar
            dof = self.dof_satt()
        else:
            raise ValueError('usevar can only be "pooled" or "unequal"')
        (tstat, pval) = _tstat_generic(d1.mean, d2.mean, stdm, dof, alternative, diff=value)
        return (tstat, pval, dof)

    def ztest_ind(self, alternative='two-sided', usevar='pooled', value=0):
        if False:
            print('Hello World!')
        "z-test for the null hypothesis of identical means\n\n        Parameters\n        ----------\n        x1 : array_like, 1-D or 2-D\n            first of the two independent samples, see notes for 2-D case\n        x2 : array_like, 1-D or 2-D\n            second of the two independent samples, see notes for 2-D case\n        alternative : str\n            The alternative hypothesis, H1, has to be one of the following\n            'two-sided': H1: difference in means not equal to value (default)\n            'larger' :   H1: difference in means larger than value\n            'smaller' :  H1: difference in means smaller than value\n\n        usevar : str, 'pooled' or 'unequal'\n            If ``pooled``, then the standard deviation of the samples is assumed to be\n            the same. If ``unequal``, then the standard deviations of the samples may\n            be different.\n        value : float\n            difference between the means under the Null hypothesis.\n\n        Returns\n        -------\n        tstat : float\n            test statistic\n        pvalue : float\n            pvalue of the z-test\n\n        "
        d1 = self.d1
        d2 = self.d2
        if usevar == 'pooled':
            stdm = self.std_meandiff_pooledvar
        elif usevar == 'unequal':
            stdm = self.std_meandiff_separatevar
        else:
            raise ValueError('usevar can only be "pooled" or "unequal"')
        (tstat, pval) = _zstat_generic(d1.mean, d2.mean, stdm, alternative, diff=value)
        return (tstat, pval)

    def tconfint_diff(self, alpha=0.05, alternative='two-sided', usevar='pooled'):
        if False:
            while True:
                i = 10
        "confidence interval for the difference in means\n\n        Parameters\n        ----------\n        alpha : float\n            significance level for the confidence interval, coverage is\n            ``1-alpha``\n        alternative : str\n            This specifies the alternative hypothesis for the test that\n            corresponds to the confidence interval.\n            The alternative hypothesis, H1, has to be one of the following :\n\n            'two-sided': H1: difference in means not equal to value (default)\n            'larger' :   H1: difference in means larger than value\n            'smaller' :  H1: difference in means smaller than value\n\n        usevar : str, 'pooled' or 'unequal'\n            If ``pooled``, then the standard deviation of the samples is assumed to be\n            the same. If ``unequal``, then Welch ttest with Satterthwait degrees\n            of freedom is used\n\n        Returns\n        -------\n        lower, upper : floats\n            lower and upper limits of the confidence interval\n\n        Notes\n        -----\n        The result is independent of the user specified ddof.\n\n        "
        d1 = self.d1
        d2 = self.d2
        diff = d1.mean - d2.mean
        if usevar == 'pooled':
            std_diff = self.std_meandiff_pooledvar
            dof = d1.nobs - 1 + d2.nobs - 1
        elif usevar == 'unequal':
            std_diff = self.std_meandiff_separatevar
            dof = self.dof_satt()
        else:
            raise ValueError('usevar can only be "pooled" or "unequal"')
        res = _tconfint_generic(diff, std_diff, dof, alpha=alpha, alternative=alternative)
        return res

    def zconfint_diff(self, alpha=0.05, alternative='two-sided', usevar='pooled'):
        if False:
            print('Hello World!')
        "confidence interval for the difference in means\n\n        Parameters\n        ----------\n        alpha : float\n            significance level for the confidence interval, coverage is\n            ``1-alpha``\n        alternative : str\n            This specifies the alternative hypothesis for the test that\n            corresponds to the confidence interval.\n            The alternative hypothesis, H1, has to be one of the following :\n\n            'two-sided': H1: difference in means not equal to value (default)\n            'larger' :   H1: difference in means larger than value\n            'smaller' :  H1: difference in means smaller than value\n\n        usevar : str, 'pooled' or 'unequal'\n            If ``pooled``, then the standard deviation of the samples is assumed to be\n            the same. If ``unequal``, then Welch ttest with Satterthwait degrees\n            of freedom is used\n\n        Returns\n        -------\n        lower, upper : floats\n            lower and upper limits of the confidence interval\n\n        Notes\n        -----\n        The result is independent of the user specified ddof.\n\n        "
        d1 = self.d1
        d2 = self.d2
        diff = d1.mean - d2.mean
        if usevar == 'pooled':
            std_diff = self.std_meandiff_pooledvar
        elif usevar == 'unequal':
            std_diff = self.std_meandiff_separatevar
        else:
            raise ValueError('usevar can only be "pooled" or "unequal"')
        res = _zconfint_generic(diff, std_diff, alpha=alpha, alternative=alternative)
        return res

    def ttost_ind(self, low, upp, usevar='pooled'):
        if False:
            return 10
        "\n        test of equivalence for two independent samples, base on t-test\n\n        Parameters\n        ----------\n        low, upp : float\n            equivalence interval low < m1 - m2 < upp\n        usevar : str, 'pooled' or 'unequal'\n            If ``pooled``, then the standard deviation of the samples is assumed to be\n            the same. If ``unequal``, then Welch ttest with Satterthwait degrees\n            of freedom is used\n\n        Returns\n        -------\n        pvalue : float\n            pvalue of the non-equivalence test\n        t1, pv1 : tuple of floats\n            test statistic and pvalue for lower threshold test\n        t2, pv2 : tuple of floats\n            test statistic and pvalue for upper threshold test\n        "
        tt1 = self.ttest_ind(alternative='larger', usevar=usevar, value=low)
        tt2 = self.ttest_ind(alternative='smaller', usevar=usevar, value=upp)
        return (np.maximum(tt1[1], tt2[1]), (tt1, tt2))

    def ztost_ind(self, low, upp, usevar='pooled'):
        if False:
            return 10
        "\n        test of equivalence for two independent samples, based on z-test\n\n        Parameters\n        ----------\n        low, upp : float\n            equivalence interval low < m1 - m2 < upp\n        usevar : str, 'pooled' or 'unequal'\n            If ``pooled``, then the standard deviation of the samples is assumed to be\n            the same. If ``unequal``, then Welch ttest with Satterthwait degrees\n            of freedom is used\n\n        Returns\n        -------\n        pvalue : float\n            pvalue of the non-equivalence test\n        t1, pv1 : tuple of floats\n            test statistic and pvalue for lower threshold test\n        t2, pv2 : tuple of floats\n            test statistic and pvalue for upper threshold test\n        "
        tt1 = self.ztest_ind(alternative='larger', usevar=usevar, value=low)
        tt2 = self.ztest_ind(alternative='smaller', usevar=usevar, value=upp)
        return (np.maximum(tt1[1], tt2[1]), tt1, tt2)

def ttest_ind(x1, x2, alternative='two-sided', usevar='pooled', weights=(None, None), value=0):
    if False:
        for i in range(10):
            print('nop')
    "ttest independent sample\n\n    Convenience function that uses the classes and throws away the intermediate\n    results,\n    compared to scipy stats: drops axis option, adds alternative, usevar, and\n    weights option.\n\n    Parameters\n    ----------\n    x1 : array_like, 1-D or 2-D\n        first of the two independent samples, see notes for 2-D case\n    x2 : array_like, 1-D or 2-D\n        second of the two independent samples, see notes for 2-D case\n    alternative : str\n        The alternative hypothesis, H1, has to be one of the following\n\n           * 'two-sided' (default): H1: difference in means not equal to value\n           * 'larger' :   H1: difference in means larger than value\n           * 'smaller' :  H1: difference in means smaller than value\n\n    usevar : str, 'pooled' or 'unequal'\n        If ``pooled``, then the standard deviation of the samples is assumed to be\n        the same. If ``unequal``, then Welch ttest with Satterthwait degrees\n        of freedom is used\n    weights : tuple of None or ndarrays\n        Case weights for the two samples. For details on weights see\n        ``DescrStatsW``\n    value : float\n        difference between the means under the Null hypothesis.\n\n\n    Returns\n    -------\n    tstat : float\n        test statistic\n    pvalue : float\n        pvalue of the t-test\n    df : int or float\n        degrees of freedom used in the t-test\n\n    "
    cm = CompareMeans(DescrStatsW(x1, weights=weights[0], ddof=0), DescrStatsW(x2, weights=weights[1], ddof=0))
    (tstat, pval, dof) = cm.ttest_ind(alternative=alternative, usevar=usevar, value=value)
    return (tstat, pval, dof)

def ttost_ind(x1, x2, low, upp, usevar='pooled', weights=(None, None), transform=None):
    if False:
        for i in range(10):
            print('nop')
    "test of (non-)equivalence for two independent samples\n\n    TOST: two one-sided t tests\n\n    null hypothesis:  m1 - m2 < low or m1 - m2 > upp\n    alternative hypothesis:  low < m1 - m2 < upp\n\n    where m1, m2 are the means, expected values of the two samples.\n\n    If the pvalue is smaller than a threshold, say 0.05, then we reject the\n    hypothesis that the difference between the two samples is larger than the\n    the thresholds given by low and upp.\n\n    Parameters\n    ----------\n    x1 : array_like, 1-D or 2-D\n        first of the two independent samples, see notes for 2-D case\n    x2 : array_like, 1-D or 2-D\n        second of the two independent samples, see notes for 2-D case\n    low, upp : float\n        equivalence interval low < m1 - m2 < upp\n    usevar : str, 'pooled' or 'unequal'\n        If ``pooled``, then the standard deviation of the samples is assumed to be\n        the same. If ``unequal``, then Welch ttest with Satterthwait degrees\n        of freedom is used\n    weights : tuple of None or ndarrays\n        Case weights for the two samples. For details on weights see\n        ``DescrStatsW``\n    transform : None or function\n        If None (default), then the data is not transformed. Given a function,\n        sample data and thresholds are transformed. If transform is log, then\n        the equivalence interval is in ratio: low < m1 / m2 < upp\n\n    Returns\n    -------\n    pvalue : float\n        pvalue of the non-equivalence test\n    t1, pv1 : tuple of floats\n        test statistic and pvalue for lower threshold test\n    t2, pv2 : tuple of floats\n        test statistic and pvalue for upper threshold test\n\n    Notes\n    -----\n    The test rejects if the 2*alpha confidence interval for the difference\n    is contained in the ``(low, upp)`` interval.\n\n    This test works also for multi-endpoint comparisons: If d1 and d2\n    have the same number of columns, then each column of the data in d1 is\n    compared with the corresponding column in d2. This is the same as\n    comparing each of the corresponding columns separately. Currently no\n    multi-comparison correction is used. The raw p-values reported here can\n    be correction with the functions in ``multitest``.\n\n    "
    if transform:
        if transform is np.log:
            x1 = transform(x1)
            x2 = transform(x2)
        else:
            xx = transform(np.concatenate((x1, x2), 0))
            x1 = xx[:len(x1)]
            x2 = xx[len(x1):]
        low = transform(low)
        upp = transform(upp)
    cm = CompareMeans(DescrStatsW(x1, weights=weights[0], ddof=0), DescrStatsW(x2, weights=weights[1], ddof=0))
    (pval, res) = cm.ttost_ind(low, upp, usevar=usevar)
    return (pval, res[0], res[1])

def ttost_paired(x1, x2, low, upp, transform=None, weights=None):
    if False:
        print('Hello World!')
    'test of (non-)equivalence for two dependent, paired sample\n\n    TOST: two one-sided t tests\n\n    null hypothesis:  md < low or md > upp\n    alternative hypothesis:  low < md < upp\n\n    where md is the mean, expected value of the difference x1 - x2\n\n    If the pvalue is smaller than a threshold,say 0.05, then we reject the\n    hypothesis that the difference between the two samples is larger than the\n    the thresholds given by low and upp.\n\n    Parameters\n    ----------\n    x1 : array_like\n        first of the two independent samples\n    x2 : array_like\n        second of the two independent samples\n    low, upp : float\n        equivalence interval low < mean of difference < upp\n    weights : None or ndarray\n        case weights for the two samples. For details on weights see\n        ``DescrStatsW``\n    transform : None or function\n        If None (default), then the data is not transformed. Given a function\n        sample data and thresholds are transformed. If transform is log the\n        the equivalence interval is in ratio: low < x1 / x2 < upp\n\n    Returns\n    -------\n    pvalue : float\n        pvalue of the non-equivalence test\n    t1, pv1, df1 : tuple\n        test statistic, pvalue and degrees of freedom for lower threshold test\n    t2, pv2, df2 : tuple\n        test statistic, pvalue and degrees of freedom for upper threshold test\n\n    '
    if transform:
        if transform is np.log:
            x1 = transform(x1)
            x2 = transform(x2)
        else:
            xx = transform(np.concatenate((x1, x2), 0))
            x1 = xx[:len(x1)]
            x2 = xx[len(x1):]
        low = transform(low)
        upp = transform(upp)
    dd = DescrStatsW(x1 - x2, weights=weights, ddof=0)
    (t1, pv1, df1) = dd.ttest_mean(low, alternative='larger')
    (t2, pv2, df2) = dd.ttest_mean(upp, alternative='smaller')
    return (np.maximum(pv1, pv2), (t1, pv1, df1), (t2, pv2, df2))

def ztest(x1, x2=None, value=0, alternative='two-sided', usevar='pooled', ddof=1.0):
    if False:
        while True:
            i = 10
    "test for mean based on normal distribution, one or two samples\n\n    In the case of two samples, the samples are assumed to be independent.\n\n    Parameters\n    ----------\n    x1 : array_like, 1-D or 2-D\n        first of the two independent samples\n    x2 : array_like, 1-D or 2-D\n        second of the two independent samples\n    value : float\n        In the one sample case, value is the mean of x1 under the Null\n        hypothesis.\n        In the two sample case, value is the difference between mean of x1 and\n        mean of x2 under the Null hypothesis. The test statistic is\n        `x1_mean - x2_mean - value`.\n    alternative : str\n        The alternative hypothesis, H1, has to be one of the following\n\n           'two-sided': H1: difference in means not equal to value (default)\n           'larger' :   H1: difference in means larger than value\n           'smaller' :  H1: difference in means smaller than value\n\n    usevar : str, 'pooled' or 'unequal'\n        If ``pooled``, then the standard deviation of the samples is assumed to be\n        the same. If ``unequal``, then the standard deviation of the sample is\n        assumed to be different.\n    ddof : int\n        Degrees of freedom use in the calculation of the variance of the mean\n        estimate. In the case of comparing means this is one, however it can\n        be adjusted for testing other statistics (proportion, correlation)\n\n    Returns\n    -------\n    tstat : float\n        test statistic\n    pvalue : float\n        pvalue of the t-test\n\n    Notes\n    -----\n    usevar can be pooled or unequal in two sample case\n\n    "
    if usevar not in {'pooled', 'unequal'}:
        raise NotImplementedError('usevar can only be "pooled" or "unequal"')
    x1 = np.asarray(x1)
    nobs1 = x1.shape[0]
    x1_mean = x1.mean(0)
    x1_var = x1.var(0)
    if x2 is not None:
        x2 = np.asarray(x2)
        nobs2 = x2.shape[0]
        x2_mean = x2.mean(0)
        x2_var = x2.var(0)
        if usevar == 'pooled':
            var = nobs1 * x1_var + nobs2 * x2_var
            var /= nobs1 + nobs2 - 2 * ddof
            var *= 1.0 / nobs1 + 1.0 / nobs2
        elif usevar == 'unequal':
            var = x1_var / (nobs1 - ddof) + x2_var / (nobs2 - ddof)
    else:
        var = x1_var / (nobs1 - ddof)
        x2_mean = 0
    std_diff = np.sqrt(var)
    return _zstat_generic(x1_mean, x2_mean, std_diff, alternative, diff=value)

def zconfint(x1, x2=None, value=0, alpha=0.05, alternative='two-sided', usevar='pooled', ddof=1.0):
    if False:
        i = 10
        return i + 15
    "confidence interval based on normal distribution z-test\n\n    Parameters\n    ----------\n    x1 : array_like, 1-D or 2-D\n        first of the two independent samples, see notes for 2-D case\n    x2 : array_like, 1-D or 2-D\n        second of the two independent samples, see notes for 2-D case\n    value : float\n        In the one sample case, value is the mean of x1 under the Null\n        hypothesis.\n        In the two sample case, value is the difference between mean of x1 and\n        mean of x2 under the Null hypothesis. The test statistic is\n        `x1_mean - x2_mean - value`.\n    usevar : str, 'pooled'\n        Currently, only 'pooled' is implemented.\n        If ``pooled``, then the standard deviation of the samples is assumed to be\n        the same. see CompareMeans.ztest_ind for different options.\n    ddof : int\n        Degrees of freedom use in the calculation of the variance of the mean\n        estimate. In the case of comparing means this is one, however it can\n        be adjusted for testing other statistics (proportion, correlation)\n\n    Notes\n    -----\n    checked only for 1 sample case\n\n    usevar not implemented, is always pooled in two sample case\n\n    ``value`` shifts the confidence interval so it is centered at\n    `x1_mean - x2_mean - value`\n\n    See Also\n    --------\n    ztest\n    CompareMeans\n\n    "
    if usevar != 'pooled':
        raise NotImplementedError('only usevar="pooled" is implemented')
    x1 = np.asarray(x1)
    nobs1 = x1.shape[0]
    x1_mean = x1.mean(0)
    x1_var = x1.var(0)
    if x2 is not None:
        x2 = np.asarray(x2)
        nobs2 = x2.shape[0]
        x2_mean = x2.mean(0)
        x2_var = x2.var(0)
        var_pooled = nobs1 * x1_var + nobs2 * x2_var
        var_pooled /= nobs1 + nobs2 - 2 * ddof
        var_pooled *= 1.0 / nobs1 + 1.0 / nobs2
    else:
        var_pooled = x1_var / (nobs1 - ddof)
        x2_mean = 0
    std_diff = np.sqrt(var_pooled)
    ci = _zconfint_generic(x1_mean - x2_mean - value, std_diff, alpha, alternative)
    return ci

def ztost(x1, low, upp, x2=None, usevar='pooled', ddof=1.0):
    if False:
        while True:
            i = 10
    "Equivalence test based on normal distribution\n\n    Parameters\n    ----------\n    x1 : array_like\n        one sample or first sample for 2 independent samples\n    low, upp : float\n        equivalence interval low < m1 - m2 < upp\n    x1 : array_like or None\n        second sample for 2 independent samples test. If None, then a\n        one-sample test is performed.\n    usevar : str, 'pooled'\n        If `pooled`, then the standard deviation of the samples is assumed to be\n        the same. Only `pooled` is currently implemented.\n\n    Returns\n    -------\n    pvalue : float\n        pvalue of the non-equivalence test\n    t1, pv1 : tuple of floats\n        test statistic and pvalue for lower threshold test\n    t2, pv2 : tuple of floats\n        test statistic and pvalue for upper threshold test\n\n    Notes\n    -----\n    checked only for 1 sample case\n\n    "
    tt1 = ztest(x1, x2, alternative='larger', usevar=usevar, value=low, ddof=ddof)
    tt2 = ztest(x1, x2, alternative='smaller', usevar=usevar, value=upp, ddof=ddof)
    return (np.maximum(tt1[1], tt2[1]), tt1, tt2)