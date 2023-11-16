"""
Rank based methods for inferential statistics

Created on Sat Aug 15 10:18:53 2020

Author: Josef Perktold
License: BSD-3

"""
import numpy as np
from scipy import stats
from scipy.stats import rankdata
from statsmodels.stats.base import HolderTuple
from statsmodels.stats.weightstats import _tconfint_generic, _tstat_generic, _zconfint_generic, _zstat_generic

def rankdata_2samp(x1, x2):
    if False:
        print('Hello World!')
    'Compute midranks for two samples\n\n    Parameters\n    ----------\n    x1, x2 : array_like\n        Original data for two samples that will be converted to midranks.\n\n    Returns\n    -------\n    rank1 : ndarray\n        Midranks of the first sample in the pooled sample.\n    rank2 : ndarray\n        Midranks of the second sample in the pooled sample.\n    ranki1 : ndarray\n        Internal midranks of the first sample.\n    ranki2 : ndarray\n        Internal midranks of the second sample.\n\n    '
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    nobs1 = len(x1)
    nobs2 = len(x2)
    if nobs1 == 0 or nobs2 == 0:
        raise ValueError('one sample has zero length')
    x_combined = np.concatenate((x1, x2))
    if x_combined.ndim > 1:
        rank = np.apply_along_axis(rankdata, 0, x_combined)
    else:
        rank = rankdata(x_combined)
    rank1 = rank[:nobs1]
    rank2 = rank[nobs1:]
    if x_combined.ndim > 1:
        ranki1 = np.apply_along_axis(rankdata, 0, x1)
        ranki2 = np.apply_along_axis(rankdata, 0, x2)
    else:
        ranki1 = rankdata(x1)
        ranki2 = rankdata(x2)
    return (rank1, rank2, ranki1, ranki2)

class RankCompareResult(HolderTuple):
    """Results for rank comparison

    This is a subclass of HolderTuple that includes results from intermediate
    computations, as well as methods for hypothesis tests, confidence intervals
    and summary.
    """

    def conf_int(self, value=None, alpha=0.05, alternative='two-sided'):
        if False:
            return 10
        '\n        Confidence interval for probability that sample 1 has larger values\n\n        Confidence interval is for the shifted probability\n\n            P(x1 > x2) + 0.5 * P(x1 = x2) - value\n\n        Parameters\n        ----------\n        value : float\n            Value, default 0, shifts the confidence interval,\n            e.g. ``value=0.5`` centers the confidence interval at zero.\n        alpha : float\n            Significance level for the confidence interval, coverage is\n            ``1-alpha``\n        alternative : str\n            The alternative hypothesis, H1, has to be one of the following\n\n               * \'two-sided\' : H1: ``prob - value`` not equal to 0.\n               * \'larger\' :   H1: ``prob - value > 0``\n               * \'smaller\' :  H1: ``prob - value < 0``\n\n        Returns\n        -------\n        lower : float or ndarray\n            Lower confidence limit. This is -inf for the one-sided alternative\n            "smaller".\n        upper : float or ndarray\n            Upper confidence limit. This is inf for the one-sided alternative\n            "larger".\n\n        '
        p0 = value
        if p0 is None:
            p0 = 0
        diff = self.prob1 - p0
        std_diff = np.sqrt(self.var / self.nobs)
        if self.use_t is False:
            return _zconfint_generic(diff, std_diff, alpha, alternative)
        else:
            return _tconfint_generic(diff, std_diff, self.df, alpha, alternative)

    def test_prob_superior(self, value=0.5, alternative='two-sided'):
        if False:
            while True:
                i = 10
        "test for superiority probability\n\n        H0: P(x1 > x2) + 0.5 * P(x1 = x2) = value\n\n        The alternative is that the probability is either not equal, larger\n        or smaller than the null-value depending on the chosen alternative.\n\n        Parameters\n        ----------\n        value : float\n            Value of the probability under the Null hypothesis.\n        alternative : str\n            The alternative hypothesis, H1, has to be one of the following\n\n               * 'two-sided' : H1: ``prob - value`` not equal to 0.\n               * 'larger' :   H1: ``prob - value > 0``\n               * 'smaller' :  H1: ``prob - value < 0``\n\n        Returns\n        -------\n        res : HolderTuple\n            HolderTuple instance with the following main attributes\n\n            statistic : float\n                Test statistic for z- or t-test\n            pvalue : float\n                Pvalue of the test based on either normal or t distribution.\n\n        "
        p0 = value
        std_diff = np.sqrt(self.var / self.nobs)
        if not self.use_t:
            (stat, pv) = _zstat_generic(self.prob1, p0, std_diff, alternative, diff=0)
            distr = 'normal'
        else:
            (stat, pv) = _tstat_generic(self.prob1, p0, std_diff, self.df, alternative, diff=0)
            distr = 't'
        res = HolderTuple(statistic=stat, pvalue=pv, df=self.df, distribution=distr)
        return res

    def tost_prob_superior(self, low, upp):
        if False:
            for i in range(10):
                print('nop')
        'test of stochastic (non-)equivalence of p = P(x1 > x2)\n\n        Null hypothesis:  p < low or p > upp\n        Alternative hypothesis:  low < p < upp\n\n        where p is the probability that a random draw from the population of\n        the first sample has a larger value than a random draw from the\n        population of the second sample, specifically\n\n            p = P(x1 > x2) + 0.5 * P(x1 = x2)\n\n        If the pvalue is smaller than a threshold, say 0.05, then we reject the\n        hypothesis that the probability p that distribution 1 is stochastically\n        superior to distribution 2 is outside of the interval given by\n        thresholds low and upp.\n\n        Parameters\n        ----------\n        low, upp : float\n            equivalence interval low < mean < upp\n\n        Returns\n        -------\n        res : HolderTuple\n            HolderTuple instance with the following main attributes\n\n            pvalue : float\n                Pvalue of the equivalence test given by the larger pvalue of\n                the two one-sided tests.\n            statistic : float\n                Test statistic of the one-sided test that has the larger\n                pvalue.\n            results_larger : HolderTuple\n                Results instanc with test statistic, pvalue and degrees of\n                freedom for lower threshold test.\n            results_smaller : HolderTuple\n                Results instanc with test statistic, pvalue and degrees of\n                freedom for upper threshold test.\n\n        '
        t1 = self.test_prob_superior(low, alternative='larger')
        t2 = self.test_prob_superior(upp, alternative='smaller')
        idx_max = np.asarray(t1.pvalue < t2.pvalue, int)
        title = 'Equivalence test for Prob(x1 > x2) + 0.5 Prob(x1 = x2) '
        res = HolderTuple(statistic=np.choose(idx_max, [t1.statistic, t2.statistic]), pvalue=np.choose(idx_max, [t1.pvalue, t2.pvalue]), results_larger=t1, results_smaller=t2, title=title)
        return res

    def confint_lintransf(self, const=-1, slope=2, alpha=0.05, alternative='two-sided'):
        if False:
            return 10
        'confidence interval of a linear transformation of prob1\n\n        This computes the confidence interval for\n\n            d = const + slope * prob1\n\n        Default values correspond to Somers\' d.\n\n        Parameters\n        ----------\n        const, slope : float\n            Constant and slope for linear (affine) transformation.\n        alpha : float\n            Significance level for the confidence interval, coverage is\n            ``1-alpha``\n        alternative : str\n            The alternative hypothesis, H1, has to be one of the following\n\n               * \'two-sided\' : H1: ``prob - value`` not equal to 0.\n               * \'larger\' :   H1: ``prob - value > 0``\n               * \'smaller\' :  H1: ``prob - value < 0``\n\n        Returns\n        -------\n        lower : float or ndarray\n            Lower confidence limit. This is -inf for the one-sided alternative\n            "smaller".\n        upper : float or ndarray\n            Upper confidence limit. This is inf for the one-sided alternative\n            "larger".\n\n        '
        (low_p, upp_p) = self.conf_int(alpha=alpha, alternative=alternative)
        low = const + slope * low_p
        upp = const + slope * upp_p
        if slope < 0:
            (low, upp) = (upp, low)
        return (low, upp)

    def effectsize_normal(self, prob=None):
        if False:
            while True:
                i = 10
        "\n        Cohen's d, standardized mean difference under normality assumption.\n\n        This computes the standardized mean difference, Cohen's d, effect size\n        that is equivalent to the rank based probability ``p`` of being\n        stochastically larger if we assume that the data is normally\n        distributed, given by\n\n            :math: `d = F^{-1}(p) * \\sqrt{2}`\n\n        where :math:`F^{-1}` is the inverse of the cdf of the normal\n        distribution.\n\n        Parameters\n        ----------\n        prob : float in (0, 1)\n            Probability to be converted to Cohen's d effect size.\n            If prob is None, then the ``prob1`` attribute is used.\n\n        Returns\n        -------\n        equivalent Cohen's d effect size under normality assumption.\n\n        "
        if prob is None:
            prob = self.prob1
        return stats.norm.ppf(prob) * np.sqrt(2)

    def summary(self, alpha=0.05, xname=None):
        if False:
            while True:
                i = 10
        'summary table for probability that random draw x1 is larger than x2\n\n        Parameters\n        ----------\n        alpha : float\n            Significance level for confidence intervals. Coverage is 1 - alpha\n        xname : None or list of str\n            If None, then each row has a name column with generic names.\n            If xname is a list of strings, then it will be included as part\n            of those names.\n\n        Returns\n        -------\n        SimpleTable instance with methods to convert to different output\n        formats.\n        '
        yname = 'None'
        effect = np.atleast_1d(self.prob1)
        if self.pvalue is None:
            (statistic, pvalue) = self.test_prob_superior()
        else:
            pvalue = self.pvalue
            statistic = self.statistic
        pvalues = np.atleast_1d(pvalue)
        ci = np.atleast_2d(self.conf_int(alpha=alpha))
        if ci.shape[0] > 1:
            ci = ci.T
        use_t = self.use_t
        sd = np.atleast_1d(np.sqrt(self.var_prob))
        statistic = np.atleast_1d(statistic)
        if xname is None:
            xname = ['c%d' % ii for ii in range(len(effect))]
        xname2 = ['prob(x1>x2) %s' % ii for ii in xname]
        title = 'Probability sample 1 is stochastically larger'
        from statsmodels.iolib.summary import summary_params
        summ = summary_params((self, effect, sd, statistic, pvalues, ci), yname=yname, xname=xname2, use_t=use_t, title=title, alpha=alpha)
        return summ

def rank_compare_2indep(x1, x2, use_t=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Statistics and tests for the probability that x1 has larger values than x2.\n\n    p is the probability that a random draw from the population of\n    the first sample has a larger value than a random draw from the\n    population of the second sample, specifically\n\n            p = P(x1 > x2) + 0.5 * P(x1 = x2)\n\n    This is a measure underlying Wilcoxon-Mann-Whitney\'s U test,\n    Fligner-Policello test and Brunner-Munzel test, and\n    Inference is based on the asymptotic distribution of the Brunner-Munzel\n    test. The half probability for ties corresponds to the use of midranks\n    and make it valid for discrete variables.\n\n    The Null hypothesis for stochastic equality is p = 0.5, which corresponds\n    to the Brunner-Munzel test.\n\n    Parameters\n    ----------\n    x1, x2 : array_like\n        Array of samples, should be one-dimensional.\n    use_t : boolean\n        If use_t is true, the t distribution with Welch-Satterthwaite type\n        degrees of freedom is used for p-value and confidence interval.\n        If use_t is false, then the normal distribution is used.\n\n    Returns\n    -------\n    res : RankCompareResult\n        The results instance contains the results for the Brunner-Munzel test\n        and has methods for hypothesis tests, confidence intervals and summary.\n\n        statistic : float\n            The Brunner-Munzel W statistic.\n        pvalue : float\n            p-value assuming an t distribution. One-sided or\n            two-sided, depending on the choice of `alternative` and `use_t`.\n\n    See Also\n    --------\n    RankCompareResult\n    scipy.stats.brunnermunzel : Brunner-Munzel test for stochastic equality\n    scipy.stats.mannwhitneyu : Mann-Whitney rank test on two samples.\n\n    Notes\n    -----\n    Wilcoxon-Mann-Whitney assumes equal variance or equal distribution under\n    the Null hypothesis. Fligner-Policello test allows for unequal variances\n    but assumes continuous distribution, i.e. no ties.\n    Brunner-Munzel extend the test to allow for unequal variance and discrete\n    or ordered categorical random variables.\n\n    Brunner and Munzel recommended to estimate the p-value by t-distribution\n    when the size of data is 50 or less. If the size is lower than 10, it would\n    be better to use permuted Brunner Munzel test (see [2]_) for the test\n    of stochastic equality.\n\n    This measure has been introduced in the literature under many different\n    names relying on a variety of assumptions.\n    In psychology, McGraw and Wong (1992) introduced it as Common Language\n    effect size for the continuous, normal distribution case,\n    Vargha and Delaney (2000) [3]_ extended it to the nonparametric\n    continuous distribution case as in Fligner-Policello.\n\n    WMW and related tests can only be interpreted as test of medians or tests\n    of central location only under very restrictive additional assumptions\n    such as both distribution are identical under the equality null hypothesis\n    (assumed by Mann-Whitney) or both distributions are symmetric (shown by\n    Fligner-Policello). If the distribution of the two samples can differ in\n    an arbitrary way, then the equality Null hypothesis corresponds to p=0.5\n    against an alternative p != 0.5.  see for example Conroy (2012) [4]_ and\n    Divine et al (2018) [5]_ .\n\n    Note: Brunner-Munzel and related literature define the probability that x1\n    is stochastically smaller than x2, while here we use stochastically larger.\n    This equivalent to switching x1 and x2 in the two sample case.\n\n    References\n    ----------\n    .. [1] Brunner, E. and Munzel, U. "The nonparametric Benhrens-Fisher\n           problem: Asymptotic theory and a small-sample approximation".\n           Biometrical Journal. Vol. 42(2000): 17-25.\n    .. [2] Neubert, K. and Brunner, E. "A studentized permutation test for the\n           non-parametric Behrens-Fisher problem". Computational Statistics and\n           Data Analysis. Vol. 51(2007): 5192-5204.\n    .. [3] Vargha, András, and Harold D. Delaney. 2000. “A Critique and\n           Improvement of the CL Common Language Effect Size Statistics of\n           McGraw and Wong.” Journal of Educational and Behavioral Statistics\n           25 (2): 101–32. https://doi.org/10.3102/10769986025002101.\n    .. [4] Conroy, Ronán M. 2012. “What Hypotheses Do ‘Nonparametric’ Two-Group\n           Tests Actually Test?” The Stata Journal: Promoting Communications on\n           Statistics and Stata 12 (2): 182–90.\n           https://doi.org/10.1177/1536867X1201200202.\n    .. [5] Divine, George W., H. James Norton, Anna E. Barón, and Elizabeth\n           Juarez-Colunga. 2018. “The Wilcoxon–Mann–Whitney Procedure Fails as\n           a Test of Medians.” The American Statistician 72 (3): 278–86.\n           https://doi.org/10.1080/00031305.2017.1305291.\n\n    '
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    nobs1 = len(x1)
    nobs2 = len(x2)
    nobs = nobs1 + nobs2
    if nobs1 == 0 or nobs2 == 0:
        raise ValueError('one sample has zero length')
    (rank1, rank2, ranki1, ranki2) = rankdata_2samp(x1, x2)
    meanr1 = np.mean(rank1, axis=0)
    meanr2 = np.mean(rank2, axis=0)
    meanri1 = np.mean(ranki1, axis=0)
    meanri2 = np.mean(ranki2, axis=0)
    S1 = np.sum(np.power(rank1 - ranki1 - meanr1 + meanri1, 2.0), axis=0)
    S1 /= nobs1 - 1
    S2 = np.sum(np.power(rank2 - ranki2 - meanr2 + meanri2, 2.0), axis=0)
    S2 /= nobs2 - 1
    wbfn = nobs1 * nobs2 * (meanr1 - meanr2)
    wbfn /= (nobs1 + nobs2) * np.sqrt(nobs1 * S1 + nobs2 * S2)
    if use_t:
        df_numer = np.power(nobs1 * S1 + nobs2 * S2, 2.0)
        df_denom = np.power(nobs1 * S1, 2.0) / (nobs1 - 1)
        df_denom += np.power(nobs2 * S2, 2.0) / (nobs2 - 1)
        df = df_numer / df_denom
        pvalue = 2 * stats.t.sf(np.abs(wbfn), df)
    else:
        pvalue = 2 * stats.norm.sf(np.abs(wbfn))
        df = None
    var1 = S1 / (nobs - nobs1) ** 2
    var2 = S2 / (nobs - nobs2) ** 2
    var_prob = var1 / nobs1 + var2 / nobs2
    var = nobs * (var1 / nobs1 + var2 / nobs2)
    prob1 = (meanr1 - (nobs1 + 1) / 2) / nobs2
    prob2 = (meanr2 - (nobs2 + 1) / 2) / nobs1
    return RankCompareResult(statistic=wbfn, pvalue=pvalue, s1=S1, s2=S2, var1=var1, var2=var2, var=var, var_prob=var_prob, nobs1=nobs1, nobs2=nobs2, nobs=nobs, mean1=meanr1, mean2=meanr2, prob1=prob1, prob2=prob2, somersd1=prob1 * 2 - 1, somersd2=prob2 * 2 - 1, df=df, use_t=use_t)

def rank_compare_2ordinal(count1, count2, ddof=1, use_t=True):
    if False:
        while True:
            i = 10
    '\n    Stochastically larger probability for 2 independent ordinal samples.\n\n    This is a special case of `rank_compare_2indep` when the data are given as\n    counts of two independent ordinal, i.e. ordered multinomial, samples.\n\n    The statistic of interest is the probability that a random draw from the\n    population of the first sample has a larger value than a random draw from\n    the population of the second sample, specifically\n\n        p = P(x1 > x2) + 0.5 * P(x1 = x2)\n\n    Parameters\n    ----------\n    count1 : array_like\n        Counts of the first sample, categories are assumed to be ordered.\n    count2 : array_like\n        Counts of the second sample, number of categories and ordering needs\n        to be the same as for sample 1.\n    ddof : scalar\n        Degrees of freedom correction for variance estimation. The default\n        ddof=1 corresponds to `rank_compare_2indep`.\n    use_t : bool\n        If use_t is true, the t distribution with Welch-Satterthwaite type\n        degrees of freedom is used for p-value and confidence interval.\n        If use_t is false, then the normal distribution is used.\n\n    Returns\n    -------\n    res : RankCompareResult\n        This includes methods for hypothesis tests and confidence intervals\n        for the probability that sample 1 is stochastically larger than\n        sample 2.\n\n    See Also\n    --------\n    rank_compare_2indep\n    RankCompareResult\n\n    Notes\n    -----\n    The implementation is based on the appendix of Munzel and Hauschke (2003)\n    with the addition of ``ddof`` so that the results match the general\n    function `rank_compare_2indep`.\n\n    '
    count1 = np.asarray(count1)
    count2 = np.asarray(count2)
    (nobs1, nobs2) = (count1.sum(), count2.sum())
    freq1 = count1 / nobs1
    freq2 = count2 / nobs2
    cdf1 = np.concatenate(([0], freq1)).cumsum(axis=0)
    cdf2 = np.concatenate(([0], freq2)).cumsum(axis=0)
    cdfm1 = (cdf1[1:] + cdf1[:-1]) / 2
    cdfm2 = (cdf2[1:] + cdf2[:-1]) / 2
    prob1 = (cdfm2 * freq1).sum()
    prob2 = (cdfm1 * freq2).sum()
    var1 = (cdfm2 ** 2 * freq1).sum() - prob1 ** 2
    var2 = (cdfm1 ** 2 * freq2).sum() - prob2 ** 2
    var_prob = var1 / (nobs1 - ddof) + var2 / (nobs2 - ddof)
    nobs = nobs1 + nobs2
    var = nobs * var_prob
    vn1 = var1 * nobs2 * nobs1 / (nobs1 - ddof)
    vn2 = var2 * nobs1 * nobs2 / (nobs2 - ddof)
    df = (vn1 + vn2) ** 2 / (vn1 ** 2 / (nobs1 - 1) + vn2 ** 2 / (nobs2 - 1))
    res = RankCompareResult(statistic=None, pvalue=None, s1=None, s2=None, var1=var1, var2=var2, var=var, var_prob=var_prob, nobs1=nobs1, nobs2=nobs2, nobs=nobs, mean1=None, mean2=None, prob1=prob1, prob2=prob2, somersd1=prob1 * 2 - 1, somersd2=prob2 * 2 - 1, df=df, use_t=use_t)
    return res

def prob_larger_continuous(distr1, distr2):
    if False:
        print('Hello World!')
    '\n    Probability indicating that distr1 is stochastically larger than distr2.\n\n    This computes\n\n        p = P(x1 > x2)\n\n    for two continuous distributions, where `distr1` and `distr2` are the\n    distributions of random variables x1 and x2 respectively.\n\n    Parameters\n    ----------\n    distr1, distr2 : distributions\n        Two instances of scipy.stats.distributions. The required methods are\n        cdf of the second distribution and expect of the first distribution.\n\n    Returns\n    -------\n    p : probability x1 is larger than x2\n\n\n    Notes\n    -----\n    This is a one-liner that is added mainly as reference.\n\n    Examples\n    --------\n    >>> from scipy import stats\n    >>> prob_larger_continuous(stats.norm, stats.t(5))\n    0.4999999999999999\n\n    # which is the same as\n    >>> stats.norm.expect(stats.t(5).cdf)\n    0.4999999999999999\n\n    # distribution 1 with smaller mean (loc) than distribution 2\n    >>> prob_larger_continuous(stats.norm, stats.norm(loc=1))\n    0.23975006109347669\n\n    '
    return distr1.expect(distr2.cdf)

def cohensd2problarger(d):
    if False:
        for i in range(10):
            print('nop')
    "\n    Convert Cohen's d effect size to stochastically-larger-probability.\n\n    This assumes observations are normally distributed.\n\n    Computed as\n\n        p = Prob(x1 > x2) = F(d / sqrt(2))\n\n    where `F` is cdf of normal distribution. Cohen's d is defined as\n\n        d = (mean1 - mean2) / std\n\n    where ``std`` is the pooled within standard deviation.\n\n    Parameters\n    ----------\n    d : float or array_like\n        Cohen's d effect size for difference mean1 - mean2.\n\n    Returns\n    -------\n    prob : float or ndarray\n        Prob(x1 > x2)\n    "
    return stats.norm.cdf(d / np.sqrt(2))