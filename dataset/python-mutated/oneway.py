"""
Created on Wed Mar 18 10:33:38 2020

Author: Josef Perktold
License: BSD-3

"""
import numpy as np
from scipy import stats
from scipy.special import ncfdtrinc
from statsmodels.stats.power import ncf_cdf, ncf_ppf
from statsmodels.stats.robust_compare import TrimmedMean, scale_transform
from statsmodels.tools.testing import Holder
from statsmodels.stats.base import HolderTuple

def effectsize_oneway(means, vars_, nobs, use_var='unequal', ddof_between=0):
    if False:
        return 10
    '\n    Effect size corresponding to Cohen\'s f = nc / nobs for oneway anova\n\n    This contains adjustment for Welch and Brown-Forsythe Anova so that\n    effect size can be used with FTestAnovaPower.\n\n    Parameters\n    ----------\n    means : array_like\n        Mean of samples to be compared\n    vars_ : float or array_like\n        Residual (within) variance of each sample or pooled\n        If ``vars_`` is scalar, then it is interpreted as pooled variance that\n        is the same for all samples, ``use_var`` will be ignored.\n        Otherwise, the variances are used depending on the ``use_var`` keyword.\n    nobs : int or array_like\n        Number of observations for the samples.\n        If nobs is scalar, then it is assumed that all samples have the same\n        number ``nobs`` of observation, i.e. a balanced sample case.\n        Otherwise, statistics will be weighted corresponding to nobs.\n        Only relative sizes are relevant, any proportional change to nobs does\n        not change the effect size.\n    use_var : {"unequal", "equal", "bf"}\n        If ``use_var`` is "unequal", then the variances can differ across\n        samples and the effect size for Welch anova will be computed.\n    ddof_between : int\n        Degrees of freedom correction for the weighted between sum of squares.\n        The denominator is ``nobs_total - ddof_between``\n        This can be used to match differences across reference literature.\n\n    Returns\n    -------\n    f2 : float\n        Effect size corresponding to squared Cohen\'s f, which is also equal\n        to the noncentrality divided by total number of observations.\n\n    Notes\n    -----\n    This currently handles the following cases for oneway anova\n\n    - balanced sample with homoscedastic variances\n    - samples with different number of observations and with homoscedastic\n      variances\n    - samples with different number of observations and with heteroskedastic\n      variances. This corresponds to Welch anova\n\n    In the case of "unequal" and "bf" methods for unequal variances, the\n    effect sizes do not directly correspond to the test statistic in Anova.\n    Both have correction terms dropped or added, so the effect sizes match up\n    with using FTestAnovaPower.\n    If all variances are equal, then all three methods result in the same\n    effect size. If variances are unequal, then the three methods produce\n    small differences in effect size.\n\n    Note, the effect size and power computation for BF Anova was not found in\n    the literature. The correction terms were added so that FTestAnovaPower\n    provides a good approximation to the power.\n\n    Status: experimental\n    We might add additional returns, if those are needed to support power\n    and sample size applications.\n\n    Examples\n    --------\n    The following shows how to compute effect size and power for each of the\n    three anova methods. The null hypothesis is that the means are equal which\n    corresponds to a zero effect size. Under the alternative, means differ\n    with two sample means at a distance delta from the mean. We assume the\n    variance is the same under the null and alternative hypothesis.\n\n    ``nobs`` for the samples defines the fraction of observations in the\n    samples. ``nobs`` in the power method defines the total sample size.\n\n    In simulations, the computed power for standard anova,\n    i.e.``use_var="equal"`` overestimates the simulated power by a few percent.\n    The equal variance assumption does not hold in this example.\n\n    >>> from statsmodels.stats.oneway import effectsize_oneway\n    >>> from statsmodels.stats.power import FTestAnovaPower\n    >>>\n    >>> nobs = np.array([10, 12, 13, 15])\n    >>> delta = 0.5\n    >>> means_alt = np.array([-1, 0, 0, 1]) * delta\n    >>> vars_ = np.arange(1, len(means_alt) + 1)\n    >>>\n    >>> f2_alt = effectsize_oneway(means_alt, vars_, nobs, use_var="equal")\n    >>> f2_alt\n    0.04581300813008131\n    >>>\n    >>> kwds = {\'effect_size\': np.sqrt(f2_alt), \'nobs\': 100, \'alpha\': 0.05,\n    ...         \'k_groups\': 4}\n    >>> power = FTestAnovaPower().power(**kwds)\n    >>> power\n    0.39165892158983273\n    >>>\n    >>> f2_alt = effectsize_oneway(means_alt, vars_, nobs, use_var="unequal")\n    >>> f2_alt\n    0.060640138408304504\n    >>>\n    >>> kwds[\'effect_size\'] = np.sqrt(f2_alt)\n    >>> power = FTestAnovaPower().power(**kwds)\n    >>> power\n    0.5047366512800622\n    >>>\n    >>> f2_alt = effectsize_oneway(means_alt, vars_, nobs, use_var="bf")\n    >>> f2_alt\n    0.04391324307956788\n    >>>\n    >>> kwds[\'effect_size\'] = np.sqrt(f2_alt)\n    >>> power = FTestAnovaPower().power(**kwds)\n    >>> power\n    0.3765792117047725\n\n    '
    means = np.asarray(means)
    n_groups = means.shape[0]
    if np.size(nobs) == 1:
        nobs = np.ones(n_groups) * nobs
    nobs_t = nobs.sum()
    if use_var == 'equal':
        if np.size(vars_) == 1:
            var_resid = vars_
        else:
            vars_ = np.asarray(vars_)
            var_resid = ((nobs - 1) * vars_).sum() / (nobs_t - n_groups)
        vars_ = var_resid
    weights = nobs / vars_
    w_total = weights.sum()
    w_rel = weights / w_total
    meanw_t = w_rel @ means
    f2 = np.dot(weights, (means - meanw_t) ** 2) / (nobs_t - ddof_between)
    if use_var.lower() == 'bf':
        weights = nobs
        w_total = weights.sum()
        w_rel = weights / w_total
        meanw_t = w_rel @ means
        tmp = ((1.0 - nobs / nobs_t) * vars_).sum()
        statistic = 1.0 * (nobs * (means - meanw_t) ** 2).sum()
        statistic /= tmp
        f2 = statistic * (1.0 - nobs / nobs_t).sum() / nobs_t
        df_num2 = n_groups - 1
        df_num = tmp ** 2 / ((vars_ ** 2).sum() + (nobs / nobs_t * vars_).sum() ** 2 - 2 * (nobs / nobs_t * vars_ ** 2).sum())
        f2 *= df_num / df_num2
    return f2

def convert_effectsize_fsqu(f2=None, eta2=None):
    if False:
        i = 10
        return i + 15
    "Convert squared effect sizes in f family\n\n    f2 is signal to noise ratio, var_explained / var_residual\n\n    eta2 is proportion of explained variance, var_explained / var_total\n\n    uses the relationship:\n    f2 = eta2 / (1 - eta2)\n\n    Parameters\n    ----------\n    f2 : None or float\n       Squared Cohen's F effect size. If f2 is not None, then eta2 will be\n       computed.\n    eta2 : None or float\n       Squared eta effect size. If f2 is None and eta2 is not None, then f2 is\n       computed.\n\n    Returns\n    -------\n    res : Holder instance\n        An instance of the Holder class with f2 and eta2 as attributes.\n\n    "
    if f2 is not None:
        eta2 = 1 / (1 + 1 / f2)
    elif eta2 is not None:
        f2 = eta2 / (1 - eta2)
    res = Holder(f2=f2, eta2=eta2)
    return res

def _fstat2effectsize(f_stat, df):
    if False:
        i = 10
        return i + 15
    'Compute anova effect size from F-statistic\n\n    This might be combined with convert_effectsize_fsqu\n\n    Parameters\n    ----------\n    f_stat : array_like\n        Test statistic of an F-test\n    df : tuple\n        degrees of freedom ``df = (df1, df2)`` where\n         - df1 : numerator degrees of freedom, number of constraints\n         - df2 : denominator degrees of freedom, df_resid\n\n    Returns\n    -------\n    res : Holder instance\n        This instance contains effect size measures f2, eta2, omega2 and eps2\n        as attributes.\n\n    Notes\n    -----\n    This uses the following definitions:\n\n    - f2 = f_stat * df1 / df2\n    - eta2 = f2 / (f2 + 1)\n    - omega2 = (f2 - df1 / df2) / (f2 + 2)\n    - eps2 = (f2 - df1 / df2) / (f2 + 1)\n\n    This differs from effect size measures in other function which define\n    ``f2 = f_stat * df1 / nobs``\n    or an equivalent expression for power computation. The noncentrality\n    index for the hypothesis test is in those cases given by\n    ``nc = f_stat * df1``.\n\n    Currently omega2 and eps2 are computed in two different ways. Those\n    values agree for regular cases but can show different behavior in corner\n    cases (e.g. zero division).\n\n    '
    (df1, df2) = df
    f2 = f_stat * df1 / df2
    eta2 = f2 / (f2 + 1)
    omega2_ = (f_stat - 1) / (f_stat + (df2 + 1) / df1)
    omega2 = (f2 - df1 / df2) / (f2 + 1 + 1 / df2)
    eps2_ = (f_stat - 1) / (f_stat + df2 / df1)
    eps2 = (f2 - df1 / df2) / (f2 + 1)
    return Holder(f2=f2, eta2=eta2, omega2=omega2, eps2=eps2, eps2_=eps2_, omega2_=omega2_)

def wellek_to_f2(eps, n_groups):
    if False:
        return 10
    "Convert Wellek's effect size (sqrt) to Cohen's f-squared\n\n    This computes the following effect size :\n\n       f2 = 1 / n_groups * eps**2\n\n    Parameters\n    ----------\n    eps : float or ndarray\n        Wellek's effect size used in anova equivalence test\n    n_groups : int\n        Number of groups in oneway comparison\n\n    Returns\n    -------\n    f2 : effect size Cohen's f-squared\n\n    "
    f2 = 1 / n_groups * eps ** 2
    return f2

def f2_to_wellek(f2, n_groups):
    if False:
        while True:
            i = 10
    "Convert Cohen's f-squared to Wellek's effect size (sqrt)\n\n    This computes the following effect size :\n\n       eps = sqrt(n_groups * f2)\n\n    Parameters\n    ----------\n    f2 : float or ndarray\n        Effect size Cohen's f-squared\n    n_groups : int\n        Number of groups in oneway comparison\n\n    Returns\n    -------\n    eps : float or ndarray\n        Wellek's effect size used in anova equivalence test\n    "
    eps = np.sqrt(n_groups * f2)
    return eps

def fstat_to_wellek(f_stat, n_groups, nobs_mean):
    if False:
        i = 10
        return i + 15
    "Convert F statistic to wellek's effect size eps squared\n\n    This computes the following effect size :\n\n       es = f_stat * (n_groups - 1) / nobs_mean\n\n    Parameters\n    ----------\n    f_stat : float or ndarray\n        Test statistic of an F-test.\n    n_groups : int\n        Number of groups in oneway comparison\n    nobs_mean : float or ndarray\n        Average number of observations across groups.\n\n    Returns\n    -------\n    eps : float or ndarray\n        Wellek's effect size used in anova equivalence test\n\n    "
    es = f_stat * (n_groups - 1) / nobs_mean
    return es

def confint_noncentrality(f_stat, df, alpha=0.05, alternative='two-sided'):
    if False:
        return 10
    '\n    Confidence interval for noncentrality parameter in F-test\n\n    This does not yet handle non-negativity constraint on nc.\n    Currently only two-sided alternative is supported.\n\n    Parameters\n    ----------\n    f_stat : float\n    df : tuple\n        degrees of freedom ``df = (df1, df2)`` where\n\n        - df1 : numerator degrees of freedom, number of constraints\n        - df2 : denominator degrees of freedom, df_resid\n\n    alpha : float, default 0.05\n    alternative : {"two-sided"}\n        Other alternatives have not been implements.\n\n    Returns\n    -------\n    float\n        The end point of the confidence interval.\n\n    Notes\n    -----\n    The algorithm inverts the cdf of the noncentral F distribution with\n    respect to the noncentrality parameters.\n    See Steiger 2004 and references cited in it.\n\n    References\n    ----------\n    .. [1] Steiger, James H. 2004. “Beyond the F Test: Effect Size Confidence\n       Intervals and Tests of Close Fit in the Analysis of Variance and\n       Contrast Analysis.” Psychological Methods 9 (2): 164–82.\n       https://doi.org/10.1037/1082-989X.9.2.164.\n\n    See Also\n    --------\n    confint_effectsize_oneway\n    '
    (df1, df2) = df
    if alternative in ['two-sided', '2s', 'ts']:
        alpha1s = alpha / 2
        ci = ncfdtrinc(df1, df2, [1 - alpha1s, alpha1s], f_stat)
    else:
        raise NotImplementedError
    return ci

def confint_effectsize_oneway(f_stat, df, alpha=0.05, nobs=None):
    if False:
        print('Hello World!')
    '\n    Confidence interval for effect size in oneway anova for F distribution\n\n    This does not yet handle non-negativity constraint on nc.\n    Currently only two-sided alternative is supported.\n\n    Parameters\n    ----------\n    f_stat : float\n    df : tuple\n        degrees of freedom ``df = (df1, df2)`` where\n\n        - df1 : numerator degrees of freedom, number of constraints\n        - df2 : denominator degrees of freedom, df_resid\n\n    alpha : float, default 0.05\n    nobs : int, default None\n\n    Returns\n    -------\n    Holder\n        Class with effect size and confidence attributes\n\n    Notes\n    -----\n    The confidence interval for the noncentrality parameter is obtained by\n    inverting the cdf of the noncentral F distribution. Confidence intervals\n    for other effect sizes are computed by endpoint transformation.\n\n\n    R package ``effectsize`` does not compute the confidence intervals in the\n    same way. Their confidence intervals can be replicated with\n\n    >>> ci_nc = confint_noncentrality(f_stat, df1, df2, alpha=0.1)\n    >>> ci_es = smo._fstat2effectsize(ci_nc / df1, df1, df2)\n\n    See Also\n    --------\n    confint_noncentrality\n    '
    (df1, df2) = df
    if nobs is None:
        nobs = df1 + df2 + 1
    ci_nc = confint_noncentrality(f_stat, df, alpha=alpha)
    ci_f2 = ci_nc / nobs
    ci_res = convert_effectsize_fsqu(f2=ci_f2)
    ci_res.ci_omega2 = (ci_f2 - df1 / df2) / (ci_f2 + 1 + 1 / df2)
    ci_res.ci_nc = ci_nc
    ci_res.ci_f = np.sqrt(ci_res.f2)
    ci_res.ci_eta = np.sqrt(ci_res.eta2)
    ci_res.ci_f_corrected = np.sqrt(ci_res.f2 * (df1 + 1) / df1)
    return ci_res

def anova_generic(means, variances, nobs, use_var='unequal', welch_correction=True, info=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Oneway Anova based on summary statistics\n\n    Parameters\n    ----------\n    means : array_like\n        Mean of samples to be compared\n    variances : float or array_like\n        Residual (within) variance of each sample or pooled.\n        If ``variances`` is scalar, then it is interpreted as pooled variance\n        that is the same for all samples, ``use_var`` will be ignored.\n        Otherwise, the variances are used depending on the ``use_var`` keyword.\n    nobs : int or array_like\n        Number of observations for the samples.\n        If nobs is scalar, then it is assumed that all samples have the same\n        number ``nobs`` of observation, i.e. a balanced sample case.\n        Otherwise, statistics will be weighted corresponding to nobs.\n        Only relative sizes are relevant, any proportional change to nobs does\n        not change the effect size.\n    use_var : {"unequal", "equal", "bf"}\n        If ``use_var`` is "unequal", then the variances can differ across\n        samples and the effect size for Welch anova will be computed.\n    welch_correction : bool\n        If this is false, then the Welch correction to the test statistic is\n        not included. This allows the computation of an effect size measure\n        that corresponds more closely to Cohen\'s f.\n    info : not used yet\n\n    Returns\n    -------\n    res : results instance\n        This includes `statistic` and `pvalue`.\n\n    '
    options = {'use_var': use_var, 'welch_correction': welch_correction}
    if means.ndim != 1:
        raise ValueError('data (means, ...) has to be one-dimensional')
    nobs_t = nobs.sum()
    n_groups = len(means)
    if use_var == 'unequal':
        weights = nobs / variances
    else:
        weights = nobs
    w_total = weights.sum()
    w_rel = weights / w_total
    meanw_t = w_rel @ means
    statistic = np.dot(weights, (means - meanw_t) ** 2) / (n_groups - 1.0)
    df_num = n_groups - 1.0
    if use_var == 'unequal':
        tmp = ((1 - w_rel) ** 2 / (nobs - 1)).sum() / (n_groups ** 2 - 1)
        if welch_correction:
            statistic /= 1 + 2 * (n_groups - 2) * tmp
        df_denom = 1.0 / (3.0 * tmp)
    elif use_var == 'equal':
        tmp = ((nobs - 1) * variances).sum() / (nobs_t - n_groups)
        statistic /= tmp
        df_denom = nobs_t - n_groups
    elif use_var == 'bf':
        tmp = ((1.0 - nobs / nobs_t) * variances).sum()
        statistic = 1.0 * (nobs * (means - meanw_t) ** 2).sum()
        statistic /= tmp
        df_num2 = n_groups - 1
        df_denom = tmp ** 2 / ((1.0 - nobs / nobs_t) ** 2 * variances ** 2 / (nobs - 1)).sum()
        df_num = tmp ** 2 / ((variances ** 2).sum() + (nobs / nobs_t * variances).sum() ** 2 - 2 * (nobs / nobs_t * variances ** 2).sum())
        pval2 = stats.f.sf(statistic, df_num2, df_denom)
        options['df2'] = (df_num2, df_denom)
        options['df_num2'] = df_num2
        options['pvalue2'] = pval2
    else:
        raise ValueError('use_var is to be one of "unequal", "equal" or "bf"')
    pval = stats.f.sf(statistic, df_num, df_denom)
    res = HolderTuple(statistic=statistic, pvalue=pval, df=(df_num, df_denom), df_num=df_num, df_denom=df_denom, nobs_t=nobs_t, n_groups=n_groups, means=means, nobs=nobs, vars_=variances, **options)
    return res

def anova_oneway(data, groups=None, use_var='unequal', welch_correction=True, trim_frac=0):
    if False:
        return 10
    'Oneway Anova\n\n    This implements standard anova, Welch and Brown-Forsythe, and trimmed\n    (Yuen) variants of those.\n\n    Parameters\n    ----------\n    data : tuple of array_like or DataFrame or Series\n        Data for k independent samples, with k >= 2.\n        The data can be provided as a tuple or list of arrays or in long\n        format with outcome observations in ``data`` and group membership in\n        ``groups``.\n    groups : ndarray or Series\n        If data is in long format, then groups is needed as indicator to which\n        group or sample and observations belongs.\n    use_var : {"unequal", "equal" or "bf"}\n        `use_var` specified how to treat heteroscedasticity, unequal variance,\n        across samples. Three approaches are available\n\n        "unequal" : Variances are not assumed to be equal across samples.\n            Heteroscedasticity is taken into account with Welch Anova and\n            Satterthwaite-Welch degrees of freedom.\n            This is the default.\n        "equal" : Variances are assumed to be equal across samples.\n            This is the standard Anova.\n        "bf: Variances are not assumed to be equal across samples.\n            The method is Browne-Forsythe (1971) for testing equality of means\n            with the corrected degrees of freedom by Merothra. The original BF\n            degrees of freedom are available as additional attributes in the\n            results instance, ``df_denom2`` and ``p_value2``.\n\n    welch_correction : bool\n        If this is false, then the Welch correction to the test statistic is\n        not included. This allows the computation of an effect size measure\n        that corresponds more closely to Cohen\'s f.\n    trim_frac : float in [0, 0.5)\n        Optional trimming for Anova with trimmed mean and winsorized variances.\n        With the default trim_frac equal to zero, the oneway Anova statistics\n        are computed without trimming. If `trim_frac` is larger than zero,\n        then the largest and smallest observations in each sample are trimmed.\n        The number of trimmed observations is the fraction of number of\n        observations in the sample truncated to the next lower integer.\n        `trim_frac` has to be smaller than 0.5, however, if the fraction is\n        so large that there are not enough observations left over, then `nan`\n        will be returned.\n\n    Returns\n    -------\n    res : results instance\n        The returned HolderTuple instance has the following main attributes\n        and some additional information in other attributes.\n\n        statistic : float\n            Test statistic for k-sample mean comparison which is approximately\n            F-distributed.\n        pvalue : float\n            If ``use_var="bf"``, then the p-value is based on corrected\n            degrees of freedom following Mehrotra 1997.\n        pvalue2 : float\n            This is the p-value based on degrees of freedom as in\n            Brown-Forsythe 1974 and is only available if ``use_var="bf"``.\n        df = (df_denom, df_num) : tuple of floats\n            Degreeds of freedom for the F-distribution depend on ``use_var``.\n            If ``use_var="bf"``, then `df_denom` is for Mehrotra p-values\n            `df_denom2` is available for Brown-Forsythe 1974 p-values.\n            `df_num` is the same numerator degrees of freedom for both\n            p-values.\n\n    Notes\n    -----\n    Welch\'s anova is correctly sized (not liberal or conservative) in smaller\n    samples if the distribution of the samples is not very far away from the\n    normal distribution. The test can become liberal if the data is strongly\n    skewed. Welch\'s Anova can also be correctly sized for discrete\n    distributions with finite support, like Lickert scale data.\n    The trimmed version is robust to many non-normal distributions, it stays\n    correctly sized in many cases, and is more powerful in some cases with\n    skewness or heavy tails.\n\n    Trimming is currently based on the integer part of ``nobs * trim_frac``.\n    The default might change to including fractional observations as in the\n    original articles by Yuen.\n\n\n    See Also\n    --------\n    anova_generic\n\n    References\n    ----------\n    Brown, Morton B., and Alan B. Forsythe. 1974. “The Small Sample Behavior\n    of Some Statistics Which Test the Equality of Several Means.”\n    Technometrics 16 (1) (February 1): 129–132. doi:10.2307/1267501.\n\n    Mehrotra, Devan V. 1997. “Improving the Brown-Forsythe Solution to the\n    Generalized Behrens-Fisher Problem.” Communications in Statistics -\n    Simulation and Computation 26 (3): 1139–1145.\n    doi:10.1080/03610919708813431.\n    '
    if groups is not None:
        uniques = np.unique(groups)
        data = [data[groups == uni] for uni in uniques]
    else:
        pass
    args = list(map(np.asarray, data))
    if any([x.ndim != 1 for x in args]):
        raise ValueError('data arrays have to be one-dimensional')
    nobs = np.array([len(x) for x in args], float)
    if trim_frac == 0:
        means = np.array([x.mean() for x in args])
        vars_ = np.array([x.var(ddof=1) for x in args])
    else:
        tms = [TrimmedMean(x, trim_frac) for x in args]
        means = np.array([tm.mean_trimmed for tm in tms])
        vars_ = np.array([tm.var_winsorized * (tm.nobs - 1) / (tm.nobs_reduced - 1) for tm in tms])
        nobs = np.array([tm.nobs_reduced for tm in tms])
    res = anova_generic(means, vars_, nobs, use_var=use_var, welch_correction=welch_correction)
    return res

def equivalence_oneway_generic(f_stat, n_groups, nobs, equiv_margin, df, alpha=0.05, margin_type='f2'):
    if False:
        for i in range(10):
            print('nop')
    'Equivalence test for oneway anova (Wellek and extensions)\n\n    This is an helper function when summary statistics are available.\n    Use `equivalence_oneway` instead.\n\n    The null hypothesis is that the means differ by more than `equiv_margin`\n    in the anova distance measure.\n    If the Null is rejected, then the data supports that means are equivalent,\n    i.e. within a given distance.\n\n    Parameters\n    ----------\n    f_stat : float\n        F-statistic\n    n_groups : int\n        Number of groups in oneway comparison.\n    nobs : ndarray\n        Array of number of observations in groups.\n    equiv_margin : float\n        Equivalence margin in terms of effect size. Effect size can be chosen\n        with `margin_type`. default is squared Cohen\'s f.\n    df : tuple\n        degrees of freedom ``df = (df1, df2)`` where\n\n        - df1 : numerator degrees of freedom, number of constraints\n        - df2 : denominator degrees of freedom, df_resid\n\n    alpha : float in (0, 1)\n        Significance level for the hypothesis test.\n    margin_type : "f2" or "wellek"\n        Type of effect size used for equivalence margin.\n\n    Returns\n    -------\n    results : instance of HolderTuple class\n        The two main attributes are test statistic `statistic` and p-value\n        `pvalue`.\n\n    Notes\n    -----\n    Equivalence in this function is defined in terms of a squared distance\n    measure similar to Mahalanobis distance.\n    Alternative definitions for the oneway case are based on maximum difference\n    between pairs of means or similar pairwise distances.\n\n    The equivalence margin is used for the noncentrality parameter in the\n    noncentral F distribution for the test statistic. In samples with unequal\n    variances estimated using Welch or Brown-Forsythe Anova, the f-statistic\n    depends on the unequal variances and corrections to the test statistic.\n    This means that the equivalence margins are not fully comparable across\n    methods for treating unequal variances.\n\n    References\n    ----------\n    Wellek, Stefan. 2010. Testing Statistical Hypotheses of Equivalence and\n    Noninferiority. 2nd ed. Boca Raton: CRC Press.\n\n    Cribbie, Robert A., Chantal A. Arpin-Cribbie, and Jamie A. Gruman. 2009.\n    “Tests of Equivalence for One-Way Independent Groups Designs.” The Journal\n    of Experimental Education 78 (1): 1–13.\n    https://doi.org/10.1080/00220970903224552.\n\n    Jan, Show-Li, and Gwowen Shieh. 2019. “On the Extended Welch Test for\n    Assessing Equivalence of Standardized Means.” Statistics in\n    Biopharmaceutical Research 0 (0): 1–8.\n    https://doi.org/10.1080/19466315.2019.1654915.\n\n    '
    nobs_t = nobs.sum()
    nobs_mean = nobs_t / n_groups
    if margin_type == 'wellek':
        nc_null = nobs_mean * equiv_margin ** 2
        es = f_stat * (n_groups - 1) / nobs_mean
        type_effectsize = "Wellek's psi_squared"
    elif margin_type in ['f2', 'fsqu', 'fsquared']:
        nc_null = nobs_t * equiv_margin
        es = f_stat / nobs_t
        type_effectsize = "Cohen's f_squared"
    else:
        raise ValueError('`margin_type` should be "f2" or "wellek"')
    crit_f = ncf_ppf(alpha, df[0], df[1], nc_null)
    if margin_type == 'wellek':
        crit_es = crit_f * (n_groups - 1) / nobs_mean
    elif margin_type in ['f2', 'fsqu', 'fsquared']:
        crit_es = crit_f / nobs_t
    reject = es < crit_es
    pv = ncf_cdf(f_stat, df[0], df[1], nc_null)
    pwr = ncf_cdf(crit_f, df[0], df[1], 1e-13)
    res = HolderTuple(statistic=f_stat, pvalue=pv, effectsize=es, crit_f=crit_f, crit_es=crit_es, reject=reject, power_zero=pwr, df=df, f_stat=f_stat, type_effectsize=type_effectsize)
    return res

def equivalence_oneway(data, equiv_margin, groups=None, use_var='unequal', welch_correction=True, trim_frac=0, margin_type='f2'):
    if False:
        while True:
            i = 10
    'equivalence test for oneway anova (Wellek\'s Anova)\n\n    The null hypothesis is that the means differ by more than `equiv_margin`\n    in the anova distance measure.\n    If the Null is rejected, then the data supports that means are equivalent,\n    i.e. within a given distance.\n\n    Parameters\n    ----------\n    data : tuple of array_like or DataFrame or Series\n        Data for k independent samples, with k >= 2.\n        The data can be provided as a tuple or list of arrays or in long\n        format with outcome observations in ``data`` and group membership in\n        ``groups``.\n    equiv_margin : float\n        Equivalence margin in terms of effect size. Effect size can be chosen\n        with `margin_type`. default is squared Cohen\'s f.\n    groups : ndarray or Series\n        If data is in long format, then groups is needed as indicator to which\n        group or sample and observations belongs.\n    use_var : {"unequal", "equal" or "bf"}\n        `use_var` specified how to treat heteroscedasticity, unequal variance,\n        across samples. Three approaches are available\n\n        "unequal" : Variances are not assumed to be equal across samples.\n            Heteroscedasticity is taken into account with Welch Anova and\n            Satterthwaite-Welch degrees of freedom.\n            This is the default.\n        "equal" : Variances are assumed to be equal across samples.\n            This is the standard Anova.\n        "bf: Variances are not assumed to be equal across samples.\n            The method is Browne-Forsythe (1971) for testing equality of means\n            with the corrected degrees of freedom by Merothra. The original BF\n            degrees of freedom are available as additional attributes in the\n            results instance, ``df_denom2`` and ``p_value2``.\n\n    welch_correction : bool\n        If this is false, then the Welch correction to the test statistic is\n        not included. This allows the computation of an effect size measure\n        that corresponds more closely to Cohen\'s f.\n    trim_frac : float in [0, 0.5)\n        Optional trimming for Anova with trimmed mean and winsorized variances.\n        With the default trim_frac equal to zero, the oneway Anova statistics\n        are computed without trimming. If `trim_frac` is larger than zero,\n        then the largest and smallest observations in each sample are trimmed.\n        The number of trimmed observations is the fraction of number of\n        observations in the sample truncated to the next lower integer.\n        `trim_frac` has to be smaller than 0.5, however, if the fraction is\n        so large that there are not enough observations left over, then `nan`\n        will be returned.\n    margin_type : "f2" or "wellek"\n        Type of effect size used for equivalence margin, either squared\n        Cohen\'s f or Wellek\'s psi. Default is "f2".\n\n    Returns\n    -------\n    results : instance of HolderTuple class\n        The two main attributes are test statistic `statistic` and p-value\n        `pvalue`.\n\n    See Also\n    --------\n    anova_oneway\n    equivalence_scale_oneway\n    '
    res0 = anova_oneway(data, groups=groups, use_var=use_var, welch_correction=welch_correction, trim_frac=trim_frac)
    f_stat = res0.statistic
    res = equivalence_oneway_generic(f_stat, res0.n_groups, res0.nobs_t, equiv_margin, res0.df, alpha=0.05, margin_type=margin_type)
    return res

def _power_equivalence_oneway_emp(f_stat, n_groups, nobs, eps, df, alpha=0.05):
    if False:
        return 10
    "Empirical power of oneway equivalence test\n\n    This only returns post-hoc, empirical power.\n\n    Warning: eps is currently effect size margin as defined as in Wellek, and\n    not the signal to noise ratio (Cohen's f family).\n\n    Parameters\n    ----------\n    f_stat : float\n        F-statistic from oneway anova, used to compute empirical effect size\n    n_groups : int\n        Number of groups in oneway comparison.\n    nobs : ndarray\n        Array of number of observations in groups.\n    eps : float\n        Equivalence margin in terms of effect size given by Wellek's psi.\n    df : tuple\n        Degrees of freedom for F distribution.\n    alpha : float in (0, 1)\n        Significance level for the hypothesis test.\n\n    Returns\n    -------\n    pow : float\n        Ex-post, post-hoc or empirical power at f-statistic of the equivalence\n        test.\n    "
    res = equivalence_oneway_generic(f_stat, n_groups, nobs, eps, df, alpha=alpha, margin_type='wellek')
    nobs_mean = nobs.sum() / n_groups
    fn = f_stat
    esn = fn * (n_groups - 1) / nobs_mean
    pow_ = ncf_cdf(res.crit_f, df[0], df[1], nobs_mean * esn)
    return pow_

def power_equivalence_oneway(f2_alt, equiv_margin, nobs_t, n_groups=None, df=None, alpha=0.05, margin_type='f2'):
    if False:
        while True:
            i = 10
    '\n    Power of  oneway equivalence test\n\n    Parameters\n    ----------\n    f2_alt : float\n        Effect size, squared Cohen\'s f, under the alternative.\n    equiv_margin : float\n        Equivalence margin in terms of effect size. Effect size can be chosen\n        with `margin_type`. default is squared Cohen\'s f.\n    nobs_t : ndarray\n        Total number of observations summed over all groups.\n    n_groups : int\n        Number of groups in oneway comparison. If margin_type is "wellek",\n        then either ``n_groups`` or ``df`` has to be given.\n    df : tuple\n        Degrees of freedom for F distribution,\n        ``df = (n_groups - 1, nobs_t - n_groups)``\n    alpha : float in (0, 1)\n        Significance level for the hypothesis test.\n    margin_type : "f2" or "wellek"\n        Type of effect size used for equivalence margin, either squared\n        Cohen\'s f or Wellek\'s psi. Default is "f2".\n\n    Returns\n    -------\n    pow_alt : float\n        Power of the equivalence test at given equivalence effect size under\n        the alternative.\n    '
    if df is None:
        if n_groups is None:
            raise ValueError('either df or n_groups has to be provided')
        df = (n_groups - 1, nobs_t - n_groups)
    if f2_alt == 0:
        f2_alt = 1e-13
    if margin_type in ['f2', 'fsqu', 'fsquared']:
        f2_null = equiv_margin
    elif margin_type == 'wellek':
        if n_groups is None:
            raise ValueError('If margin_type is wellek, then n_groups has to be provided')
        nobs_mean = nobs_t / n_groups
        f2_null = nobs_mean * equiv_margin ** 2 / nobs_t
        f2_alt = nobs_mean * f2_alt ** 2 / nobs_t
    else:
        raise ValueError('`margin_type` should be "f2" or "wellek"')
    crit_f_margin = ncf_ppf(alpha, df[0], df[1], nobs_t * f2_null)
    pwr_alt = ncf_cdf(crit_f_margin, df[0], df[1], nobs_t * f2_alt)
    return pwr_alt

def simulate_power_equivalence_oneway(means, nobs, equiv_margin, vars_=None, k_mc=1000, trim_frac=0, options_var=None, margin_type='f2'):
    if False:
        return 10
    "Simulate Power for oneway equivalence test (Wellek's Anova)\n\n    This function is experimental and written to evaluate asymptotic power\n    function. This function will change without backwards compatibility\n    constraints. The only part that is stable is `pvalue` attribute in results.\n\n    Effect size for equivalence margin\n\n    "
    if options_var is None:
        options_var = ['unequal', 'equal', 'bf']
    if vars_ is not None:
        stds = np.sqrt(vars_)
    else:
        stds = np.ones(len(means))
    nobs_mean = nobs.mean()
    n_groups = len(nobs)
    res_mc = []
    f_mc = []
    reject_mc = []
    other_mc = []
    for _ in range(k_mc):
        (y0, y1, y2, y3) = [m + std * np.random.randn(n) for (n, m, std) in zip(nobs, means, stds)]
        res_i = []
        f_i = []
        reject_i = []
        other_i = []
        for uv in options_var:
            res0 = anova_oneway([y0, y1, y2, y3], use_var=uv, trim_frac=trim_frac)
            f_stat = res0.statistic
            res1 = equivalence_oneway_generic(f_stat, n_groups, nobs.sum(), equiv_margin, res0.df, alpha=0.05, margin_type=margin_type)
            res_i.append(res1.pvalue)
            es_wellek = f_stat * (n_groups - 1) / nobs_mean
            f_i.append(es_wellek)
            reject_i.append(res1.reject)
            other_i.extend([res1.crit_f, res1.crit_es, res1.power_zero])
        res_mc.append(res_i)
        f_mc.append(f_i)
        reject_mc.append(reject_i)
        other_mc.append(other_i)
    f_mc = np.asarray(f_mc)
    other_mc = np.asarray(other_mc)
    res_mc = np.asarray(res_mc)
    reject_mc = np.asarray(reject_mc)
    res = Holder(f_stat=f_mc, other=other_mc, pvalue=res_mc, reject=reject_mc)
    return res

def test_scale_oneway(data, method='bf', center='median', transform='abs', trim_frac_mean=0.1, trim_frac_anova=0.0):
    if False:
        while True:
            i = 10
    'Oneway Anova test for equal scale, variance or dispersion\n\n    This hypothesis test performs a oneway anova test on transformed data and\n    includes Levene and Brown-Forsythe tests for equal variances as special\n    cases.\n\n    Parameters\n    ----------\n    data : tuple of array_like or DataFrame or Series\n        Data for k independent samples, with k >= 2. The data can be provided\n        as a tuple or list of arrays or in long format with outcome\n        observations in ``data`` and group membership in ``groups``.\n    method : {"unequal", "equal" or "bf"}\n        How to treat heteroscedasticity across samples. This is used as\n        `use_var` option in `anova_oneway` and refers to the variance of the\n        transformed data, i.e. assumption is on 4th moment if squares are used\n        as transform.\n        Three approaches are available:\n\n        "unequal" : Variances are not assumed to be equal across samples.\n            Heteroscedasticity is taken into account with Welch Anova and\n            Satterthwaite-Welch degrees of freedom.\n            This is the default.\n        "equal" : Variances are assumed to be equal across samples.\n            This is the standard Anova.\n        "bf" : Variances are not assumed to be equal across samples.\n            The method is Browne-Forsythe (1971) for testing equality of means\n            with the corrected degrees of freedom by Merothra. The original BF\n            degrees of freedom are available as additional attributes in the\n            results instance, ``df_denom2`` and ``p_value2``.\n\n    center : "median", "mean", "trimmed" or float\n        Statistic used for centering observations. If a float, then this\n        value is used to center. Default is median.\n    transform : "abs", "square" or callable\n        Transformation for the centered observations. If a callable, then this\n        function is called on the centered data.\n        Default is absolute value.\n    trim_frac_mean=0.1 : float in [0, 0.5)\n        Trim fraction for the trimmed mean when `center` is "trimmed"\n    trim_frac_anova : float in [0, 0.5)\n        Optional trimming for Anova with trimmed mean and Winsorized variances.\n        With the default trim_frac equal to zero, the oneway Anova statistics\n        are computed without trimming. If `trim_frac` is larger than zero,\n        then the largest and smallest observations in each sample are trimmed.\n        see ``trim_frac`` option in `anova_oneway`\n\n    Returns\n    -------\n    res : results instance\n        The returned HolderTuple instance has the following main attributes\n        and some additional information in other attributes.\n\n        statistic : float\n            Test statistic for k-sample mean comparison which is approximately\n            F-distributed.\n        pvalue : float\n            If ``method="bf"``, then the p-value is based on corrected\n            degrees of freedom following Mehrotra 1997.\n        pvalue2 : float\n            This is the p-value based on degrees of freedom as in\n            Brown-Forsythe 1974 and is only available if ``method="bf"``.\n        df : (df_denom, df_num)\n            Tuple containing degrees of freedom for the F-distribution depend\n            on ``method``. If ``method="bf"``, then `df_denom` is for Mehrotra\n            p-values `df_denom2` is available for Brown-Forsythe 1974 p-values.\n            `df_num` is the same numerator degrees of freedom for both\n            p-values.\n\n    See Also\n    --------\n    anova_oneway\n    scale_transform\n    '
    data = map(np.asarray, data)
    xxd = [scale_transform(x, center=center, transform=transform, trim_frac=trim_frac_mean) for x in data]
    res = anova_oneway(xxd, groups=None, use_var=method, welch_correction=True, trim_frac=trim_frac_anova)
    res.data_transformed = xxd
    return res

def equivalence_scale_oneway(data, equiv_margin, method='bf', center='median', transform='abs', trim_frac_mean=0.0, trim_frac_anova=0.0):
    if False:
        return 10
    'Oneway Anova test for equivalence of scale, variance or dispersion\n\n    This hypothesis test performs a oneway equivalence anova test on\n    transformed data.\n\n    Note, the interpretation of the equivalence margin `equiv_margin` will\n    depend on the transformation of the data. Transformations like\n    absolute deviation are not scaled to correspond to the variance under\n    normal distribution.\n\n    Parameters\n    ----------\n    data : tuple of array_like or DataFrame or Series\n        Data for k independent samples, with k >= 2. The data can be provided\n        as a tuple or list of arrays or in long format with outcome\n        observations in ``data`` and group membership in ``groups``.\n    equiv_margin : float\n        Equivalence margin in terms of effect size. Effect size can be chosen\n        with `margin_type`. default is squared Cohen\'s f.\n    method : {"unequal", "equal" or "bf"}\n        How to treat heteroscedasticity across samples. This is used as\n        `use_var` option in `anova_oneway` and refers to the variance of the\n        transformed data, i.e. assumption is on 4th moment if squares are used\n        as transform.\n        Three approaches are available:\n\n        "unequal" : Variances are not assumed to be equal across samples.\n            Heteroscedasticity is taken into account with Welch Anova and\n            Satterthwaite-Welch degrees of freedom.\n            This is the default.\n        "equal" : Variances are assumed to be equal across samples.\n            This is the standard Anova.\n        "bf" : Variances are not assumed to be equal across samples.\n            The method is Browne-Forsythe (1971) for testing equality of means\n            with the corrected degrees of freedom by Merothra. The original BF\n            degrees of freedom are available as additional attributes in the\n            results instance, ``df_denom2`` and ``p_value2``.\n    center : "median", "mean", "trimmed" or float\n        Statistic used for centering observations. If a float, then this\n        value is used to center. Default is median.\n    transform : "abs", "square" or callable\n        Transformation for the centered observations. If a callable, then this\n        function is called on the centered data.\n        Default is absolute value.\n    trim_frac_mean : float in [0, 0.5)\n        Trim fraction for the trimmed mean when `center` is "trimmed"\n    trim_frac_anova : float in [0, 0.5)\n        Optional trimming for Anova with trimmed mean and Winsorized variances.\n        With the default trim_frac equal to zero, the oneway Anova statistics\n        are computed without trimming. If `trim_frac` is larger than zero,\n        then the largest and smallest observations in each sample are trimmed.\n        see ``trim_frac`` option in `anova_oneway`\n\n    Returns\n    -------\n    results : instance of HolderTuple class\n        The two main attributes are test statistic `statistic` and p-value\n        `pvalue`.\n\n    See Also\n    --------\n    anova_oneway\n    scale_transform\n    equivalence_oneway\n    '
    data = map(np.asarray, data)
    xxd = [scale_transform(x, center=center, transform=transform, trim_frac=trim_frac_mean) for x in data]
    res = equivalence_oneway(xxd, equiv_margin, use_var=method, welch_correction=True, trim_frac=trim_frac_anova)
    res.x_transformed = xxd
    return res