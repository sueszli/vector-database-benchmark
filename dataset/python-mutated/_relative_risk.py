import operator
from dataclasses import dataclass
import numpy as np
from scipy.special import ndtri
from ._common import ConfidenceInterval

def _validate_int(n, bound, name):
    if False:
        i = 10
        return i + 15
    msg = f'{name} must be an integer not less than {bound}, but got {n!r}'
    try:
        n = operator.index(n)
    except TypeError:
        raise TypeError(msg) from None
    if n < bound:
        raise ValueError(msg)
    return n

@dataclass
class RelativeRiskResult:
    """
    Result of `scipy.stats.contingency.relative_risk`.

    Attributes
    ----------
    relative_risk : float
        This is::

            (exposed_cases/exposed_total) / (control_cases/control_total)

    exposed_cases : int
        The number of "cases" (i.e. occurrence of disease or other event
        of interest) among the sample of "exposed" individuals.
    exposed_total : int
        The total number of "exposed" individuals in the sample.
    control_cases : int
        The number of "cases" among the sample of "control" or non-exposed
        individuals.
    control_total : int
        The total number of "control" individuals in the sample.

    Methods
    -------
    confidence_interval :
        Compute the confidence interval for the relative risk estimate.
    """
    relative_risk: float
    exposed_cases: int
    exposed_total: int
    control_cases: int
    control_total: int

    def confidence_interval(self, confidence_level=0.95):
        if False:
            return 10
        '\n        Compute the confidence interval for the relative risk.\n\n        The confidence interval is computed using the Katz method\n        (i.e. "Method C" of [1]_; see also [2]_, section 3.1.2).\n\n        Parameters\n        ----------\n        confidence_level : float, optional\n            The confidence level to use for the confidence interval.\n            Default is 0.95.\n\n        Returns\n        -------\n        ci : ConfidenceInterval instance\n            The return value is an object with attributes ``low`` and\n            ``high`` that hold the confidence interval.\n\n        References\n        ----------\n        .. [1] D. Katz, J. Baptista, S. P. Azen and M. C. Pike, "Obtaining\n               confidence intervals for the risk ratio in cohort studies",\n               Biometrics, 34, 469-474 (1978).\n        .. [2] Hardeo Sahai and Anwer Khurshid, Statistics in Epidemiology,\n               CRC Press LLC, Boca Raton, FL, USA (1996).\n\n\n        Examples\n        --------\n        >>> from scipy.stats.contingency import relative_risk\n        >>> result = relative_risk(exposed_cases=10, exposed_total=75,\n        ...                        control_cases=12, control_total=225)\n        >>> result.relative_risk\n        2.5\n        >>> result.confidence_interval()\n        ConfidenceInterval(low=1.1261564003469628, high=5.549850800541033)\n        '
        if not 0 <= confidence_level <= 1:
            raise ValueError('confidence_level must be in the interval [0, 1].')
        if self.exposed_cases == 0 and self.control_cases == 0:
            return ConfidenceInterval(low=np.nan, high=np.nan)
        elif self.exposed_cases == 0:
            return ConfidenceInterval(low=0.0, high=np.nan)
        elif self.control_cases == 0:
            return ConfidenceInterval(low=np.nan, high=np.inf)
        alpha = 1 - confidence_level
        z = ndtri(1 - alpha / 2)
        rr = self.relative_risk
        se = np.sqrt(1 / self.exposed_cases - 1 / self.exposed_total + 1 / self.control_cases - 1 / self.control_total)
        delta = z * se
        katz_lo = rr * np.exp(-delta)
        katz_hi = rr * np.exp(delta)
        return ConfidenceInterval(low=katz_lo, high=katz_hi)

def relative_risk(exposed_cases, exposed_total, control_cases, control_total):
    if False:
        i = 10
        return i + 15
    '\n    Compute the relative risk (also known as the risk ratio).\n\n    This function computes the relative risk associated with a 2x2\n    contingency table ([1]_, section 2.2.3; [2]_, section 3.1.2). Instead\n    of accepting a table as an argument, the individual numbers that are\n    used to compute the relative risk are given as separate parameters.\n    This is to avoid the ambiguity of which row or column of the contingency\n    table corresponds to the "exposed" cases and which corresponds to the\n    "control" cases.  Unlike, say, the odds ratio, the relative risk is not\n    invariant under an interchange of the rows or columns.\n\n    Parameters\n    ----------\n    exposed_cases : nonnegative int\n        The number of "cases" (i.e. occurrence of disease or other event\n        of interest) among the sample of "exposed" individuals.\n    exposed_total : positive int\n        The total number of "exposed" individuals in the sample.\n    control_cases : nonnegative int\n        The number of "cases" among the sample of "control" or non-exposed\n        individuals.\n    control_total : positive int\n        The total number of "control" individuals in the sample.\n\n    Returns\n    -------\n    result : instance of `~scipy.stats._result_classes.RelativeRiskResult`\n        The object has the float attribute ``relative_risk``, which is::\n\n            rr = (exposed_cases/exposed_total) / (control_cases/control_total)\n\n        The object also has the method ``confidence_interval`` to compute\n        the confidence interval of the relative risk for a given confidence\n        level.\n\n    See Also\n    --------\n    odds_ratio\n\n    Notes\n    -----\n    The R package epitools has the function `riskratio`, which accepts\n    a table with the following layout::\n\n                        disease=0   disease=1\n        exposed=0 (ref)    n00         n01\n        exposed=1          n10         n11\n\n    With a 2x2 table in the above format, the estimate of the CI is\n    computed by `riskratio` when the argument method="wald" is given,\n    or with the function `riskratio.wald`.\n\n    For example, in a test of the incidence of lung cancer among a\n    sample of smokers and nonsmokers, the "exposed" category would\n    correspond to "is a smoker" and the "disease" category would\n    correspond to "has or had lung cancer".\n\n    To pass the same data to ``relative_risk``, use::\n\n        relative_risk(n11, n10 + n11, n01, n00 + n01)\n\n    .. versionadded:: 1.7.0\n\n    References\n    ----------\n    .. [1] Alan Agresti, An Introduction to Categorical Data Analysis\n           (second edition), Wiley, Hoboken, NJ, USA (2007).\n    .. [2] Hardeo Sahai and Anwer Khurshid, Statistics in Epidemiology,\n           CRC Press LLC, Boca Raton, FL, USA (1996).\n\n    Examples\n    --------\n    >>> from scipy.stats.contingency import relative_risk\n\n    This example is from Example 3.1 of [2]_.  The results of a heart\n    disease study are summarized in the following table::\n\n                 High CAT   Low CAT    Total\n                 --------   -------    -----\n        CHD         27         44        71\n        No CHD      95        443       538\n\n        Total      122        487       609\n\n    CHD is coronary heart disease, and CAT refers to the level of\n    circulating catecholamine.  CAT is the "exposure" variable, and\n    high CAT is the "exposed" category. So the data from the table\n    to be passed to ``relative_risk`` is::\n\n        exposed_cases = 27\n        exposed_total = 122\n        control_cases = 44\n        control_total = 487\n\n    >>> result = relative_risk(27, 122, 44, 487)\n    >>> result.relative_risk\n    2.4495156482861398\n\n    Find the confidence interval for the relative risk.\n\n    >>> result.confidence_interval(confidence_level=0.95)\n    ConfidenceInterval(low=1.5836990926700116, high=3.7886786315466354)\n\n    The interval does not contain 1, so the data supports the statement\n    that high CAT is associated with greater risk of CHD.\n    '
    exposed_cases = _validate_int(exposed_cases, 0, 'exposed_cases')
    exposed_total = _validate_int(exposed_total, 1, 'exposed_total')
    control_cases = _validate_int(control_cases, 0, 'control_cases')
    control_total = _validate_int(control_total, 1, 'control_total')
    if exposed_cases > exposed_total:
        raise ValueError('exposed_cases must not exceed exposed_total.')
    if control_cases > control_total:
        raise ValueError('control_cases must not exceed control_total.')
    if exposed_cases == 0 and control_cases == 0:
        rr = np.nan
    elif exposed_cases == 0:
        rr = 0.0
    elif control_cases == 0:
        rr = np.inf
    else:
        p1 = exposed_cases / exposed_total
        p2 = control_cases / control_total
        rr = p1 / p2
    return RelativeRiskResult(relative_risk=rr, exposed_cases=exposed_cases, exposed_total=exposed_total, control_cases=control_cases, control_total=control_total)