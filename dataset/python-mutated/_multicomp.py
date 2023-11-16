from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.stats._common import ConfidenceInterval
from scipy.stats._qmc import check_random_state
from scipy.stats._stats_py import _var
if TYPE_CHECKING:
    import numpy.typing as npt
    from scipy._lib._util import DecimalNumber, SeedType
    from typing import Literal, Sequence
__all__ = ['dunnett']

@dataclass
class DunnettResult:
    """Result object returned by `scipy.stats.dunnett`.

    Attributes
    ----------
    statistic : float ndarray
        The computed statistic of the test for each comparison. The element
        at index ``i`` is the statistic for the comparison between
        groups ``i`` and the control.
    pvalue : float ndarray
        The computed p-value of the test for each comparison. The element
        at index ``i`` is the p-value for the comparison between
        group ``i`` and the control.
    """
    statistic: np.ndarray
    pvalue: np.ndarray
    _alternative: Literal['two-sided', 'less', 'greater'] = field(repr=False)
    _rho: np.ndarray = field(repr=False)
    _df: int = field(repr=False)
    _std: float = field(repr=False)
    _mean_samples: np.ndarray = field(repr=False)
    _mean_control: np.ndarray = field(repr=False)
    _n_samples: np.ndarray = field(repr=False)
    _n_control: int = field(repr=False)
    _rng: SeedType = field(repr=False)
    _ci: ConfidenceInterval | None = field(default=None, repr=False)
    _ci_cl: DecimalNumber | None = field(default=None, repr=False)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        if self._ci is None:
            self.confidence_interval(confidence_level=0.95)
        s = f"Dunnett's test ({self._ci_cl * 100:.1f}% Confidence Interval)\nComparison               Statistic  p-value  Lower CI  Upper CI\n"
        for i in range(self.pvalue.size):
            s += f' (Sample {i} - Control) {self.statistic[i]:>10.3f}{self.pvalue[i]:>10.3f}{self._ci.low[i]:>10.3f}{self._ci.high[i]:>10.3f}\n'
        return s

    def _allowance(self, confidence_level: DecimalNumber=0.95, tol: DecimalNumber=0.001) -> float:
        if False:
            for i in range(10):
                print('nop')
        'Allowance.\n\n        It is the quantity to add/subtract from the observed difference\n        between the means of observed groups and the mean of the control\n        group. The result gives confidence limits.\n\n        Parameters\n        ----------\n        confidence_level : float, optional\n            Confidence level for the computed confidence interval.\n            Default is .95.\n        tol : float, optional\n            A tolerance for numerical optimization: the allowance will produce\n            a confidence within ``10*tol*(1 - confidence_level)`` of the\n            specified level, or a warning will be emitted. Tight tolerances\n            may be impractical due to noisy evaluation of the objective.\n            Default is 1e-3.\n\n        Returns\n        -------\n        allowance : float\n            Allowance around the mean.\n        '
        alpha = 1 - confidence_level

        def pvalue_from_stat(statistic):
            if False:
                for i in range(10):
                    print('nop')
            statistic = np.array(statistic)
            sf = _pvalue_dunnett(rho=self._rho, df=self._df, statistic=statistic, alternative=self._alternative, rng=self._rng)
            return abs(sf - alpha) / alpha
        res = minimize_scalar(pvalue_from_stat, method='brent', tol=tol)
        critical_value = res.x
        if res.success is False or res.fun >= tol * 10:
            warnings.warn(f'Computation of the confidence interval did not converge to the desired level. The confidence level corresponding with the returned interval is approximately {alpha * (1 + res.fun)}.', stacklevel=3)
        allowance = critical_value * self._std * np.sqrt(1 / self._n_samples + 1 / self._n_control)
        return abs(allowance)

    def confidence_interval(self, confidence_level: DecimalNumber=0.95) -> ConfidenceInterval:
        if False:
            print('Hello World!')
        'Compute the confidence interval for the specified confidence level.\n\n        Parameters\n        ----------\n        confidence_level : float, optional\n            Confidence level for the computed confidence interval.\n            Default is .95.\n\n        Returns\n        -------\n        ci : ``ConfidenceInterval`` object\n            The object has attributes ``low`` and ``high`` that hold the\n            lower and upper bounds of the confidence intervals for each\n            comparison. The high and low values are accessible for each\n            comparison at index ``i`` for each group ``i``.\n\n        '
        if self._ci is not None and confidence_level == self._ci_cl:
            return self._ci
        if not 0 < confidence_level < 1:
            raise ValueError('Confidence level must be between 0 and 1.')
        allowance = self._allowance(confidence_level=confidence_level)
        diff_means = self._mean_samples - self._mean_control
        low = diff_means - allowance
        high = diff_means + allowance
        if self._alternative == 'greater':
            high = [np.inf] * len(diff_means)
        elif self._alternative == 'less':
            low = [-np.inf] * len(diff_means)
        self._ci_cl = confidence_level
        self._ci = ConfidenceInterval(low=low, high=high)
        return self._ci

def dunnett(*samples: npt.ArrayLike, control: npt.ArrayLike, alternative: Literal['two-sided', 'less', 'greater']='two-sided', random_state: SeedType=None) -> DunnettResult:
    if False:
        return 10
    'Dunnett\'s test: multiple comparisons of means against a control group.\n\n    This is an implementation of Dunnett\'s original, single-step test as\n    described in [1]_.\n\n    Parameters\n    ----------\n    sample1, sample2, ... : 1D array_like\n        The sample measurements for each experimental group.\n    control : 1D array_like\n        The sample measurements for the control group.\n    alternative : {\'two-sided\', \'less\', \'greater\'}, optional\n        Defines the alternative hypothesis.\n\n        The null hypothesis is that the means of the distributions underlying\n        the samples and control are equal. The following alternative\n        hypotheses are available (default is \'two-sided\'):\n\n        * \'two-sided\': the means of the distributions underlying the samples\n          and control are unequal.\n        * \'less\': the means of the distributions underlying the samples\n          are less than the mean of the distribution underlying the control.\n        * \'greater\': the means of the distributions underlying the\n          samples are greater than the mean of the distribution underlying\n          the control.\n    random_state : {None, int, `numpy.random.Generator`}, optional\n        If `random_state` is an int or None, a new `numpy.random.Generator` is\n        created using ``np.random.default_rng(random_state)``.\n        If `random_state` is already a ``Generator`` instance, then the\n        provided instance is used.\n\n        The random number generator is used to control the randomized\n        Quasi-Monte Carlo integration of the multivariate-t distribution.\n\n    Returns\n    -------\n    res : `~scipy.stats._result_classes.DunnettResult`\n        An object containing attributes:\n\n        statistic : float ndarray\n            The computed statistic of the test for each comparison. The element\n            at index ``i`` is the statistic for the comparison between\n            groups ``i`` and the control.\n        pvalue : float ndarray\n            The computed p-value of the test for each comparison. The element\n            at index ``i`` is the p-value for the comparison between\n            group ``i`` and the control.\n\n        And the following method:\n\n        confidence_interval(confidence_level=0.95) :\n            Compute the difference in means of the groups\n            with the control +- the allowance.\n\n    See Also\n    --------\n    tukey_hsd : performs pairwise comparison of means.\n\n    Notes\n    -----\n    Like the independent-sample t-test, Dunnett\'s test [1]_ is used to make\n    inferences about the means of distributions from which samples were drawn.\n    However, when multiple t-tests are performed at a fixed significance level,\n    the "family-wise error rate" - the probability of incorrectly rejecting the\n    null hypothesis in at least one test - will exceed the significance level.\n    Dunnett\'s test is designed to perform multiple comparisons while\n    controlling the family-wise error rate.\n\n    Dunnett\'s test compares the means of multiple experimental groups\n    against a single control group. Tukey\'s Honestly Significant Difference Test\n    is another multiple-comparison test that controls the family-wise error\n    rate, but `tukey_hsd` performs *all* pairwise comparisons between groups.\n    When pairwise comparisons between experimental groups are not needed,\n    Dunnett\'s test is preferable due to its higher power.\n\n\n    The use of this test relies on several assumptions.\n\n    1. The observations are independent within and among groups.\n    2. The observations within each group are normally distributed.\n    3. The distributions from which the samples are drawn have the same finite\n       variance.\n\n    References\n    ----------\n    .. [1] Charles W. Dunnett. "A Multiple Comparison Procedure for Comparing\n       Several Treatments with a Control."\n       Journal of the American Statistical Association, 50:272, 1096-1121,\n       :doi:`10.1080/01621459.1955.10501294`, 1955.\n\n    Examples\n    --------\n    In [1]_, the influence of drugs on blood count measurements on three groups\n    of animal is investigated.\n\n    The following table summarizes the results of the experiment in which\n    two groups received different drugs, and one group acted as a control.\n    Blood counts (in millions of cells per cubic millimeter) were recorded::\n\n    >>> import numpy as np\n    >>> control = np.array([7.40, 8.50, 7.20, 8.24, 9.84, 8.32])\n    >>> drug_a = np.array([9.76, 8.80, 7.68, 9.36])\n    >>> drug_b = np.array([12.80, 9.68, 12.16, 9.20, 10.55])\n\n    We would like to see if the means between any of the groups are\n    significantly different. First, visually examine a box and whisker plot.\n\n    >>> import matplotlib.pyplot as plt\n    >>> fig, ax = plt.subplots(1, 1)\n    >>> ax.boxplot([control, drug_a, drug_b])\n    >>> ax.set_xticklabels(["Control", "Drug A", "Drug B"])  # doctest: +SKIP\n    >>> ax.set_ylabel("mean")  # doctest: +SKIP\n    >>> plt.show()\n\n    Note the overlapping interquartile ranges of the drug A group and control\n    group and the apparent separation between the drug B group and control\n    group.\n\n    Next, we will use Dunnett\'s test to assess whether the difference\n    between group means is significant while controlling the family-wise error\n    rate: the probability of making any false discoveries.\n    Let the null hypothesis be that the experimental groups have the same\n    mean as the control and the alternative be that an experimental group does\n    not have the same mean as the control. We will consider a 5% family-wise\n    error rate to be acceptable, and therefore we choose 0.05 as the threshold\n    for significance.\n\n    >>> from scipy.stats import dunnett\n    >>> res = dunnett(drug_a, drug_b, control=control)\n    >>> res.pvalue\n    array([0.62004941, 0.0059035 ])  # may vary\n\n    The p-value corresponding with the comparison between group A and control\n    exceeds 0.05, so we do not reject the null hypothesis for that comparison.\n    However, the p-value corresponding with the comparison between group B\n    and control is less than 0.05, so we consider the experimental results\n    to be evidence against the null hypothesis in favor of the alternative:\n    group B has a different mean than the control group.\n\n    '
    (samples_, control_, rng) = _iv_dunnett(samples=samples, control=control, alternative=alternative, random_state=random_state)
    (rho, df, n_group, n_samples, n_control) = _params_dunnett(samples=samples_, control=control_)
    (statistic, std, mean_control, mean_samples) = _statistic_dunnett(samples_, control_, df, n_samples, n_control)
    pvalue = _pvalue_dunnett(rho=rho, df=df, statistic=statistic, alternative=alternative, rng=rng)
    return DunnettResult(statistic=statistic, pvalue=pvalue, _alternative=alternative, _rho=rho, _df=df, _std=std, _mean_samples=mean_samples, _mean_control=mean_control, _n_samples=n_samples, _n_control=n_control, _rng=rng)

def _iv_dunnett(samples: Sequence[npt.ArrayLike], control: npt.ArrayLike, alternative: Literal['two-sided', 'less', 'greater'], random_state: SeedType) -> tuple[list[np.ndarray], np.ndarray, SeedType]:
    if False:
        i = 10
        return i + 15
    "Input validation for Dunnett's test."
    rng = check_random_state(random_state)
    if alternative not in {'two-sided', 'less', 'greater'}:
        raise ValueError("alternative must be 'less', 'greater' or 'two-sided'")
    ndim_msg = 'Control and samples groups must be 1D arrays'
    n_obs_msg = 'Control and samples groups must have at least 1 observation'
    control = np.asarray(control)
    samples_ = [np.asarray(sample) for sample in samples]
    samples_control: list[np.ndarray] = samples_ + [control]
    for sample in samples_control:
        if sample.ndim > 1:
            raise ValueError(ndim_msg)
        if sample.size < 1:
            raise ValueError(n_obs_msg)
    return (samples_, control, rng)

def _params_dunnett(samples: list[np.ndarray], control: np.ndarray) -> tuple[np.ndarray, int, int, np.ndarray, int]:
    if False:
        print('Hello World!')
    "Specific parameters for Dunnett's test.\n\n    Degree of freedom is the number of observations minus the number of groups\n    including the control.\n    "
    n_samples = np.array([sample.size for sample in samples])
    n_sample = n_samples.sum()
    n_control = control.size
    n = n_sample + n_control
    n_groups = len(samples)
    df = n - n_groups - 1
    rho = n_control / n_samples + 1
    rho = 1 / np.sqrt(rho[:, None] * rho[None, :])
    np.fill_diagonal(rho, 1)
    return (rho, df, n_groups, n_samples, n_control)

def _statistic_dunnett(samples: list[np.ndarray], control: np.ndarray, df: int, n_samples: np.ndarray, n_control: int) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    if False:
        print('Hello World!')
    "Statistic of Dunnett's test.\n\n    Computation based on the original single-step test from [1].\n    "
    mean_control = np.mean(control)
    mean_samples = np.array([np.mean(sample) for sample in samples])
    all_samples = [control] + samples
    all_means = np.concatenate([[mean_control], mean_samples])
    s2 = np.sum([_var(sample, mean=mean) * sample.size for (sample, mean) in zip(all_samples, all_means)]) / df
    std = np.sqrt(s2)
    z = (mean_samples - mean_control) / np.sqrt(1 / n_samples + 1 / n_control)
    return (z / std, std, mean_control, mean_samples)

def _pvalue_dunnett(rho: np.ndarray, df: int, statistic: np.ndarray, alternative: Literal['two-sided', 'less', 'greater'], rng: SeedType=None) -> np.ndarray:
    if False:
        return 10
    'pvalue from the multivariate t-distribution.\n\n    Critical values come from the multivariate student-t distribution.\n    '
    statistic = statistic.reshape(-1, 1)
    mvt = stats.multivariate_t(shape=rho, df=df, seed=rng)
    if alternative == 'two-sided':
        statistic = abs(statistic)
        pvalue = 1 - mvt.cdf(statistic, lower_limit=-statistic)
    elif alternative == 'greater':
        pvalue = 1 - mvt.cdf(statistic, lower_limit=-np.inf)
    else:
        pvalue = 1 - mvt.cdf(np.inf, lower_limit=statistic)
    return np.atleast_1d(pvalue)