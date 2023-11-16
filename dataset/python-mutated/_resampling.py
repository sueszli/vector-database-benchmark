from __future__ import annotations
import warnings
import numpy as np
from itertools import combinations, permutations, product
from collections.abc import Sequence
import inspect
from scipy._lib._util import check_random_state, _rename_parameter
from scipy.special import ndtr, ndtri, comb, factorial
from scipy._lib._util import rng_integers
from dataclasses import dataclass
from ._common import ConfidenceInterval
from ._axis_nan_policy import _broadcast_concatenate, _broadcast_arrays
from ._warnings_errors import DegenerateDataWarning
__all__ = ['bootstrap', 'monte_carlo_test', 'permutation_test']

def _vectorize_statistic(statistic):
    if False:
        print('Hello World!')
    'Vectorize an n-sample statistic'

    def stat_nd(*data, axis=0):
        if False:
            return 10
        lengths = [sample.shape[axis] for sample in data]
        split_indices = np.cumsum(lengths)[:-1]
        z = _broadcast_concatenate(data, axis)
        z = np.moveaxis(z, axis, 0)

        def stat_1d(z):
            if False:
                while True:
                    i = 10
            data = np.split(z, split_indices)
            return statistic(*data)
        return np.apply_along_axis(stat_1d, 0, z)[()]
    return stat_nd

def _jackknife_resample(sample, batch=None):
    if False:
        for i in range(10):
            print('nop')
    'Jackknife resample the sample. Only one-sample stats for now.'
    n = sample.shape[-1]
    batch_nominal = batch or n
    for k in range(0, n, batch_nominal):
        batch_actual = min(batch_nominal, n - k)
        j = np.ones((batch_actual, n), dtype=bool)
        np.fill_diagonal(j[:, k:k + batch_actual], False)
        i = np.arange(n)
        i = np.broadcast_to(i, (batch_actual, n))
        i = i[j].reshape((batch_actual, n - 1))
        resamples = sample[..., i]
        yield resamples

def _bootstrap_resample(sample, n_resamples=None, random_state=None):
    if False:
        return 10
    'Bootstrap resample the sample.'
    n = sample.shape[-1]
    i = rng_integers(random_state, 0, n, (n_resamples, n))
    resamples = sample[..., i]
    return resamples

def _percentile_of_score(a, score, axis):
    if False:
        i = 10
        return i + 15
    "Vectorized, simplified `scipy.stats.percentileofscore`.\n    Uses logic of the 'mean' value of percentileofscore's kind parameter.\n\n    Unlike `stats.percentileofscore`, the percentile returned is a fraction\n    in [0, 1].\n    "
    B = a.shape[axis]
    return ((a < score).sum(axis=axis) + (a <= score).sum(axis=axis)) / (2 * B)

def _percentile_along_axis(theta_hat_b, alpha):
    if False:
        return 10
    '`np.percentile` with different percentile for each slice.'
    shape = theta_hat_b.shape[:-1]
    alpha = np.broadcast_to(alpha, shape)
    percentiles = np.zeros_like(alpha, dtype=np.float64)
    for (indices, alpha_i) in np.ndenumerate(alpha):
        if np.isnan(alpha_i):
            msg = 'The BCa confidence interval cannot be calculated. This problem is known to occur when the distribution is degenerate or the statistic is np.min.'
            warnings.warn(DegenerateDataWarning(msg))
            percentiles[indices] = np.nan
        else:
            theta_hat_b_i = theta_hat_b[indices]
            percentiles[indices] = np.percentile(theta_hat_b_i, alpha_i)
    return percentiles[()]

def _bca_interval(data, statistic, axis, alpha, theta_hat_b, batch):
    if False:
        print('Hello World!')
    'Bias-corrected and accelerated interval.'
    theta_hat = np.asarray(statistic(*data, axis=axis))[..., None]
    percentile = _percentile_of_score(theta_hat_b, theta_hat, axis=-1)
    z0_hat = ndtri(percentile)
    theta_hat_ji = []
    for (j, sample) in enumerate(data):
        samples = [np.expand_dims(sample, -2) for sample in data]
        theta_hat_i = []
        for jackknife_sample in _jackknife_resample(sample, batch):
            samples[j] = jackknife_sample
            broadcasted = _broadcast_arrays(samples, axis=-1)
            theta_hat_i.append(statistic(*broadcasted, axis=-1))
        theta_hat_ji.append(theta_hat_i)
    theta_hat_ji = [np.concatenate(theta_hat_i, axis=-1) for theta_hat_i in theta_hat_ji]
    n_j = [theta_hat_i.shape[-1] for theta_hat_i in theta_hat_ji]
    theta_hat_j_dot = [theta_hat_i.mean(axis=-1, keepdims=True) for theta_hat_i in theta_hat_ji]
    U_ji = [(n - 1) * (theta_hat_dot - theta_hat_i) for (theta_hat_dot, theta_hat_i, n) in zip(theta_hat_j_dot, theta_hat_ji, n_j)]
    nums = [(U_i ** 3).sum(axis=-1) / n ** 3 for (U_i, n) in zip(U_ji, n_j)]
    dens = [(U_i ** 2).sum(axis=-1) / n ** 2 for (U_i, n) in zip(U_ji, n_j)]
    a_hat = 1 / 6 * sum(nums) / sum(dens) ** (3 / 2)
    z_alpha = ndtri(alpha)
    z_1alpha = -z_alpha
    num1 = z0_hat + z_alpha
    alpha_1 = ndtr(z0_hat + num1 / (1 - a_hat * num1))
    num2 = z0_hat + z_1alpha
    alpha_2 = ndtr(z0_hat + num2 / (1 - a_hat * num2))
    return (alpha_1, alpha_2, a_hat)

def _bootstrap_iv(data, statistic, vectorized, paired, axis, confidence_level, alternative, n_resamples, batch, method, bootstrap_result, random_state):
    if False:
        print('Hello World!')
    'Input validation and standardization for `bootstrap`.'
    if vectorized not in {True, False, None}:
        raise ValueError('`vectorized` must be `True`, `False`, or `None`.')
    if vectorized is None:
        vectorized = 'axis' in inspect.signature(statistic).parameters
    if not vectorized:
        statistic = _vectorize_statistic(statistic)
    axis_int = int(axis)
    if axis != axis_int:
        raise ValueError('`axis` must be an integer.')
    n_samples = 0
    try:
        n_samples = len(data)
    except TypeError:
        raise ValueError('`data` must be a sequence of samples.')
    if n_samples == 0:
        raise ValueError('`data` must contain at least one sample.')
    data_iv = []
    for sample in data:
        sample = np.atleast_1d(sample)
        if sample.shape[axis_int] <= 1:
            raise ValueError('each sample in `data` must contain two or more observations along `axis`.')
        sample = np.moveaxis(sample, axis_int, -1)
        data_iv.append(sample)
    if paired not in {True, False}:
        raise ValueError('`paired` must be `True` or `False`.')
    if paired:
        n = data_iv[0].shape[-1]
        for sample in data_iv[1:]:
            if sample.shape[-1] != n:
                message = 'When `paired is True`, all samples must have the same length along `axis`'
                raise ValueError(message)

        def statistic(i, axis=-1, data=data_iv, unpaired_statistic=statistic):
            if False:
                for i in range(10):
                    print('nop')
            data = [sample[..., i] for sample in data]
            return unpaired_statistic(*data, axis=axis)
        data_iv = [np.arange(n)]
    confidence_level_float = float(confidence_level)
    alternative = alternative.lower()
    alternatives = {'two-sided', 'less', 'greater'}
    if alternative not in alternatives:
        raise ValueError(f'`alternative` must be one of {alternatives}')
    n_resamples_int = int(n_resamples)
    if n_resamples != n_resamples_int or n_resamples_int < 0:
        raise ValueError('`n_resamples` must be a non-negative integer.')
    if batch is None:
        batch_iv = batch
    else:
        batch_iv = int(batch)
        if batch != batch_iv or batch_iv <= 0:
            raise ValueError('`batch` must be a positive integer or None.')
    methods = {'percentile', 'basic', 'bca'}
    method = method.lower()
    if method not in methods:
        raise ValueError(f'`method` must be in {methods}')
    message = "`bootstrap_result` must have attribute `bootstrap_distribution'"
    if bootstrap_result is not None and (not hasattr(bootstrap_result, 'bootstrap_distribution')):
        raise ValueError(message)
    message = 'Either `bootstrap_result.bootstrap_distribution.size` or `n_resamples` must be positive.'
    if (not bootstrap_result or not bootstrap_result.bootstrap_distribution.size) and n_resamples_int == 0:
        raise ValueError(message)
    random_state = check_random_state(random_state)
    return (data_iv, statistic, vectorized, paired, axis_int, confidence_level_float, alternative, n_resamples_int, batch_iv, method, bootstrap_result, random_state)

@dataclass
class BootstrapResult:
    """Result object returned by `scipy.stats.bootstrap`.

    Attributes
    ----------
    confidence_interval : ConfidenceInterval
        The bootstrap confidence interval as an instance of
        `collections.namedtuple` with attributes `low` and `high`.
    bootstrap_distribution : ndarray
        The bootstrap distribution, that is, the value of `statistic` for
        each resample. The last dimension corresponds with the resamples
        (e.g. ``res.bootstrap_distribution.shape[-1] == n_resamples``).
    standard_error : float or ndarray
        The bootstrap standard error, that is, the sample standard
        deviation of the bootstrap distribution.

    """
    confidence_interval: ConfidenceInterval
    bootstrap_distribution: np.ndarray
    standard_error: float | np.ndarray

def bootstrap(data, statistic, *, n_resamples=9999, batch=None, vectorized=None, paired=False, axis=0, confidence_level=0.95, alternative='two-sided', method='BCa', bootstrap_result=None, random_state=None):
    if False:
        while True:
            i = 10
    '\n    Compute a two-sided bootstrap confidence interval of a statistic.\n\n    When `method` is ``\'percentile\'`` and `alternative` is ``\'two-sided\'``,\n    a bootstrap confidence interval is computed according to the following\n    procedure.\n\n    1. Resample the data: for each sample in `data` and for each of\n       `n_resamples`, take a random sample of the original sample\n       (with replacement) of the same size as the original sample.\n\n    2. Compute the bootstrap distribution of the statistic: for each set of\n       resamples, compute the test statistic.\n\n    3. Determine the confidence interval: find the interval of the bootstrap\n       distribution that is\n\n       - symmetric about the median and\n       - contains `confidence_level` of the resampled statistic values.\n\n    While the ``\'percentile\'`` method is the most intuitive, it is rarely\n    used in practice. Two more common methods are available, ``\'basic\'``\n    (\'reverse percentile\') and ``\'BCa\'`` (\'bias-corrected and accelerated\');\n    they differ in how step 3 is performed.\n\n    If the samples in `data` are  taken at random from their respective\n    distributions :math:`n` times, the confidence interval returned by\n    `bootstrap` will contain the true value of the statistic for those\n    distributions approximately `confidence_level`:math:`\\, \\times \\, n` times.\n\n    Parameters\n    ----------\n    data : sequence of array-like\n         Each element of data is a sample from an underlying distribution.\n    statistic : callable\n        Statistic for which the confidence interval is to be calculated.\n        `statistic` must be a callable that accepts ``len(data)`` samples\n        as separate arguments and returns the resulting statistic.\n        If `vectorized` is set ``True``,\n        `statistic` must also accept a keyword argument `axis` and be\n        vectorized to compute the statistic along the provided `axis`.\n    n_resamples : int, default: ``9999``\n        The number of resamples performed to form the bootstrap distribution\n        of the statistic.\n    batch : int, optional\n        The number of resamples to process in each vectorized call to\n        `statistic`. Memory usage is O( `batch` * ``n`` ), where ``n`` is the\n        sample size. Default is ``None``, in which case ``batch = n_resamples``\n        (or ``batch = max(n_resamples, n)`` for ``method=\'BCa\'``).\n    vectorized : bool, optional\n        If `vectorized` is set ``False``, `statistic` will not be passed\n        keyword argument `axis` and is expected to calculate the statistic\n        only for 1D samples. If ``True``, `statistic` will be passed keyword\n        argument `axis` and is expected to calculate the statistic along `axis`\n        when passed an ND sample array. If ``None`` (default), `vectorized`\n        will be set ``True`` if ``axis`` is a parameter of `statistic`. Use of\n        a vectorized statistic typically reduces computation time.\n    paired : bool, default: ``False``\n        Whether the statistic treats corresponding elements of the samples\n        in `data` as paired.\n    axis : int, default: ``0``\n        The axis of the samples in `data` along which the `statistic` is\n        calculated.\n    confidence_level : float, default: ``0.95``\n        The confidence level of the confidence interval.\n    alternative : {\'two-sided\', \'less\', \'greater\'}, default: ``\'two-sided\'``\n        Choose ``\'two-sided\'`` (default) for a two-sided confidence interval,\n        ``\'less\'`` for a one-sided confidence interval with the lower bound\n        at ``-np.inf``, and ``\'greater\'`` for a one-sided confidence interval\n        with the upper bound at ``np.inf``. The other bound of the one-sided\n        confidence intervals is the same as that of a two-sided confidence\n        interval with `confidence_level` twice as far from 1.0; e.g. the upper\n        bound of a 95% ``\'less\'``  confidence interval is the same as the upper\n        bound of a 90% ``\'two-sided\'`` confidence interval.\n    method : {\'percentile\', \'basic\', \'bca\'}, default: ``\'BCa\'``\n        Whether to return the \'percentile\' bootstrap confidence interval\n        (``\'percentile\'``), the \'basic\' (AKA \'reverse\') bootstrap confidence\n        interval (``\'basic\'``), or the bias-corrected and accelerated bootstrap\n        confidence interval (``\'BCa\'``).\n    bootstrap_result : BootstrapResult, optional\n        Provide the result object returned by a previous call to `bootstrap`\n        to include the previous bootstrap distribution in the new bootstrap\n        distribution. This can be used, for example, to change\n        `confidence_level`, change `method`, or see the effect of performing\n        additional resampling without repeating computations.\n    random_state : {None, int, `numpy.random.Generator`,\n                    `numpy.random.RandomState`}, optional\n\n        Pseudorandom number generator state used to generate resamples.\n\n        If `random_state` is ``None`` (or `np.random`), the\n        `numpy.random.RandomState` singleton is used.\n        If `random_state` is an int, a new ``RandomState`` instance is used,\n        seeded with `random_state`.\n        If `random_state` is already a ``Generator`` or ``RandomState``\n        instance then that instance is used.\n\n    Returns\n    -------\n    res : BootstrapResult\n        An object with attributes:\n\n        confidence_interval : ConfidenceInterval\n            The bootstrap confidence interval as an instance of\n            `collections.namedtuple` with attributes `low` and `high`.\n        bootstrap_distribution : ndarray\n            The bootstrap distribution, that is, the value of `statistic` for\n            each resample. The last dimension corresponds with the resamples\n            (e.g. ``res.bootstrap_distribution.shape[-1] == n_resamples``).\n        standard_error : float or ndarray\n            The bootstrap standard error, that is, the sample standard\n            deviation of the bootstrap distribution.\n\n    Warns\n    -----\n    `~scipy.stats.DegenerateDataWarning`\n        Generated when ``method=\'BCa\'`` and the bootstrap distribution is\n        degenerate (e.g. all elements are identical).\n\n    Notes\n    -----\n    Elements of the confidence interval may be NaN for ``method=\'BCa\'`` if\n    the bootstrap distribution is degenerate (e.g. all elements are identical).\n    In this case, consider using another `method` or inspecting `data` for\n    indications that other analysis may be more appropriate (e.g. all\n    observations are identical).\n\n    References\n    ----------\n    .. [1] B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap,\n       Chapman & Hall/CRC, Boca Raton, FL, USA (1993)\n    .. [2] Nathaniel E. Helwig, "Bootstrap Confidence Intervals",\n       http://users.stat.umn.edu/~helwig/notes/bootci-Notes.pdf\n    .. [3] Bootstrapping (statistics), Wikipedia,\n       https://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29\n\n    Examples\n    --------\n    Suppose we have sampled data from an unknown distribution.\n\n    >>> import numpy as np\n    >>> rng = np.random.default_rng()\n    >>> from scipy.stats import norm\n    >>> dist = norm(loc=2, scale=4)  # our "unknown" distribution\n    >>> data = dist.rvs(size=100, random_state=rng)\n\n    We are interested in the standard deviation of the distribution.\n\n    >>> std_true = dist.std()      # the true value of the statistic\n    >>> print(std_true)\n    4.0\n    >>> std_sample = np.std(data)  # the sample statistic\n    >>> print(std_sample)\n    3.9460644295563863\n\n    The bootstrap is used to approximate the variability we would expect if we\n    were to repeatedly sample from the unknown distribution and calculate the\n    statistic of the sample each time. It does this by repeatedly resampling\n    values *from the original sample* with replacement and calculating the\n    statistic of each resample. This results in a "bootstrap distribution" of\n    the statistic.\n\n    >>> import matplotlib.pyplot as plt\n    >>> from scipy.stats import bootstrap\n    >>> data = (data,)  # samples must be in a sequence\n    >>> res = bootstrap(data, np.std, confidence_level=0.9,\n    ...                 random_state=rng)\n    >>> fig, ax = plt.subplots()\n    >>> ax.hist(res.bootstrap_distribution, bins=25)\n    >>> ax.set_title(\'Bootstrap Distribution\')\n    >>> ax.set_xlabel(\'statistic value\')\n    >>> ax.set_ylabel(\'frequency\')\n    >>> plt.show()\n\n    The standard error quantifies this variability. It is calculated as the\n    standard deviation of the bootstrap distribution.\n\n    >>> res.standard_error\n    0.24427002125829136\n    >>> res.standard_error == np.std(res.bootstrap_distribution, ddof=1)\n    True\n\n    The bootstrap distribution of the statistic is often approximately normal\n    with scale equal to the standard error.\n\n    >>> x = np.linspace(3, 5)\n    >>> pdf = norm.pdf(x, loc=std_sample, scale=res.standard_error)\n    >>> fig, ax = plt.subplots()\n    >>> ax.hist(res.bootstrap_distribution, bins=25, density=True)\n    >>> ax.plot(x, pdf)\n    >>> ax.set_title(\'Normal Approximation of the Bootstrap Distribution\')\n    >>> ax.set_xlabel(\'statistic value\')\n    >>> ax.set_ylabel(\'pdf\')\n    >>> plt.show()\n\n    This suggests that we could construct a 90% confidence interval on the\n    statistic based on quantiles of this normal distribution.\n\n    >>> norm.interval(0.9, loc=std_sample, scale=res.standard_error)\n    (3.5442759991341726, 4.3478528599786)\n\n    Due to central limit theorem, this normal approximation is accurate for a\n    variety of statistics and distributions underlying the samples; however,\n    the approximation is not reliable in all cases. Because `bootstrap` is\n    designed to work with arbitrary underlying distributions and statistics,\n    it uses more advanced techniques to generate an accurate confidence\n    interval.\n\n    >>> print(res.confidence_interval)\n    ConfidenceInterval(low=3.57655333533867, high=4.382043696342881)\n\n    If we sample from the original distribution 1000 times and form a bootstrap\n    confidence interval for each sample, the confidence interval\n    contains the true value of the statistic approximately 90% of the time.\n\n    >>> n_trials = 1000\n    >>> ci_contains_true_std = 0\n    >>> for i in range(n_trials):\n    ...    data = (dist.rvs(size=100, random_state=rng),)\n    ...    ci = bootstrap(data, np.std, confidence_level=0.9, n_resamples=1000,\n    ...                   random_state=rng).confidence_interval\n    ...    if ci[0] < std_true < ci[1]:\n    ...        ci_contains_true_std += 1\n    >>> print(ci_contains_true_std)\n    875\n\n    Rather than writing a loop, we can also determine the confidence intervals\n    for all 1000 samples at once.\n\n    >>> data = (dist.rvs(size=(n_trials, 100), random_state=rng),)\n    >>> res = bootstrap(data, np.std, axis=-1, confidence_level=0.9,\n    ...                 n_resamples=1000, random_state=rng)\n    >>> ci_l, ci_u = res.confidence_interval\n\n    Here, `ci_l` and `ci_u` contain the confidence interval for each of the\n    ``n_trials = 1000`` samples.\n\n    >>> print(ci_l[995:])\n    [3.77729695 3.75090233 3.45829131 3.34078217 3.48072829]\n    >>> print(ci_u[995:])\n    [4.88316666 4.86924034 4.32032996 4.2822427  4.59360598]\n\n    And again, approximately 90% contain the true value, ``std_true = 4``.\n\n    >>> print(np.sum((ci_l < std_true) & (std_true < ci_u)))\n    900\n\n    `bootstrap` can also be used to estimate confidence intervals of\n    multi-sample statistics, including those calculated by hypothesis\n    tests. `scipy.stats.mood` perform\'s Mood\'s test for equal scale parameters,\n    and it returns two outputs: a statistic, and a p-value. To get a\n    confidence interval for the test statistic, we first wrap\n    `scipy.stats.mood` in a function that accepts two sample arguments,\n    accepts an `axis` keyword argument, and returns only the statistic.\n\n    >>> from scipy.stats import mood\n    >>> def my_statistic(sample1, sample2, axis):\n    ...     statistic, _ = mood(sample1, sample2, axis=-1)\n    ...     return statistic\n\n    Here, we use the \'percentile\' method with the default 95% confidence level.\n\n    >>> sample1 = norm.rvs(scale=1, size=100, random_state=rng)\n    >>> sample2 = norm.rvs(scale=2, size=100, random_state=rng)\n    >>> data = (sample1, sample2)\n    >>> res = bootstrap(data, my_statistic, method=\'basic\', random_state=rng)\n    >>> print(mood(sample1, sample2)[0])  # element 0 is the statistic\n    -5.521109549096542\n    >>> print(res.confidence_interval)\n    ConfidenceInterval(low=-7.255994487314675, high=-4.016202624747605)\n\n    The bootstrap estimate of the standard error is also available.\n\n    >>> print(res.standard_error)\n    0.8344963846318795\n\n    Paired-sample statistics work, too. For example, consider the Pearson\n    correlation coefficient.\n\n    >>> from scipy.stats import pearsonr\n    >>> n = 100\n    >>> x = np.linspace(0, 10, n)\n    >>> y = x + rng.uniform(size=n)\n    >>> print(pearsonr(x, y)[0])  # element 0 is the statistic\n    0.9962357936065914\n\n    We wrap `pearsonr` so that it returns only the statistic.\n\n    >>> def my_statistic(x, y):\n    ...     return pearsonr(x, y)[0]\n\n    We call `bootstrap` using ``paired=True``.\n    Also, since ``my_statistic`` isn\'t vectorized to calculate the statistic\n    along a given axis, we pass in ``vectorized=False``.\n\n    >>> res = bootstrap((x, y), my_statistic, vectorized=False, paired=True,\n    ...                 random_state=rng)\n    >>> print(res.confidence_interval)\n    ConfidenceInterval(low=0.9950085825848624, high=0.9971212407917498)\n\n    The result object can be passed back into `bootstrap` to perform additional\n    resampling:\n\n    >>> len(res.bootstrap_distribution)\n    9999\n    >>> res = bootstrap((x, y), my_statistic, vectorized=False, paired=True,\n    ...                 n_resamples=1001, random_state=rng,\n    ...                 bootstrap_result=res)\n    >>> len(res.bootstrap_distribution)\n    11000\n\n    or to change the confidence interval options:\n\n    >>> res2 = bootstrap((x, y), my_statistic, vectorized=False, paired=True,\n    ...                  n_resamples=0, random_state=rng, bootstrap_result=res,\n    ...                  method=\'percentile\', confidence_level=0.9)\n    >>> np.testing.assert_equal(res2.bootstrap_distribution,\n    ...                         res.bootstrap_distribution)\n    >>> res.confidence_interval\n    ConfidenceInterval(low=0.9950035351407804, high=0.9971170323404578)\n\n    without repeating computation of the original bootstrap distribution.\n\n    '
    args = _bootstrap_iv(data, statistic, vectorized, paired, axis, confidence_level, alternative, n_resamples, batch, method, bootstrap_result, random_state)
    (data, statistic, vectorized, paired, axis, confidence_level, alternative, n_resamples, batch, method, bootstrap_result, random_state) = args
    theta_hat_b = [] if bootstrap_result is None else [bootstrap_result.bootstrap_distribution]
    batch_nominal = batch or n_resamples or 1
    for k in range(0, n_resamples, batch_nominal):
        batch_actual = min(batch_nominal, n_resamples - k)
        resampled_data = []
        for sample in data:
            resample = _bootstrap_resample(sample, n_resamples=batch_actual, random_state=random_state)
            resampled_data.append(resample)
        theta_hat_b.append(statistic(*resampled_data, axis=-1))
    theta_hat_b = np.concatenate(theta_hat_b, axis=-1)
    alpha = (1 - confidence_level) / 2 if alternative == 'two-sided' else 1 - confidence_level
    if method == 'bca':
        interval = _bca_interval(data, statistic, axis=-1, alpha=alpha, theta_hat_b=theta_hat_b, batch=batch)[:2]
        percentile_fun = _percentile_along_axis
    else:
        interval = (alpha, 1 - alpha)

        def percentile_fun(a, q):
            if False:
                i = 10
                return i + 15
            return np.percentile(a=a, q=q, axis=-1)
    ci_l = percentile_fun(theta_hat_b, interval[0] * 100)
    ci_u = percentile_fun(theta_hat_b, interval[1] * 100)
    if method == 'basic':
        theta_hat = statistic(*data, axis=-1)
        (ci_l, ci_u) = (2 * theta_hat - ci_u, 2 * theta_hat - ci_l)
    if alternative == 'less':
        ci_l = np.full_like(ci_l, -np.inf)
    elif alternative == 'greater':
        ci_u = np.full_like(ci_u, np.inf)
    return BootstrapResult(confidence_interval=ConfidenceInterval(ci_l, ci_u), bootstrap_distribution=theta_hat_b, standard_error=np.std(theta_hat_b, ddof=1, axis=-1))

def _monte_carlo_test_iv(data, rvs, statistic, vectorized, n_resamples, batch, alternative, axis):
    if False:
        return 10
    'Input validation for `monte_carlo_test`.'
    axis_int = int(axis)
    if axis != axis_int:
        raise ValueError('`axis` must be an integer.')
    if vectorized not in {True, False, None}:
        raise ValueError('`vectorized` must be `True`, `False`, or `None`.')
    if not isinstance(rvs, Sequence):
        rvs = (rvs,)
        data = (data,)
    for rvs_i in rvs:
        if not callable(rvs_i):
            raise TypeError('`rvs` must be callable or sequence of callables.')
    if not len(rvs) == len(data):
        message = 'If `rvs` is a sequence, `len(rvs)` must equal `len(data)`.'
        raise ValueError(message)
    if not callable(statistic):
        raise TypeError('`statistic` must be callable.')
    if vectorized is None:
        vectorized = 'axis' in inspect.signature(statistic).parameters
    if not vectorized:
        statistic_vectorized = _vectorize_statistic(statistic)
    else:
        statistic_vectorized = statistic
    data = _broadcast_arrays(data, axis)
    data_iv = []
    for sample in data:
        sample = np.atleast_1d(sample)
        sample = np.moveaxis(sample, axis_int, -1)
        data_iv.append(sample)
    n_resamples_int = int(n_resamples)
    if n_resamples != n_resamples_int or n_resamples_int <= 0:
        raise ValueError('`n_resamples` must be a positive integer.')
    if batch is None:
        batch_iv = batch
    else:
        batch_iv = int(batch)
        if batch != batch_iv or batch_iv <= 0:
            raise ValueError('`batch` must be a positive integer or None.')
    alternatives = {'two-sided', 'greater', 'less'}
    alternative = alternative.lower()
    if alternative not in alternatives:
        raise ValueError(f'`alternative` must be in {alternatives}')
    return (data_iv, rvs, statistic_vectorized, vectorized, n_resamples_int, batch_iv, alternative, axis_int)

@dataclass
class MonteCarloTestResult:
    """Result object returned by `scipy.stats.monte_carlo_test`.

    Attributes
    ----------
    statistic : float or ndarray
        The observed test statistic of the sample.
    pvalue : float or ndarray
        The p-value for the given alternative.
    null_distribution : ndarray
        The values of the test statistic generated under the null
        hypothesis.
    """
    statistic: float | np.ndarray
    pvalue: float | np.ndarray
    null_distribution: np.ndarray

@_rename_parameter('sample', 'data')
def monte_carlo_test(data, rvs, statistic, *, vectorized=None, n_resamples=9999, batch=None, alternative='two-sided', axis=0):
    if False:
        for i in range(10):
            print('nop')
    'Perform a Monte Carlo hypothesis test.\n\n    `data` contains a sample or a sequence of one or more samples. `rvs`\n    specifies the distribution(s) of the sample(s) in `data` under the null\n    hypothesis. The value of `statistic` for the given `data` is compared\n    against a Monte Carlo null distribution: the value of the statistic for\n    each of `n_resamples` sets of samples generated using `rvs`. This gives\n    the p-value, the probability of observing such an extreme value of the\n    test statistic under the null hypothesis.\n\n    Parameters\n    ----------\n    data : array-like or sequence of array-like\n        An array or sequence of arrays of observations.\n    rvs : callable or tuple of callables\n        A callable or sequence of callables that generates random variates\n        under the null hypothesis. Each element of `rvs` must be a callable\n        that accepts keyword argument ``size`` (e.g. ``rvs(size=(m, n))``) and\n        returns an N-d array sample of that shape. If `rvs` is a sequence, the\n        number of callables in `rvs` must match the number of samples in\n        `data`, i.e. ``len(rvs) == len(data)``. If `rvs` is a single callable,\n        `data` is treated as a single sample.\n    statistic : callable\n        Statistic for which the p-value of the hypothesis test is to be\n        calculated. `statistic` must be a callable that accepts a sample\n        (e.g. ``statistic(sample)``) or ``len(rvs)`` separate samples (e.g.\n        ``statistic(samples1, sample2)`` if `rvs` contains two callables and\n        `data` contains two samples) and returns the resulting statistic.\n        If `vectorized` is set ``True``, `statistic` must also accept a keyword\n        argument `axis` and be vectorized to compute the statistic along the\n        provided `axis` of the samples in `data`.\n    vectorized : bool, optional\n        If `vectorized` is set ``False``, `statistic` will not be passed\n        keyword argument `axis` and is expected to calculate the statistic\n        only for 1D samples. If ``True``, `statistic` will be passed keyword\n        argument `axis` and is expected to calculate the statistic along `axis`\n        when passed ND sample arrays. If ``None`` (default), `vectorized`\n        will be set ``True`` if ``axis`` is a parameter of `statistic`. Use of\n        a vectorized statistic typically reduces computation time.\n    n_resamples : int, default: 9999\n        Number of samples drawn from each of the callables of `rvs`.\n        Equivalently, the number statistic values under the null hypothesis\n        used as the Monte Carlo null distribution.\n    batch : int, optional\n        The number of Monte Carlo samples to process in each call to\n        `statistic`. Memory usage is O( `batch` * ``sample.size[axis]`` ). Default\n        is ``None``, in which case `batch` equals `n_resamples`.\n    alternative : {\'two-sided\', \'less\', \'greater\'}\n        The alternative hypothesis for which the p-value is calculated.\n        For each alternative, the p-value is defined as follows.\n\n        - ``\'greater\'`` : the percentage of the null distribution that is\n          greater than or equal to the observed value of the test statistic.\n        - ``\'less\'`` : the percentage of the null distribution that is\n          less than or equal to the observed value of the test statistic.\n        - ``\'two-sided\'`` : twice the smaller of the p-values above.\n\n    axis : int, default: 0\n        The axis of `data` (or each sample within `data`) over which to\n        calculate the statistic.\n\n    Returns\n    -------\n    res : MonteCarloTestResult\n        An object with attributes:\n\n        statistic : float or ndarray\n            The test statistic of the observed `data`.\n        pvalue : float or ndarray\n            The p-value for the given alternative.\n        null_distribution : ndarray\n            The values of the test statistic generated under the null\n            hypothesis.\n\n    References\n    ----------\n\n    .. [1] B. Phipson and G. K. Smyth. "Permutation P-values Should Never Be\n       Zero: Calculating Exact P-values When Permutations Are Randomly Drawn."\n       Statistical Applications in Genetics and Molecular Biology 9.1 (2010).\n\n    Examples\n    --------\n\n    Suppose we wish to test whether a small sample has been drawn from a normal\n    distribution. We decide that we will use the skew of the sample as a\n    test statistic, and we will consider a p-value of 0.05 to be statistically\n    significant.\n\n    >>> import numpy as np\n    >>> from scipy import stats\n    >>> def statistic(x, axis):\n    ...     return stats.skew(x, axis)\n\n    After collecting our data, we calculate the observed value of the test\n    statistic.\n\n    >>> rng = np.random.default_rng()\n    >>> x = stats.skewnorm.rvs(a=1, size=50, random_state=rng)\n    >>> statistic(x, axis=0)\n    0.12457412450240658\n\n    To determine the probability of observing such an extreme value of the\n    skewness by chance if the sample were drawn from the normal distribution,\n    we can perform a Monte Carlo hypothesis test. The test will draw many\n    samples at random from their normal distribution, calculate the skewness\n    of each sample, and compare our original skewness against this\n    distribution to determine an approximate p-value.\n\n    >>> from scipy.stats import monte_carlo_test\n    >>> # because our statistic is vectorized, we pass `vectorized=True`\n    >>> rvs = lambda size: stats.norm.rvs(size=size, random_state=rng)\n    >>> res = monte_carlo_test(x, rvs, statistic, vectorized=True)\n    >>> print(res.statistic)\n    0.12457412450240658\n    >>> print(res.pvalue)\n    0.7012\n\n    The probability of obtaining a test statistic less than or equal to the\n    observed value under the null hypothesis is ~70%. This is greater than\n    our chosen threshold of 5%, so we cannot consider this to be significant\n    evidence against the null hypothesis.\n\n    Note that this p-value essentially matches that of\n    `scipy.stats.skewtest`, which relies on an asymptotic distribution of a\n    test statistic based on the sample skewness.\n\n    >>> stats.skewtest(x).pvalue\n    0.6892046027110614\n\n    This asymptotic approximation is not valid for small sample sizes, but\n    `monte_carlo_test` can be used with samples of any size.\n\n    >>> x = stats.skewnorm.rvs(a=1, size=7, random_state=rng)\n    >>> # stats.skewtest(x) would produce an error due to small sample\n    >>> res = monte_carlo_test(x, rvs, statistic, vectorized=True)\n\n    The Monte Carlo distribution of the test statistic is provided for\n    further investigation.\n\n    >>> import matplotlib.pyplot as plt\n    >>> fig, ax = plt.subplots()\n    >>> ax.hist(res.null_distribution, bins=50)\n    >>> ax.set_title("Monte Carlo distribution of test statistic")\n    >>> ax.set_xlabel("Value of Statistic")\n    >>> ax.set_ylabel("Frequency")\n    >>> plt.show()\n\n    '
    args = _monte_carlo_test_iv(data, rvs, statistic, vectorized, n_resamples, batch, alternative, axis)
    (data, rvs, statistic, vectorized, n_resamples, batch, alternative, axis) = args
    observed = np.asarray(statistic(*data, axis=-1))[()]
    n_observations = [sample.shape[-1] for sample in data]
    batch_nominal = batch or n_resamples
    null_distribution = []
    for k in range(0, n_resamples, batch_nominal):
        batch_actual = min(batch_nominal, n_resamples - k)
        resamples = [rvs_i(size=(batch_actual, n_observations_i)) for (rvs_i, n_observations_i) in zip(rvs, n_observations)]
        null_distribution.append(statistic(*resamples, axis=-1))
    null_distribution = np.concatenate(null_distribution)
    null_distribution = null_distribution.reshape([-1] + [1] * observed.ndim)

    def less(null_distribution, observed):
        if False:
            for i in range(10):
                print('nop')
        cmps = null_distribution <= observed
        pvalues = (cmps.sum(axis=0) + 1) / (n_resamples + 1)
        return pvalues

    def greater(null_distribution, observed):
        if False:
            while True:
                i = 10
        cmps = null_distribution >= observed
        pvalues = (cmps.sum(axis=0) + 1) / (n_resamples + 1)
        return pvalues

    def two_sided(null_distribution, observed):
        if False:
            print('Hello World!')
        pvalues_less = less(null_distribution, observed)
        pvalues_greater = greater(null_distribution, observed)
        pvalues = np.minimum(pvalues_less, pvalues_greater) * 2
        return pvalues
    compare = {'less': less, 'greater': greater, 'two-sided': two_sided}
    pvalues = compare[alternative](null_distribution, observed)
    pvalues = np.clip(pvalues, 0, 1)
    return MonteCarloTestResult(observed, pvalues, null_distribution)

@dataclass
class PermutationTestResult:
    """Result object returned by `scipy.stats.permutation_test`.

    Attributes
    ----------
    statistic : float or ndarray
        The observed test statistic of the data.
    pvalue : float or ndarray
        The p-value for the given alternative.
    null_distribution : ndarray
        The values of the test statistic generated under the null
        hypothesis.
    """
    statistic: float | np.ndarray
    pvalue: float | np.ndarray
    null_distribution: np.ndarray

def _all_partitions_concatenated(ns):
    if False:
        i = 10
        return i + 15
    '\n    Generate all partitions of indices of groups of given sizes, concatenated\n\n    `ns` is an iterable of ints.\n    '

    def all_partitions(z, n):
        if False:
            return 10
        for c in combinations(z, n):
            x0 = set(c)
            x1 = z - x0
            yield [x0, x1]

    def all_partitions_n(z, ns):
        if False:
            i = 10
            return i + 15
        if len(ns) == 0:
            yield [z]
            return
        for c in all_partitions(z, ns[0]):
            for d in all_partitions_n(c[1], ns[1:]):
                yield (c[0:1] + d)
    z = set(range(np.sum(ns)))
    for partitioning in all_partitions_n(z, ns[:]):
        x = np.concatenate([list(partition) for partition in partitioning]).astype(int)
        yield x

def _batch_generator(iterable, batch):
    if False:
        print('Hello World!')
    'A generator that yields batches of elements from an iterable'
    iterator = iter(iterable)
    if batch <= 0:
        raise ValueError('`batch` must be positive.')
    z = [item for (i, item) in zip(range(batch), iterator)]
    while z:
        yield z
        z = [item for (i, item) in zip(range(batch), iterator)]

def _pairings_permutations_gen(n_permutations, n_samples, n_obs_sample, batch, random_state):
    if False:
        return 10
    batch = min(batch, n_permutations)
    if hasattr(random_state, 'permuted'):

        def batched_perm_generator():
            if False:
                print('Hello World!')
            indices = np.arange(n_obs_sample)
            indices = np.tile(indices, (batch, n_samples, 1))
            for k in range(0, n_permutations, batch):
                batch_actual = min(batch, n_permutations - k)
                permuted_indices = random_state.permuted(indices, axis=-1)
                yield permuted_indices[:batch_actual]
    else:

        def batched_perm_generator():
            if False:
                for i in range(10):
                    print('nop')
            for k in range(0, n_permutations, batch):
                batch_actual = min(batch, n_permutations - k)
                size = (batch_actual, n_samples, n_obs_sample)
                x = random_state.random(size=size)
                yield np.argsort(x, axis=-1)[:batch_actual]
    return batched_perm_generator()

def _calculate_null_both(data, statistic, n_permutations, batch, random_state=None):
    if False:
        while True:
            i = 10
    '\n    Calculate null distribution for independent sample tests.\n    '
    n_samples = len(data)
    n_obs_i = [sample.shape[-1] for sample in data]
    n_obs_ic = np.cumsum(n_obs_i)
    n_obs = n_obs_ic[-1]
    n_max = np.prod([comb(n_obs_ic[i], n_obs_ic[i - 1]) for i in range(n_samples - 1, 0, -1)])
    if n_permutations >= n_max:
        exact_test = True
        n_permutations = n_max
        perm_generator = _all_partitions_concatenated(n_obs_i)
    else:
        exact_test = False
        perm_generator = (random_state.permutation(n_obs) for i in range(n_permutations))
    batch = batch or int(n_permutations)
    null_distribution = []
    data = np.concatenate(data, axis=-1)
    for indices in _batch_generator(perm_generator, batch=batch):
        indices = np.array(indices)
        data_batch = data[..., indices]
        data_batch = np.moveaxis(data_batch, -2, 0)
        data_batch = np.split(data_batch, n_obs_ic[:-1], axis=-1)
        null_distribution.append(statistic(*data_batch, axis=-1))
    null_distribution = np.concatenate(null_distribution, axis=0)
    return (null_distribution, n_permutations, exact_test)

def _calculate_null_pairings(data, statistic, n_permutations, batch, random_state=None):
    if False:
        while True:
            i = 10
    '\n    Calculate null distribution for association tests.\n    '
    n_samples = len(data)
    n_obs_sample = data[0].shape[-1]
    n_max = factorial(n_obs_sample) ** n_samples
    if n_permutations >= n_max:
        exact_test = True
        n_permutations = n_max
        batch = batch or int(n_permutations)
        perm_generator = product(*(permutations(range(n_obs_sample)) for i in range(n_samples)))
        batched_perm_generator = _batch_generator(perm_generator, batch=batch)
    else:
        exact_test = False
        batch = batch or int(n_permutations)
        args = (n_permutations, n_samples, n_obs_sample, batch, random_state)
        batched_perm_generator = _pairings_permutations_gen(*args)
    null_distribution = []
    for indices in batched_perm_generator:
        indices = np.array(indices)
        indices = np.swapaxes(indices, 0, 1)
        data_batch = [None] * n_samples
        for i in range(n_samples):
            data_batch[i] = data[i][..., indices[i]]
            data_batch[i] = np.moveaxis(data_batch[i], -2, 0)
        null_distribution.append(statistic(*data_batch, axis=-1))
    null_distribution = np.concatenate(null_distribution, axis=0)
    return (null_distribution, n_permutations, exact_test)

def _calculate_null_samples(data, statistic, n_permutations, batch, random_state=None):
    if False:
        print('Hello World!')
    '\n    Calculate null distribution for paired-sample tests.\n    '
    n_samples = len(data)
    if n_samples == 1:
        data = [data[0], -data[0]]
    data = np.swapaxes(data, 0, -1)

    def statistic_wrapped(*data, axis):
        if False:
            i = 10
            return i + 15
        data = np.swapaxes(data, 0, -1)
        if n_samples == 1:
            data = data[0:1]
        return statistic(*data, axis=axis)
    return _calculate_null_pairings(data, statistic_wrapped, n_permutations, batch, random_state)

def _permutation_test_iv(data, statistic, permutation_type, vectorized, n_resamples, batch, alternative, axis, random_state):
    if False:
        print('Hello World!')
    'Input validation for `permutation_test`.'
    axis_int = int(axis)
    if axis != axis_int:
        raise ValueError('`axis` must be an integer.')
    permutation_types = {'samples', 'pairings', 'independent'}
    permutation_type = permutation_type.lower()
    if permutation_type not in permutation_types:
        raise ValueError(f'`permutation_type` must be in {permutation_types}.')
    if vectorized not in {True, False, None}:
        raise ValueError('`vectorized` must be `True`, `False`, or `None`.')
    if vectorized is None:
        vectorized = 'axis' in inspect.signature(statistic).parameters
    if not vectorized:
        statistic = _vectorize_statistic(statistic)
    message = '`data` must be a tuple containing at least two samples'
    try:
        if len(data) < 2 and permutation_type == 'independent':
            raise ValueError(message)
    except TypeError:
        raise TypeError(message)
    data = _broadcast_arrays(data, axis)
    data_iv = []
    for sample in data:
        sample = np.atleast_1d(sample)
        if sample.shape[axis] <= 1:
            raise ValueError('each sample in `data` must contain two or more observations along `axis`.')
        sample = np.moveaxis(sample, axis_int, -1)
        data_iv.append(sample)
    n_resamples_int = int(n_resamples) if not np.isinf(n_resamples) else np.inf
    if n_resamples != n_resamples_int or n_resamples_int <= 0:
        raise ValueError('`n_resamples` must be a positive integer.')
    if batch is None:
        batch_iv = batch
    else:
        batch_iv = int(batch)
        if batch != batch_iv or batch_iv <= 0:
            raise ValueError('`batch` must be a positive integer or None.')
    alternatives = {'two-sided', 'greater', 'less'}
    alternative = alternative.lower()
    if alternative not in alternatives:
        raise ValueError(f'`alternative` must be in {alternatives}')
    random_state = check_random_state(random_state)
    return (data_iv, statistic, permutation_type, vectorized, n_resamples_int, batch_iv, alternative, axis_int, random_state)

def permutation_test(data, statistic, *, permutation_type='independent', vectorized=None, n_resamples=9999, batch=None, alternative='two-sided', axis=0, random_state=None):
    if False:
        print('Hello World!')
    '\n    Performs a permutation test of a given statistic on provided data.\n\n    For independent sample statistics, the null hypothesis is that the data are\n    randomly sampled from the same distribution.\n    For paired sample statistics, two null hypothesis can be tested:\n    that the data are paired at random or that the data are assigned to samples\n    at random.\n\n    Parameters\n    ----------\n    data : iterable of array-like\n        Contains the samples, each of which is an array of observations.\n        Dimensions of sample arrays must be compatible for broadcasting except\n        along `axis`.\n    statistic : callable\n        Statistic for which the p-value of the hypothesis test is to be\n        calculated. `statistic` must be a callable that accepts samples\n        as separate arguments (e.g. ``statistic(*data)``) and returns the\n        resulting statistic.\n        If `vectorized` is set ``True``, `statistic` must also accept a keyword\n        argument `axis` and be vectorized to compute the statistic along the\n        provided `axis` of the sample arrays.\n    permutation_type : {\'independent\', \'samples\', \'pairings\'}, optional\n        The type of permutations to be performed, in accordance with the\n        null hypothesis. The first two permutation types are for paired sample\n        statistics, in which all samples contain the same number of\n        observations and observations with corresponding indices along `axis`\n        are considered to be paired; the third is for independent sample\n        statistics.\n\n        - ``\'samples\'`` : observations are assigned to different samples\n          but remain paired with the same observations from other samples.\n          This permutation type is appropriate for paired sample hypothesis\n          tests such as the Wilcoxon signed-rank test and the paired t-test.\n        - ``\'pairings\'`` : observations are paired with different observations,\n          but they remain within the same sample. This permutation type is\n          appropriate for association/correlation tests with statistics such\n          as Spearman\'s :math:`\\rho`, Kendall\'s :math:`\\tau`, and Pearson\'s\n          :math:`r`.\n        - ``\'independent\'`` (default) : observations are assigned to different\n          samples. Samples may contain different numbers of observations. This\n          permutation type is appropriate for independent sample hypothesis\n          tests such as the Mann-Whitney :math:`U` test and the independent\n          sample t-test.\n\n          Please see the Notes section below for more detailed descriptions\n          of the permutation types.\n\n    vectorized : bool, optional\n        If `vectorized` is set ``False``, `statistic` will not be passed\n        keyword argument `axis` and is expected to calculate the statistic\n        only for 1D samples. If ``True``, `statistic` will be passed keyword\n        argument `axis` and is expected to calculate the statistic along `axis`\n        when passed an ND sample array. If ``None`` (default), `vectorized`\n        will be set ``True`` if ``axis`` is a parameter of `statistic`. Use\n        of a vectorized statistic typically reduces computation time.\n    n_resamples : int or np.inf, default: 9999\n        Number of random permutations (resamples) used to approximate the null\n        distribution. If greater than or equal to the number of distinct\n        permutations, the exact null distribution will be computed.\n        Note that the number of distinct permutations grows very rapidly with\n        the sizes of samples, so exact tests are feasible only for very small\n        data sets.\n    batch : int, optional\n        The number of permutations to process in each call to `statistic`.\n        Memory usage is O( `batch` * ``n`` ), where ``n`` is the total size\n        of all samples, regardless of the value of `vectorized`. Default is\n        ``None``, in which case ``batch`` is the number of permutations.\n    alternative : {\'two-sided\', \'less\', \'greater\'}, optional\n        The alternative hypothesis for which the p-value is calculated.\n        For each alternative, the p-value is defined for exact tests as\n        follows.\n\n        - ``\'greater\'`` : the percentage of the null distribution that is\n          greater than or equal to the observed value of the test statistic.\n        - ``\'less\'`` : the percentage of the null distribution that is\n          less than or equal to the observed value of the test statistic.\n        - ``\'two-sided\'`` (default) : twice the smaller of the p-values above.\n\n        Note that p-values for randomized tests are calculated according to the\n        conservative (over-estimated) approximation suggested in [2]_ and [3]_\n        rather than the unbiased estimator suggested in [4]_. That is, when\n        calculating the proportion of the randomized null distribution that is\n        as extreme as the observed value of the test statistic, the values in\n        the numerator and denominator are both increased by one. An\n        interpretation of this adjustment is that the observed value of the\n        test statistic is always included as an element of the randomized\n        null distribution.\n        The convention used for two-sided p-values is not universal;\n        the observed test statistic and null distribution are returned in\n        case a different definition is preferred.\n\n    axis : int, default: 0\n        The axis of the (broadcasted) samples over which to calculate the\n        statistic. If samples have a different number of dimensions,\n        singleton dimensions are prepended to samples with fewer dimensions\n        before `axis` is considered.\n    random_state : {None, int, `numpy.random.Generator`,\n                    `numpy.random.RandomState`}, optional\n\n        Pseudorandom number generator state used to generate permutations.\n\n        If `random_state` is ``None`` (default), the\n        `numpy.random.RandomState` singleton is used.\n        If `random_state` is an int, a new ``RandomState`` instance is used,\n        seeded with `random_state`.\n        If `random_state` is already a ``Generator`` or ``RandomState``\n        instance then that instance is used.\n\n    Returns\n    -------\n    res : PermutationTestResult\n        An object with attributes:\n\n        statistic : float or ndarray\n            The observed test statistic of the data.\n        pvalue : float or ndarray\n            The p-value for the given alternative.\n        null_distribution : ndarray\n            The values of the test statistic generated under the null\n            hypothesis.\n\n    Notes\n    -----\n\n    The three types of permutation tests supported by this function are\n    described below.\n\n    **Unpaired statistics** (``permutation_type=\'independent\'``):\n\n    The null hypothesis associated with this permutation type is that all\n    observations are sampled from the same underlying distribution and that\n    they have been assigned to one of the samples at random.\n\n    Suppose ``data`` contains two samples; e.g. ``a, b = data``.\n    When ``1 < n_resamples < binom(n, k)``, where\n\n    * ``k`` is the number of observations in ``a``,\n    * ``n`` is the total number of observations in ``a`` and ``b``, and\n    * ``binom(n, k)`` is the binomial coefficient (``n`` choose ``k``),\n\n    the data are pooled (concatenated), randomly assigned to either the first\n    or second sample, and the statistic is calculated. This process is\n    performed repeatedly, `permutation` times, generating a distribution of the\n    statistic under the null hypothesis. The statistic of the original\n    data is compared to this distribution to determine the p-value.\n\n    When ``n_resamples >= binom(n, k)``, an exact test is performed: the data\n    are *partitioned* between the samples in each distinct way exactly once,\n    and the exact null distribution is formed.\n    Note that for a given partitioning of the data between the samples,\n    only one ordering/permutation of the data *within* each sample is\n    considered. For statistics that do not depend on the order of the data\n    within samples, this dramatically reduces computational cost without\n    affecting the shape of the null distribution (because the frequency/count\n    of each value is affected by the same factor).\n\n    For ``a = [a1, a2, a3, a4]`` and ``b = [b1, b2, b3]``, an example of this\n    permutation type is ``x = [b3, a1, a2, b2]`` and ``y = [a4, b1, a3]``.\n    Because only one ordering/permutation of the data *within* each sample\n    is considered in an exact test, a resampling like ``x = [b3, a1, b2, a2]``\n    and ``y = [a4, a3, b1]`` would *not* be considered distinct from the\n    example above.\n\n    ``permutation_type=\'independent\'`` does not support one-sample statistics,\n    but it can be applied to statistics with more than two samples. In this\n    case, if ``n`` is an array of the number of observations within each\n    sample, the number of distinct partitions is::\n\n        np.prod([binom(sum(n[i:]), sum(n[i+1:])) for i in range(len(n)-1)])\n\n    **Paired statistics, permute pairings** (``permutation_type=\'pairings\'``):\n\n    The null hypothesis associated with this permutation type is that\n    observations within each sample are drawn from the same underlying\n    distribution and that pairings with elements of other samples are\n    assigned at random.\n\n    Suppose ``data`` contains only one sample; e.g. ``a, = data``, and we\n    wish to consider all possible pairings of elements of ``a`` with elements\n    of a second sample, ``b``. Let ``n`` be the number of observations in\n    ``a``, which must also equal the number of observations in ``b``.\n\n    When ``1 < n_resamples < factorial(n)``, the elements of ``a`` are\n    randomly permuted. The user-supplied statistic accepts one data argument,\n    say ``a_perm``, and calculates the statistic considering ``a_perm`` and\n    ``b``. This process is performed repeatedly, `permutation` times,\n    generating a distribution of the statistic under the null hypothesis.\n    The statistic of the original data is compared to this distribution to\n    determine the p-value.\n\n    When ``n_resamples >= factorial(n)``, an exact test is performed:\n    ``a`` is permuted in each distinct way exactly once. Therefore, the\n    `statistic` is computed for each unique pairing of samples between ``a``\n    and ``b`` exactly once.\n\n    For ``a = [a1, a2, a3]`` and ``b = [b1, b2, b3]``, an example of this\n    permutation type is ``a_perm = [a3, a1, a2]`` while ``b`` is left\n    in its original order.\n\n    ``permutation_type=\'pairings\'`` supports ``data`` containing any number\n    of samples, each of which must contain the same number of observations.\n    All samples provided in ``data`` are permuted *independently*. Therefore,\n    if ``m`` is the number of samples and ``n`` is the number of observations\n    within each sample, then the number of permutations in an exact test is::\n\n        factorial(n)**m\n\n    Note that if a two-sample statistic, for example, does not inherently\n    depend on the order in which observations are provided - only on the\n    *pairings* of observations - then only one of the two samples should be\n    provided in ``data``. This dramatically reduces computational cost without\n    affecting the shape of the null distribution (because the frequency/count\n    of each value is affected by the same factor).\n\n    **Paired statistics, permute samples** (``permutation_type=\'samples\'``):\n\n    The null hypothesis associated with this permutation type is that\n    observations within each pair are drawn from the same underlying\n    distribution and that the sample to which they are assigned is random.\n\n    Suppose ``data`` contains two samples; e.g. ``a, b = data``.\n    Let ``n`` be the number of observations in ``a``, which must also equal\n    the number of observations in ``b``.\n\n    When ``1 < n_resamples < 2**n``, the elements of ``a`` are ``b`` are\n    randomly swapped between samples (maintaining their pairings) and the\n    statistic is calculated. This process is performed repeatedly,\n    `permutation` times,  generating a distribution of the statistic under the\n    null hypothesis. The statistic of the original data is compared to this\n    distribution to determine the p-value.\n\n    When ``n_resamples >= 2**n``, an exact test is performed: the observations\n    are assigned to the two samples in each distinct way (while maintaining\n    pairings) exactly once.\n\n    For ``a = [a1, a2, a3]`` and ``b = [b1, b2, b3]``, an example of this\n    permutation type is ``x = [b1, a2, b3]`` and ``y = [a1, b2, a3]``.\n\n    ``permutation_type=\'samples\'`` supports ``data`` containing any number\n    of samples, each of which must contain the same number of observations.\n    If ``data`` contains more than one sample, paired observations within\n    ``data`` are exchanged between samples *independently*. Therefore, if ``m``\n    is the number of samples and ``n`` is the number of observations within\n    each sample, then the number of permutations in an exact test is::\n\n        factorial(m)**n\n\n    Several paired-sample statistical tests, such as the Wilcoxon signed rank\n    test and paired-sample t-test, can be performed considering only the\n    *difference* between two paired elements. Accordingly, if ``data`` contains\n    only one sample, then the null distribution is formed by independently\n    changing the *sign* of each observation.\n\n    .. warning::\n        The p-value is calculated by counting the elements of the null\n        distribution that are as extreme or more extreme than the observed\n        value of the statistic. Due to the use of finite precision arithmetic,\n        some statistic functions return numerically distinct values when the\n        theoretical values would be exactly equal. In some cases, this could\n        lead to a large error in the calculated p-value. `permutation_test`\n        guards against this by considering elements in the null distribution\n        that are "close" (within a factor of ``1+1e-14``) to the observed\n        value of the test statistic as equal to the observed value of the\n        test statistic. However, the user is advised to inspect the null\n        distribution to assess whether this method of comparison is\n        appropriate, and if not, calculate the p-value manually. See example\n        below.\n\n    References\n    ----------\n\n    .. [1] R. A. Fisher. The Design of Experiments, 6th Ed (1951).\n    .. [2] B. Phipson and G. K. Smyth. "Permutation P-values Should Never Be\n       Zero: Calculating Exact P-values When Permutations Are Randomly Drawn."\n       Statistical Applications in Genetics and Molecular Biology 9.1 (2010).\n    .. [3] M. D. Ernst. "Permutation Methods: A Basis for Exact Inference".\n       Statistical Science (2004).\n    .. [4] B. Efron and R. J. Tibshirani. An Introduction to the Bootstrap\n       (1993).\n\n    Examples\n    --------\n\n    Suppose we wish to test whether two samples are drawn from the same\n    distribution. Assume that the underlying distributions are unknown to us,\n    and that before observing the data, we hypothesized that the mean of the\n    first sample would be less than that of the second sample. We decide that\n    we will use the difference between the sample means as a test statistic,\n    and we will consider a p-value of 0.05 to be statistically significant.\n\n    For efficiency, we write the function defining the test statistic in a\n    vectorized fashion: the samples ``x`` and ``y`` can be ND arrays, and the\n    statistic will be calculated for each axis-slice along `axis`.\n\n    >>> import numpy as np\n    >>> def statistic(x, y, axis):\n    ...     return np.mean(x, axis=axis) - np.mean(y, axis=axis)\n\n    After collecting our data, we calculate the observed value of the test\n    statistic.\n\n    >>> from scipy.stats import norm\n    >>> rng = np.random.default_rng()\n    >>> x = norm.rvs(size=5, random_state=rng)\n    >>> y = norm.rvs(size=6, loc = 3, random_state=rng)\n    >>> statistic(x, y, 0)\n    -3.5411688580987266\n\n    Indeed, the test statistic is negative, suggesting that the true mean of\n    the distribution underlying ``x`` is less than that of the distribution\n    underlying ``y``. To determine the probability of this occurring by chance\n    if the two samples were drawn from the same distribution, we perform\n    a permutation test.\n\n    >>> from scipy.stats import permutation_test\n    >>> # because our statistic is vectorized, we pass `vectorized=True`\n    >>> # `n_resamples=np.inf` indicates that an exact test is to be performed\n    >>> res = permutation_test((x, y), statistic, vectorized=True,\n    ...                        n_resamples=np.inf, alternative=\'less\')\n    >>> print(res.statistic)\n    -3.5411688580987266\n    >>> print(res.pvalue)\n    0.004329004329004329\n\n    The probability of obtaining a test statistic less than or equal to the\n    observed value under the null hypothesis is 0.4329%. This is less than our\n    chosen threshold of 5%, so we consider this to be significant evidence\n    against the null hypothesis in favor of the alternative.\n\n    Because the size of the samples above was small, `permutation_test` could\n    perform an exact test. For larger samples, we resort to a randomized\n    permutation test.\n\n    >>> x = norm.rvs(size=100, random_state=rng)\n    >>> y = norm.rvs(size=120, loc=0.3, random_state=rng)\n    >>> res = permutation_test((x, y), statistic, n_resamples=100000,\n    ...                        vectorized=True, alternative=\'less\',\n    ...                        random_state=rng)\n    >>> print(res.statistic)\n    -0.5230459671240913\n    >>> print(res.pvalue)\n    0.00016999830001699983\n\n    The approximate probability of obtaining a test statistic less than or\n    equal to the observed value under the null hypothesis is 0.0225%. This is\n    again less than our chosen threshold of 5%, so again we have significant\n    evidence to reject the null hypothesis in favor of the alternative.\n\n    For large samples and number of permutations, the result is comparable to\n    that of the corresponding asymptotic test, the independent sample t-test.\n\n    >>> from scipy.stats import ttest_ind\n    >>> res_asymptotic = ttest_ind(x, y, alternative=\'less\')\n    >>> print(res_asymptotic.pvalue)\n    0.00012688101537979522\n\n    The permutation distribution of the test statistic is provided for\n    further investigation.\n\n    >>> import matplotlib.pyplot as plt\n    >>> plt.hist(res.null_distribution, bins=50)\n    >>> plt.title("Permutation distribution of test statistic")\n    >>> plt.xlabel("Value of Statistic")\n    >>> plt.ylabel("Frequency")\n    >>> plt.show()\n\n    Inspection of the null distribution is essential if the statistic suffers\n    from inaccuracy due to limited machine precision. Consider the following\n    case:\n\n    >>> from scipy.stats import pearsonr\n    >>> x = [1, 2, 4, 3]\n    >>> y = [2, 4, 6, 8]\n    >>> def statistic(x, y):\n    ...     return pearsonr(x, y).statistic\n    >>> res = permutation_test((x, y), statistic, vectorized=False,\n    ...                        permutation_type=\'pairings\',\n    ...                        alternative=\'greater\')\n    >>> r, pvalue, null = res.statistic, res.pvalue, res.null_distribution\n\n    In this case, some elements of the null distribution differ from the\n    observed value of the correlation coefficient ``r`` due to numerical noise.\n    We manually inspect the elements of the null distribution that are nearly\n    the same as the observed value of the test statistic.\n\n    >>> r\n    0.8\n    >>> unique = np.unique(null)\n    >>> unique\n    array([-1. , -0.8, -0.8, -0.6, -0.4, -0.2, -0.2,  0. ,  0.2,  0.2,  0.4,\n            0.6,  0.8,  0.8,  1. ]) # may vary\n    >>> unique[np.isclose(r, unique)].tolist()\n    [0.7999999999999999, 0.8]\n\n    If `permutation_test` were to perform the comparison naively, the\n    elements of the null distribution with value ``0.7999999999999999`` would\n    not be considered as extreme or more extreme as the observed value of the\n    statistic, so the calculated p-value would be too small.\n\n    >>> incorrect_pvalue = np.count_nonzero(null >= r) / len(null)\n    >>> incorrect_pvalue\n    0.1111111111111111  # may vary\n\n    Instead, `permutation_test` treats elements of the null distribution that\n    are within ``max(1e-14, abs(r)*1e-14)`` of the observed value of the\n    statistic ``r`` to be equal to ``r``.\n\n    >>> correct_pvalue = np.count_nonzero(null >= r - 1e-14) / len(null)\n    >>> correct_pvalue\n    0.16666666666666666\n    >>> res.pvalue == correct_pvalue\n    True\n\n    This method of comparison is expected to be accurate in most practical\n    situations, but the user is advised to assess this by inspecting the\n    elements of the null distribution that are close to the observed value\n    of the statistic. Also, consider the use of statistics that can be\n    calculated using exact arithmetic (e.g. integer statistics).\n\n    '
    args = _permutation_test_iv(data, statistic, permutation_type, vectorized, n_resamples, batch, alternative, axis, random_state)
    (data, statistic, permutation_type, vectorized, n_resamples, batch, alternative, axis, random_state) = args
    observed = statistic(*data, axis=-1)
    null_calculators = {'pairings': _calculate_null_pairings, 'samples': _calculate_null_samples, 'independent': _calculate_null_both}
    null_calculator_args = (data, statistic, n_resamples, batch, random_state)
    calculate_null = null_calculators[permutation_type]
    (null_distribution, n_resamples, exact_test) = calculate_null(*null_calculator_args)
    adjustment = 0 if exact_test else 1
    eps = 1e-14
    gamma = np.maximum(eps, np.abs(eps * observed))

    def less(null_distribution, observed):
        if False:
            for i in range(10):
                print('nop')
        cmps = null_distribution <= observed + gamma
        pvalues = (cmps.sum(axis=0) + adjustment) / (n_resamples + adjustment)
        return pvalues

    def greater(null_distribution, observed):
        if False:
            i = 10
            return i + 15
        cmps = null_distribution >= observed - gamma
        pvalues = (cmps.sum(axis=0) + adjustment) / (n_resamples + adjustment)
        return pvalues

    def two_sided(null_distribution, observed):
        if False:
            while True:
                i = 10
        pvalues_less = less(null_distribution, observed)
        pvalues_greater = greater(null_distribution, observed)
        pvalues = np.minimum(pvalues_less, pvalues_greater) * 2
        return pvalues
    compare = {'less': less, 'greater': greater, 'two-sided': two_sided}
    pvalues = compare[alternative](null_distribution, observed)
    pvalues = np.clip(pvalues, 0, 1)
    return PermutationTestResult(observed, pvalues, null_distribution)

@dataclass
class ResamplingMethod:
    """Configuration information for a statistical resampling method.

    Instances of this class can be passed into the `method` parameter of some
    hypothesis test functions to perform a resampling or Monte Carlo version
    of the hypothesis test.

    Attributes
    ----------
    n_resamples : int
        The number of resamples to perform or Monte Carlo samples to draw.
    batch : int, optional
        The number of resamples to process in each vectorized call to
        the statistic. Batch sizes >>1 tend to be faster when the statistic
        is vectorized, but memory usage scales linearly with the batch size.
        Default is ``None``, which processes all resamples in a single batch.
    """
    n_resamples: int = 9999
    batch: int = None

@dataclass
class MonteCarloMethod(ResamplingMethod):
    """Configuration information for a Monte Carlo hypothesis test.

    Instances of this class can be passed into the `method` parameter of some
    hypothesis test functions to perform a Monte Carlo version of the
    hypothesis tests.

    Attributes
    ----------
    n_resamples : int, optional
        The number of Monte Carlo samples to draw. Default is 9999.
    batch : int, optional
        The number of Monte Carlo samples to process in each vectorized call to
        the statistic. Batch sizes >>1 tend to be faster when the statistic
        is vectorized, but memory usage scales linearly with the batch size.
        Default is ``None``, which processes all samples in a single batch.
    rvs : callable or tuple of callables, optional
        A callable or sequence of callables that generates random variates
        under the null hypothesis. Each element of `rvs` must be a callable
        that accepts keyword argument ``size`` (e.g. ``rvs(size=(m, n))``) and
        returns an N-d array sample of that shape. If `rvs` is a sequence, the
        number of callables in `rvs` must match the number of samples passed
        to the hypothesis test in which the `MonteCarloMethod` is used. Default
        is ``None``, in which case the hypothesis test function chooses values
        to match the standard version of the hypothesis test. For example,
        the null hypothesis of `scipy.stats.pearsonr` is typically that the
        samples are drawn from the standard normal distribution, so
        ``rvs = (rng.normal, rng.normal)`` where
        ``rng = np.random.default_rng()``.
    """
    rvs: object = None

    def _asdict(self):
        if False:
            while True:
                i = 10
        return dict(n_resamples=self.n_resamples, batch=self.batch, rvs=self.rvs)

@dataclass
class PermutationMethod(ResamplingMethod):
    """Configuration information for a permutation hypothesis test.

    Instances of this class can be passed into the `method` parameter of some
    hypothesis test functions to perform a permutation version of the
    hypothesis tests.

    Attributes
    ----------
    n_resamples : int, optional
        The number of resamples to perform. Default is 9999.
    batch : int, optional
        The number of resamples to process in each vectorized call to
        the statistic. Batch sizes >>1 tend to be faster when the statistic
        is vectorized, but memory usage scales linearly with the batch size.
        Default is ``None``, which processes all resamples in a single batch.
    random_state : {None, int, `numpy.random.Generator`,
                    `numpy.random.RandomState`}, optional

        Pseudorandom number generator state used to generate resamples.

        If `random_state` is already a ``Generator`` or ``RandomState``
        instance, then that instance is used.
        If `random_state` is an int, a new ``RandomState`` instance is used,
        seeded with `random_state`.
        If `random_state` is ``None`` (default), the
        `numpy.random.RandomState` singleton is used.
    """
    random_state: object = None

    def _asdict(self):
        if False:
            for i in range(10):
                print('nop')
        return dict(n_resamples=self.n_resamples, batch=self.batch, random_state=self.random_state)

@dataclass
class BootstrapMethod(ResamplingMethod):
    """Configuration information for a bootstrap confidence interval.

    Instances of this class can be passed into the `method` parameter of some
    confidence interval methods to generate a bootstrap confidence interval.

    Attributes
    ----------
    n_resamples : int, optional
        The number of resamples to perform. Default is 9999.
    batch : int, optional
        The number of resamples to process in each vectorized call to
        the statistic. Batch sizes >>1 tend to be faster when the statistic
        is vectorized, but memory usage scales linearly with the batch size.
        Default is ``None``, which processes all resamples in a single batch.
    random_state : {None, int, `numpy.random.Generator`,
                    `numpy.random.RandomState`}, optional

        Pseudorandom number generator state used to generate resamples.

        If `random_state` is already a ``Generator`` or ``RandomState``
        instance, then that instance is used.
        If `random_state` is an int, a new ``RandomState`` instance is used,
        seeded with `random_state`.
        If `random_state` is ``None`` (default), the
        `numpy.random.RandomState` singleton is used.

    method : {'bca', 'percentile', 'basic'}
        Whether to use the 'percentile' bootstrap ('percentile'), the 'basic'
        (AKA 'reverse') bootstrap ('basic'), or the bias-corrected and
        accelerated bootstrap ('BCa', default).
    """
    random_state: object = None
    method: str = 'BCa'

    def _asdict(self):
        if False:
            return 10
        return dict(n_resamples=self.n_resamples, batch=self.batch, random_state=self.random_state, method=self.method)