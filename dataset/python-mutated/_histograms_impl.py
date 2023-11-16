"""
Histogram-related functions
"""
import contextlib
import functools
import operator
import warnings
import numpy as np
from numpy._core import overrides
__all__ = ['histogram', 'histogramdd', 'histogram_bin_edges']
array_function_dispatch = functools.partial(overrides.array_function_dispatch, module='numpy')
_range = range

def _ptp(x):
    if False:
        print('Hello World!')
    "Peak-to-peak value of x.\n\n    This implementation avoids the problem of signed integer arrays having a\n    peak-to-peak value that cannot be represented with the array's data type.\n    This function returns an unsigned value for signed integer arrays.\n    "
    return _unsigned_subtract(x.max(), x.min())

def _hist_bin_sqrt(x, range):
    if False:
        i = 10
        return i + 15
    '\n    Square root histogram bin estimator.\n\n    Bin width is inversely proportional to the data size. Used by many\n    programs for its simplicity.\n\n    Parameters\n    ----------\n    x : array_like\n        Input data that is to be histogrammed, trimmed to range. May not\n        be empty.\n\n    Returns\n    -------\n    h : An estimate of the optimal bin width for the given data.\n    '
    del range
    return _ptp(x) / np.sqrt(x.size)

def _hist_bin_sturges(x, range):
    if False:
        i = 10
        return i + 15
    '\n    Sturges histogram bin estimator.\n\n    A very simplistic estimator based on the assumption of normality of\n    the data. This estimator has poor performance for non-normal data,\n    which becomes especially obvious for large data sets. The estimate\n    depends only on size of the data.\n\n    Parameters\n    ----------\n    x : array_like\n        Input data that is to be histogrammed, trimmed to range. May not\n        be empty.\n\n    Returns\n    -------\n    h : An estimate of the optimal bin width for the given data.\n    '
    del range
    return _ptp(x) / (np.log2(x.size) + 1.0)

def _hist_bin_rice(x, range):
    if False:
        print('Hello World!')
    '\n    Rice histogram bin estimator.\n\n    Another simple estimator with no normality assumption. It has better\n    performance for large data than Sturges, but tends to overestimate\n    the number of bins. The number of bins is proportional to the cube\n    root of data size (asymptotically optimal). The estimate depends\n    only on size of the data.\n\n    Parameters\n    ----------\n    x : array_like\n        Input data that is to be histogrammed, trimmed to range. May not\n        be empty.\n\n    Returns\n    -------\n    h : An estimate of the optimal bin width for the given data.\n    '
    del range
    return _ptp(x) / (2.0 * x.size ** (1.0 / 3))

def _hist_bin_scott(x, range):
    if False:
        i = 10
        return i + 15
    '\n    Scott histogram bin estimator.\n\n    The binwidth is proportional to the standard deviation of the data\n    and inversely proportional to the cube root of data size\n    (asymptotically optimal).\n\n    Parameters\n    ----------\n    x : array_like\n        Input data that is to be histogrammed, trimmed to range. May not\n        be empty.\n\n    Returns\n    -------\n    h : An estimate of the optimal bin width for the given data.\n    '
    del range
    return (24.0 * np.pi ** 0.5 / x.size) ** (1.0 / 3.0) * np.std(x)

def _hist_bin_stone(x, range):
    if False:
        for i in range(10):
            print('nop')
    "\n    Histogram bin estimator based on minimizing the estimated integrated squared error (ISE).\n\n    The number of bins is chosen by minimizing the estimated ISE against the unknown true distribution.\n    The ISE is estimated using cross-validation and can be regarded as a generalization of Scott's rule.\n    https://en.wikipedia.org/wiki/Histogram#Scott.27s_normal_reference_rule\n\n    This paper by Stone appears to be the origination of this rule.\n    https://digitalassets.lib.berkeley.edu/sdtr/ucb/text/34.pdf\n\n    Parameters\n    ----------\n    x : array_like\n        Input data that is to be histogrammed, trimmed to range. May not\n        be empty.\n    range : (float, float)\n        The lower and upper range of the bins.\n\n    Returns\n    -------\n    h : An estimate of the optimal bin width for the given data.\n    "
    n = x.size
    ptp_x = _ptp(x)
    if n <= 1 or ptp_x == 0:
        return 0

    def jhat(nbins):
        if False:
            for i in range(10):
                print('nop')
        hh = ptp_x / nbins
        p_k = np.histogram(x, bins=nbins, range=range)[0] / n
        return (2 - (n + 1) * p_k.dot(p_k)) / hh
    nbins_upper_bound = max(100, int(np.sqrt(n)))
    nbins = min(_range(1, nbins_upper_bound + 1), key=jhat)
    if nbins == nbins_upper_bound:
        warnings.warn('The number of bins estimated may be suboptimal.', RuntimeWarning, stacklevel=3)
    return ptp_x / nbins

def _hist_bin_doane(x, range):
    if False:
        while True:
            i = 10
    "\n    Doane's histogram bin estimator.\n\n    Improved version of Sturges' formula which works better for\n    non-normal data. See\n    stats.stackexchange.com/questions/55134/doanes-formula-for-histogram-binning\n\n    Parameters\n    ----------\n    x : array_like\n        Input data that is to be histogrammed, trimmed to range. May not\n        be empty.\n\n    Returns\n    -------\n    h : An estimate of the optimal bin width for the given data.\n    "
    del range
    if x.size > 2:
        sg1 = np.sqrt(6.0 * (x.size - 2) / ((x.size + 1.0) * (x.size + 3)))
        sigma = np.std(x)
        if sigma > 0.0:
            temp = x - np.mean(x)
            np.true_divide(temp, sigma, temp)
            np.power(temp, 3, temp)
            g1 = np.mean(temp)
            return _ptp(x) / (1.0 + np.log2(x.size) + np.log2(1.0 + np.absolute(g1) / sg1))
    return 0.0

def _hist_bin_fd(x, range):
    if False:
        while True:
            i = 10
    '\n    The Freedman-Diaconis histogram bin estimator.\n\n    The Freedman-Diaconis rule uses interquartile range (IQR) to\n    estimate binwidth. It is considered a variation of the Scott rule\n    with more robustness as the IQR is less affected by outliers than\n    the standard deviation. However, the IQR depends on fewer points\n    than the standard deviation, so it is less accurate, especially for\n    long tailed distributions.\n\n    If the IQR is 0, this function returns 0 for the bin width.\n    Binwidth is inversely proportional to the cube root of data size\n    (asymptotically optimal).\n\n    Parameters\n    ----------\n    x : array_like\n        Input data that is to be histogrammed, trimmed to range. May not\n        be empty.\n\n    Returns\n    -------\n    h : An estimate of the optimal bin width for the given data.\n    '
    del range
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    return 2.0 * iqr * x.size ** (-1.0 / 3.0)

def _hist_bin_auto(x, range):
    if False:
        return 10
    "\n    Histogram bin estimator that uses the minimum width of the\n    Freedman-Diaconis and Sturges estimators if the FD bin width is non-zero.\n    If the bin width from the FD estimator is 0, the Sturges estimator is used.\n\n    The FD estimator is usually the most robust method, but its width\n    estimate tends to be too large for small `x` and bad for data with limited\n    variance. The Sturges estimator is quite good for small (<1000) datasets\n    and is the default in the R language. This method gives good off-the-shelf\n    behaviour.\n\n    .. versionchanged:: 1.15.0\n    If there is limited variance the IQR can be 0, which results in the\n    FD bin width being 0 too. This is not a valid bin width, so\n    ``np.histogram_bin_edges`` chooses 1 bin instead, which may not be optimal.\n    If the IQR is 0, it's unlikely any variance-based estimators will be of\n    use, so we revert to the Sturges estimator, which only uses the size of the\n    dataset in its calculation.\n\n    Parameters\n    ----------\n    x : array_like\n        Input data that is to be histogrammed, trimmed to range. May not\n        be empty.\n\n    Returns\n    -------\n    h : An estimate of the optimal bin width for the given data.\n\n    See Also\n    --------\n    _hist_bin_fd, _hist_bin_sturges\n    "
    fd_bw = _hist_bin_fd(x, range)
    sturges_bw = _hist_bin_sturges(x, range)
    del range
    if fd_bw:
        return min(fd_bw, sturges_bw)
    else:
        return sturges_bw
_hist_bin_selectors = {'stone': _hist_bin_stone, 'auto': _hist_bin_auto, 'doane': _hist_bin_doane, 'fd': _hist_bin_fd, 'rice': _hist_bin_rice, 'scott': _hist_bin_scott, 'sqrt': _hist_bin_sqrt, 'sturges': _hist_bin_sturges}

def _ravel_and_check_weights(a, weights):
    if False:
        print('Hello World!')
    ' Check a and weights have matching shapes, and ravel both '
    a = np.asarray(a)
    if a.dtype == np.bool_:
        warnings.warn('Converting input from {} to {} for compatibility.'.format(a.dtype, np.uint8), RuntimeWarning, stacklevel=3)
        a = a.astype(np.uint8)
    if weights is not None:
        weights = np.asarray(weights)
        if weights.shape != a.shape:
            raise ValueError('weights should have the same shape as a.')
        weights = weights.ravel()
    a = a.ravel()
    return (a, weights)

def _get_outer_edges(a, range):
    if False:
        return 10
    '\n    Determine the outer bin edges to use, from either the data or the range\n    argument\n    '
    if range is not None:
        (first_edge, last_edge) = range
        if first_edge > last_edge:
            raise ValueError('max must be larger than min in range parameter.')
        if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
            raise ValueError('supplied range of [{}, {}] is not finite'.format(first_edge, last_edge))
    elif a.size == 0:
        (first_edge, last_edge) = (0, 1)
    else:
        (first_edge, last_edge) = (a.min(), a.max())
        if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
            raise ValueError('autodetected range of [{}, {}] is not finite'.format(first_edge, last_edge))
    if first_edge == last_edge:
        first_edge = first_edge - 0.5
        last_edge = last_edge + 0.5
    return (first_edge, last_edge)

def _unsigned_subtract(a, b):
    if False:
        i = 10
        return i + 15
    '\n    Subtract two values where a >= b, and produce an unsigned result\n\n    This is needed when finding the difference between the upper and lower\n    bound of an int16 histogram\n    '
    signed_to_unsigned = {np.byte: np.ubyte, np.short: np.ushort, np.intc: np.uintc, np.int_: np.uint, np.longlong: np.ulonglong}
    dt = np.result_type(a, b)
    try:
        unsigned_dt = signed_to_unsigned[dt.type]
    except KeyError:
        return np.subtract(a, b, dtype=dt)
    else:
        return np.subtract(np.asarray(a, dtype=dt), np.asarray(b, dtype=dt), casting='unsafe', dtype=unsigned_dt)

def _get_bin_edges(a, bins, range, weights):
    if False:
        print('Hello World!')
    '\n    Computes the bins used internally by `histogram`.\n\n    Parameters\n    ==========\n    a : ndarray\n        Ravelled data array\n    bins, range\n        Forwarded arguments from `histogram`.\n    weights : ndarray, optional\n        Ravelled weights array, or None\n\n    Returns\n    =======\n    bin_edges : ndarray\n        Array of bin edges\n    uniform_bins : (Number, Number, int):\n        The upper bound, lowerbound, and number of bins, used in the optimized\n        implementation of `histogram` that works on uniform bins.\n    '
    n_equal_bins = None
    bin_edges = None
    if isinstance(bins, str):
        bin_name = bins
        if bin_name not in _hist_bin_selectors:
            raise ValueError('{!r} is not a valid estimator for `bins`'.format(bin_name))
        if weights is not None:
            raise TypeError('Automated estimation of the number of bins is not supported for weighted data')
        (first_edge, last_edge) = _get_outer_edges(a, range)
        if range is not None:
            keep = a >= first_edge
            keep &= a <= last_edge
            if not np.logical_and.reduce(keep):
                a = a[keep]
        if a.size == 0:
            n_equal_bins = 1
        else:
            width = _hist_bin_selectors[bin_name](a, (first_edge, last_edge))
            if width:
                n_equal_bins = int(np.ceil(_unsigned_subtract(last_edge, first_edge) / width))
            else:
                n_equal_bins = 1
    elif np.ndim(bins) == 0:
        try:
            n_equal_bins = operator.index(bins)
        except TypeError as e:
            raise TypeError('`bins` must be an integer, a string, or an array') from e
        if n_equal_bins < 1:
            raise ValueError('`bins` must be positive, when an integer')
        (first_edge, last_edge) = _get_outer_edges(a, range)
    elif np.ndim(bins) == 1:
        bin_edges = np.asarray(bins)
        if np.any(bin_edges[:-1] > bin_edges[1:]):
            raise ValueError('`bins` must increase monotonically, when an array')
    else:
        raise ValueError('`bins` must be 1d, when an array')
    if n_equal_bins is not None:
        bin_type = np.result_type(first_edge, last_edge, a)
        if np.issubdtype(bin_type, np.integer):
            bin_type = np.result_type(bin_type, float)
        bin_edges = np.linspace(first_edge, last_edge, n_equal_bins + 1, endpoint=True, dtype=bin_type)
        return (bin_edges, (first_edge, last_edge, n_equal_bins))
    else:
        return (bin_edges, None)

def _search_sorted_inclusive(a, v):
    if False:
        for i in range(10):
            print('nop')
    '\n    Like `searchsorted`, but where the last item in `v` is placed on the right.\n\n    In the context of a histogram, this makes the last bin edge inclusive\n    '
    return np.concatenate((a.searchsorted(v[:-1], 'left'), a.searchsorted(v[-1:], 'right')))

def _histogram_bin_edges_dispatcher(a, bins=None, range=None, weights=None):
    if False:
        print('Hello World!')
    return (a, bins, weights)

@array_function_dispatch(_histogram_bin_edges_dispatcher)
def histogram_bin_edges(a, bins=10, range=None, weights=None):
    if False:
        return 10
    "\n    Function to calculate only the edges of the bins used by the `histogram`\n    function.\n\n    Parameters\n    ----------\n    a : array_like\n        Input data. The histogram is computed over the flattened array.\n    bins : int or sequence of scalars or str, optional\n        If `bins` is an int, it defines the number of equal-width\n        bins in the given range (10, by default). If `bins` is a\n        sequence, it defines the bin edges, including the rightmost\n        edge, allowing for non-uniform bin widths.\n\n        If `bins` is a string from the list below, `histogram_bin_edges` will use\n        the method chosen to calculate the optimal bin width and\n        consequently the number of bins (see `Notes` for more detail on\n        the estimators) from the data that falls within the requested\n        range. While the bin width will be optimal for the actual data\n        in the range, the number of bins will be computed to fill the\n        entire range, including the empty portions. For visualisation,\n        using the 'auto' option is suggested. Weighted data is not\n        supported for automated bin size selection.\n\n        'auto'\n            Maximum of the 'sturges' and 'fd' estimators. Provides good\n            all around performance.\n\n        'fd' (Freedman Diaconis Estimator)\n            Robust (resilient to outliers) estimator that takes into\n            account data variability and data size.\n\n        'doane'\n            An improved version of Sturges' estimator that works better\n            with non-normal datasets.\n\n        'scott'\n            Less robust estimator that takes into account data variability\n            and data size.\n\n        'stone'\n            Estimator based on leave-one-out cross-validation estimate of\n            the integrated squared error. Can be regarded as a generalization\n            of Scott's rule.\n\n        'rice'\n            Estimator does not take variability into account, only data\n            size. Commonly overestimates number of bins required.\n\n        'sturges'\n            R's default method, only accounts for data size. Only\n            optimal for gaussian data and underestimates number of bins\n            for large non-gaussian datasets.\n\n        'sqrt'\n            Square root (of data size) estimator, used by Excel and\n            other programs for its speed and simplicity.\n\n    range : (float, float), optional\n        The lower and upper range of the bins.  If not provided, range\n        is simply ``(a.min(), a.max())``.  Values outside the range are\n        ignored. The first element of the range must be less than or\n        equal to the second. `range` affects the automatic bin\n        computation as well. While bin width is computed to be optimal\n        based on the actual data within `range`, the bin count will fill\n        the entire range including portions containing no data.\n\n    weights : array_like, optional\n        An array of weights, of the same shape as `a`.  Each value in\n        `a` only contributes its associated weight towards the bin count\n        (instead of 1). This is currently not used by any of the bin estimators,\n        but may be in the future.\n\n    Returns\n    -------\n    bin_edges : array of dtype float\n        The edges to pass into `histogram`\n\n    See Also\n    --------\n    histogram\n\n    Notes\n    -----\n    The methods to estimate the optimal number of bins are well founded\n    in literature, and are inspired by the choices R provides for\n    histogram visualisation. Note that having the number of bins\n    proportional to :math:`n^{1/3}` is asymptotically optimal, which is\n    why it appears in most estimators. These are simply plug-in methods\n    that give good starting points for number of bins. In the equations\n    below, :math:`h` is the binwidth and :math:`n_h` is the number of\n    bins. All estimators that compute bin counts are recast to bin width\n    using the `ptp` of the data. The final bin count is obtained from\n    ``np.round(np.ceil(range / h))``. The final bin width is often less\n    than what is returned by the estimators below.\n\n    'auto' (maximum of the 'sturges' and 'fd' estimators)\n        A compromise to get a good value. For small datasets the Sturges\n        value will usually be chosen, while larger datasets will usually\n        default to FD.  Avoids the overly conservative behaviour of FD\n        and Sturges for small and large datasets respectively.\n        Switchover point is usually :math:`a.size \\approx 1000`.\n\n    'fd' (Freedman Diaconis Estimator)\n        .. math:: h = 2 \\frac{IQR}{n^{1/3}}\n\n        The binwidth is proportional to the interquartile range (IQR)\n        and inversely proportional to cube root of a.size. Can be too\n        conservative for small datasets, but is quite good for large\n        datasets. The IQR is very robust to outliers.\n\n    'scott'\n        .. math:: h = \\sigma \\sqrt[3]{\\frac{24 \\sqrt{\\pi}}{n}}\n\n        The binwidth is proportional to the standard deviation of the\n        data and inversely proportional to cube root of ``x.size``. Can\n        be too conservative for small datasets, but is quite good for\n        large datasets. The standard deviation is not very robust to\n        outliers. Values are very similar to the Freedman-Diaconis\n        estimator in the absence of outliers.\n\n    'rice'\n        .. math:: n_h = 2n^{1/3}\n\n        The number of bins is only proportional to cube root of\n        ``a.size``. It tends to overestimate the number of bins and it\n        does not take into account data variability.\n\n    'sturges'\n        .. math:: n_h = \\log _{2}(n) + 1\n\n        The number of bins is the base 2 log of ``a.size``.  This\n        estimator assumes normality of data and is too conservative for\n        larger, non-normal datasets. This is the default method in R's\n        ``hist`` method.\n\n    'doane'\n        .. math:: n_h = 1 + \\log_{2}(n) +\n                        \\log_{2}\\left(1 + \\frac{|g_1|}{\\sigma_{g_1}}\\right)\n\n            g_1 = mean\\left[\\left(\\frac{x - \\mu}{\\sigma}\\right)^3\\right]\n\n            \\sigma_{g_1} = \\sqrt{\\frac{6(n - 2)}{(n + 1)(n + 3)}}\n\n        An improved version of Sturges' formula that produces better\n        estimates for non-normal datasets. This estimator attempts to\n        account for the skew of the data.\n\n    'sqrt'\n        .. math:: n_h = \\sqrt n\n\n        The simplest and fastest estimator. Only takes into account the\n        data size.\n\n    Examples\n    --------\n    >>> arr = np.array([0, 0, 0, 1, 2, 3, 3, 4, 5])\n    >>> np.histogram_bin_edges(arr, bins='auto', range=(0, 1))\n    array([0.  , 0.25, 0.5 , 0.75, 1.  ])\n    >>> np.histogram_bin_edges(arr, bins=2)\n    array([0. , 2.5, 5. ])\n\n    For consistency with histogram, an array of pre-computed bins is\n    passed through unmodified:\n\n    >>> np.histogram_bin_edges(arr, [1, 2])\n    array([1, 2])\n\n    This function allows one set of bins to be computed, and reused across\n    multiple histograms:\n\n    >>> shared_bins = np.histogram_bin_edges(arr, bins='auto')\n    >>> shared_bins\n    array([0., 1., 2., 3., 4., 5.])\n\n    >>> group_id = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1])\n    >>> hist_0, _ = np.histogram(arr[group_id == 0], bins=shared_bins)\n    >>> hist_1, _ = np.histogram(arr[group_id == 1], bins=shared_bins)\n\n    >>> hist_0; hist_1\n    array([1, 1, 0, 1, 0])\n    array([2, 0, 1, 1, 2])\n\n    Which gives more easily comparable results than using separate bins for\n    each histogram:\n\n    >>> hist_0, bins_0 = np.histogram(arr[group_id == 0], bins='auto')\n    >>> hist_1, bins_1 = np.histogram(arr[group_id == 1], bins='auto')\n    >>> hist_0; hist_1\n    array([1, 1, 1])\n    array([2, 1, 1, 2])\n    >>> bins_0; bins_1\n    array([0., 1., 2., 3.])\n    array([0.  , 1.25, 2.5 , 3.75, 5.  ])\n\n    "
    (a, weights) = _ravel_and_check_weights(a, weights)
    (bin_edges, _) = _get_bin_edges(a, bins, range, weights)
    return bin_edges

def _histogram_dispatcher(a, bins=None, range=None, density=None, weights=None):
    if False:
        return 10
    return (a, bins, weights)

@array_function_dispatch(_histogram_dispatcher)
def histogram(a, bins=10, range=None, density=None, weights=None):
    if False:
        i = 10
        return i + 15
    '\n    Compute the histogram of a dataset.\n\n    Parameters\n    ----------\n    a : array_like\n        Input data. The histogram is computed over the flattened array.\n    bins : int or sequence of scalars or str, optional\n        If `bins` is an int, it defines the number of equal-width\n        bins in the given range (10, by default). If `bins` is a\n        sequence, it defines a monotonically increasing array of bin edges,\n        including the rightmost edge, allowing for non-uniform bin widths.\n\n        .. versionadded:: 1.11.0\n\n        If `bins` is a string, it defines the method used to calculate the\n        optimal bin width, as defined by `histogram_bin_edges`.\n\n    range : (float, float), optional\n        The lower and upper range of the bins.  If not provided, range\n        is simply ``(a.min(), a.max())``.  Values outside the range are\n        ignored. The first element of the range must be less than or\n        equal to the second. `range` affects the automatic bin\n        computation as well. While bin width is computed to be optimal\n        based on the actual data within `range`, the bin count will fill\n        the entire range including portions containing no data.\n    weights : array_like, optional\n        An array of weights, of the same shape as `a`.  Each value in\n        `a` only contributes its associated weight towards the bin count\n        (instead of 1). If `density` is True, the weights are\n        normalized, so that the integral of the density over the range\n        remains 1.\n    density : bool, optional\n        If ``False``, the result will contain the number of samples in\n        each bin. If ``True``, the result is the value of the\n        probability *density* function at the bin, normalized such that\n        the *integral* over the range is 1. Note that the sum of the\n        histogram values will not be equal to 1 unless bins of unity\n        width are chosen; it is not a probability *mass* function.\n\n    Returns\n    -------\n    hist : array\n        The values of the histogram. See `density` and `weights` for a\n        description of the possible semantics.\n    bin_edges : array of dtype float\n        Return the bin edges ``(length(hist)+1)``.\n\n\n    See Also\n    --------\n    histogramdd, bincount, searchsorted, digitize, histogram_bin_edges\n\n    Notes\n    -----\n    All but the last (righthand-most) bin is half-open.  In other words,\n    if `bins` is::\n\n      [1, 2, 3, 4]\n\n    then the first bin is ``[1, 2)`` (including 1, but excluding 2) and\n    the second ``[2, 3)``.  The last bin, however, is ``[3, 4]``, which\n    *includes* 4.\n\n\n    Examples\n    --------\n    >>> np.histogram([1, 2, 1], bins=[0, 1, 2, 3])\n    (array([0, 2, 1]), array([0, 1, 2, 3]))\n    >>> np.histogram(np.arange(4), bins=np.arange(5), density=True)\n    (array([0.25, 0.25, 0.25, 0.25]), array([0, 1, 2, 3, 4]))\n    >>> np.histogram([[1, 2, 1], [1, 0, 1]], bins=[0,1,2,3])\n    (array([1, 4, 1]), array([0, 1, 2, 3]))\n\n    >>> a = np.arange(5)\n    >>> hist, bin_edges = np.histogram(a, density=True)\n    >>> hist\n    array([0.5, 0. , 0.5, 0. , 0. , 0.5, 0. , 0.5, 0. , 0.5])\n    >>> hist.sum()\n    2.4999999999999996\n    >>> np.sum(hist * np.diff(bin_edges))\n    1.0\n\n    .. versionadded:: 1.11.0\n\n    Automated Bin Selection Methods example, using 2 peak random data\n    with 2000 points.\n\n    .. plot::\n        :include-source:\n\n        import matplotlib.pyplot as plt\n        import numpy as np\n\n        rng = np.random.RandomState(10)  # deterministic random data\n        a = np.hstack((rng.normal(size=1000),\n                       rng.normal(loc=5, scale=2, size=1000)))\n        plt.hist(a, bins=\'auto\')  # arguments are passed to np.histogram\n        plt.title("Histogram with \'auto\' bins")\n        plt.show()\n\n    '
    (a, weights) = _ravel_and_check_weights(a, weights)
    (bin_edges, uniform_bins) = _get_bin_edges(a, bins, range, weights)
    if weights is None:
        ntype = np.dtype(np.intp)
    else:
        ntype = weights.dtype
    BLOCK = 65536
    simple_weights = weights is None or np.can_cast(weights.dtype, np.double) or np.can_cast(weights.dtype, complex)
    if uniform_bins is not None and simple_weights:
        (first_edge, last_edge, n_equal_bins) = uniform_bins
        n = np.zeros(n_equal_bins, ntype)
        norm_numerator = n_equal_bins
        norm_denom = _unsigned_subtract(last_edge, first_edge)
        for i in _range(0, len(a), BLOCK):
            tmp_a = a[i:i + BLOCK]
            if weights is None:
                tmp_w = None
            else:
                tmp_w = weights[i:i + BLOCK]
            keep = tmp_a >= first_edge
            keep &= tmp_a <= last_edge
            if not np.logical_and.reduce(keep):
                tmp_a = tmp_a[keep]
                if tmp_w is not None:
                    tmp_w = tmp_w[keep]
            tmp_a = tmp_a.astype(bin_edges.dtype, copy=False)
            f_indices = _unsigned_subtract(tmp_a, first_edge) / norm_denom * norm_numerator
            indices = f_indices.astype(np.intp)
            indices[indices == n_equal_bins] -= 1
            decrement = tmp_a < bin_edges[indices]
            indices[decrement] -= 1
            increment = (tmp_a >= bin_edges[indices + 1]) & (indices != n_equal_bins - 1)
            indices[increment] += 1
            if ntype.kind == 'c':
                n.real += np.bincount(indices, weights=tmp_w.real, minlength=n_equal_bins)
                n.imag += np.bincount(indices, weights=tmp_w.imag, minlength=n_equal_bins)
            else:
                n += np.bincount(indices, weights=tmp_w, minlength=n_equal_bins).astype(ntype)
    else:
        cum_n = np.zeros(bin_edges.shape, ntype)
        if weights is None:
            for i in _range(0, len(a), BLOCK):
                sa = np.sort(a[i:i + BLOCK])
                cum_n += _search_sorted_inclusive(sa, bin_edges)
        else:
            zero = np.zeros(1, dtype=ntype)
            for i in _range(0, len(a), BLOCK):
                tmp_a = a[i:i + BLOCK]
                tmp_w = weights[i:i + BLOCK]
                sorting_index = np.argsort(tmp_a)
                sa = tmp_a[sorting_index]
                sw = tmp_w[sorting_index]
                cw = np.concatenate((zero, sw.cumsum()))
                bin_index = _search_sorted_inclusive(sa, bin_edges)
                cum_n += cw[bin_index]
        n = np.diff(cum_n)
    if density:
        db = np.array(np.diff(bin_edges), float)
        return (n / db / n.sum(), bin_edges)
    return (n, bin_edges)

def _histogramdd_dispatcher(sample, bins=None, range=None, density=None, weights=None):
    if False:
        for i in range(10):
            print('nop')
    if hasattr(sample, 'shape'):
        yield sample
    else:
        yield from sample
    with contextlib.suppress(TypeError):
        yield from bins
    yield weights

@array_function_dispatch(_histogramdd_dispatcher)
def histogramdd(sample, bins=10, range=None, density=None, weights=None):
    if False:
        while True:
            i = 10
    '\n    Compute the multidimensional histogram of some data.\n\n    Parameters\n    ----------\n    sample : (N, D) array, or (N, D) array_like\n        The data to be histogrammed.\n\n        Note the unusual interpretation of sample when an array_like:\n\n        * When an array, each row is a coordinate in a D-dimensional space -\n          such as ``histogramdd(np.array([p1, p2, p3]))``.\n        * When an array_like, each element is the list of values for single\n          coordinate - such as ``histogramdd((X, Y, Z))``.\n\n        The first form should be preferred.\n\n    bins : sequence or int, optional\n        The bin specification:\n\n        * A sequence of arrays describing the monotonically increasing bin\n          edges along each dimension.\n        * The number of bins for each dimension (nx, ny, ... =bins)\n        * The number of bins for all dimensions (nx=ny=...=bins).\n\n    range : sequence, optional\n        A sequence of length D, each an optional (lower, upper) tuple giving\n        the outer bin edges to be used if the edges are not given explicitly in\n        `bins`.\n        An entry of None in the sequence results in the minimum and maximum\n        values being used for the corresponding dimension.\n        The default, None, is equivalent to passing a tuple of D None values.\n    density : bool, optional\n        If False, the default, returns the number of samples in each bin.\n        If True, returns the probability *density* function at the bin,\n        ``bin_count / sample_count / bin_volume``.\n    weights : (N,) array_like, optional\n        An array of values `w_i` weighing each sample `(x_i, y_i, z_i, ...)`.\n        Weights are normalized to 1 if density is True. If density is False,\n        the values of the returned histogram are equal to the sum of the\n        weights belonging to the samples falling into each bin.\n\n    Returns\n    -------\n    H : ndarray\n        The multidimensional histogram of sample x. See density and weights\n        for the different possible semantics.\n    edges : list\n        A list of D arrays describing the bin edges for each dimension.\n\n    See Also\n    --------\n    histogram: 1-D histogram\n    histogram2d: 2-D histogram\n\n    Examples\n    --------\n    >>> r = np.random.randn(100,3)\n    >>> H, edges = np.histogramdd(r, bins = (5, 8, 4))\n    >>> H.shape, edges[0].size, edges[1].size, edges[2].size\n    ((5, 8, 4), 6, 9, 5)\n\n    '
    try:
        (N, D) = sample.shape
    except (AttributeError, ValueError):
        sample = np.atleast_2d(sample).T
        (N, D) = sample.shape
    nbin = np.empty(D, np.intp)
    edges = D * [None]
    dedges = D * [None]
    if weights is not None:
        weights = np.asarray(weights)
    try:
        M = len(bins)
        if M != D:
            raise ValueError('The dimension of bins must be equal to the dimension of the sample x.')
    except TypeError:
        bins = D * [bins]
    if range is None:
        range = (None,) * D
    elif len(range) != D:
        raise ValueError('range argument must have one entry per dimension')
    for i in _range(D):
        if np.ndim(bins[i]) == 0:
            if bins[i] < 1:
                raise ValueError('`bins[{}]` must be positive, when an integer'.format(i))
            (smin, smax) = _get_outer_edges(sample[:, i], range[i])
            try:
                n = operator.index(bins[i])
            except TypeError as e:
                raise TypeError('`bins[{}]` must be an integer, when a scalar'.format(i)) from e
            edges[i] = np.linspace(smin, smax, n + 1)
        elif np.ndim(bins[i]) == 1:
            edges[i] = np.asarray(bins[i])
            if np.any(edges[i][:-1] > edges[i][1:]):
                raise ValueError('`bins[{}]` must be monotonically increasing, when an array'.format(i))
        else:
            raise ValueError('`bins[{}]` must be a scalar or 1d array'.format(i))
        nbin[i] = len(edges[i]) + 1
        dedges[i] = np.diff(edges[i])
    Ncount = tuple((np.searchsorted(edges[i], sample[:, i], side='right') for i in _range(D)))
    for i in _range(D):
        on_edge = sample[:, i] == edges[i][-1]
        Ncount[i][on_edge] -= 1
    xy = np.ravel_multi_index(Ncount, nbin)
    hist = np.bincount(xy, weights, minlength=nbin.prod())
    hist = hist.reshape(nbin)
    hist = hist.astype(float, casting='safe')
    core = D * (slice(1, -1),)
    hist = hist[core]
    if density:
        s = hist.sum()
        for i in _range(D):
            shape = np.ones(D, int)
            shape[i] = nbin[i] - 2
            hist = hist / dedges[i].reshape(shape)
        hist /= s
    if (hist.shape != nbin - 2).any():
        raise RuntimeError('Internal Shape Error')
    return (hist, edges)