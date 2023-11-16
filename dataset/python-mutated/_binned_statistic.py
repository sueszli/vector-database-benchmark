import builtins
from warnings import catch_warnings, simplefilter
import numpy as np
from operator import index
from collections import namedtuple
__all__ = ['binned_statistic', 'binned_statistic_2d', 'binned_statistic_dd']
BinnedStatisticResult = namedtuple('BinnedStatisticResult', ('statistic', 'bin_edges', 'binnumber'))

def binned_statistic(x, values, statistic='mean', bins=10, range=None):
    if False:
        while True:
            i = 10
    "\n    Compute a binned statistic for one or more sets of data.\n\n    This is a generalization of a histogram function.  A histogram divides\n    the space into bins, and returns the count of the number of points in\n    each bin.  This function allows the computation of the sum, mean, median,\n    or other statistic of the values (or set of values) within each bin.\n\n    Parameters\n    ----------\n    x : (N,) array_like\n        A sequence of values to be binned.\n    values : (N,) array_like or list of (N,) array_like\n        The data on which the statistic will be computed.  This must be\n        the same shape as `x`, or a set of sequences - each the same shape as\n        `x`.  If `values` is a set of sequences, the statistic will be computed\n        on each independently.\n    statistic : string or callable, optional\n        The statistic to compute (default is 'mean').\n        The following statistics are available:\n\n          * 'mean' : compute the mean of values for points within each bin.\n            Empty bins will be represented by NaN.\n          * 'std' : compute the standard deviation within each bin. This\n            is implicitly calculated with ddof=0.\n          * 'median' : compute the median of values for points within each\n            bin. Empty bins will be represented by NaN.\n          * 'count' : compute the count of points within each bin.  This is\n            identical to an unweighted histogram.  `values` array is not\n            referenced.\n          * 'sum' : compute the sum of values for points within each bin.\n            This is identical to a weighted histogram.\n          * 'min' : compute the minimum of values for points within each bin.\n            Empty bins will be represented by NaN.\n          * 'max' : compute the maximum of values for point within each bin.\n            Empty bins will be represented by NaN.\n          * function : a user-defined function which takes a 1D array of\n            values, and outputs a single numerical statistic. This function\n            will be called on the values in each bin.  Empty bins will be\n            represented by function([]), or NaN if this returns an error.\n\n    bins : int or sequence of scalars, optional\n        If `bins` is an int, it defines the number of equal-width bins in the\n        given range (10 by default).  If `bins` is a sequence, it defines the\n        bin edges, including the rightmost edge, allowing for non-uniform bin\n        widths.  Values in `x` that are smaller than lowest bin edge are\n        assigned to bin number 0, values beyond the highest bin are assigned to\n        ``bins[-1]``.  If the bin edges are specified, the number of bins will\n        be, (nx = len(bins)-1).\n    range : (float, float) or [(float, float)], optional\n        The lower and upper range of the bins.  If not provided, range\n        is simply ``(x.min(), x.max())``.  Values outside the range are\n        ignored.\n\n    Returns\n    -------\n    statistic : array\n        The values of the selected statistic in each bin.\n    bin_edges : array of dtype float\n        Return the bin edges ``(length(statistic)+1)``.\n    binnumber: 1-D ndarray of ints\n        Indices of the bins (corresponding to `bin_edges`) in which each value\n        of `x` belongs.  Same length as `values`.  A binnumber of `i` means the\n        corresponding value is between (bin_edges[i-1], bin_edges[i]).\n\n    See Also\n    --------\n    numpy.digitize, numpy.histogram, binned_statistic_2d, binned_statistic_dd\n\n    Notes\n    -----\n    All but the last (righthand-most) bin is half-open.  In other words, if\n    `bins` is ``[1, 2, 3, 4]``, then the first bin is ``[1, 2)`` (including 1,\n    but excluding 2) and the second ``[2, 3)``.  The last bin, however, is\n    ``[3, 4]``, which *includes* 4.\n\n    .. versionadded:: 0.11.0\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy import stats\n    >>> import matplotlib.pyplot as plt\n\n    First some basic examples:\n\n    Create two evenly spaced bins in the range of the given sample, and sum the\n    corresponding values in each of those bins:\n\n    >>> values = [1.0, 1.0, 2.0, 1.5, 3.0]\n    >>> stats.binned_statistic([1, 1, 2, 5, 7], values, 'sum', bins=2)\n    BinnedStatisticResult(statistic=array([4. , 4.5]),\n            bin_edges=array([1., 4., 7.]), binnumber=array([1, 1, 1, 2, 2]))\n\n    Multiple arrays of values can also be passed.  The statistic is calculated\n    on each set independently:\n\n    >>> values = [[1.0, 1.0, 2.0, 1.5, 3.0], [2.0, 2.0, 4.0, 3.0, 6.0]]\n    >>> stats.binned_statistic([1, 1, 2, 5, 7], values, 'sum', bins=2)\n    BinnedStatisticResult(statistic=array([[4. , 4.5],\n           [8. , 9. ]]), bin_edges=array([1., 4., 7.]),\n           binnumber=array([1, 1, 1, 2, 2]))\n\n    >>> stats.binned_statistic([1, 2, 1, 2, 4], np.arange(5), statistic='mean',\n    ...                        bins=3)\n    BinnedStatisticResult(statistic=array([1., 2., 4.]),\n            bin_edges=array([1., 2., 3., 4.]),\n            binnumber=array([1, 2, 1, 2, 3]))\n\n    As a second example, we now generate some random data of sailing boat speed\n    as a function of wind speed, and then determine how fast our boat is for\n    certain wind speeds:\n\n    >>> rng = np.random.default_rng()\n    >>> windspeed = 8 * rng.random(500)\n    >>> boatspeed = .3 * windspeed**.5 + .2 * rng.random(500)\n    >>> bin_means, bin_edges, binnumber = stats.binned_statistic(windspeed,\n    ...                 boatspeed, statistic='median', bins=[1,2,3,4,5,6,7])\n    >>> plt.figure()\n    >>> plt.plot(windspeed, boatspeed, 'b.', label='raw data')\n    >>> plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='g', lw=5,\n    ...            label='binned statistic of data')\n    >>> plt.legend()\n\n    Now we can use ``binnumber`` to select all datapoints with a windspeed\n    below 1:\n\n    >>> low_boatspeed = boatspeed[binnumber == 0]\n\n    As a final example, we will use ``bin_edges`` and ``binnumber`` to make a\n    plot of a distribution that shows the mean and distribution around that\n    mean per bin, on top of a regular histogram and the probability\n    distribution function:\n\n    >>> x = np.linspace(0, 5, num=500)\n    >>> x_pdf = stats.maxwell.pdf(x)\n    >>> samples = stats.maxwell.rvs(size=10000)\n\n    >>> bin_means, bin_edges, binnumber = stats.binned_statistic(x, x_pdf,\n    ...         statistic='mean', bins=25)\n    >>> bin_width = (bin_edges[1] - bin_edges[0])\n    >>> bin_centers = bin_edges[1:] - bin_width/2\n\n    >>> plt.figure()\n    >>> plt.hist(samples, bins=50, density=True, histtype='stepfilled',\n    ...          alpha=0.2, label='histogram of data')\n    >>> plt.plot(x, x_pdf, 'r-', label='analytical pdf')\n    >>> plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='g', lw=2,\n    ...            label='binned statistic of data')\n    >>> plt.plot((binnumber - 0.5) * bin_width, x_pdf, 'g.', alpha=0.5)\n    >>> plt.legend(fontsize=10)\n    >>> plt.show()\n\n    "
    try:
        N = len(bins)
    except TypeError:
        N = 1
    if N != 1:
        bins = [np.asarray(bins, float)]
    if range is not None:
        if len(range) == 2:
            range = [range]
    (medians, edges, binnumbers) = binned_statistic_dd([x], values, statistic, bins, range)
    return BinnedStatisticResult(medians, edges[0], binnumbers)
BinnedStatistic2dResult = namedtuple('BinnedStatistic2dResult', ('statistic', 'x_edge', 'y_edge', 'binnumber'))

def binned_statistic_2d(x, y, values, statistic='mean', bins=10, range=None, expand_binnumbers=False):
    if False:
        return 10
    "\n    Compute a bidimensional binned statistic for one or more sets of data.\n\n    This is a generalization of a histogram2d function.  A histogram divides\n    the space into bins, and returns the count of the number of points in\n    each bin.  This function allows the computation of the sum, mean, median,\n    or other statistic of the values (or set of values) within each bin.\n\n    Parameters\n    ----------\n    x : (N,) array_like\n        A sequence of values to be binned along the first dimension.\n    y : (N,) array_like\n        A sequence of values to be binned along the second dimension.\n    values : (N,) array_like or list of (N,) array_like\n        The data on which the statistic will be computed.  This must be\n        the same shape as `x`, or a list of sequences - each with the same\n        shape as `x`.  If `values` is such a list, the statistic will be\n        computed on each independently.\n    statistic : string or callable, optional\n        The statistic to compute (default is 'mean').\n        The following statistics are available:\n\n          * 'mean' : compute the mean of values for points within each bin.\n            Empty bins will be represented by NaN.\n          * 'std' : compute the standard deviation within each bin. This\n            is implicitly calculated with ddof=0.\n          * 'median' : compute the median of values for points within each\n            bin. Empty bins will be represented by NaN.\n          * 'count' : compute the count of points within each bin.  This is\n            identical to an unweighted histogram.  `values` array is not\n            referenced.\n          * 'sum' : compute the sum of values for points within each bin.\n            This is identical to a weighted histogram.\n          * 'min' : compute the minimum of values for points within each bin.\n            Empty bins will be represented by NaN.\n          * 'max' : compute the maximum of values for point within each bin.\n            Empty bins will be represented by NaN.\n          * function : a user-defined function which takes a 1D array of\n            values, and outputs a single numerical statistic. This function\n            will be called on the values in each bin.  Empty bins will be\n            represented by function([]), or NaN if this returns an error.\n\n    bins : int or [int, int] or array_like or [array, array], optional\n        The bin specification:\n\n          * the number of bins for the two dimensions (nx = ny = bins),\n          * the number of bins in each dimension (nx, ny = bins),\n          * the bin edges for the two dimensions (x_edge = y_edge = bins),\n          * the bin edges in each dimension (x_edge, y_edge = bins).\n\n        If the bin edges are specified, the number of bins will be,\n        (nx = len(x_edge)-1, ny = len(y_edge)-1).\n\n    range : (2,2) array_like, optional\n        The leftmost and rightmost edges of the bins along each dimension\n        (if not specified explicitly in the `bins` parameters):\n        [[xmin, xmax], [ymin, ymax]]. All values outside of this range will be\n        considered outliers and not tallied in the histogram.\n    expand_binnumbers : bool, optional\n        'False' (default): the returned `binnumber` is a shape (N,) array of\n        linearized bin indices.\n        'True': the returned `binnumber` is 'unraveled' into a shape (2,N)\n        ndarray, where each row gives the bin numbers in the corresponding\n        dimension.\n        See the `binnumber` returned value, and the `Examples` section.\n\n        .. versionadded:: 0.17.0\n\n    Returns\n    -------\n    statistic : (nx, ny) ndarray\n        The values of the selected statistic in each two-dimensional bin.\n    x_edge : (nx + 1) ndarray\n        The bin edges along the first dimension.\n    y_edge : (ny + 1) ndarray\n        The bin edges along the second dimension.\n    binnumber : (N,) array of ints or (2,N) ndarray of ints\n        This assigns to each element of `sample` an integer that represents the\n        bin in which this observation falls.  The representation depends on the\n        `expand_binnumbers` argument.  See `Notes` for details.\n\n\n    See Also\n    --------\n    numpy.digitize, numpy.histogram2d, binned_statistic, binned_statistic_dd\n\n    Notes\n    -----\n    Binedges:\n    All but the last (righthand-most) bin is half-open.  In other words, if\n    `bins` is ``[1, 2, 3, 4]``, then the first bin is ``[1, 2)`` (including 1,\n    but excluding 2) and the second ``[2, 3)``.  The last bin, however, is\n    ``[3, 4]``, which *includes* 4.\n\n    `binnumber`:\n    This returned argument assigns to each element of `sample` an integer that\n    represents the bin in which it belongs.  The representation depends on the\n    `expand_binnumbers` argument. If 'False' (default): The returned\n    `binnumber` is a shape (N,) array of linearized indices mapping each\n    element of `sample` to its corresponding bin (using row-major ordering).\n    Note that the returned linearized bin indices are used for an array with\n    extra bins on the outer binedges to capture values outside of the defined\n    bin bounds.\n    If 'True': The returned `binnumber` is a shape (2,N) ndarray where\n    each row indicates bin placements for each dimension respectively.  In each\n    dimension, a binnumber of `i` means the corresponding value is between\n    (D_edge[i-1], D_edge[i]), where 'D' is either 'x' or 'y'.\n\n    .. versionadded:: 0.11.0\n\n    Examples\n    --------\n    >>> from scipy import stats\n\n    Calculate the counts with explicit bin-edges:\n\n    >>> x = [0.1, 0.1, 0.1, 0.6]\n    >>> y = [2.1, 2.6, 2.1, 2.1]\n    >>> binx = [0.0, 0.5, 1.0]\n    >>> biny = [2.0, 2.5, 3.0]\n    >>> ret = stats.binned_statistic_2d(x, y, None, 'count', bins=[binx, biny])\n    >>> ret.statistic\n    array([[2., 1.],\n           [1., 0.]])\n\n    The bin in which each sample is placed is given by the `binnumber`\n    returned parameter.  By default, these are the linearized bin indices:\n\n    >>> ret.binnumber\n    array([5, 6, 5, 9])\n\n    The bin indices can also be expanded into separate entries for each\n    dimension using the `expand_binnumbers` parameter:\n\n    >>> ret = stats.binned_statistic_2d(x, y, None, 'count', bins=[binx, biny],\n    ...                                 expand_binnumbers=True)\n    >>> ret.binnumber\n    array([[1, 1, 1, 2],\n           [1, 2, 1, 1]])\n\n    Which shows that the first three elements belong in the xbin 1, and the\n    fourth into xbin 2; and so on for y.\n\n    "
    try:
        N = len(bins)
    except TypeError:
        N = 1
    if N != 1 and N != 2:
        xedges = yedges = np.asarray(bins, float)
        bins = [xedges, yedges]
    (medians, edges, binnumbers) = binned_statistic_dd([x, y], values, statistic, bins, range, expand_binnumbers=expand_binnumbers)
    return BinnedStatistic2dResult(medians, edges[0], edges[1], binnumbers)
BinnedStatisticddResult = namedtuple('BinnedStatisticddResult', ('statistic', 'bin_edges', 'binnumber'))

def _bincount(x, weights):
    if False:
        i = 10
        return i + 15
    if np.iscomplexobj(weights):
        a = np.bincount(x, np.real(weights))
        b = np.bincount(x, np.imag(weights))
        z = a + b * 1j
    else:
        z = np.bincount(x, weights)
    return z

def binned_statistic_dd(sample, values, statistic='mean', bins=10, range=None, expand_binnumbers=False, binned_statistic_result=None):
    if False:
        print('Hello World!')
    "\n    Compute a multidimensional binned statistic for a set of data.\n\n    This is a generalization of a histogramdd function.  A histogram divides\n    the space into bins, and returns the count of the number of points in\n    each bin.  This function allows the computation of the sum, mean, median,\n    or other statistic of the values within each bin.\n\n    Parameters\n    ----------\n    sample : array_like\n        Data to histogram passed as a sequence of N arrays of length D, or\n        as an (N,D) array.\n    values : (N,) array_like or list of (N,) array_like\n        The data on which the statistic will be computed.  This must be\n        the same shape as `sample`, or a list of sequences - each with the\n        same shape as `sample`.  If `values` is such a list, the statistic\n        will be computed on each independently.\n    statistic : string or callable, optional\n        The statistic to compute (default is 'mean').\n        The following statistics are available:\n\n          * 'mean' : compute the mean of values for points within each bin.\n            Empty bins will be represented by NaN.\n          * 'median' : compute the median of values for points within each\n            bin. Empty bins will be represented by NaN.\n          * 'count' : compute the count of points within each bin.  This is\n            identical to an unweighted histogram.  `values` array is not\n            referenced.\n          * 'sum' : compute the sum of values for points within each bin.\n            This is identical to a weighted histogram.\n          * 'std' : compute the standard deviation within each bin. This\n            is implicitly calculated with ddof=0. If the number of values\n            within a given bin is 0 or 1, the computed standard deviation value\n            will be 0 for the bin.\n          * 'min' : compute the minimum of values for points within each bin.\n            Empty bins will be represented by NaN.\n          * 'max' : compute the maximum of values for point within each bin.\n            Empty bins will be represented by NaN.\n          * function : a user-defined function which takes a 1D array of\n            values, and outputs a single numerical statistic. This function\n            will be called on the values in each bin.  Empty bins will be\n            represented by function([]), or NaN if this returns an error.\n\n    bins : sequence or positive int, optional\n        The bin specification must be in one of the following forms:\n\n          * A sequence of arrays describing the bin edges along each dimension.\n          * The number of bins for each dimension (nx, ny, ... = bins).\n          * The number of bins for all dimensions (nx = ny = ... = bins).\n    range : sequence, optional\n        A sequence of lower and upper bin edges to be used if the edges are\n        not given explicitly in `bins`. Defaults to the minimum and maximum\n        values along each dimension.\n    expand_binnumbers : bool, optional\n        'False' (default): the returned `binnumber` is a shape (N,) array of\n        linearized bin indices.\n        'True': the returned `binnumber` is 'unraveled' into a shape (D,N)\n        ndarray, where each row gives the bin numbers in the corresponding\n        dimension.\n        See the `binnumber` returned value, and the `Examples` section of\n        `binned_statistic_2d`.\n    binned_statistic_result : binnedStatisticddResult\n        Result of a previous call to the function in order to reuse bin edges\n        and bin numbers with new values and/or a different statistic.\n        To reuse bin numbers, `expand_binnumbers` must have been set to False\n        (the default)\n\n        .. versionadded:: 0.17.0\n\n    Returns\n    -------\n    statistic : ndarray, shape(nx1, nx2, nx3,...)\n        The values of the selected statistic in each two-dimensional bin.\n    bin_edges : list of ndarrays\n        A list of D arrays describing the (nxi + 1) bin edges for each\n        dimension.\n    binnumber : (N,) array of ints or (D,N) ndarray of ints\n        This assigns to each element of `sample` an integer that represents the\n        bin in which this observation falls.  The representation depends on the\n        `expand_binnumbers` argument.  See `Notes` for details.\n\n\n    See Also\n    --------\n    numpy.digitize, numpy.histogramdd, binned_statistic, binned_statistic_2d\n\n    Notes\n    -----\n    Binedges:\n    All but the last (righthand-most) bin is half-open in each dimension.  In\n    other words, if `bins` is ``[1, 2, 3, 4]``, then the first bin is\n    ``[1, 2)`` (including 1, but excluding 2) and the second ``[2, 3)``.  The\n    last bin, however, is ``[3, 4]``, which *includes* 4.\n\n    `binnumber`:\n    This returned argument assigns to each element of `sample` an integer that\n    represents the bin in which it belongs.  The representation depends on the\n    `expand_binnumbers` argument. If 'False' (default): The returned\n    `binnumber` is a shape (N,) array of linearized indices mapping each\n    element of `sample` to its corresponding bin (using row-major ordering).\n    If 'True': The returned `binnumber` is a shape (D,N) ndarray where\n    each row indicates bin placements for each dimension respectively.  In each\n    dimension, a binnumber of `i` means the corresponding value is between\n    (bin_edges[D][i-1], bin_edges[D][i]), for each dimension 'D'.\n\n    .. versionadded:: 0.11.0\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy import stats\n    >>> import matplotlib.pyplot as plt\n    >>> from mpl_toolkits.mplot3d import Axes3D\n\n    Take an array of 600 (x, y) coordinates as an example.\n    `binned_statistic_dd` can handle arrays of higher dimension `D`. But a plot\n    of dimension `D+1` is required.\n\n    >>> mu = np.array([0., 1.])\n    >>> sigma = np.array([[1., -0.5],[-0.5, 1.5]])\n    >>> multinormal = stats.multivariate_normal(mu, sigma)\n    >>> data = multinormal.rvs(size=600, random_state=235412)\n    >>> data.shape\n    (600, 2)\n\n    Create bins and count how many arrays fall in each bin:\n\n    >>> N = 60\n    >>> x = np.linspace(-3, 3, N)\n    >>> y = np.linspace(-3, 4, N)\n    >>> ret = stats.binned_statistic_dd(data, np.arange(600), bins=[x, y],\n    ...                                 statistic='count')\n    >>> bincounts = ret.statistic\n\n    Set the volume and the location of bars:\n\n    >>> dx = x[1] - x[0]\n    >>> dy = y[1] - y[0]\n    >>> x, y = np.meshgrid(x[:-1]+dx/2, y[:-1]+dy/2)\n    >>> z = 0\n\n    >>> bincounts = bincounts.ravel()\n    >>> x = x.ravel()\n    >>> y = y.ravel()\n\n    >>> fig = plt.figure()\n    >>> ax = fig.add_subplot(111, projection='3d')\n    >>> with np.errstate(divide='ignore'):   # silence random axes3d warning\n    ...     ax.bar3d(x, y, z, dx, dy, bincounts)\n\n    Reuse bin numbers and bin edges with new values:\n\n    >>> ret2 = stats.binned_statistic_dd(data, -np.arange(600),\n    ...                                  binned_statistic_result=ret,\n    ...                                  statistic='mean')\n    "
    known_stats = ['mean', 'median', 'count', 'sum', 'std', 'min', 'max']
    if not callable(statistic) and statistic not in known_stats:
        raise ValueError(f'invalid statistic {statistic!r}')
    try:
        bins = index(bins)
    except TypeError:
        pass
    if isinstance(bins, int) and (not np.isfinite(sample).all()):
        raise ValueError(f'{sample!r} contains non-finite values.')
    try:
        (Dlen, Ndim) = sample.shape
    except (AttributeError, ValueError):
        sample = np.atleast_2d(sample).T
        (Dlen, Ndim) = sample.shape
    values = np.asarray(values)
    input_shape = list(values.shape)
    values = np.atleast_2d(values)
    (Vdim, Vlen) = values.shape
    if statistic != 'count' and Vlen != Dlen:
        raise AttributeError('The number of `values` elements must match the length of each `sample` dimension.')
    try:
        M = len(bins)
        if M != Ndim:
            raise AttributeError('The dimension of bins must be equal to the dimension of the sample x.')
    except TypeError:
        bins = Ndim * [bins]
    if binned_statistic_result is None:
        (nbin, edges, dedges) = _bin_edges(sample, bins, range)
        binnumbers = _bin_numbers(sample, nbin, edges, dedges)
    else:
        edges = binned_statistic_result.bin_edges
        nbin = np.array([len(edges[i]) + 1 for i in builtins.range(Ndim)])
        dedges = [np.diff(edges[i]) for i in builtins.range(Ndim)]
        binnumbers = binned_statistic_result.binnumber
    result_type = np.result_type(values, np.float64)
    result = np.empty([Vdim, nbin.prod()], dtype=result_type)
    if statistic in {'mean', np.mean}:
        result.fill(np.nan)
        flatcount = _bincount(binnumbers, None)
        a = flatcount.nonzero()
        for vv in builtins.range(Vdim):
            flatsum = _bincount(binnumbers, values[vv])
            result[vv, a] = flatsum[a] / flatcount[a]
    elif statistic in {'std', np.std}:
        result.fill(np.nan)
        flatcount = _bincount(binnumbers, None)
        a = flatcount.nonzero()
        for vv in builtins.range(Vdim):
            flatsum = _bincount(binnumbers, values[vv])
            delta = values[vv] - flatsum[binnumbers] / flatcount[binnumbers]
            std = np.sqrt(_bincount(binnumbers, delta * np.conj(delta))[a] / flatcount[a])
            result[vv, a] = std
        result = np.real(result)
    elif statistic == 'count':
        result = np.empty([Vdim, nbin.prod()], dtype=np.float64)
        result.fill(0)
        flatcount = _bincount(binnumbers, None)
        a = np.arange(len(flatcount))
        result[:, a] = flatcount[np.newaxis, :]
    elif statistic in {'sum', np.sum}:
        result.fill(0)
        for vv in builtins.range(Vdim):
            flatsum = _bincount(binnumbers, values[vv])
            a = np.arange(len(flatsum))
            result[vv, a] = flatsum
    elif statistic in {'median', np.median}:
        result.fill(np.nan)
        for vv in builtins.range(Vdim):
            i = np.lexsort((values[vv], binnumbers))
            (_, j, counts) = np.unique(binnumbers[i], return_index=True, return_counts=True)
            mid = j + (counts - 1) / 2
            mid_a = values[vv, i][np.floor(mid).astype(int)]
            mid_b = values[vv, i][np.ceil(mid).astype(int)]
            medians = (mid_a + mid_b) / 2
            result[vv, binnumbers[i][j]] = medians
    elif statistic in {'min', np.min}:
        result.fill(np.nan)
        for vv in builtins.range(Vdim):
            i = np.argsort(values[vv])[::-1]
            result[vv, binnumbers[i]] = values[vv, i]
    elif statistic in {'max', np.max}:
        result.fill(np.nan)
        for vv in builtins.range(Vdim):
            i = np.argsort(values[vv])
            result[vv, binnumbers[i]] = values[vv, i]
    elif callable(statistic):
        with np.errstate(invalid='ignore'), catch_warnings():
            simplefilter('ignore', RuntimeWarning)
            try:
                null = statistic([])
            except Exception:
                null = np.nan
        if np.iscomplexobj(null):
            result = result.astype(np.complex128)
        result.fill(null)
        try:
            _calc_binned_statistic(Vdim, binnumbers, result, values, statistic)
        except ValueError:
            result = result.astype(np.complex128)
            _calc_binned_statistic(Vdim, binnumbers, result, values, statistic)
    result = result.reshape(np.append(Vdim, nbin))
    core = tuple([slice(None)] + Ndim * [slice(1, -1)])
    result = result[core]
    if expand_binnumbers and Ndim > 1:
        binnumbers = np.asarray(np.unravel_index(binnumbers, nbin))
    if np.any(result.shape[1:] != nbin - 2):
        raise RuntimeError('Internal Shape Error')
    result = result.reshape(input_shape[:-1] + list(nbin - 2))
    return BinnedStatisticddResult(result, edges, binnumbers)

def _calc_binned_statistic(Vdim, bin_numbers, result, values, stat_func):
    if False:
        print('Hello World!')
    unique_bin_numbers = np.unique(bin_numbers)
    for vv in builtins.range(Vdim):
        bin_map = _create_binned_data(bin_numbers, unique_bin_numbers, values, vv)
        for i in unique_bin_numbers:
            stat = stat_func(np.array(bin_map[i]))
            if np.iscomplexobj(stat) and (not np.iscomplexobj(result)):
                raise ValueError('The statistic function returns complex ')
            result[vv, i] = stat

def _create_binned_data(bin_numbers, unique_bin_numbers, values, vv):
    if False:
        print('Hello World!')
    ' Create hashmap of bin ids to values in bins\n    key: bin number\n    value: list of binned data\n    '
    bin_map = dict()
    for i in unique_bin_numbers:
        bin_map[i] = []
    for i in builtins.range(len(bin_numbers)):
        bin_map[bin_numbers[i]].append(values[vv, i])
    return bin_map

def _bin_edges(sample, bins=None, range=None):
    if False:
        for i in range(10):
            print('nop')
    ' Create edge arrays\n    '
    (Dlen, Ndim) = sample.shape
    nbin = np.empty(Ndim, int)
    edges = Ndim * [None]
    dedges = Ndim * [None]
    if range is None:
        smin = np.atleast_1d(np.array(sample.min(axis=0), float))
        smax = np.atleast_1d(np.array(sample.max(axis=0), float))
    else:
        if len(range) != Ndim:
            raise ValueError(f'range given for {len(range)} dimensions; {Ndim} required')
        smin = np.empty(Ndim)
        smax = np.empty(Ndim)
        for i in builtins.range(Ndim):
            if range[i][1] < range[i][0]:
                raise ValueError('In {}range, start must be <= stop'.format(f'dimension {i + 1} of ' if Ndim > 1 else ''))
            (smin[i], smax[i]) = range[i]
    for i in builtins.range(len(smin)):
        if smin[i] == smax[i]:
            smin[i] = smin[i] - 0.5
            smax[i] = smax[i] + 0.5
    edges_dtype = sample.dtype if np.issubdtype(sample.dtype, np.floating) else float
    for i in builtins.range(Ndim):
        if np.isscalar(bins[i]):
            nbin[i] = bins[i] + 2
            edges[i] = np.linspace(smin[i], smax[i], nbin[i] - 1, dtype=edges_dtype)
        else:
            edges[i] = np.asarray(bins[i], edges_dtype)
            nbin[i] = len(edges[i]) + 1
        dedges[i] = np.diff(edges[i])
    nbin = np.asarray(nbin)
    return (nbin, edges, dedges)

def _bin_numbers(sample, nbin, edges, dedges):
    if False:
        while True:
            i = 10
    'Compute the bin number each sample falls into, in each dimension\n    '
    (Dlen, Ndim) = sample.shape
    sampBin = [np.digitize(sample[:, i], edges[i]) for i in range(Ndim)]
    for i in range(Ndim):
        dedges_min = dedges[i].min()
        if dedges_min == 0:
            raise ValueError('The smallest edge difference is numerically 0.')
        decimal = int(-np.log10(dedges_min)) + 6
        on_edge = np.where((sample[:, i] >= edges[i][-1]) & (np.around(sample[:, i], decimal) == np.around(edges[i][-1], decimal)))[0]
        sampBin[i][on_edge] -= 1
    binnumbers = np.ravel_multi_index(sampBin, nbin)
    return binnumbers