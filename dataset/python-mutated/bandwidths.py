import numpy as np
from scipy.stats import scoreatpercentile
from statsmodels.compat.pandas import Substitution
from statsmodels.sandbox.nonparametric import kernels

def _select_sigma(x, percentile=25):
    if False:
        i = 10
        return i + 15
    '\n    Returns the smaller of std(X, ddof=1) or normalized IQR(X) over axis 0.\n\n    References\n    ----------\n    Silverman (1986) p.47\n    '
    normalize = 1.349
    IQR = (scoreatpercentile(x, 75) - scoreatpercentile(x, 25)) / normalize
    std_dev = np.std(x, axis=0, ddof=1)
    if IQR > 0:
        return np.minimum(std_dev, IQR)
    else:
        return std_dev

def bw_scott(x, kernel=None):
    if False:
        print('Hello World!')
    "\n    Scott's Rule of Thumb\n\n    Parameters\n    ----------\n    x : array_like\n        Array for which to get the bandwidth\n    kernel : CustomKernel object\n        Unused\n\n    Returns\n    -------\n    bw : float\n        The estimate of the bandwidth\n\n    Notes\n    -----\n    Returns 1.059 * A * n ** (-1/5.) where ::\n\n       A = min(std(x, ddof=1), IQR/1.349)\n       IQR = np.subtract.reduce(np.percentile(x, [75,25]))\n\n    References\n    ----------\n\n    Scott, D.W. (1992) Multivariate Density Estimation: Theory, Practice, and\n        Visualization.\n    "
    A = _select_sigma(x)
    n = len(x)
    return 1.059 * A * n ** (-0.2)

def bw_silverman(x, kernel=None):
    if False:
        while True:
            i = 10
    "\n    Silverman's Rule of Thumb\n\n    Parameters\n    ----------\n    x : array_like\n        Array for which to get the bandwidth\n    kernel : CustomKernel object\n        Unused\n\n    Returns\n    -------\n    bw : float\n        The estimate of the bandwidth\n\n    Notes\n    -----\n    Returns .9 * A * n ** (-1/5.) where ::\n\n       A = min(std(x, ddof=1), IQR/1.349)\n       IQR = np.subtract.reduce(np.percentile(x, [75,25]))\n\n    References\n    ----------\n\n    Silverman, B.W. (1986) `Density Estimation.`\n    "
    A = _select_sigma(x)
    n = len(x)
    return 0.9 * A * n ** (-0.2)

def bw_normal_reference(x, kernel=None):
    if False:
        return 10
    "\n    Plug-in bandwidth with kernel specific constant based on normal reference.\n\n    This bandwidth minimizes the mean integrated square error if the true\n    distribution is the normal. This choice is an appropriate bandwidth for\n    single peaked distributions that are similar to the normal distribution.\n\n    Parameters\n    ----------\n    x : array_like\n        Array for which to get the bandwidth\n    kernel : CustomKernel object\n        Used to calculate the constant for the plug-in bandwidth.\n        The default is a Gaussian kernel.\n\n    Returns\n    -------\n    bw : float\n        The estimate of the bandwidth\n\n    Notes\n    -----\n    Returns C * A * n ** (-1/5.) where ::\n\n       A = min(std(x, ddof=1), IQR/1.349)\n       IQR = np.subtract.reduce(np.percentile(x, [75,25]))\n       C = constant from Hansen (2009)\n\n    When using a Gaussian kernel this is equivalent to the 'scott' bandwidth up\n    to two decimal places. This is the accuracy to which the 'scott' constant is\n    specified.\n\n    References\n    ----------\n\n    Silverman, B.W. (1986) `Density Estimation.`\n    Hansen, B.E. (2009) `Lecture Notes on Nonparametrics.`\n    "
    if kernel is None:
        kernel = kernels.Gaussian()
    C = kernel.normal_reference_constant
    A = _select_sigma(x)
    n = len(x)
    return C * A * n ** (-0.2)
bandwidth_funcs = {'scott': bw_scott, 'silverman': bw_silverman, 'normal_reference': bw_normal_reference}

@Substitution(', '.join(sorted(bandwidth_funcs.keys())))
def select_bandwidth(x, bw, kernel):
    if False:
        i = 10
        return i + 15
    '\n    Selects bandwidth for a selection rule bw\n\n    this is a wrapper around existing bandwidth selection rules\n\n    Parameters\n    ----------\n    x : array_like\n        Array for which to get the bandwidth\n    bw : str\n        name of bandwidth selection rule, currently supported are:\n        %s\n    kernel : not used yet\n\n    Returns\n    -------\n    bw : float\n        The estimate of the bandwidth\n    '
    bw = bw.lower()
    if bw not in bandwidth_funcs:
        raise ValueError('Bandwidth %s not understood' % bw)
    bandwidth = bandwidth_funcs[bw](x, kernel)
    if np.any(bandwidth == 0):
        err = 'Selected KDE bandwidth is 0. Cannot estimate density. Either provide the bandwidth during initialization or use an alternative method.'
        raise RuntimeError(err)
    else:
        return bandwidth