"""
This module contains simple functions for dealing with circular statistics, for
instance, mean, variance, standard deviation, correlation coefficient, and so
on. This module also cover tests of uniformity, e.g., the Rayleigh and V tests.
The Maximum Likelihood Estimator for the Von Mises distribution along with the
Cramer-Rao Lower Bounds are also implemented. Almost all of the implementations
are based on reference [1]_, which is also the basis for the R package
'CircStats' [2]_.
"""
import numpy as np
from astropy.units import Quantity
__all__ = ['circmean', 'circstd', 'circvar', 'circmoment', 'circcorrcoef', 'rayleightest', 'vtest', 'vonmisesmle']
__doctest_requires__ = {'vtest': ['scipy']}

def _components(data, p=1, phi=0.0, axis=None, weights=None):
    if False:
        for i in range(10):
            print('nop')
    if weights is None:
        weights = np.ones((1,))
    try:
        weights = np.broadcast_to(weights, data.shape)
    except ValueError:
        raise ValueError('Weights and data have inconsistent shape.')
    C = np.sum(weights * np.cos(p * (data - phi)), axis) / np.sum(weights, axis)
    S = np.sum(weights * np.sin(p * (data - phi)), axis) / np.sum(weights, axis)
    return (C, S)

def _angle(data, p=1, phi=0.0, axis=None, weights=None):
    if False:
        print('Hello World!')
    (C, S) = _components(data, p, phi, axis, weights)
    theta = np.arctan2(S, C)
    if isinstance(data, Quantity):
        theta = theta.to(data.unit)
    return theta

def _length(data, p=1, phi=0.0, axis=None, weights=None):
    if False:
        print('Hello World!')
    (C, S) = _components(data, p, phi, axis, weights)
    return np.hypot(S, C)

def circmean(data, axis=None, weights=None):
    if False:
        while True:
            i = 10
    'Computes the circular mean angle of an array of circular data.\n\n    Parameters\n    ----------\n    data : ndarray or `~astropy.units.Quantity`\n        Array of circular (directional) data, which is assumed to be in\n        radians whenever ``data`` is ``numpy.ndarray``.\n    axis : int, optional\n        Axis along which circular means are computed. The default is to compute\n        the mean of the flattened array.\n    weights : numpy.ndarray, optional\n        In case of grouped data, the i-th element of ``weights`` represents a\n        weighting factor for each group such that ``sum(weights, axis)``\n        equals the number of observations. See [1]_, remark 1.4, page 22, for\n        detailed explanation.\n\n    Returns\n    -------\n    circmean : ndarray or `~astropy.units.Quantity`\n        Circular mean.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from astropy.stats import circmean\n    >>> from astropy import units as u\n    >>> data = np.array([51, 67, 40, 109, 31, 358])*u.deg\n    >>> circmean(data) # doctest: +FLOAT_CMP\n    <Quantity 48.62718088722989 deg>\n\n    References\n    ----------\n    .. [1] S. R. Jammalamadaka, A. SenGupta. "Topics in Circular Statistics".\n       Series on Multivariate Analysis, Vol. 5, 2001.\n    .. [2] C. Agostinelli, U. Lund. "Circular Statistics from \'Topics in\n       Circular Statistics (2001)\'". 2015.\n       <https://cran.r-project.org/web/packages/CircStats/CircStats.pdf>\n    '
    return _angle(data, 1, 0.0, axis, weights)

def circvar(data, axis=None, weights=None):
    if False:
        for i in range(10):
            print('nop')
    'Computes the circular variance of an array of circular data.\n\n    There are some concepts for defining measures of dispersion for circular\n    data. The variance implemented here is based on the definition given by\n    [1]_, which is also the same used by the R package \'CircStats\' [2]_.\n\n    Parameters\n    ----------\n    data : ndarray or `~astropy.units.Quantity`\n        Array of circular (directional) data, which is assumed to be in\n        radians whenever ``data`` is ``numpy.ndarray``.\n        Dimensionless, if Quantity.\n    axis : int, optional\n        Axis along which circular variances are computed. The default is to\n        compute the variance of the flattened array.\n    weights : numpy.ndarray, optional\n        In case of grouped data, the i-th element of ``weights`` represents a\n        weighting factor for each group such that ``sum(weights, axis)``\n        equals the number of observations. See [1]_, remark 1.4, page 22,\n        for detailed explanation.\n\n    Returns\n    -------\n    circvar : ndarray or `~astropy.units.Quantity` [\'dimensionless\']\n        Circular variance.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from astropy.stats import circvar\n    >>> from astropy import units as u\n    >>> data = np.array([51, 67, 40, 109, 31, 358])*u.deg\n    >>> circvar(data) # doctest: +FLOAT_CMP\n    <Quantity 0.16356352748437508>\n\n    References\n    ----------\n    .. [1] S. R. Jammalamadaka, A. SenGupta. "Topics in Circular Statistics".\n       Series on Multivariate Analysis, Vol. 5, 2001.\n    .. [2] C. Agostinelli, U. Lund. "Circular Statistics from \'Topics in\n       Circular Statistics (2001)\'". 2015.\n       <https://cran.r-project.org/web/packages/CircStats/CircStats.pdf>\n\n    Notes\n    -----\n    For Scipy < 1.9.0, ``scipy.stats.circvar`` uses a different\n    definition based on an approximation using the limit of small\n    angles that approaches the linear variance. For Scipy >= 1.9.0,\n    ``scipy.stats.cirvar`` uses a definition consistent with this\n    implementation.\n    '
    return 1.0 - _length(data, 1, 0.0, axis, weights)

def circstd(data, axis=None, weights=None, method='angular'):
    if False:
        for i in range(10):
            print('nop')
    'Computes the circular standard deviation of an array of circular data.\n\n    The standard deviation implemented here is based on the definitions given\n    by [1]_, which is also the same used by the R package \'CirStat\' [2]_.\n\n    Two methods are implemented: \'angular\' and \'circular\'. The former is\n    defined as sqrt(2 * (1 - R)) and it is bounded in [0, 2*Pi]. The\n    latter is defined as sqrt(-2 * ln(R)) and it is bounded in [0, inf].\n\n    Following \'CircStat\' the default method used to obtain the standard\n    deviation is \'angular\'.\n\n    Parameters\n    ----------\n    data : ndarray or `~astropy.units.Quantity`\n        Array of circular (directional) data, which is assumed to be in\n        radians whenever ``data`` is ``numpy.ndarray``.\n        If quantity, must be dimensionless.\n    axis : int, optional\n        Axis along which circular variances are computed. The default is to\n        compute the variance of the flattened array.\n    weights : numpy.ndarray, optional\n        In case of grouped data, the i-th element of ``weights`` represents a\n        weighting factor for each group such that ``sum(weights, axis)``\n        equals the number of observations. See [3]_, remark 1.4, page 22,\n        for detailed explanation.\n    method : str, optional\n        The method used to estimate the standard deviation:\n\n        - \'angular\' : obtains the angular deviation\n\n        - \'circular\' : obtains the circular deviation\n\n\n    Returns\n    -------\n    circstd : ndarray or `~astropy.units.Quantity` [\'dimensionless\']\n        Angular or circular standard deviation.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from astropy.stats import circstd\n    >>> from astropy import units as u\n    >>> data = np.array([51, 67, 40, 109, 31, 358])*u.deg\n    >>> circstd(data) # doctest: +FLOAT_CMP\n    <Quantity 0.57195022>\n\n    Alternatively, using the \'circular\' method:\n\n    >>> import numpy as np\n    >>> from astropy.stats import circstd\n    >>> from astropy import units as u\n    >>> data = np.array([51, 67, 40, 109, 31, 358])*u.deg\n    >>> circstd(data, method=\'circular\') # doctest: +FLOAT_CMP\n    <Quantity 0.59766999>\n\n    References\n    ----------\n    .. [1] P. Berens. "CircStat: A MATLAB Toolbox for Circular Statistics".\n       Journal of Statistical Software, vol 31, issue 10, 2009.\n    .. [2] C. Agostinelli, U. Lund. "Circular Statistics from \'Topics in\n       Circular Statistics (2001)\'". 2015.\n       <https://cran.r-project.org/web/packages/CircStats/CircStats.pdf>\n    .. [3] S. R. Jammalamadaka, A. SenGupta. "Topics in Circular Statistics".\n       Series on Multivariate Analysis, Vol. 5, 2001.\n\n    '
    if method not in ('angular', 'circular'):
        raise ValueError("method should be either 'angular' or 'circular'")
    if method == 'angular':
        return np.sqrt(2.0 * (1.0 - _length(data, 1, 0.0, axis, weights)))
    else:
        return np.sqrt(-2.0 * np.log(_length(data, 1, 0.0, axis, weights)))

def circmoment(data, p=1.0, centered=False, axis=None, weights=None):
    if False:
        while True:
            i = 10
    'Computes the ``p``-th trigonometric circular moment for an array\n    of circular data.\n\n    Parameters\n    ----------\n    data : ndarray or `~astropy.units.Quantity`\n        Array of circular (directional) data, which is assumed to be in\n        radians whenever ``data`` is ``numpy.ndarray``.\n    p : float, optional\n        Order of the circular moment.\n    centered : bool, optional\n        If ``True``, central circular moments are computed. Default value is\n        ``False``.\n    axis : int, optional\n        Axis along which circular moments are computed. The default is to\n        compute the circular moment of the flattened array.\n    weights : numpy.ndarray, optional\n        In case of grouped data, the i-th element of ``weights`` represents a\n        weighting factor for each group such that ``sum(weights, axis)``\n        equals the number of observations. See [1]_, remark 1.4, page 22,\n        for detailed explanation.\n\n    Returns\n    -------\n    circmoment : ndarray or `~astropy.units.Quantity`\n        The first and second elements correspond to the direction and length of\n        the ``p``-th circular moment, respectively.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from astropy.stats import circmoment\n    >>> from astropy import units as u\n    >>> data = np.array([51, 67, 40, 109, 31, 358])*u.deg\n    >>> circmoment(data, p=2) # doctest: +FLOAT_CMP\n    (<Quantity 90.99263082432564 deg>, <Quantity 0.48004283892950717>)\n\n    References\n    ----------\n    .. [1] S. R. Jammalamadaka, A. SenGupta. "Topics in Circular Statistics".\n       Series on Multivariate Analysis, Vol. 5, 2001.\n    .. [2] C. Agostinelli, U. Lund. "Circular Statistics from \'Topics in\n       Circular Statistics (2001)\'". 2015.\n       <https://cran.r-project.org/web/packages/CircStats/CircStats.pdf>\n    '
    if centered:
        phi = circmean(data, axis, weights)
    else:
        phi = 0.0
    return (_angle(data, p, phi, axis, weights), _length(data, p, phi, axis, weights))

def circcorrcoef(alpha, beta, axis=None, weights_alpha=None, weights_beta=None):
    if False:
        return 10
    'Computes the circular correlation coefficient between two array of\n    circular data.\n\n    Parameters\n    ----------\n    alpha : ndarray or `~astropy.units.Quantity`\n        Array of circular (directional) data, which is assumed to be in\n        radians whenever ``data`` is ``numpy.ndarray``.\n    beta : ndarray or `~astropy.units.Quantity`\n        Array of circular (directional) data, which is assumed to be in\n        radians whenever ``data`` is ``numpy.ndarray``.\n    axis : int, optional\n        Axis along which circular correlation coefficients are computed.\n        The default is the compute the circular correlation coefficient of the\n        flattened array.\n    weights_alpha : numpy.ndarray, optional\n        In case of grouped data, the i-th element of ``weights_alpha``\n        represents a weighting factor for each group such that\n        ``sum(weights_alpha, axis)`` equals the number of observations.\n        See [1]_, remark 1.4, page 22, for detailed explanation.\n    weights_beta : numpy.ndarray, optional\n        See description of ``weights_alpha``.\n\n    Returns\n    -------\n    rho : ndarray or `~astropy.units.Quantity` [\'dimensionless\']\n        Circular correlation coefficient.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from astropy.stats import circcorrcoef\n    >>> from astropy import units as u\n    >>> alpha = np.array([356, 97, 211, 232, 343, 292, 157, 302, 335, 302,\n    ...                   324, 85, 324, 340, 157, 238, 254, 146, 232, 122,\n    ...                   329])*u.deg\n    >>> beta = np.array([119, 162, 221, 259, 270, 29, 97, 292, 40, 313, 94,\n    ...                  45, 47, 108, 221, 270, 119, 248, 270, 45, 23])*u.deg\n    >>> circcorrcoef(alpha, beta) # doctest: +FLOAT_CMP\n    <Quantity 0.2704648826748831>\n\n    References\n    ----------\n    .. [1] S. R. Jammalamadaka, A. SenGupta. "Topics in Circular Statistics".\n       Series on Multivariate Analysis, Vol. 5, 2001.\n    .. [2] C. Agostinelli, U. Lund. "Circular Statistics from \'Topics in\n       Circular Statistics (2001)\'". 2015.\n       <https://cran.r-project.org/web/packages/CircStats/CircStats.pdf>\n    '
    if np.size(alpha, axis) != np.size(beta, axis):
        raise ValueError('alpha and beta must be arrays of the same size')
    mu_a = circmean(alpha, axis, weights_alpha)
    mu_b = circmean(beta, axis, weights_beta)
    sin_a = np.sin(alpha - mu_a)
    sin_b = np.sin(beta - mu_b)
    rho = np.sum(sin_a * sin_b) / np.sqrt(np.sum(sin_a * sin_a) * np.sum(sin_b * sin_b))
    return rho

def rayleightest(data, axis=None, weights=None):
    if False:
        return 10
    'Performs the Rayleigh test of uniformity.\n\n    This test is  used to identify a non-uniform distribution, i.e. it is\n    designed for detecting an unimodal deviation from uniformity. More\n    precisely, it assumes the following hypotheses:\n    - H0 (null hypothesis): The population is distributed uniformly around the\n    circle.\n    - H1 (alternative hypothesis): The population is not distributed uniformly\n    around the circle.\n    Small p-values suggest to reject the null hypothesis.\n\n    Parameters\n    ----------\n    data : ndarray or `~astropy.units.Quantity`\n        Array of circular (directional) data, which is assumed to be in\n        radians whenever ``data`` is ``numpy.ndarray``.\n    axis : int, optional\n        Axis along which the Rayleigh test will be performed.\n    weights : numpy.ndarray, optional\n        In case of grouped data, the i-th element of ``weights`` represents a\n        weighting factor for each group such that ``np.sum(weights, axis)``\n        equals the number of observations.\n        See [1]_, remark 1.4, page 22, for detailed explanation.\n\n    Returns\n    -------\n    p-value : float or `~astropy.units.Quantity` [\'dimensionless\']\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from astropy.stats import rayleightest\n    >>> from astropy import units as u\n    >>> data = np.array([130, 90, 0, 145])*u.deg\n    >>> rayleightest(data) # doctest: +FLOAT_CMP\n    <Quantity 0.2563487733797317>\n\n    References\n    ----------\n    .. [1] S. R. Jammalamadaka, A. SenGupta. "Topics in Circular Statistics".\n       Series on Multivariate Analysis, Vol. 5, 2001.\n    .. [2] C. Agostinelli, U. Lund. "Circular Statistics from \'Topics in\n       Circular Statistics (2001)\'". 2015.\n       <https://cran.r-project.org/web/packages/CircStats/CircStats.pdf>\n    .. [3] M. Chirstman., C. Miller. "Testing a Sample of Directions for\n       Uniformity." Lecture Notes, STA 6934/5805. University of Florida, 2007.\n    .. [4] D. Wilkie. "Rayleigh Test for Randomness of Circular Data". Applied\n       Statistics. 1983.\n       <http://wexler.free.fr/library/files/wilkie%20(1983)%20rayleigh%20test%20for%20randomness%20of%20circular%20data.pdf>\n    '
    n = np.size(data, axis=axis)
    Rbar = _length(data, 1, 0.0, axis, weights)
    z = n * Rbar * Rbar
    tmp = 1.0
    if n < 50:
        tmp = 1.0 + (2.0 * z - z * z) / (4.0 * n) - (24.0 * z - 132.0 * z ** 2.0 + 76.0 * z ** 3.0 - 9.0 * z ** 4.0) / (288.0 * n * n)
    p_value = np.exp(-z) * tmp
    return p_value

def vtest(data, mu=0.0, axis=None, weights=None):
    if False:
        for i in range(10):
            print('nop')
    'Performs the Rayleigh test of uniformity where the alternative\n    hypothesis H1 is assumed to have a known mean angle ``mu``.\n\n    Parameters\n    ----------\n    data : ndarray or `~astropy.units.Quantity`\n        Array of circular (directional) data, which is assumed to be in\n        radians whenever ``data`` is ``numpy.ndarray``.\n    mu : float or `~astropy.units.Quantity` [\'angle\'], optional\n        Mean angle. Assumed to be known.\n    axis : int, optional\n        Axis along which the V test will be performed.\n    weights : numpy.ndarray, optional\n        In case of grouped data, the i-th element of ``weights`` represents a\n        weighting factor for each group such that ``sum(weights, axis)``\n        equals the number of observations. See [1]_, remark 1.4, page 22,\n        for detailed explanation.\n\n    Returns\n    -------\n    p-value : float or `~astropy.units.Quantity` [\'dimensionless\']\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from astropy.stats import vtest\n    >>> from astropy import units as u\n    >>> data = np.array([130, 90, 0, 145])*u.deg\n    >>> vtest(data) # doctest: +FLOAT_CMP\n    <Quantity 0.6223678199713766>\n\n    References\n    ----------\n    .. [1] S. R. Jammalamadaka, A. SenGupta. "Topics in Circular Statistics".\n       Series on Multivariate Analysis, Vol. 5, 2001.\n    .. [2] C. Agostinelli, U. Lund. "Circular Statistics from \'Topics in\n       Circular Statistics (2001)\'". 2015.\n       <https://cran.r-project.org/web/packages/CircStats/CircStats.pdf>\n    .. [3] M. Chirstman., C. Miller. "Testing a Sample of Directions for\n       Uniformity." Lecture Notes, STA 6934/5805. University of Florida, 2007.\n    '
    from scipy.stats import norm
    if weights is None:
        weights = np.ones((1,))
    try:
        weights = np.broadcast_to(weights, data.shape)
    except ValueError:
        raise ValueError('Weights and data have inconsistent shape.')
    n = np.size(data, axis=axis)
    R0bar = np.sum(weights * np.cos(data - mu), axis) / np.sum(weights, axis)
    z = np.sqrt(2.0 * n) * R0bar
    pz = norm.cdf(z)
    fz = norm.pdf(z)
    p_value = 1 - pz + fz * ((3 * z - z ** 3) / (16.0 * n) + (15 * z + 305 * z ** 3 - 125 * z ** 5 + 9 * z ** 7) / (4608.0 * n * n))
    return p_value

def _A1inv(x):
    if False:
        print('Hello World!')
    kappa1 = np.where(np.logical_and(0 <= x, x < 0.53), 2.0 * x + x * x * x + 5.0 * x ** 5 / 6.0, 0)
    kappa2 = np.where(np.logical_and(0.53 <= x, x < 0.85), -0.4 + 1.39 * x + 0.43 / (1.0 - x), 0)
    kappa3 = np.where(np.logical_or(x < 0, 0.85 <= x), 1.0 / (x * x * x - 4.0 * x * x + 3.0 * x), 0)
    return kappa1 + kappa2 + kappa3

def vonmisesmle(data, axis=None, weights=None):
    if False:
        for i in range(10):
            print('nop')
    'Computes the Maximum Likelihood Estimator (MLE) for the parameters of\n    the von Mises distribution.\n\n    Parameters\n    ----------\n    data : ndarray or `~astropy.units.Quantity`\n        Array of circular (directional) data, which is assumed to be in\n        radians whenever ``data`` is ``numpy.ndarray``.\n    axis : int, optional\n        Axis along which the mle will be computed.\n    weights : numpy.ndarray, optional\n        In case of grouped data, the i-th element of ``weights`` represents a\n        weighting factor for each group such that ``sum(weights, axis)``\n        equals the number of observations. See [1]_, remark 1.4, page 22,\n        for detailed explanation.\n\n    Returns\n    -------\n    mu : float or `~astropy.units.Quantity`\n        The mean (aka location parameter).\n    kappa : float or `~astropy.units.Quantity` [\'dimensionless\']\n        The concentration parameter.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from astropy.stats import vonmisesmle\n    >>> from astropy import units as u\n    >>> data = np.array([130, 90, 0, 145])*u.deg\n    >>> vonmisesmle(data) # doctest: +FLOAT_CMP\n    (<Quantity 101.16894320013179 deg>, <Quantity 1.49358958737054>)\n\n    References\n    ----------\n    .. [1] S. R. Jammalamadaka, A. SenGupta. "Topics in Circular Statistics".\n       Series on Multivariate Analysis, Vol. 5, 2001.\n    .. [2] C. Agostinelli, U. Lund. "Circular Statistics from \'Topics in\n       Circular Statistics (2001)\'". 2015.\n       <https://cran.r-project.org/web/packages/CircStats/CircStats.pdf>\n    '
    mu = circmean(data, axis=axis, weights=weights)
    kappa = _A1inv(_length(data, p=1, phi=0.0, axis=axis, weights=weights))
    return (mu, kappa)