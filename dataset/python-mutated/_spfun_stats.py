"""Some more special functions which may be useful for multivariate statistical
analysis."""
import numpy as np
from scipy.special import gammaln as loggam
__all__ = ['multigammaln']

def multigammaln(a, d):
    if False:
        return 10
    'Returns the log of multivariate gamma, also sometimes called the\n    generalized gamma.\n\n    Parameters\n    ----------\n    a : ndarray\n        The multivariate gamma is computed for each item of `a`.\n    d : int\n        The dimension of the space of integration.\n\n    Returns\n    -------\n    res : ndarray\n        The values of the log multivariate gamma at the given points `a`.\n\n    Notes\n    -----\n    The formal definition of the multivariate gamma of dimension d for a real\n    `a` is\n\n    .. math::\n\n        \\Gamma_d(a) = \\int_{A>0} e^{-tr(A)} |A|^{a - (d+1)/2} dA\n\n    with the condition :math:`a > (d-1)/2`, and :math:`A > 0` being the set of\n    all the positive definite matrices of dimension `d`.  Note that `a` is a\n    scalar: the integrand only is multivariate, the argument is not (the\n    function is defined over a subset of the real set).\n\n    This can be proven to be equal to the much friendlier equation\n\n    .. math::\n\n        \\Gamma_d(a) = \\pi^{d(d-1)/4} \\prod_{i=1}^{d} \\Gamma(a - (i-1)/2).\n\n    References\n    ----------\n    R. J. Muirhead, Aspects of multivariate statistical theory (Wiley Series in\n    probability and mathematical statistics).\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.special import multigammaln, gammaln\n    >>> a = 23.5\n    >>> d = 10\n    >>> multigammaln(a, d)\n    454.1488605074416\n\n    Verify that the result agrees with the logarithm of the equation\n    shown above:\n\n    >>> d*(d-1)/4*np.log(np.pi) + gammaln(a - 0.5*np.arange(0, d)).sum()\n    454.1488605074416\n    '
    a = np.asarray(a)
    if not np.isscalar(d) or np.floor(d) != d:
        raise ValueError('d should be a positive integer (dimension)')
    if np.any(a <= 0.5 * (d - 1)):
        raise ValueError('condition a (%f) > 0.5 * (d-1) (%f) not met' % (a, 0.5 * (d - 1)))
    res = d * (d - 1) * 0.25 * np.log(np.pi)
    res += np.sum(loggam([a - (j - 1.0) / 2 for j in range(1, d + 1)]), axis=0)
    return res