import math
import cupy
from cupy import _core
gammaln = _core.create_ufunc('cupyx_scipy_special_gammaln', ('f->f', 'd->d'), '\n    if (isinf(in0) && in0 < 0) {\n        out0 = -1.0 / 0.0;\n    } else {\n        out0 = lgamma(in0);\n    }\n    ', doc='Logarithm of the absolute value of the Gamma function.\n\n    Args:\n        x (cupy.ndarray): Values on the real line at which to compute\n        ``gammaln``.\n\n    Returns:\n        cupy.ndarray: Values of ``gammaln`` at x.\n\n    .. seealso:: :data:`scipy.special.gammaln`\n\n    ')

def multigammaln(a, d):
    if False:
        print('Hello World!')
    'Returns the log of multivariate gamma, also sometimes called the\n    generalized gamma.\n\n    Parameters\n    ----------\n    a : cupy.ndarray\n        The multivariate gamma is computed for each item of `a`.\n    d : int\n        The dimension of the space of integration.\n\n    Returns\n    -------\n    res : ndarray\n        The values of the log multivariate gamma at the given points `a`.\n\n    See Also\n    --------\n    :func:`scipy.special.multigammaln`\n\n    '
    if not cupy.isscalar(d) or math.floor(d) != d:
        raise ValueError('d should be a positive integer (dimension)')
    if cupy.isscalar(a):
        a = cupy.asarray(a, dtype=float)
    if int(cupy.any(a <= 0.5 * (d - 1))):
        raise ValueError('condition a > 0.5 * (d-1) not met')
    res = d * (d - 1) * 0.25 * math.log(math.pi)
    gam0 = gammaln(a)
    if a.dtype.kind != 'f':
        gam0 = gam0.astype(cupy.float64)
    res = res + gam0
    for j in range(2, d + 1):
        res += gammaln(a - (j - 1.0) / 2)
    return res