from decimal import Decimal
import math
from numpy import isnan

def tolerant_equals(a, b, atol=1e-06, rtol=1e-06, equal_nan=False):
    if False:
        return 10
    'Check if a and b are equal with some tolerance.\n\n    Parameters\n    ----------\n    a, b : float\n        The floats to check for equality.\n    atol : float, optional\n        The absolute tolerance.\n    rtol : float, optional\n        The relative tolerance.\n    equal_nan : bool, optional\n        Should NaN compare equal?\n\n    See Also\n    --------\n    numpy.isclose\n\n    Notes\n    -----\n    This function is just a scalar version of numpy.isclose for performance.\n    See the docstring of ``isclose`` for more information about ``atol`` and\n    ``rtol``.\n    '
    if equal_nan and isnan(a) and isnan(b):
        return True
    return math.fabs(a - b) <= atol + rtol * math.fabs(b)
try:
    import bottleneck as bn
    nanmean = bn.nanmean
    nanstd = bn.nanstd
    nansum = bn.nansum
    nanmax = bn.nanmax
    nanmin = bn.nanmin
    nanargmax = bn.nanargmax
    nanargmin = bn.nanargmin
    nanmedian = bn.nanmedian
except ImportError:
    import numpy as np
    nanmean = np.nanmean
    nanstd = np.nanstd
    nansum = np.nansum
    nanmax = np.nanmax
    nanmin = np.nanmin
    nanargmax = np.nanargmax
    nanargmin = np.nanargmin
    nanmedian = np.nanmedian

def round_if_near_integer(a, epsilon=0.0001):
    if False:
        while True:
            i = 10
    '\n    Round a to the nearest integer if that integer is within an epsilon\n    of a.\n    '
    if abs(a - round(a)) <= epsilon:
        return round(a)
    else:
        return a

def number_of_decimal_places(n):
    if False:
        return 10
    "\n    Compute the number of decimal places in a number.\n\n    Examples\n    --------\n    >>> number_of_decimal_places(1)\n    0\n    >>> number_of_decimal_places(3.14)\n    2\n    >>> number_of_decimal_places('3.14')\n    2\n    "
    decimal = Decimal(str(n))
    return -decimal.as_tuple().exponent