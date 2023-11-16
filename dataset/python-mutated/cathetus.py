from math import fabs, inf, isinf, isnan, nan, sqrt
from sys import float_info

def cathetus(h, a):
    if False:
        for i in range(10):
            print('nop')
    'Given the lengths of the hypotenuse and a side of a right triangle,\n    return the length of the other side.\n\n    A companion to the C99 hypot() function.  Some care is needed to avoid\n    underflow in the case of small arguments, and overflow in the case of\n    large arguments as would occur for the naive implementation as\n    sqrt(h*h - a*a).  The behaviour with respect the non-finite arguments\n    (NaNs and infinities) is designed to be as consistent as possible with\n    the C99 hypot() specifications.\n\n    This function relies on the system ``sqrt`` function and so, like it,\n    may be inaccurate up to a relative error of (around) floating-point\n    epsilon.\n\n    Based on the C99 implementation https://github.com/jjgreen/cathetus\n    '
    if isnan(h):
        return nan
    if isinf(h):
        if isinf(a):
            return nan
        else:
            return inf
    h = fabs(h)
    a = fabs(a)
    if h < a:
        return nan
    if h > sqrt(float_info.max):
        if h > float_info.max / 2:
            b = sqrt(h - a) * sqrt(h / 2 + a / 2) * sqrt(2)
        else:
            b = sqrt(h - a) * sqrt(h + a)
    elif h < sqrt(float_info.min):
        b = sqrt(h - a) * sqrt(h + a)
    else:
        b = sqrt((h - a) * (h + a))
    return min(b, h)