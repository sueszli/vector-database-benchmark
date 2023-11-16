"""
``python-future``: pure Python implementation of Python 3 round().
"""
from __future__ import division
from future.utils import PYPY, PY26, bind_method
from decimal import Decimal, ROUND_HALF_EVEN

def newround(number, ndigits=None):
    if False:
        return 10
    "\n    See Python 3 documentation: uses Banker's Rounding.\n\n    Delegates to the __round__ method if for some reason this exists.\n\n    If not, rounds a number to a given precision in decimal digits (default\n    0 digits). This returns an int when called with one argument,\n    otherwise the same type as the number. ndigits may be negative.\n\n    See the test_round method in future/tests/test_builtins.py for\n    examples.\n    "
    return_int = False
    if ndigits is None:
        return_int = True
        ndigits = 0
    if hasattr(number, '__round__'):
        return number.__round__(ndigits)
    exponent = Decimal('10') ** (-ndigits)
    if 'numpy' in repr(type(number)):
        number = float(number)
    if isinstance(number, Decimal):
        d = number
    elif not PY26:
        d = Decimal.from_float(number)
    else:
        d = from_float_26(number)
    if ndigits < 0:
        result = newround(d / exponent) * exponent
    else:
        result = d.quantize(exponent, rounding=ROUND_HALF_EVEN)
    if return_int:
        return int(result)
    else:
        return float(result)

def from_float_26(f):
    if False:
        i = 10
        return i + 15
    "Converts a float to a decimal number, exactly.\n\n    Note that Decimal.from_float(0.1) is not the same as Decimal('0.1').\n    Since 0.1 is not exactly representable in binary floating point, the\n    value is stored as the nearest representable value which is\n    0x1.999999999999ap-4.  The exact equivalent of the value in decimal\n    is 0.1000000000000000055511151231257827021181583404541015625.\n\n    >>> Decimal.from_float(0.1)\n    Decimal('0.1000000000000000055511151231257827021181583404541015625')\n    >>> Decimal.from_float(float('nan'))\n    Decimal('NaN')\n    >>> Decimal.from_float(float('inf'))\n    Decimal('Infinity')\n    >>> Decimal.from_float(-float('inf'))\n    Decimal('-Infinity')\n    >>> Decimal.from_float(-0.0)\n    Decimal('-0')\n\n    "
    import math as _math
    from decimal import _dec_from_triple
    if isinstance(f, (int, long)):
        return Decimal(f)
    if _math.isinf(f) or _math.isnan(f):
        return Decimal(repr(f))
    if _math.copysign(1.0, f) == 1.0:
        sign = 0
    else:
        sign = 1
    (n, d) = abs(f).as_integer_ratio()

    def bit_length(d):
        if False:
            return 10
        if d != 0:
            return len(bin(abs(d))) - 2
        else:
            return 0
    k = bit_length(d) - 1
    result = _dec_from_triple(sign, str(n * 5 ** k), -k)
    return result
__all__ = ['newround']