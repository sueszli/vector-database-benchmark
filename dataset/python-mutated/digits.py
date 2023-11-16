from collections import defaultdict
from sympy.utilities.iterables import multiset, is_palindromic as _palindromic
from sympy.utilities.misc import as_int

def digits(n, b=10, digits=None):
    if False:
        while True:
            i = 10
    '\n    Return a list of the digits of ``n`` in base ``b``. The first\n    element in the list is ``b`` (or ``-b`` if ``n`` is negative).\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.digits import digits\n    >>> digits(35)\n    [10, 3, 5]\n\n    If the number is negative, the negative sign will be placed on the\n    base (which is the first element in the returned list):\n\n    >>> digits(-35)\n    [-10, 3, 5]\n\n    Bases other than 10 (and greater than 1) can be selected with ``b``:\n\n    >>> digits(27, b=2)\n    [2, 1, 1, 0, 1, 1]\n\n    Use the ``digits`` keyword if a certain number of digits is desired:\n\n    >>> digits(35, digits=4)\n    [10, 0, 0, 3, 5]\n\n    Parameters\n    ==========\n\n    n: integer\n        The number whose digits are returned.\n\n    b: integer\n        The base in which digits are computed.\n\n    digits: integer (or None for all digits)\n        The number of digits to be returned (padded with zeros, if\n        necessary).\n\n    See Also\n    ========\n    sympy.core.intfunc.num_digits, count_digits\n    '
    b = as_int(b)
    n = as_int(n)
    if b < 2:
        raise ValueError('b must be greater than 1')
    else:
        (x, y) = (abs(n), [])
        while x >= b:
            (x, r) = divmod(x, b)
            y.append(r)
        y.append(x)
        y.append(-b if n < 0 else b)
        y.reverse()
        ndig = len(y) - 1
        if digits is not None:
            if ndig > digits:
                raise ValueError('For %s, at least %s digits are needed.' % (n, ndig))
            elif ndig < digits:
                y[1:1] = [0] * (digits - ndig)
        return y

def count_digits(n, b=10):
    if False:
        while True:
            i = 10
    '\n    Return a dictionary whose keys are the digits of ``n`` in the\n    given base, ``b``, with keys indicating the digits appearing in the\n    number and values indicating how many times that digit appeared.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory import count_digits\n\n    >>> count_digits(1111339)\n    {1: 4, 3: 2, 9: 1}\n\n    The digits returned are always represented in base-10\n    but the number itself can be entered in any format that is\n    understood by Python; the base of the number can also be\n    given if it is different than 10:\n\n    >>> n = 0xFA; n\n    250\n    >>> count_digits(_)\n    {0: 1, 2: 1, 5: 1}\n    >>> count_digits(n, 16)\n    {10: 1, 15: 1}\n\n    The default dictionary will return a 0 for any digit that did\n    not appear in the number. For example, which digits appear 7\n    times in ``77!``:\n\n    >>> from sympy import factorial\n    >>> c77 = count_digits(factorial(77))\n    >>> [i for i in range(10) if c77[i] == 7]\n    [1, 3, 7, 9]\n\n    See Also\n    ========\n    sympy.core.intfunc.num_digits, digits\n    '
    rv = defaultdict(int, multiset(digits(n, b)).items())
    rv.pop(b) if b in rv else rv.pop(-b)
    return rv

def is_palindromic(n, b=10):
    if False:
        i = 10
        return i + 15
    "return True if ``n`` is the same when read from left to right\n    or right to left in the given base, ``b``.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory import is_palindromic\n\n    >>> all(is_palindromic(i) for i in (-11, 1, 22, 121))\n    True\n\n    The second argument allows you to test numbers in other\n    bases. For example, 88 is palindromic in base-10 but not\n    in base-8:\n\n    >>> is_palindromic(88, 8)\n    False\n\n    On the other hand, a number can be palindromic in base-8 but\n    not in base-10:\n\n    >>> 0o121, is_palindromic(0o121)\n    (81, False)\n\n    Or it might be palindromic in both bases:\n\n    >>> oct(121), is_palindromic(121, 8) and is_palindromic(121)\n    ('0o171', True)\n\n    "
    return _palindromic(digits(n, b), 1)