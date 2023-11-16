"""
The routines here were removed from numbers.py, power.py,
digits.py and factor_.py so they could be imported into core
without raising circular import errors.

Although the name 'intfunc' was chosen to represent functions that
work with integers, it can also be thought of as containing
internal/core functions that are needed by the classes of the core.
"""
import math
import sys
from functools import lru_cache
from .sympify import sympify
from .singleton import S
from sympy.external.gmpy import gcd as number_gcd, lcm as number_lcm, sqrt, iroot, bit_scan1, gcdext
from sympy.utilities.misc import as_int, filldedent

def num_digits(n, base=10):
    if False:
        for i in range(10):
            print('nop')
    'Return the number of digits needed to express n in give base.\n\n    Examples\n    ========\n\n    >>> from sympy.core.intfunc import num_digits\n    >>> num_digits(10)\n    2\n    >>> num_digits(10, 2)  # 1010 -> 4 digits\n    4\n    >>> num_digits(-100, 16)  # -64 -> 2 digits\n    2\n\n\n    Parameters\n    ==========\n\n    n: integer\n        The number whose digits are counted.\n\n    b: integer\n        The base in which digits are computed.\n\n    See Also\n    ========\n    sympy.ntheory.digits.digits, sympy.ntheory.digits.count_digits\n    '
    if base < 0:
        raise ValueError('base must be int greater than 1')
    if not n:
        return 1
    (e, t) = integer_log(abs(n), base)
    return 1 + e

def integer_log(n, b):
    if False:
        return 10
    '\n    Returns ``(e, bool)`` where e is the largest nonnegative integer\n    such that :math:`|n| \\geq |b^e|` and ``bool`` is True if $n = b^e$.\n\n    Examples\n    ========\n\n    >>> from sympy import integer_log\n    >>> integer_log(125, 5)\n    (3, True)\n    >>> integer_log(17, 9)\n    (1, False)\n\n    If the base is positive and the number negative the\n    return value will always be the same except for 2:\n\n    >>> integer_log(-4, 2)\n    (2, False)\n    >>> integer_log(-16, 4)\n    (0, False)\n\n    When the base is negative, the returned value\n    will only be True if the parity of the exponent is\n    correct for the sign of the base:\n\n    >>> integer_log(4, -2)\n    (2, True)\n    >>> integer_log(8, -2)\n    (3, False)\n    >>> integer_log(-8, -2)\n    (3, True)\n    >>> integer_log(-4, -2)\n    (2, False)\n\n    See Also\n    ========\n    integer_nthroot\n    sympy.ntheory.primetest.is_square\n    sympy.ntheory.factor_.multiplicity\n    sympy.ntheory.factor_.perfect_power\n    '
    n = as_int(n)
    b = as_int(b)
    if b < 0:
        (e, t) = integer_log(abs(n), -b)
        t = t and e % 2 == (n < 0)
        return (e, t)
    if b <= 1:
        raise ValueError('base must be 2 or more')
    if n < 0:
        if b != 2:
            return (0, False)
        (e, t) = integer_log(-n, b)
        return (e, False)
    if n == 0:
        raise ValueError('n cannot be 0')
    if n < b:
        return (0, n == 1)
    if b == 2:
        e = n.bit_length() - 1
        return (e, trailing(n) == e)
    t = trailing(b)
    if 2 ** t == b:
        e = int(n.bit_length() - 1) // t
        n_ = 1 << t * e
        return (e, n_ == n)
    d = math.floor(math.log10(n) / math.log10(b))
    n_ = b ** d
    while n_ <= n:
        d += 1
        n_ *= b
    return (d - (n_ > n), n_ == n or n_ // b == n)

def trailing(n):
    if False:
        for i in range(10):
            print('nop')
    'Count the number of trailing zero digits in the binary\n    representation of n, i.e. determine the largest power of 2\n    that divides n.\n\n    Examples\n    ========\n\n    >>> from sympy import trailing\n    >>> trailing(128)\n    7\n    >>> trailing(63)\n    0\n\n    See Also\n    ========\n    sympy.ntheory.factor_.multiplicity\n\n    '
    if not n:
        return 0
    return bit_scan1(int(n))

@lru_cache(1024)
def igcd(*args):
    if False:
        return 10
    "Computes nonnegative integer greatest common divisor.\n\n    Explanation\n    ===========\n\n    The algorithm is based on the well known Euclid's algorithm [1]_. To\n    improve speed, ``igcd()`` has its own caching mechanism.\n    If you do not need the cache mechanism, using ``sympy.external.gmpy.gcd``.\n\n    Examples\n    ========\n\n    >>> from sympy import igcd\n    >>> igcd(2, 4)\n    2\n    >>> igcd(5, 10, 15)\n    5\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Euclidean_algorithm\n\n    "
    if len(args) < 2:
        raise TypeError('igcd() takes at least 2 arguments (%s given)' % len(args))
    return int(number_gcd(*map(as_int, args)))
igcd2 = math.gcd

def igcd_lehmer(a, b):
    if False:
        i = 10
        return i + 15
    "Computes greatest common divisor of two integers.\n\n    Explanation\n    ===========\n\n    Euclid's algorithm for the computation of the greatest\n    common divisor ``gcd(a, b)``  of two (positive) integers\n    $a$ and $b$ is based on the division identity\n    $$ a = q \\times b + r$$,\n    where the quotient  $q$  and the remainder  $r$  are integers\n    and  $0 \\le r < b$. Then each common divisor of  $a$  and  $b$\n    divides  $r$, and it follows that  ``gcd(a, b) == gcd(b, r)``.\n    The algorithm works by constructing the sequence\n    r0, r1, r2, ..., where  r0 = a, r1 = b,  and each  rn\n    is the remainder from the division of the two preceding\n    elements.\n\n    In Python, ``q = a // b``  and  ``r = a % b``  are obtained by the\n    floor division and the remainder operations, respectively.\n    These are the most expensive arithmetic operations, especially\n    for large  a  and  b.\n\n    Lehmer's algorithm [1]_ is based on the observation that the quotients\n    ``qn = r(n-1) // rn``  are in general small integers even\n    when  a  and  b  are very large. Hence the quotients can be\n    usually determined from a relatively small number of most\n    significant bits.\n\n    The efficiency of the algorithm is further enhanced by not\n    computing each long remainder in Euclid's sequence. The remainders\n    are linear combinations of  a  and  b  with integer coefficients\n    derived from the quotients. The coefficients can be computed\n    as far as the quotients can be determined from the chosen\n    most significant parts of  a  and  b. Only then a new pair of\n    consecutive remainders is computed and the algorithm starts\n    anew with this pair.\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Lehmer%27s_GCD_algorithm\n\n    "
    (a, b) = (abs(as_int(a)), abs(as_int(b)))
    if a < b:
        (a, b) = (b, a)
    nbits = 2 * sys.int_info.bits_per_digit
    while a.bit_length() > nbits and b != 0:
        n = a.bit_length() - nbits
        (x, y) = (int(a >> n), int(b >> n))
        (A, B, C, D) = (1, 0, 0, 1)
        while True:
            if y + C <= 0:
                break
            q = (x + A) // (y + C)
            (x_qy, B_qD) = (x - q * y, B - q * D)
            if x_qy + B_qD < 0:
                break
            (x, y) = (y, x_qy)
            (A, B, C, D) = (C, D, A - q * C, B_qD)
            if y + D <= 0:
                break
            q = (x + B) // (y + D)
            (x_qy, A_qC) = (x - q * y, A - q * C)
            if x_qy + A_qC < 0:
                break
            (x, y) = (y, x_qy)
            (A, B, C, D) = (C, D, A_qC, B - q * D)
        if B == 0:
            (a, b) = (b, a % b)
            continue
        (a, b) = (A * a + B * b, C * a + D * b)
    while b:
        (a, b) = (b, a % b)
    return a

def ilcm(*args):
    if False:
        print('Hello World!')
    'Computes integer least common multiple.\n\n    Examples\n    ========\n\n    >>> from sympy import ilcm\n    >>> ilcm(5, 10)\n    10\n    >>> ilcm(7, 3)\n    21\n    >>> ilcm(5, 10, 15)\n    30\n\n    '
    if len(args) < 2:
        raise TypeError('ilcm() takes at least 2 arguments (%s given)' % len(args))
    return int(number_lcm(*map(as_int, args)))

def igcdex(a, b):
    if False:
        i = 10
        return i + 15
    'Returns x, y, g such that g = x*a + y*b = gcd(a, b).\n\n    Examples\n    ========\n\n    >>> from sympy.core.intfunc import igcdex\n    >>> igcdex(2, 3)\n    (-1, 1, 1)\n    >>> igcdex(10, 12)\n    (-1, 1, 2)\n\n    >>> x, y, g = igcdex(100, 2004)\n    >>> x, y, g\n    (-20, 1, 4)\n    >>> x*100 + y*2004\n    4\n\n    '
    if not a and (not b):
        return (0, 1, 0)
    (g, x, y) = gcdext(int(a), int(b))
    return (x, y, g)

def mod_inverse(a, m):
    if False:
        i = 10
        return i + 15
    '\n    Return the number $c$ such that, $a \\times c = 1 \\pmod{m}$\n    where $c$ has the same sign as $m$. If no such value exists,\n    a ValueError is raised.\n\n    Examples\n    ========\n\n    >>> from sympy import mod_inverse, S\n\n    Suppose we wish to find multiplicative inverse $x$ of\n    3 modulo 11. This is the same as finding $x$ such\n    that $3x = 1 \\pmod{11}$. One value of x that satisfies\n    this congruence is 4. Because $3 \\times 4 = 12$ and $12 = 1 \\pmod{11}$.\n    This is the value returned by ``mod_inverse``:\n\n    >>> mod_inverse(3, 11)\n    4\n    >>> mod_inverse(-3, 11)\n    7\n\n    When there is a common factor between the numerators of\n    `a` and `m` the inverse does not exist:\n\n    >>> mod_inverse(2, 4)\n    Traceback (most recent call last):\n    ...\n    ValueError: inverse of 2 mod 4 does not exist\n\n    >>> mod_inverse(S(2)/7, S(5)/2)\n    7/2\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Modular_multiplicative_inverse\n    .. [2] https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm\n    '
    c = None
    try:
        (a, m) = (as_int(a), as_int(m))
        if m != 1 and m != -1:
            (x, _, g) = igcdex(a, m)
            if g == 1:
                c = x % m
    except ValueError:
        (a, m) = (sympify(a), sympify(m))
        if not (a.is_number and m.is_number):
            raise TypeError(filldedent('\n                Expected numbers for arguments; symbolic `mod_inverse`\n                is not implemented\n                but symbolic expressions can be handled with the\n                similar function,\n                sympy.polys.polytools.invert'))
        big = m > 1
        if big not in (S.true, S.false):
            raise ValueError('m > 1 did not evaluate; try to simplify %s' % m)
        elif big:
            c = 1 / a
    if c is None:
        raise ValueError('inverse of %s (mod %s) does not exist' % (a, m))
    return c

def isqrt(n):
    if False:
        print('Hello World!')
    ' Return the largest integer less than or equal to `\\sqrt{n}`.\n\n    Parameters\n    ==========\n\n    n : non-negative integer\n\n    Returns\n    =======\n\n    int : `\\left\\lfloor\\sqrt{n}\\right\\rfloor`\n\n    Raises\n    ======\n\n    ValueError\n        If n is negative.\n    TypeError\n        If n is of a type that cannot be compared to ``int``.\n        Therefore, a TypeError is raised for ``str``, but not for ``float``.\n\n    Examples\n    ========\n\n    >>> from sympy.core.intfunc import isqrt\n    >>> isqrt(0)\n    0\n    >>> isqrt(9)\n    3\n    >>> isqrt(10)\n    3\n    >>> isqrt("30")\n    Traceback (most recent call last):\n        ...\n    TypeError: \'<\' not supported between instances of \'str\' and \'int\'\n    >>> from sympy.core.numbers import Rational\n    >>> isqrt(Rational(-1, 2))\n    Traceback (most recent call last):\n        ...\n    ValueError: n must be nonnegative\n\n    '
    if n < 0:
        raise ValueError('n must be nonnegative')
    return int(sqrt(int(n)))

def integer_nthroot(y, n):
    if False:
        return 10
    '\n    Return a tuple containing x = floor(y**(1/n))\n    and a boolean indicating whether the result is exact (that is,\n    whether x**n == y).\n\n    Examples\n    ========\n\n    >>> from sympy import integer_nthroot\n    >>> integer_nthroot(16, 2)\n    (4, True)\n    >>> integer_nthroot(26, 2)\n    (5, False)\n\n    To simply determine if a number is a perfect square, the is_square\n    function should be used:\n\n    >>> from sympy.ntheory.primetest import is_square\n    >>> is_square(26)\n    False\n\n    See Also\n    ========\n    sympy.ntheory.primetest.is_square\n    integer_log\n    '
    (x, b) = iroot(as_int(y), as_int(n))
    return (int(x), b)