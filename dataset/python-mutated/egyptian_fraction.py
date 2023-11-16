from sympy.core.containers import Tuple
from sympy.core.numbers import Integer, Rational
from sympy.core.singleton import S
import sympy.polys
from math import gcd

def egyptian_fraction(r, algorithm='Greedy'):
    if False:
        i = 10
        return i + 15
    '\n    Return the list of denominators of an Egyptian fraction\n    expansion [1]_ of the said rational `r`.\n\n    Parameters\n    ==========\n\n    r : Rational or (p, q)\n        a positive rational number, ``p/q``.\n    algorithm : { "Greedy", "Graham Jewett", "Takenouchi", "Golomb" }, optional\n        Denotes the algorithm to be used (the default is "Greedy").\n\n    Examples\n    ========\n\n    >>> from sympy import Rational\n    >>> from sympy.ntheory.egyptian_fraction import egyptian_fraction\n    >>> egyptian_fraction(Rational(3, 7))\n    [3, 11, 231]\n    >>> egyptian_fraction((3, 7), "Graham Jewett")\n    [7, 8, 9, 56, 57, 72, 3192]\n    >>> egyptian_fraction((3, 7), "Takenouchi")\n    [4, 7, 28]\n    >>> egyptian_fraction((3, 7), "Golomb")\n    [3, 15, 35]\n    >>> egyptian_fraction((11, 5), "Golomb")\n    [1, 2, 3, 4, 9, 234, 1118, 2580]\n\n    See Also\n    ========\n\n    sympy.core.numbers.Rational\n\n    Notes\n    =====\n\n    Currently the following algorithms are supported:\n\n    1) Greedy Algorithm\n\n       Also called the Fibonacci-Sylvester algorithm [2]_.\n       At each step, extract the largest unit fraction less\n       than the target and replace the target with the remainder.\n\n       It has some distinct properties:\n\n       a) Given `p/q` in lowest terms, generates an expansion of maximum\n          length `p`. Even as the numerators get large, the number of\n          terms is seldom more than a handful.\n\n       b) Uses minimal memory.\n\n       c) The terms can blow up (standard examples of this are 5/121 and\n          31/311).  The denominator is at most squared at each step\n          (doubly-exponential growth) and typically exhibits\n          singly-exponential growth.\n\n    2) Graham Jewett Algorithm\n\n       The algorithm suggested by the result of Graham and Jewett.\n       Note that this has a tendency to blow up: the length of the\n       resulting expansion is always ``2**(x/gcd(x, y)) - 1``.  See [3]_.\n\n    3) Takenouchi Algorithm\n\n       The algorithm suggested by Takenouchi (1921).\n       Differs from the Graham-Jewett algorithm only in the handling\n       of duplicates.  See [3]_.\n\n    4) Golomb\'s Algorithm\n\n       A method given by Golumb (1962), using modular arithmetic and\n       inverses.  It yields the same results as a method using continued\n       fractions proposed by Bleicher (1972).  See [4]_.\n\n    If the given rational is greater than or equal to 1, a greedy algorithm\n    of summing the harmonic sequence 1/1 + 1/2 + 1/3 + ... is used, taking\n    all the unit fractions of this sequence until adding one more would be\n    greater than the given number.  This list of denominators is prefixed\n    to the result from the requested algorithm used on the remainder.  For\n    example, if r is 8/3, using the Greedy algorithm, we get [1, 2, 3, 4,\n    5, 6, 7, 14, 420], where the beginning of the sequence, [1, 2, 3, 4, 5,\n    6, 7] is part of the harmonic sequence summing to 363/140, leaving a\n    remainder of 31/420, which yields [14, 420] by the Greedy algorithm.\n    The result of egyptian_fraction(Rational(8, 3), "Golomb") is [1, 2, 3,\n    4, 5, 6, 7, 14, 574, 2788, 6460, 11590, 33062, 113820], and so on.\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Egyptian_fraction\n    .. [2] https://en.wikipedia.org/wiki/Greedy_algorithm_for_Egyptian_fractions\n    .. [3] https://www.ics.uci.edu/~eppstein/numth/egypt/conflict.html\n    .. [4] https://web.archive.org/web/20180413004012/https://ami.ektf.hu/uploads/papers/finalpdf/AMI_42_from129to134.pdf\n\n    '
    if not isinstance(r, Rational):
        if isinstance(r, (Tuple, tuple)) and len(r) == 2:
            r = Rational(*r)
        else:
            raise ValueError('Value must be a Rational or tuple of ints')
    if r <= 0:
        raise ValueError('Value must be positive')
    (x, y) = r.as_numer_denom()
    if y == 1 and x == 2:
        return [Integer(i) for i in [1, 2, 3, 6]]
    if x == y + 1:
        return [S.One, y]
    (prefix, rem) = egypt_harmonic(r)
    if rem == 0:
        return prefix
    (x, y) = (rem.p, rem.q)
    if algorithm == 'Greedy':
        postfix = egypt_greedy(x, y)
    elif algorithm == 'Graham Jewett':
        postfix = egypt_graham_jewett(x, y)
    elif algorithm == 'Takenouchi':
        postfix = egypt_takenouchi(x, y)
    elif algorithm == 'Golomb':
        postfix = egypt_golomb(x, y)
    else:
        raise ValueError('Entered invalid algorithm')
    return prefix + [Integer(i) for i in postfix]

def egypt_greedy(x, y):
    if False:
        i = 10
        return i + 15
    if x == 1:
        return [y]
    else:
        a = -y % x
        b = y * (y // x + 1)
        c = gcd(a, b)
        if c > 1:
            (num, denom) = (a // c, b // c)
        else:
            (num, denom) = (a, b)
        return [y // x + 1] + egypt_greedy(num, denom)

def egypt_graham_jewett(x, y):
    if False:
        for i in range(10):
            print('nop')
    l = [y] * x
    while len(l) != len(set(l)):
        l.sort()
        for i in range(len(l) - 1):
            if l[i] == l[i + 1]:
                break
        l[i + 1] = l[i] + 1
        l.append(l[i] * (l[i] + 1))
    return sorted(l)

def egypt_takenouchi(x, y):
    if False:
        i = 10
        return i + 15
    if x == 3:
        if y % 2 == 0:
            return [y // 2, y]
        i = (y - 1) // 2
        j = i + 1
        k = j + i
        return [j, k, j * k]
    l = [y] * x
    while len(l) != len(set(l)):
        l.sort()
        for i in range(len(l) - 1):
            if l[i] == l[i + 1]:
                break
        k = l[i]
        if k % 2 == 0:
            l[i] = l[i] // 2
            del l[i + 1]
        else:
            (l[i], l[i + 1]) = ((k + 1) // 2, k * (k + 1) // 2)
    return sorted(l)

def egypt_golomb(x, y):
    if False:
        return 10
    if x == 1:
        return [y]
    xp = sympy.polys.ZZ.invert(int(x), int(y))
    rv = [xp * y]
    rv.extend(egypt_golomb((x * xp - 1) // y, xp))
    return sorted(rv)

def egypt_harmonic(r):
    if False:
        for i in range(10):
            print('nop')
    rv = []
    d = S.One
    acc = S.Zero
    while acc + 1 / d <= r:
        acc += 1 / d
        rv.append(d)
        d += 1
    return (rv, r - acc)