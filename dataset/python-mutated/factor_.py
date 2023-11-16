"""
Integer factorization
"""
from collections import defaultdict
from functools import reduce
import math
from sympy.core import sympify
from sympy.core.containers import Dict
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.logic import fuzzy_and
from sympy.core.mul import Mul
from sympy.core.numbers import Rational, Integer
from sympy.core.intfunc import num_digits
from sympy.core.power import Pow
from sympy.core.random import _randint
from sympy.core.singleton import S
from sympy.external.gmpy import SYMPY_INTS, gcd, lcm, sqrt as isqrt, sqrtrem, iroot, bit_scan1, remove
from .primetest import isprime, MERSENNE_PRIME_EXPONENTS, is_mersenne_prime
from .generate import sieve, primerange, nextprime
from .digits import digits
from sympy.utilities.iterables import flatten
from sympy.utilities.misc import as_int, filldedent
from .ecm import _ecm_one_factor

def smoothness(n):
    if False:
        i = 10
        return i + 15
    '\n    Return the B-smooth and B-power smooth values of n.\n\n    The smoothness of n is the largest prime factor of n; the power-\n    smoothness is the largest divisor raised to its multiplicity.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.factor_ import smoothness\n    >>> smoothness(2**7*3**2)\n    (3, 128)\n    >>> smoothness(2**4*13)\n    (13, 16)\n    >>> smoothness(2)\n    (2, 2)\n\n    See Also\n    ========\n\n    factorint, smoothness_p\n    '
    if n == 1:
        return (1, 1)
    facs = factorint(n)
    return (max(facs), max((m ** facs[m] for m in facs)))

def smoothness_p(n, m=-1, power=0, visual=None):
    if False:
        print('Hello World!')
    "\n    Return a list of [m, (p, (M, sm(p + m), psm(p + m)))...]\n    where:\n\n    1. p**M is the base-p divisor of n\n    2. sm(p + m) is the smoothness of p + m (m = -1 by default)\n    3. psm(p + m) is the power smoothness of p + m\n\n    The list is sorted according to smoothness (default) or by power smoothness\n    if power=1.\n\n    The smoothness of the numbers to the left (m = -1) or right (m = 1) of a\n    factor govern the results that are obtained from the p +/- 1 type factoring\n    methods.\n\n        >>> from sympy.ntheory.factor_ import smoothness_p, factorint\n        >>> smoothness_p(10431, m=1)\n        (1, [(3, (2, 2, 4)), (19, (1, 5, 5)), (61, (1, 31, 31))])\n        >>> smoothness_p(10431)\n        (-1, [(3, (2, 2, 2)), (19, (1, 3, 9)), (61, (1, 5, 5))])\n        >>> smoothness_p(10431, power=1)\n        (-1, [(3, (2, 2, 2)), (61, (1, 5, 5)), (19, (1, 3, 9))])\n\n    If visual=True then an annotated string will be returned:\n\n        >>> print(smoothness_p(21477639576571, visual=1))\n        p**i=4410317**1 has p-1 B=1787, B-pow=1787\n        p**i=4869863**1 has p-1 B=2434931, B-pow=2434931\n\n    This string can also be generated directly from a factorization dictionary\n    and vice versa:\n\n        >>> factorint(17*9)\n        {3: 2, 17: 1}\n        >>> smoothness_p(_)\n        'p**i=3**2 has p-1 B=2, B-pow=2\\np**i=17**1 has p-1 B=2, B-pow=16'\n        >>> smoothness_p(_)\n        {3: 2, 17: 1}\n\n    The table of the output logic is:\n\n        ====== ====== ======= =======\n        |              Visual\n        ------ ----------------------\n        Input  True   False   other\n        ====== ====== ======= =======\n        dict    str    tuple   str\n        str     str    tuple   dict\n        tuple   str    tuple   str\n        n       str    tuple   tuple\n        mul     str    tuple   tuple\n        ====== ====== ======= =======\n\n    See Also\n    ========\n\n    factorint, smoothness\n    "
    if visual in (1, 0):
        visual = bool(visual)
    elif visual not in (True, False):
        visual = None
    if isinstance(n, str):
        if visual:
            return n
        d = {}
        for li in n.splitlines():
            (k, v) = [int(i) for i in li.split('has')[0].split('=')[1].split('**')]
            d[k] = v
        if visual is not True and visual is not False:
            return d
        return smoothness_p(d, visual=False)
    elif not isinstance(n, tuple):
        facs = factorint(n, visual=False)
    if power:
        k = -1
    else:
        k = 1
    if isinstance(n, tuple):
        rv = n
    else:
        rv = (m, sorted([(f, tuple([M] + list(smoothness(f + m)))) for (f, M) in list(facs.items())], key=lambda x: (x[1][k], x[0])))
    if visual is False or (visual is not True and type(n) in [int, Mul]):
        return rv
    lines = []
    for dat in rv[1]:
        dat = flatten(dat)
        dat.insert(2, m)
        lines.append('p**i=%i**%i has p%+i B=%i, B-pow=%i' % tuple(dat))
    return '\n'.join(lines)

def multiplicity(p, n):
    if False:
        while True:
            i = 10
    '\n    Find the greatest integer m such that p**m divides n.\n\n    Examples\n    ========\n\n    >>> from sympy import multiplicity, Rational\n    >>> [multiplicity(5, n) for n in [8, 5, 25, 125, 250]]\n    [0, 1, 2, 3, 3]\n    >>> multiplicity(3, Rational(1, 9))\n    -2\n\n    Note: when checking for the multiplicity of a number in a\n    large factorial it is most efficient to send it as an unevaluated\n    factorial or to call ``multiplicity_in_factorial`` directly:\n\n    >>> from sympy.ntheory import multiplicity_in_factorial\n    >>> from sympy import factorial\n    >>> p = factorial(25)\n    >>> n = 2**100\n    >>> nfac = factorial(n, evaluate=False)\n    >>> multiplicity(p, nfac)\n    52818775009509558395695966887\n    >>> _ == multiplicity_in_factorial(p, n)\n    True\n\n    See Also\n    ========\n\n    trailing\n\n    '
    try:
        (p, n) = (as_int(p), as_int(n))
    except ValueError:
        from sympy.functions.combinatorial.factorials import factorial
        if all((isinstance(i, (SYMPY_INTS, Rational)) for i in (p, n))):
            p = Rational(p)
            n = Rational(n)
            if p.q == 1:
                if n.p == 1:
                    return -multiplicity(p.p, n.q)
                return multiplicity(p.p, n.p) - multiplicity(p.p, n.q)
            elif p.p == 1:
                return multiplicity(p.q, n.q)
            else:
                like = min(multiplicity(p.p, n.p), multiplicity(p.q, n.q))
                cross = min(multiplicity(p.q, n.p), multiplicity(p.p, n.q))
                return like - cross
        elif isinstance(p, (SYMPY_INTS, Integer)) and isinstance(n, factorial) and isinstance(n.args[0], Integer) and (n.args[0] >= 0):
            return multiplicity_in_factorial(p, n.args[0])
        raise ValueError('expecting ints or fractions, got %s and %s' % (p, n))
    if n == 0:
        raise ValueError('no such integer exists: multiplicity of %s is not-defined' % n)
    return remove(n, p)[1]

def multiplicity_in_factorial(p, n):
    if False:
        return 10
    'return the largest integer ``m`` such that ``p**m`` divides ``n!``\n    without calculating the factorial of ``n``.\n\n    Parameters\n    ==========\n\n    p : Integer\n        positive integer\n    n : Integer\n        non-negative integer\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory import multiplicity_in_factorial\n    >>> from sympy import factorial\n\n    >>> multiplicity_in_factorial(2, 3)\n    1\n\n    An instructive use of this is to tell how many trailing zeros\n    a given factorial has. For example, there are 6 in 25!:\n\n    >>> factorial(25)\n    15511210043330985984000000\n    >>> multiplicity_in_factorial(10, 25)\n    6\n\n    For large factorials, it is much faster/feasible to use\n    this function rather than computing the actual factorial:\n\n    >>> multiplicity_in_factorial(factorial(25), 2**100)\n    52818775009509558395695966887\n\n    See Also\n    ========\n\n    multiplicity\n\n    '
    (p, n) = (as_int(p), as_int(n))
    if p <= 0:
        raise ValueError('expecting positive integer got %s' % p)
    if n < 0:
        raise ValueError('expecting non-negative integer got %s' % n)
    f = defaultdict(int)
    for (k, v) in factorint(p).items():
        f[v] = max(k, f[v])
    return min(((n + k - sum(digits(n, k))) // (k - 1) // v for (v, k) in f.items()))

def _perfect_power(n, k=2):
    if False:
        return 10
    ' Return integers ``(b, e)`` such that ``n == b**e`` if ``n`` is a unique\n    perfect power with ``e > 1``, else ``False`` (e.g. 1 is not a perfect power).\n\n    Explanation\n    ===========\n\n    This is a low-level helper for ``perfect_power``, for internal use.\n\n    Parameters\n    ==========\n\n    n : int\n        assume that n is a nonnegative integer\n    k : int\n        Assume that n has no factor less than k.\n        i.e., all(n % p for p in range(2, k)) is True\n\n    Examples\n    ========\n    >>> from sympy.ntheory.factor_ import _perfect_power\n    >>> _perfect_power(16)\n    (2, 4)\n    >>> _perfect_power(17)\n    False\n\n    '
    if n <= 3:
        return False
    factors = {}
    g = 0
    multi = 1

    def done(n, factors, g, multi):
        if False:
            while True:
                i = 10
        g = gcd(g, multi)
        if g == 1:
            return False
        factors[n] = multi
        return (math.prod((p ** (e // g) for (p, e) in factors.items())), g)
    if n <= 1000000:
        n = _factorint_small(factors, n, 1000, 1000)[0]
        if n > 1:
            return False
        g = gcd(*factors.values())
        if g == 1:
            return False
        return (math.prod((p ** (e // g) for (p, e) in factors.items())), g)
    if k < 3:
        g = bit_scan1(n)
        if g:
            if g == 1:
                return False
            n >>= g
            factors[2] = g
            if n == 1:
                return (2, g)
            else:
                (m, _exact) = iroot(n, g)
                if _exact:
                    return (2 * m, g)
                elif isprime(g):
                    return False
        k = 3
    while n & 7 == 1:
        (m, _exact) = iroot(n, 2)
        if _exact:
            n = m
            multi <<= 1
        else:
            break
    if n < k ** 3:
        return done(n, factors, g, multi)
    tf_max = n.bit_length() // 27 + 24
    if k < tf_max:
        for p in primerange(k, tf_max):
            (m, t) = remove(n, p)
            if t:
                n = m
                t *= multi
                _g = gcd(g, t)
                if _g == 1:
                    return False
                factors[p] = t
                if n == 1:
                    return (math.prod((p ** (e // _g) for (p, e) in factors.items())), _g)
                elif g == 0 or _g < g:
                    g = _g
                    (m, _exact) = iroot(n ** multi, g)
                    if _exact:
                        return (m * math.prod((p ** (e // g) for (p, e) in factors.items())), g)
                    elif isprime(g):
                        return False
        k = tf_max
    if n < k ** 3:
        return done(n, factors, g, multi)
    if g:
        prime_iter = sorted(factorint(g >> bit_scan1(g)).keys())
    else:
        prime_iter = primerange(3, int(math.log(n, k)) + 2)
    logn = math.log2(n)
    threshold = logn / 40
    for p in prime_iter:
        if threshold < p:
            while True:
                b = pow(2, logn / p)
                rb = int(b + 0.5)
                if abs(rb - b) < 0.01 and rb ** p == n:
                    n = rb
                    multi *= p
                    logn = math.log2(n)
                else:
                    break
        else:
            while True:
                (m, _exact) = iroot(n, p)
                if _exact:
                    n = m
                    multi *= p
                    logn = math.log2(n)
                else:
                    break
        if n < k ** (p + 2):
            break
    return done(n, factors, g, multi)

def perfect_power(n, candidates=None, big=True, factor=True):
    if False:
        return 10
    '\n    Return ``(b, e)`` such that ``n`` == ``b**e`` if ``n`` is a unique\n    perfect power with ``e > 1``, else ``False`` (e.g. 1 is not a\n    perfect power). A ValueError is raised if ``n`` is not Rational.\n\n    By default, the base is recursively decomposed and the exponents\n    collected so the largest possible ``e`` is sought. If ``big=False``\n    then the smallest possible ``e`` (thus prime) will be chosen.\n\n    If ``factor=True`` then simultaneous factorization of ``n`` is\n    attempted since finding a factor indicates the only possible root\n    for ``n``. This is True by default since only a few small factors will\n    be tested in the course of searching for the perfect power.\n\n    The use of ``candidates`` is primarily for internal use; if provided,\n    False will be returned if ``n`` cannot be written as a power with one\n    of the candidates as an exponent and factoring (beyond testing for\n    a factor of 2) will not be attempted.\n\n    Examples\n    ========\n\n    >>> from sympy import perfect_power, Rational\n    >>> perfect_power(16)\n    (2, 4)\n    >>> perfect_power(16, big=False)\n    (4, 2)\n\n    Negative numbers can only have odd perfect powers:\n\n    >>> perfect_power(-4)\n    False\n    >>> perfect_power(-8)\n    (-2, 3)\n\n    Rationals are also recognized:\n\n    >>> perfect_power(Rational(1, 2)**3)\n    (1/2, 3)\n    >>> perfect_power(Rational(-3, 2)**3)\n    (-3/2, 3)\n\n    Notes\n    =====\n\n    To know whether an integer is a perfect power of 2 use\n\n        >>> is2pow = lambda n: bool(n and not n & (n - 1))\n        >>> [(i, is2pow(i)) for i in range(5)]\n        [(0, False), (1, True), (2, True), (3, False), (4, True)]\n\n    It is not necessary to provide ``candidates``. When provided\n    it will be assumed that they are ints. The first one that is\n    larger than the computed maximum possible exponent will signal\n    failure for the routine.\n\n        >>> perfect_power(3**8, [9])\n        False\n        >>> perfect_power(3**8, [2, 4, 8])\n        (3, 8)\n        >>> perfect_power(3**8, [4, 8], big=False)\n        (9, 4)\n\n    See Also\n    ========\n    sympy.core.intfunc.integer_nthroot\n    sympy.ntheory.primetest.is_square\n    '
    if isinstance(n, Rational) and (not n.is_Integer):
        (p, q) = n.as_numer_denom()
        if p is S.One:
            pp = perfect_power(q)
            if pp:
                pp = (n.func(1, pp[0]), pp[1])
        else:
            pp = perfect_power(p)
            if pp:
                (num, e) = pp
                pq = perfect_power(q, [e])
                if pq:
                    (den, _) = pq
                    pp = (n.func(num, den), e)
        return pp
    n = as_int(n)
    if n < 0:
        pp = perfect_power(-n)
        if pp:
            (b, e) = pp
            if e % 2:
                return (-b, e)
        return False
    if candidates is None and big:
        return _perfect_power(n)
    if n <= 3:
        return False
    logn = math.log(n, 2)
    max_possible = int(logn) + 2
    not_square = n % 10 in [2, 3, 7, 8]
    min_possible = 2 + not_square
    if not candidates:
        candidates = primerange(min_possible, max_possible)
    else:
        candidates = sorted([i for i in candidates if min_possible <= i < max_possible])
        if n % 2 == 0:
            e = bit_scan1(n)
            candidates = [i for i in candidates if e % i == 0]
        if big:
            candidates = reversed(candidates)
        for e in candidates:
            (r, ok) = iroot(n, e)
            if ok:
                return (int(r), e)
        return False

    def _factors():
        if False:
            print('Hello World!')
        rv = 2 + n % 2
        while True:
            yield rv
            rv = nextprime(rv)
    for (fac, e) in zip(_factors(), candidates):
        if factor and n % fac == 0:
            e = remove(n, fac)[1]
            if e == 1:
                return False
            (r, exact) = iroot(n, e)
            if not exact:
                m = n // fac ** e
                rE = perfect_power(m, candidates=divisors(e, generator=True))
                if not rE:
                    return False
                else:
                    (r, E) = rE
                    (r, e) = (fac ** (e // E) * r, E)
            if not big:
                e0 = primefactors(e)
                if e0[0] != e:
                    (r, e) = (r ** (e // e0[0]), e0[0])
            return (int(r), e)
        if logn / e < 40:
            b = 2.0 ** (logn / e)
            if abs(int(b + 0.5) - b) > 0.01:
                continue
        (r, exact) = iroot(n, e)
        if exact:
            if big:
                m = perfect_power(r, big=big, factor=factor)
                if m:
                    (r, e) = (m[0], e * m[1])
            return (int(r), e)
    return False

def pollard_rho(n, s=2, a=1, retries=5, seed=1234, max_steps=None, F=None):
    if False:
        print('Hello World!')
    '\n    Use Pollard\'s rho method to try to extract a nontrivial factor\n    of ``n``. The returned factor may be a composite number. If no\n    factor is found, ``None`` is returned.\n\n    The algorithm generates pseudo-random values of x with a generator\n    function, replacing x with F(x). If F is not supplied then the\n    function x**2 + ``a`` is used. The first value supplied to F(x) is ``s``.\n    Upon failure (if ``retries`` is > 0) a new ``a`` and ``s`` will be\n    supplied; the ``a`` will be ignored if F was supplied.\n\n    The sequence of numbers generated by such functions generally have a\n    a lead-up to some number and then loop around back to that number and\n    begin to repeat the sequence, e.g. 1, 2, 3, 4, 5, 3, 4, 5 -- this leader\n    and loop look a bit like the Greek letter rho, and thus the name, \'rho\'.\n\n    For a given function, very different leader-loop values can be obtained\n    so it is a good idea to allow for retries:\n\n    >>> from sympy.ntheory.generate import cycle_length\n    >>> n = 16843009\n    >>> F = lambda x:(2048*pow(x, 2, n) + 32767) % n\n    >>> for s in range(5):\n    ...     print(\'loop length = %4i; leader length = %3i\' % next(cycle_length(F, s)))\n    ...\n    loop length = 2489; leader length =  42\n    loop length =   78; leader length = 120\n    loop length = 1482; leader length =  99\n    loop length = 1482; leader length = 285\n    loop length = 1482; leader length = 100\n\n    Here is an explicit example where there is a two element leadup to\n    a sequence of 3 numbers (11, 14, 4) that then repeat:\n\n    >>> x=2\n    >>> for i in range(9):\n    ...     x=(x**2+12)%17\n    ...     print(x)\n    ...\n    16\n    13\n    11\n    14\n    4\n    11\n    14\n    4\n    11\n    >>> next(cycle_length(lambda x: (x**2+12)%17, 2))\n    (3, 2)\n    >>> list(cycle_length(lambda x: (x**2+12)%17, 2, values=True))\n    [16, 13, 11, 14, 4]\n\n    Instead of checking the differences of all generated values for a gcd\n    with n, only the kth and 2*kth numbers are checked, e.g. 1st and 2nd,\n    2nd and 4th, 3rd and 6th until it has been detected that the loop has been\n    traversed. Loops may be many thousands of steps long before rho finds a\n    factor or reports failure. If ``max_steps`` is specified, the iteration\n    is cancelled with a failure after the specified number of steps.\n\n    Examples\n    ========\n\n    >>> from sympy import pollard_rho\n    >>> n=16843009\n    >>> F=lambda x:(2048*pow(x,2,n) + 32767) % n\n    >>> pollard_rho(n, F=F)\n    257\n\n    Use the default setting with a bad value of ``a`` and no retries:\n\n    >>> pollard_rho(n, a=n-2, retries=0)\n\n    If retries is > 0 then perhaps the problem will correct itself when\n    new values are generated for a:\n\n    >>> pollard_rho(n, a=n-2, retries=1)\n    257\n\n    References\n    ==========\n\n    .. [1] Richard Crandall & Carl Pomerance (2005), "Prime Numbers:\n           A Computational Perspective", Springer, 2nd edition, 229-231\n\n    '
    n = int(n)
    if n < 5:
        raise ValueError('pollard_rho should receive n > 4')
    randint = _randint(seed + retries)
    V = s
    for i in range(retries + 1):
        U = V
        if not F:
            F = lambda x: (pow(x, 2, n) + a) % n
        j = 0
        while 1:
            if max_steps and j > max_steps:
                break
            j += 1
            U = F(U)
            V = F(F(V))
            g = gcd(U - V, n)
            if g == 1:
                continue
            if g == n:
                break
            return int(g)
        V = randint(0, n - 1)
        a = randint(1, n - 3)
        F = None
    return None

def pollard_pm1(n, B=10, a=2, retries=0, seed=1234):
    if False:
        return 10
    '\n    Use Pollard\'s p-1 method to try to extract a nontrivial factor\n    of ``n``. Either a divisor (perhaps composite) or ``None`` is returned.\n\n    The value of ``a`` is the base that is used in the test gcd(a**M - 1, n).\n    The default is 2.  If ``retries`` > 0 then if no factor is found after the\n    first attempt, a new ``a`` will be generated randomly (using the ``seed``)\n    and the process repeated.\n\n    Note: the value of M is lcm(1..B) = reduce(ilcm, range(2, B + 1)).\n\n    A search is made for factors next to even numbers having a power smoothness\n    less than ``B``. Choosing a larger B increases the likelihood of finding a\n    larger factor but takes longer. Whether a factor of n is found or not\n    depends on ``a`` and the power smoothness of the even number just less than\n    the factor p (hence the name p - 1).\n\n    Although some discussion of what constitutes a good ``a`` some\n    descriptions are hard to interpret. At the modular.math site referenced\n    below it is stated that if gcd(a**M - 1, n) = N then a**M % q**r is 1\n    for every prime power divisor of N. But consider the following:\n\n        >>> from sympy.ntheory.factor_ import smoothness_p, pollard_pm1\n        >>> n=257*1009\n        >>> smoothness_p(n)\n        (-1, [(257, (1, 2, 256)), (1009, (1, 7, 16))])\n\n    So we should (and can) find a root with B=16:\n\n        >>> pollard_pm1(n, B=16, a=3)\n        1009\n\n    If we attempt to increase B to 256 we find that it does not work:\n\n        >>> pollard_pm1(n, B=256)\n        >>>\n\n    But if the value of ``a`` is changed we find that only multiples of\n    257 work, e.g.:\n\n        >>> pollard_pm1(n, B=256, a=257)\n        1009\n\n    Checking different ``a`` values shows that all the ones that did not\n    work had a gcd value not equal to ``n`` but equal to one of the\n    factors:\n\n        >>> from sympy import ilcm, igcd, factorint, Pow\n        >>> M = 1\n        >>> for i in range(2, 256):\n        ...     M = ilcm(M, i)\n        ...\n        >>> set([igcd(pow(a, M, n) - 1, n) for a in range(2, 256) if\n        ...      igcd(pow(a, M, n) - 1, n) != n])\n        {1009}\n\n    But does aM % d for every divisor of n give 1?\n\n        >>> aM = pow(255, M, n)\n        >>> [(d, aM%Pow(*d.args)) for d in factorint(n, visual=True).args]\n        [(257**1, 1), (1009**1, 1)]\n\n    No, only one of them. So perhaps the principle is that a root will\n    be found for a given value of B provided that:\n\n    1) the power smoothness of the p - 1 value next to the root\n       does not exceed B\n    2) a**M % p != 1 for any of the divisors of n.\n\n    By trying more than one ``a`` it is possible that one of them\n    will yield a factor.\n\n    Examples\n    ========\n\n    With the default smoothness bound, this number cannot be cracked:\n\n        >>> from sympy.ntheory import pollard_pm1\n        >>> pollard_pm1(21477639576571)\n\n    Increasing the smoothness bound helps:\n\n        >>> pollard_pm1(21477639576571, B=2000)\n        4410317\n\n    Looking at the smoothness of the factors of this number we find:\n\n        >>> from sympy.ntheory.factor_ import smoothness_p, factorint\n        >>> print(smoothness_p(21477639576571, visual=1))\n        p**i=4410317**1 has p-1 B=1787, B-pow=1787\n        p**i=4869863**1 has p-1 B=2434931, B-pow=2434931\n\n    The B and B-pow are the same for the p - 1 factorizations of the divisors\n    because those factorizations had a very large prime factor:\n\n        >>> factorint(4410317 - 1)\n        {2: 2, 617: 1, 1787: 1}\n        >>> factorint(4869863-1)\n        {2: 1, 2434931: 1}\n\n    Note that until B reaches the B-pow value of 1787, the number is not cracked;\n\n        >>> pollard_pm1(21477639576571, B=1786)\n        >>> pollard_pm1(21477639576571, B=1787)\n        4410317\n\n    The B value has to do with the factors of the number next to the divisor,\n    not the divisors themselves. A worst case scenario is that the number next\n    to the factor p has a large prime divisisor or is a perfect power. If these\n    conditions apply then the power-smoothness will be about p/2 or p. The more\n    realistic is that there will be a large prime factor next to p requiring\n    a B value on the order of p/2. Although primes may have been searched for\n    up to this level, the p/2 is a factor of p - 1, something that we do not\n    know. The modular.math reference below states that 15% of numbers in the\n    range of 10**15 to 15**15 + 10**4 are 10**6 power smooth so a B of 10**6\n    will fail 85% of the time in that range. From 10**8 to 10**8 + 10**3 the\n    percentages are nearly reversed...but in that range the simple trial\n    division is quite fast.\n\n    References\n    ==========\n\n    .. [1] Richard Crandall & Carl Pomerance (2005), "Prime Numbers:\n           A Computational Perspective", Springer, 2nd edition, 236-238\n    .. [2] https://web.archive.org/web/20150716201437/http://modular.math.washington.edu/edu/2007/spring/ent/ent-html/node81.html\n    .. [3] https://www.cs.toronto.edu/~yuvalf/Factorization.pdf\n    '
    n = int(n)
    if n < 4 or B < 3:
        raise ValueError('pollard_pm1 should receive n > 3 and B > 2')
    randint = _randint(seed + B)
    for i in range(retries + 1):
        aM = a
        for p in sieve.primerange(2, B + 1):
            e = int(math.log(B, p))
            aM = pow(aM, pow(p, e), n)
        g = gcd(aM - 1, n)
        if 1 < g < n:
            return int(g)
        a = randint(2, n - 2)

def _trial(factors, n, candidates, verbose=False):
    if False:
        i = 10
        return i + 15
    '\n    Helper function for integer factorization. Trial factors ``n`\n    against all integers given in the sequence ``candidates``\n    and updates the dict ``factors`` in-place. Returns the reduced\n    value of ``n`` and a flag indicating whether any factors were found.\n    '
    if verbose:
        factors0 = list(factors.keys())
    nfactors = len(factors)
    for d in candidates:
        if n % d == 0:
            (n, m) = remove(n, d)
            factors[d] = m
    if verbose:
        for k in sorted(set(factors).difference(set(factors0))):
            print(factor_msg % (k, factors[k]))
    return (int(n), len(factors) != nfactors)

def _check_termination(factors, n, limitp1, use_trial, use_rho, use_pm1, verbose):
    if False:
        i = 10
        return i + 15
    '\n    Helper function for integer factorization. Checks if ``n``\n    is a prime or a perfect power, and in those cases updates\n    the factorization and raises ``StopIteration``.\n    '
    if verbose:
        print('Check for termination')
    if n == 1:
        if verbose:
            print(complete_msg)
        return True
    p = _perfect_power(n)
    if p is not False:
        (base, exp) = p
        if limitp1:
            limit = limitp1 - 1
        else:
            limit = limitp1
        facs = factorint(base, limit, use_trial, use_rho, use_pm1, verbose=False)
        for (b, e) in facs.items():
            if verbose:
                print(factor_msg % (b, e))
            factors[b] = int(exp * e)
        if verbose:
            print(complete_msg)
        return True
    if isprime(n):
        factors[int(n)] = 1
        if verbose:
            print(complete_msg)
        return True
    return False
trial_int_msg = 'Trial division with ints [%i ... %i] and fail_max=%i'
trial_msg = 'Trial division with primes [%i ... %i]'
rho_msg = "Pollard's rho with retries %i, max_steps %i and seed %i"
pm1_msg = "Pollard's p-1 with smoothness bound %i and seed %i"
ecm_msg = 'Elliptic Curve with B1 bound %i, B2 bound %i, num_curves %i'
factor_msg = '\t%i ** %i'
fermat_msg = 'Close factors satisying Fermat condition found.'
complete_msg = 'Factorization is complete.'

def _factorint_small(factors, n, limit, fail_max):
    if False:
        return 10
    '\n    Return the value of n and either a 0 (indicating that factorization up\n    to the limit was complete) or else the next near-prime that would have\n    been tested.\n\n    Factoring stops if there are fail_max unsuccessful tests in a row.\n\n    If factors of n were found they will be in the factors dictionary as\n    {factor: multiplicity} and the returned value of n will have had those\n    factors removed. The factors dictionary is modified in-place.\n\n    '

    def done(n, d):
        if False:
            print('Hello World!')
        'return n, d if the sqrt(n) was not reached yet, else\n           n, 0 indicating that factoring is done.\n        '
        if d * d <= n:
            return (n, d)
        return (n, 0)
    d = 2
    m = bit_scan1(n)
    if m:
        factors[d] = m
        n >>= m
    d = 3
    if limit < d:
        if n > 1:
            factors[n] = 1
        return done(n, d)
    m = 0
    while n % d == 0:
        n //= d
        m += 1
        if m == 20:
            (n, mm) = remove(n, d)
            m += mm
            break
    if m:
        factors[d] = m
    if limit * limit > n:
        maxx = 0
    else:
        maxx = limit * limit
    dd = maxx or n
    d = 5
    fails = 0
    while fails < fail_max:
        if d * d > dd:
            break
        m = 0
        while n % d == 0:
            n //= d
            m += 1
            if m == 20:
                (n, mm) = remove(n, d)
                m += mm
                break
        if m:
            factors[d] = m
            dd = maxx or n
            fails = 0
        else:
            fails += 1
        d += 2
        if d * d > dd:
            break
        m = 0
        while n % d == 0:
            n //= d
            m += 1
            if m == 20:
                (n, mm) = remove(n, d)
                m += mm
                break
        if m:
            factors[d] = m
            dd = maxx or n
            fails = 0
        else:
            fails += 1
        d += 4
    return done(n, d)

def factorint(n, limit=None, use_trial=True, use_rho=True, use_pm1=True, use_ecm=True, verbose=False, visual=None, multiple=False):
    if False:
        return 10
    "\n    Given a positive integer ``n``, ``factorint(n)`` returns a dict containing\n    the prime factors of ``n`` as keys and their respective multiplicities\n    as values. For example:\n\n    >>> from sympy.ntheory import factorint\n    >>> factorint(2000)    # 2000 = (2**4) * (5**3)\n    {2: 4, 5: 3}\n    >>> factorint(65537)   # This number is prime\n    {65537: 1}\n\n    For input less than 2, factorint behaves as follows:\n\n        - ``factorint(1)`` returns the empty factorization, ``{}``\n        - ``factorint(0)`` returns ``{0:1}``\n        - ``factorint(-n)`` adds ``-1:1`` to the factors and then factors ``n``\n\n    Partial Factorization:\n\n    If ``limit`` (> 3) is specified, the search is stopped after performing\n    trial division up to (and including) the limit (or taking a\n    corresponding number of rho/p-1 steps). This is useful if one has\n    a large number and only is interested in finding small factors (if\n    any). Note that setting a limit does not prevent larger factors\n    from being found early; it simply means that the largest factor may\n    be composite. Since checking for perfect power is relatively cheap, it is\n    done regardless of the limit setting.\n\n    This number, for example, has two small factors and a huge\n    semi-prime factor that cannot be reduced easily:\n\n    >>> from sympy.ntheory import isprime\n    >>> a = 1407633717262338957430697921446883\n    >>> f = factorint(a, limit=10000)\n    >>> f == {991: 1, int(202916782076162456022877024859): 1, 7: 1}\n    True\n    >>> isprime(max(f))\n    False\n\n    This number has a small factor and a residual perfect power whose\n    base is greater than the limit:\n\n    >>> factorint(3*101**7, limit=5)\n    {3: 1, 101: 7}\n\n    List of Factors:\n\n    If ``multiple`` is set to ``True`` then a list containing the\n    prime factors including multiplicities is returned.\n\n    >>> factorint(24, multiple=True)\n    [2, 2, 2, 3]\n\n    Visual Factorization:\n\n    If ``visual`` is set to ``True``, then it will return a visual\n    factorization of the integer.  For example:\n\n    >>> from sympy import pprint\n    >>> pprint(factorint(4200, visual=True))\n     3  1  2  1\n    2 *3 *5 *7\n\n    Note that this is achieved by using the evaluate=False flag in Mul\n    and Pow. If you do other manipulations with an expression where\n    evaluate=False, it may evaluate.  Therefore, you should use the\n    visual option only for visualization, and use the normal dictionary\n    returned by visual=False if you want to perform operations on the\n    factors.\n\n    You can easily switch between the two forms by sending them back to\n    factorint:\n\n    >>> from sympy import Mul\n    >>> regular = factorint(1764); regular\n    {2: 2, 3: 2, 7: 2}\n    >>> pprint(factorint(regular))\n     2  2  2\n    2 *3 *7\n\n    >>> visual = factorint(1764, visual=True); pprint(visual)\n     2  2  2\n    2 *3 *7\n    >>> print(factorint(visual))\n    {2: 2, 3: 2, 7: 2}\n\n    If you want to send a number to be factored in a partially factored form\n    you can do so with a dictionary or unevaluated expression:\n\n    >>> factorint(factorint({4: 2, 12: 3})) # twice to toggle to dict form\n    {2: 10, 3: 3}\n    >>> factorint(Mul(4, 12, evaluate=False))\n    {2: 4, 3: 1}\n\n    The table of the output logic is:\n\n        ====== ====== ======= =======\n                       Visual\n        ------ ----------------------\n        Input  True   False   other\n        ====== ====== ======= =======\n        dict    mul    dict    mul\n        n       mul    dict    dict\n        mul     mul    dict    dict\n        ====== ====== ======= =======\n\n    Notes\n    =====\n\n    Algorithm:\n\n    The function switches between multiple algorithms. Trial division\n    quickly finds small factors (of the order 1-5 digits), and finds\n    all large factors if given enough time. The Pollard rho and p-1\n    algorithms are used to find large factors ahead of time; they\n    will often find factors of the order of 10 digits within a few\n    seconds:\n\n    >>> factors = factorint(12345678910111213141516)\n    >>> for base, exp in sorted(factors.items()):\n    ...     print('%s %s' % (base, exp))\n    ...\n    2 2\n    2507191691 1\n    1231026625769 1\n\n    Any of these methods can optionally be disabled with the following\n    boolean parameters:\n\n        - ``use_trial``: Toggle use of trial division\n        - ``use_rho``: Toggle use of Pollard's rho method\n        - ``use_pm1``: Toggle use of Pollard's p-1 method\n\n    ``factorint`` also periodically checks if the remaining part is\n    a prime number or a perfect power, and in those cases stops.\n\n    For unevaluated factorial, it uses Legendre's formula(theorem).\n\n\n    If ``verbose`` is set to ``True``, detailed progress is printed.\n\n    See Also\n    ========\n\n    smoothness, smoothness_p, divisors\n\n    "
    if isinstance(n, Dict):
        n = dict(n)
    if multiple:
        fac = factorint(n, limit=limit, use_trial=use_trial, use_rho=use_rho, use_pm1=use_pm1, verbose=verbose, visual=False, multiple=False)
        factorlist = sum(([p] * fac[p] if fac[p] > 0 else [S.One / p] * -fac[p] for p in sorted(fac)), [])
        return factorlist
    factordict = {}
    if visual and (not isinstance(n, (Mul, dict))):
        factordict = factorint(n, limit=limit, use_trial=use_trial, use_rho=use_rho, use_pm1=use_pm1, verbose=verbose, visual=False)
    elif isinstance(n, Mul):
        factordict = {int(k): int(v) for (k, v) in n.as_powers_dict().items()}
    elif isinstance(n, dict):
        factordict = n
    if factordict and isinstance(n, (Mul, dict)):
        for key in list(factordict.keys()):
            if isprime(key):
                continue
            e = factordict.pop(key)
            d = factorint(key, limit=limit, use_trial=use_trial, use_rho=use_rho, use_pm1=use_pm1, verbose=verbose, visual=False)
            for (k, v) in d.items():
                if k in factordict:
                    factordict[k] += v * e
                else:
                    factordict[k] = v * e
    if visual or (type(n) is dict and visual is not True and (visual is not False)):
        if factordict == {}:
            return S.One
        if -1 in factordict:
            factordict.pop(-1)
            args = [S.NegativeOne]
        else:
            args = []
        args.extend([Pow(*i, evaluate=False) for i in sorted(factordict.items())])
        return Mul(*args, evaluate=False)
    elif isinstance(n, (dict, Mul)):
        return factordict
    assert use_trial or use_rho or use_pm1 or use_ecm
    from sympy.functions.combinatorial.factorials import factorial
    if isinstance(n, factorial):
        x = as_int(n.args[0])
        if x >= 20:
            factors = {}
            m = 2
            for p in sieve.primerange(2, x + 1):
                if m > 1:
                    (m, q) = (0, x // p)
                    while q != 0:
                        m += q
                        q //= p
                factors[p] = m
            if factors and verbose:
                for k in sorted(factors):
                    print(factor_msg % (k, factors[k]))
            if verbose:
                print(complete_msg)
            return factors
        else:
            n = n.func(x)
    n = as_int(n)
    if limit:
        limit = int(limit)
        use_ecm = False
    if n < 0:
        factors = factorint(-n, limit=limit, use_trial=use_trial, use_rho=use_rho, use_pm1=use_pm1, verbose=verbose, visual=False)
        factors[-1] = 1
        return factors
    if limit and limit < 2:
        if n == 1:
            return {}
        return {n: 1}
    elif n < 10:
        return [{0: 1}, {}, {2: 1}, {3: 1}, {2: 2}, {5: 1}, {2: 1, 3: 1}, {7: 1}, {2: 3}, {3: 2}][n]
    factors = {}
    if verbose:
        sn = str(n)
        if len(sn) > 50:
            print('Factoring %s' % sn[:5] + '..(%i other digits)..' % (len(sn) - 10) + sn[-5:])
        else:
            print('Factoring', n)
    if use_trial:
        small = 2 ** 15
        fail_max = 600
        small = min(small, limit or small)
        if verbose:
            print(trial_int_msg % (2, small, fail_max))
        (n, next_p) = _factorint_small(factors, n, small, fail_max)
    else:
        next_p = 2
    if factors and verbose:
        for k in sorted(factors):
            print(factor_msg % (k, factors[k]))
    if next_p == 0:
        if n > 1:
            factors[int(n)] = 1
        if verbose:
            print(complete_msg)
        return factors
    if limit and next_p > limit:
        if verbose:
            print('Exceeded limit:', limit)
        if _check_termination(factors, n, limit, use_trial, use_rho, use_pm1, verbose):
            return factors
        if n > 1:
            factors[int(n)] = 1
        return factors
    if _check_termination(factors, n, limit, use_trial, use_rho, use_pm1, verbose):
        return factors
    sqrt_n = isqrt(n)
    a = sqrt_n + 1
    a2 = a ** 2
    b2 = a2 - n
    for _ in range(3):
        (b, fermat) = sqrtrem(b2)
        if not fermat:
            if verbose:
                print(fermat_msg)
            if limit:
                limit -= 1
            for r in [a - b, a + b]:
                facs = factorint(r, limit=limit, use_trial=use_trial, use_rho=use_rho, use_pm1=use_pm1, verbose=verbose)
                for (k, v) in facs.items():
                    factors[k] = factors.get(k, 0) + v
            if verbose:
                print(complete_msg)
            return factors
        b2 += 2 * a + 1
        a += 1
    (low, high) = (next_p, 2 * next_p)
    limit = limit or sqrt_n
    limit += 1
    iteration = 0
    while 1:
        high_ = high
        if limit < high_:
            high_ = limit
        if use_trial:
            if verbose:
                print(trial_msg % (low, high_))
            ps = sieve.primerange(low, high_)
            (n, found_trial) = _trial(factors, n, ps, verbose)
            if found_trial and _check_termination(factors, n, limit, use_trial, use_rho, use_pm1, verbose):
                return factors
        else:
            found_trial = False
        if high > limit:
            if verbose:
                print('Exceeded limit:', limit)
            if n > 1:
                factors[int(n)] = 1
            if verbose:
                print(complete_msg)
            return factors
        if not found_trial and (use_pm1 or use_rho):
            high_root = max(int(math.log(high_ ** 0.7)), low, 3)
            if use_pm1:
                if verbose:
                    print(pm1_msg % (high_root, high_))
                c = pollard_pm1(n, B=high_root, seed=high_)
                if c:
                    ps = factorint(c, limit=limit - 1, use_trial=use_trial, use_rho=use_rho, use_pm1=use_pm1, use_ecm=use_ecm, verbose=verbose)
                    (n, _) = _trial(factors, n, ps, verbose=False)
                    if _check_termination(factors, n, limit, use_trial, use_rho, use_pm1, verbose):
                        return factors
            if use_rho:
                max_steps = high_root
                if verbose:
                    print(rho_msg % (1, max_steps, high_))
                c = pollard_rho(n, retries=1, max_steps=max_steps, seed=high_)
                if c:
                    ps = factorint(c, limit=limit - 1, use_trial=use_trial, use_rho=use_rho, use_pm1=use_pm1, use_ecm=use_ecm, verbose=verbose)
                    (n, _) = _trial(factors, n, ps, verbose=False)
                    if _check_termination(factors, n, limit, use_trial, use_rho, use_pm1, verbose):
                        return factors
        iteration += 1
        if use_ecm and iteration >= 3 and (num_digits(n) >= 24):
            break
        (low, high) = (high, high * 2)
    B1 = 10000
    B2 = 100 * B1
    num_curves = 50
    while 1:
        if verbose:
            print(ecm_msg % (B1, B2, num_curves))
        factor = _ecm_one_factor(n, B1, B2, num_curves, seed=B1)
        if factor:
            ps = factorint(factor, limit=limit - 1, use_trial=use_trial, use_rho=use_rho, use_pm1=use_pm1, use_ecm=use_ecm, verbose=verbose)
            (n, _) = _trial(factors, n, ps, verbose=False)
            if _check_termination(factors, n, limit, use_trial, use_rho, use_pm1, verbose):
                return factors
        B1 *= 5
        B2 = 100 * B1
        num_curves *= 4

def factorrat(rat, limit=None, use_trial=True, use_rho=True, use_pm1=True, verbose=False, visual=None, multiple=False):
    if False:
        return 10
    "\n    Given a Rational ``r``, ``factorrat(r)`` returns a dict containing\n    the prime factors of ``r`` as keys and their respective multiplicities\n    as values. For example:\n\n    >>> from sympy import factorrat, S\n    >>> factorrat(S(8)/9)    # 8/9 = (2**3) * (3**-2)\n    {2: 3, 3: -2}\n    >>> factorrat(S(-1)/987)    # -1/789 = -1 * (3**-1) * (7**-1) * (47**-1)\n    {-1: 1, 3: -1, 7: -1, 47: -1}\n\n    Please see the docstring for ``factorint`` for detailed explanations\n    and examples of the following keywords:\n\n        - ``limit``: Integer limit up to which trial division is done\n        - ``use_trial``: Toggle use of trial division\n        - ``use_rho``: Toggle use of Pollard's rho method\n        - ``use_pm1``: Toggle use of Pollard's p-1 method\n        - ``verbose``: Toggle detailed printing of progress\n        - ``multiple``: Toggle returning a list of factors or dict\n        - ``visual``: Toggle product form of output\n    "
    if multiple:
        fac = factorrat(rat, limit=limit, use_trial=use_trial, use_rho=use_rho, use_pm1=use_pm1, verbose=verbose, visual=False, multiple=False)
        factorlist = sum(([p] * fac[p] if fac[p] > 0 else [S.One / p] * -fac[p] for (p, _) in sorted(fac.items(), key=lambda elem: elem[0] if elem[1] > 0 else 1 / elem[0])), [])
        return factorlist
    f = factorint(rat.p, limit=limit, use_trial=use_trial, use_rho=use_rho, use_pm1=use_pm1, verbose=verbose).copy()
    f = defaultdict(int, f)
    for (p, e) in factorint(rat.q, limit=limit, use_trial=use_trial, use_rho=use_rho, use_pm1=use_pm1, verbose=verbose).items():
        f[p] += -e
    if len(f) > 1 and 1 in f:
        del f[1]
    if not visual:
        return dict(f)
    else:
        if -1 in f:
            f.pop(-1)
            args = [S.NegativeOne]
        else:
            args = []
        args.extend([Pow(*i, evaluate=False) for i in sorted(f.items())])
        return Mul(*args, evaluate=False)

def primefactors(n, limit=None, verbose=False, **kwargs):
    if False:
        print('Hello World!')
    "Return a sorted list of n's prime factors, ignoring multiplicity\n    and any composite factor that remains if the limit was set too low\n    for complete factorization. Unlike factorint(), primefactors() does\n    not return -1 or 0.\n\n    Parameters\n    ==========\n\n    n : integer\n    limit, verbose, **kwargs :\n        Additional keyword arguments to be passed to ``factorint``.\n        Since ``kwargs`` is new in version 1.13,\n        ``limit`` and ``verbose`` are retained for compatibility purposes.\n\n    Returns\n    =======\n\n    list(int) : List of prime numbers dividing ``n``\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory import primefactors, factorint, isprime\n    >>> primefactors(6)\n    [2, 3]\n    >>> primefactors(-5)\n    [5]\n\n    >>> sorted(factorint(123456).items())\n    [(2, 6), (3, 1), (643, 1)]\n    >>> primefactors(123456)\n    [2, 3, 643]\n\n    >>> sorted(factorint(10000000001, limit=200).items())\n    [(101, 1), (99009901, 1)]\n    >>> isprime(99009901)\n    False\n    >>> primefactors(10000000001, limit=300)\n    [101]\n\n    See Also\n    ========\n\n    factorint, divisors\n\n    "
    n = int(n)
    kwargs.update({'visual': None, 'multiple': False, 'limit': limit, 'verbose': verbose})
    factors = sorted(factorint(n=n, **kwargs).keys())
    s = [f for f in factors[:-1] if f not in [-1, 0, 1]]
    if factors and isprime(factors[-1]):
        s += [factors[-1]]
    return s

def _divisors(n, proper=False):
    if False:
        return 10
    'Helper function for divisors which generates the divisors.\n\n    Parameters\n    ==========\n\n    n : int\n        a nonnegative integer\n    proper: bool\n        If `True`, returns the generator that outputs only the proper divisor (i.e., excluding n).\n\n    '
    if n <= 1:
        if not proper and n:
            yield 1
        return
    factordict = factorint(n)
    ps = sorted(factordict.keys())

    def rec_gen(n=0):
        if False:
            for i in range(10):
                print('nop')
        if n == len(ps):
            yield 1
        else:
            pows = [1]
            for _ in range(factordict[ps[n]]):
                pows.append(pows[-1] * ps[n])
            yield from (p * q for q in rec_gen(n + 1) for p in pows)
    if proper:
        yield from (p for p in rec_gen() if p != n)
    else:
        yield from rec_gen()

def divisors(n, generator=False, proper=False):
    if False:
        return 10
    '\n    Return all divisors of n sorted from 1..n by default.\n    If generator is ``True`` an unordered generator is returned.\n\n    The number of divisors of n can be quite large if there are many\n    prime factors (counting repeated factors). If only the number of\n    factors is desired use divisor_count(n).\n\n    Examples\n    ========\n\n    >>> from sympy import divisors, divisor_count\n    >>> divisors(24)\n    [1, 2, 3, 4, 6, 8, 12, 24]\n    >>> divisor_count(24)\n    8\n\n    >>> list(divisors(120, generator=True))\n    [1, 2, 4, 8, 3, 6, 12, 24, 5, 10, 20, 40, 15, 30, 60, 120]\n\n    Notes\n    =====\n\n    This is a slightly modified version of Tim Peters referenced at:\n    https://stackoverflow.com/questions/1010381/python-factorization\n\n    See Also\n    ========\n\n    primefactors, factorint, divisor_count\n    '
    rv = _divisors(as_int(abs(n)), proper)
    return rv if generator else sorted(rv)

def divisor_count(n, modulus=1, proper=False):
    if False:
        i = 10
        return i + 15
    '\n    Return the number of divisors of ``n``. If ``modulus`` is not 1 then only\n    those that are divisible by ``modulus`` are counted. If ``proper`` is True\n    then the divisor of ``n`` will not be counted.\n\n    Examples\n    ========\n\n    >>> from sympy import divisor_count\n    >>> divisor_count(6)\n    4\n    >>> divisor_count(6, 2)\n    2\n    >>> divisor_count(6, proper=True)\n    3\n\n    See Also\n    ========\n\n    factorint, divisors, totient, proper_divisor_count\n\n    '
    if not modulus:
        return 0
    elif modulus != 1:
        (n, r) = divmod(n, modulus)
        if r:
            return 0
    if n == 0:
        return 0
    n = Mul(*[v + 1 for (k, v) in factorint(n).items() if k > 1])
    if n and proper:
        n -= 1
    return n

def proper_divisors(n, generator=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return all divisors of n except n, sorted by default.\n    If generator is ``True`` an unordered generator is returned.\n\n    Examples\n    ========\n\n    >>> from sympy import proper_divisors, proper_divisor_count\n    >>> proper_divisors(24)\n    [1, 2, 3, 4, 6, 8, 12]\n    >>> proper_divisor_count(24)\n    7\n    >>> list(proper_divisors(120, generator=True))\n    [1, 2, 4, 8, 3, 6, 12, 24, 5, 10, 20, 40, 15, 30, 60]\n\n    See Also\n    ========\n\n    factorint, divisors, proper_divisor_count\n\n    '
    return divisors(n, generator=generator, proper=True)

def proper_divisor_count(n, modulus=1):
    if False:
        print('Hello World!')
    '\n    Return the number of proper divisors of ``n``.\n\n    Examples\n    ========\n\n    >>> from sympy import proper_divisor_count\n    >>> proper_divisor_count(6)\n    3\n    >>> proper_divisor_count(6, modulus=2)\n    1\n\n    See Also\n    ========\n\n    divisors, proper_divisors, divisor_count\n\n    '
    return divisor_count(n, modulus=modulus, proper=True)

def _udivisors(n):
    if False:
        print('Hello World!')
    'Helper function for udivisors which generates the unitary divisors.\n\n    Parameters\n    ==========\n\n    n : int\n        a nonnegative integer\n\n    '
    if n <= 1:
        if n == 1:
            yield 1
        return
    factorpows = [p ** e for (p, e) in factorint(n).items()]
    for i in range(2 ** len(factorpows)):
        d = 1
        for k in range(i.bit_length()):
            if i & 1:
                d *= factorpows[k]
            i >>= 1
        yield d

def udivisors(n, generator=False):
    if False:
        while True:
            i = 10
    '\n    Return all unitary divisors of n sorted from 1..n by default.\n    If generator is ``True`` an unordered generator is returned.\n\n    The number of unitary divisors of n can be quite large if there are many\n    prime factors. If only the number of unitary divisors is desired use\n    udivisor_count(n).\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.factor_ import udivisors, udivisor_count\n    >>> udivisors(15)\n    [1, 3, 5, 15]\n    >>> udivisor_count(15)\n    4\n\n    >>> sorted(udivisors(120, generator=True))\n    [1, 3, 5, 8, 15, 24, 40, 120]\n\n    See Also\n    ========\n\n    primefactors, factorint, divisors, divisor_count, udivisor_count\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Unitary_divisor\n    .. [2] https://mathworld.wolfram.com/UnitaryDivisor.html\n\n    '
    rv = _udivisors(as_int(abs(n)))
    return rv if generator else sorted(rv)

def udivisor_count(n):
    if False:
        while True:
            i = 10
    '\n    Return the number of unitary divisors of ``n``.\n\n    Parameters\n    ==========\n\n    n : integer\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.factor_ import udivisor_count\n    >>> udivisor_count(120)\n    8\n\n    See Also\n    ========\n\n    factorint, divisors, udivisors, divisor_count, totient\n\n    References\n    ==========\n\n    .. [1] https://mathworld.wolfram.com/UnitaryDivisorFunction.html\n\n    '
    if n == 0:
        return 0
    return 2 ** len([p for p in factorint(n) if p > 1])

def _antidivisors(n):
    if False:
        while True:
            i = 10
    'Helper function for antidivisors which generates the antidivisors.\n\n    Parameters\n    ==========\n\n    n : int\n        a nonnegative integer\n\n    '
    if n <= 2:
        return
    for d in _divisors(n):
        y = 2 * d
        if n > y and n % y:
            yield y
    for d in _divisors(2 * n - 1):
        if n > d >= 2 and n % d:
            yield d
    for d in _divisors(2 * n + 1):
        if n > d >= 2 and n % d:
            yield d

def antidivisors(n, generator=False):
    if False:
        while True:
            i = 10
    '\n    Return all antidivisors of n sorted from 1..n by default.\n\n    Antidivisors [1]_ of n are numbers that do not divide n by the largest\n    possible margin.  If generator is True an unordered generator is returned.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.factor_ import antidivisors\n    >>> antidivisors(24)\n    [7, 16]\n\n    >>> sorted(antidivisors(128, generator=True))\n    [3, 5, 15, 17, 51, 85]\n\n    See Also\n    ========\n\n    primefactors, factorint, divisors, divisor_count, antidivisor_count\n\n    References\n    ==========\n\n    .. [1] definition is described in https://oeis.org/A066272/a066272a.html\n\n    '
    rv = _antidivisors(as_int(abs(n)))
    return rv if generator else sorted(rv)

def antidivisor_count(n):
    if False:
        print('Hello World!')
    '\n    Return the number of antidivisors [1]_ of ``n``.\n\n    Parameters\n    ==========\n\n    n : integer\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.factor_ import antidivisor_count\n    >>> antidivisor_count(13)\n    4\n    >>> antidivisor_count(27)\n    5\n\n    See Also\n    ========\n\n    factorint, divisors, antidivisors, divisor_count, totient\n\n    References\n    ==========\n\n    .. [1] formula from https://oeis.org/A066272\n\n    '
    n = as_int(abs(n))
    if n <= 2:
        return 0
    return divisor_count(2 * n - 1) + divisor_count(2 * n + 1) + divisor_count(n) - divisor_count(n, 2) - 5

class totient(Function):
    """
    Calculate the Euler totient function phi(n)

    ``totient(n)`` or `\\phi(n)` is the number of positive integers `\\leq` n
    that are relatively prime to n.

    Parameters
    ==========

    n : integer

    Examples
    ========

    >>> from sympy.ntheory import totient
    >>> totient(1)
    1
    >>> totient(25)
    20
    >>> totient(45) == totient(5)*totient(9)
    True

    See Also
    ========

    divisor_count

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Euler%27s_totient_function
    .. [2] https://mathworld.wolfram.com/TotientFunction.html

    """

    @classmethod
    def eval(cls, n):
        if False:
            for i in range(10):
                print('nop')
        if n.is_Integer:
            if n < 1:
                raise ValueError('n must be a positive integer')
            factors = factorint(n)
            return cls._from_factors(factors)
        elif not isinstance(n, Expr) or n.is_integer is False or n.is_positive is False:
            raise ValueError('n must be a positive integer')

    def _eval_is_integer(self):
        if False:
            i = 10
            return i + 15
        return fuzzy_and([self.args[0].is_integer, self.args[0].is_positive])

    @classmethod
    def _from_distinct_primes(self, *args):
        if False:
            print('Hello World!')
        'Subroutine to compute totient from the list of assumed\n        distinct primes\n\n        Examples\n        ========\n\n        >>> from sympy.ntheory.factor_ import totient\n        >>> totient._from_distinct_primes(5, 7)\n        24\n        '
        return reduce(lambda i, j: i * (j - 1), args, 1)

    @classmethod
    def _from_factors(self, factors):
        if False:
            while True:
                i = 10
        'Subroutine to compute totient from already-computed factors\n\n        Examples\n        ========\n\n        >>> from sympy.ntheory.factor_ import totient\n        >>> totient._from_factors({5: 2})\n        20\n        '
        t = 1
        for (p, k) in factors.items():
            t *= (p - 1) * p ** (k - 1)
        return t

class reduced_totient(Function):
    """
    Calculate the Carmichael reduced totient function lambda(n)

    ``reduced_totient(n)`` or `\\lambda(n)` is the smallest m > 0 such that
    `k^m \\equiv 1 \\mod n` for all k relatively prime to n.

    Examples
    ========

    >>> from sympy.ntheory import reduced_totient
    >>> reduced_totient(1)
    1
    >>> reduced_totient(8)
    2
    >>> reduced_totient(30)
    4

    See Also
    ========

    totient

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Carmichael_function
    .. [2] https://mathworld.wolfram.com/CarmichaelFunction.html

    """

    @classmethod
    def eval(cls, n):
        if False:
            for i in range(10):
                print('nop')
        if n.is_Integer:
            if n < 1:
                raise ValueError('n must be a positive integer')
            factors = factorint(n)
            return cls._from_factors(factors)

    @classmethod
    def _from_factors(self, factors):
        if False:
            i = 10
            return i + 15
        'Subroutine to compute totient from already-computed factors\n        '
        t = 1
        for (p, k) in factors.items():
            if p == 2 and k > 2:
                t = lcm(t, 2 ** (k - 2))
            else:
                t = lcm(t, (p - 1) * p ** (k - 1))
        return t

    @classmethod
    def _from_distinct_primes(self, *args):
        if False:
            i = 10
            return i + 15
        'Subroutine to compute totient from the list of assumed\n        distinct primes\n        '
        args = [p - 1 for p in args]
        return lcm(*args)

    def _eval_is_integer(self):
        if False:
            while True:
                i = 10
        return fuzzy_and([self.args[0].is_integer, self.args[0].is_positive])

class divisor_sigma(Function):
    """
    Calculate the divisor function `\\sigma_k(n)` for positive integer n

    ``divisor_sigma(n, k)`` is equal to ``sum([x**k for x in divisors(n)])``

    If n's prime factorization is:

    .. math ::
        n = \\prod_{i=1}^\\omega p_i^{m_i},

    then

    .. math ::
        \\sigma_k(n) = \\prod_{i=1}^\\omega (1+p_i^k+p_i^{2k}+\\cdots
        + p_i^{m_ik}).

    Parameters
    ==========

    n : integer

    k : integer, optional
        power of divisors in the sum

        for k = 0, 1:
        ``divisor_sigma(n, 0)`` is equal to ``divisor_count(n)``
        ``divisor_sigma(n, 1)`` is equal to ``sum(divisors(n))``

        Default for k is 1.

    Examples
    ========

    >>> from sympy.ntheory import divisor_sigma
    >>> divisor_sigma(18, 0)
    6
    >>> divisor_sigma(39, 1)
    56
    >>> divisor_sigma(12, 2)
    210
    >>> divisor_sigma(37)
    38

    See Also
    ========

    divisor_count, totient, divisors, factorint

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Divisor_function

    """

    @classmethod
    def eval(cls, n, k=S.One):
        if False:
            print('Hello World!')
        k = sympify(k)
        if n.is_prime:
            return 1 + n ** k
        if n.is_Integer:
            if n <= 0:
                raise ValueError('n must be a positive integer')
            elif k.is_Integer:
                k = int(k)
                return Integer(math.prod(((p ** (k * (e + 1)) - 1) // (p ** k - 1) if k != 0 else e + 1 for (p, e) in factorint(n).items())))
            else:
                return Mul(*[(p ** (k * (e + 1)) - 1) / (p ** k - 1) if k != 0 else e + 1 for (p, e) in factorint(n).items()])
        if n.is_integer:
            args = []
            for (p, e) in (_.as_base_exp() for _ in Mul.make_args(n)):
                if p.is_prime and e.is_positive:
                    args.append((p ** (k * (e + 1)) - 1) / (p ** k - 1) if k != 0 else e + 1)
                else:
                    return
            return Mul(*args)

def core(n, t=2):
    if False:
        print('Hello World!')
    "\n    Calculate core(n, t) = `core_t(n)` of a positive integer n\n\n    ``core_2(n)`` is equal to the squarefree part of n\n\n    If n's prime factorization is:\n\n    .. math ::\n        n = \\prod_{i=1}^\\omega p_i^{m_i},\n\n    then\n\n    .. math ::\n        core_t(n) = \\prod_{i=1}^\\omega p_i^{m_i \\mod t}.\n\n    Parameters\n    ==========\n\n    n : integer\n\n    t : integer\n        core(n, t) calculates the t-th power free part of n\n\n        ``core(n, 2)`` is the squarefree part of ``n``\n        ``core(n, 3)`` is the cubefree part of ``n``\n\n        Default for t is 2.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.factor_ import core\n    >>> core(24, 2)\n    6\n    >>> core(9424, 3)\n    1178\n    >>> core(379238)\n    379238\n    >>> core(15**11, 10)\n    15\n\n    See Also\n    ========\n\n    factorint, sympy.solvers.diophantine.diophantine.square_factor\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Square-free_integer#Squarefree_core\n\n    "
    n = as_int(n)
    t = as_int(t)
    if n <= 0:
        raise ValueError('n must be a positive integer')
    elif t <= 1:
        raise ValueError('t must be >= 2')
    else:
        y = 1
        for (p, e) in factorint(n).items():
            y *= p ** (e % t)
        return y

class udivisor_sigma(Function):
    """
    Calculate the unitary divisor function `\\sigma_k^*(n)` for positive integer n

    ``udivisor_sigma(n, k)`` is equal to ``sum([x**k for x in udivisors(n)])``

    If n's prime factorization is:

    .. math ::
        n = \\prod_{i=1}^\\omega p_i^{m_i},

    then

    .. math ::
        \\sigma_k^*(n) = \\prod_{i=1}^\\omega (1+ p_i^{m_ik}).

    Parameters
    ==========

    k : power of divisors in the sum

        for k = 0, 1:
        ``udivisor_sigma(n, 0)`` is equal to ``udivisor_count(n)``
        ``udivisor_sigma(n, 1)`` is equal to ``sum(udivisors(n))``

        Default for k is 1.

    Examples
    ========

    >>> from sympy.ntheory.factor_ import udivisor_sigma
    >>> udivisor_sigma(18, 0)
    4
    >>> udivisor_sigma(74, 1)
    114
    >>> udivisor_sigma(36, 3)
    47450
    >>> udivisor_sigma(111)
    152

    See Also
    ========

    divisor_count, totient, divisors, udivisors, udivisor_count, divisor_sigma,
    factorint

    References
    ==========

    .. [1] https://mathworld.wolfram.com/UnitaryDivisorFunction.html

    """

    @classmethod
    def eval(cls, n, k=S.One):
        if False:
            while True:
                i = 10
        k = sympify(k)
        if n.is_prime:
            return 1 + n ** k
        if n.is_Integer:
            if n <= 0:
                raise ValueError('n must be a positive integer')
            else:
                return Mul(*[1 + p ** (k * e) for (p, e) in factorint(n).items()])

class primenu(Function):
    """
    Calculate the number of distinct prime factors for a positive integer n.

    If n's prime factorization is:

    .. math ::
        n = \\prod_{i=1}^k p_i^{m_i},

    then ``primenu(n)`` or `\\nu(n)` is:

    .. math ::
        \\nu(n) = k.

    Examples
    ========

    >>> from sympy.ntheory.factor_ import primenu
    >>> primenu(1)
    0
    >>> primenu(30)
    3

    See Also
    ========

    factorint

    References
    ==========

    .. [1] https://mathworld.wolfram.com/PrimeFactor.html

    """

    @classmethod
    def eval(cls, n):
        if False:
            for i in range(10):
                print('nop')
        if n.is_Integer:
            if n <= 0:
                raise ValueError('n must be a positive integer')
            else:
                return len(factorint(n).keys())

class primeomega(Function):
    """
    Calculate the number of prime factors counting multiplicities for a
    positive integer n.

    If n's prime factorization is:

    .. math ::
        n = \\prod_{i=1}^k p_i^{m_i},

    then ``primeomega(n)``  or `\\Omega(n)` is:

    .. math ::
        \\Omega(n) = \\sum_{i=1}^k m_i.

    Examples
    ========

    >>> from sympy.ntheory.factor_ import primeomega
    >>> primeomega(1)
    0
    >>> primeomega(20)
    3

    See Also
    ========

    factorint

    References
    ==========

    .. [1] https://mathworld.wolfram.com/PrimeFactor.html

    """

    @classmethod
    def eval(cls, n):
        if False:
            i = 10
            return i + 15
        if n.is_Integer:
            if n <= 0:
                raise ValueError('n must be a positive integer')
            else:
                return sum(factorint(n).values())

def mersenne_prime_exponent(nth):
    if False:
        i = 10
        return i + 15
    'Returns the exponent ``i`` for the nth Mersenne prime (which\n    has the form `2^i - 1`).\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.factor_ import mersenne_prime_exponent\n    >>> mersenne_prime_exponent(1)\n    2\n    >>> mersenne_prime_exponent(20)\n    4423\n    '
    n = as_int(nth)
    if n < 1:
        raise ValueError('nth must be a positive integer; mersenne_prime_exponent(1) == 2')
    if n > 51:
        raise ValueError('There are only 51 perfect numbers; nth must be less than or equal to 51')
    return MERSENNE_PRIME_EXPONENTS[n - 1]

def is_perfect(n):
    if False:
        print('Hello World!')
    'Returns True if ``n`` is a perfect number, else False.\n\n    A perfect number is equal to the sum of its positive, proper divisors.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.factor_ import is_perfect, divisors, divisor_sigma\n    >>> is_perfect(20)\n    False\n    >>> is_perfect(6)\n    True\n    >>> 6 == divisor_sigma(6) - 6 == sum(divisors(6)[:-1])\n    True\n\n    References\n    ==========\n\n    .. [1] https://mathworld.wolfram.com/PerfectNumber.html\n    .. [2] https://en.wikipedia.org/wiki/Perfect_number\n\n    '
    n = as_int(n)
    if n < 1:
        return False
    if n % 2 == 0:
        m = n.bit_length() + 1 >> 1
        if (1 << m - 1) * ((1 << m) - 1) != n:
            return False
        return m in MERSENNE_PRIME_EXPONENTS or is_mersenne_prime(2 ** m - 1)
    if n < 10 ** 2000:
        return False
    if n % 105 == 0:
        return False
    if all((n % m != r for (m, r) in [(12, 1), (468, 117), (324, 81)])):
        return False
    result = divisor_sigma(n) == 2 * n
    if result:
        raise ValueError(filldedent('In 1888, Sylvester stated: "\n            ...a prolonged meditation on the subject has satisfied\n            me that the existence of any one such [odd perfect number]\n            -- its escape, so to say, from the complex web of conditions\n            which hem it in on all sides -- would be little short of a\n            miracle." I guess SymPy just found that miracle and it\n            factors like this: %s' % factorint(n)))
    return result

def abundance(n):
    if False:
        i = 10
        return i + 15
    'Returns the difference between the sum of the positive\n    proper divisors of a number and the number.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory import abundance, is_perfect, is_abundant\n    >>> abundance(6)\n    0\n    >>> is_perfect(6)\n    True\n    >>> abundance(10)\n    -2\n    >>> is_abundant(10)\n    False\n    '
    return divisor_sigma(n, 1) - 2 * n

def is_abundant(n):
    if False:
        i = 10
        return i + 15
    'Returns True if ``n`` is an abundant number, else False.\n\n    A abundant number is smaller than the sum of its positive proper divisors.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.factor_ import is_abundant\n    >>> is_abundant(20)\n    True\n    >>> is_abundant(15)\n    False\n\n    References\n    ==========\n\n    .. [1] https://mathworld.wolfram.com/AbundantNumber.html\n\n    '
    n = as_int(n)
    if is_perfect(n):
        return False
    return n % 6 == 0 or bool(abundance(n) > 0)

def is_deficient(n):
    if False:
        i = 10
        return i + 15
    'Returns True if ``n`` is a deficient number, else False.\n\n    A deficient number is greater than the sum of its positive proper divisors.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.factor_ import is_deficient\n    >>> is_deficient(20)\n    False\n    >>> is_deficient(15)\n    True\n\n    References\n    ==========\n\n    .. [1] https://mathworld.wolfram.com/DeficientNumber.html\n\n    '
    n = as_int(n)
    if is_perfect(n):
        return False
    return bool(abundance(n) < 0)

def is_amicable(m, n):
    if False:
        return 10
    'Returns True if the numbers `m` and `n` are "amicable", else False.\n\n    Amicable numbers are two different numbers so related that the sum\n    of the proper divisors of each is equal to that of the other.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.factor_ import is_amicable, divisor_sigma\n    >>> is_amicable(220, 284)\n    True\n    >>> divisor_sigma(220) == divisor_sigma(284)\n    True\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Amicable_numbers\n\n    '
    if m == n:
        return False
    (a, b) = (divisor_sigma(i) for i in (m, n))
    return a == b == m + n

def dra(n, b):
    if False:
        while True:
            i = 10
    '\n    Returns the additive digital root of a natural number ``n`` in base ``b``\n    which is a single digit value obtained by an iterative process of summing\n    digits, on each iteration using the result from the previous iteration to\n    compute a digit sum.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.factor_ import dra\n    >>> dra(3110, 12)\n    8\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Digital_root\n\n    '
    num = abs(as_int(n))
    b = as_int(b)
    if b <= 1:
        raise ValueError('Base should be an integer greater than 1')
    if num == 0:
        return 0
    return 1 + (num - 1) % (b - 1)

def drm(n, b):
    if False:
        while True:
            i = 10
    '\n    Returns the multiplicative digital root of a natural number ``n`` in a given\n    base ``b`` which is a single digit value obtained by an iterative process of\n    multiplying digits, on each iteration using the result from the previous\n    iteration to compute the digit multiplication.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.factor_ import drm\n    >>> drm(9876, 10)\n    0\n\n    >>> drm(49, 10)\n    8\n\n    References\n    ==========\n\n    .. [1] https://mathworld.wolfram.com/MultiplicativeDigitalRoot.html\n\n    '
    n = abs(as_int(n))
    b = as_int(b)
    if b <= 1:
        raise ValueError('Base should be an integer greater than 1')
    while n > b:
        mul = 1
        while n > 1:
            (n, r) = divmod(n, b)
            if r == 0:
                return 0
            mul *= r
        n = mul
    return n