from math import prod
from sympy.external.gmpy import gcd, gcdext
from sympy.ntheory.primetest import isprime
from sympy.polys.domains import ZZ
from sympy.polys.galoistools import gf_crt, gf_crt1, gf_crt2
from sympy.utilities.misc import as_int

def symmetric_residue(a, m):
    if False:
        return 10
    'Return the residual mod m such that it is within half of the modulus.\n\n    >>> from sympy.ntheory.modular import symmetric_residue\n    >>> symmetric_residue(1, 6)\n    1\n    >>> symmetric_residue(4, 6)\n    -2\n    '
    if a <= m // 2:
        return a
    return a - m

def crt(m, v, symmetric=False, check=True):
    if False:
        return 10
    "Chinese Remainder Theorem.\n\n    The moduli in m are assumed to be pairwise coprime.  The output\n    is then an integer f, such that f = v_i mod m_i for each pair out\n    of v and m. If ``symmetric`` is False a positive integer will be\n    returned, else \\|f\\| will be less than or equal to the LCM of the\n    moduli, and thus f may be negative.\n\n    If the moduli are not co-prime the correct result will be returned\n    if/when the test of the result is found to be incorrect. This result\n    will be None if there is no solution.\n\n    The keyword ``check`` can be set to False if it is known that the moduli\n    are coprime.\n\n    Examples\n    ========\n\n    As an example consider a set of residues ``U = [49, 76, 65]``\n    and a set of moduli ``M = [99, 97, 95]``. Then we have::\n\n       >>> from sympy.ntheory.modular import crt\n\n       >>> crt([99, 97, 95], [49, 76, 65])\n       (639985, 912285)\n\n    This is the correct result because::\n\n       >>> [639985 % m for m in [99, 97, 95]]\n       [49, 76, 65]\n\n    If the moduli are not co-prime, you may receive an incorrect result\n    if you use ``check=False``:\n\n       >>> crt([12, 6, 17], [3, 4, 2], check=False)\n       (954, 1224)\n       >>> [954 % m for m in [12, 6, 17]]\n       [6, 0, 2]\n       >>> crt([12, 6, 17], [3, 4, 2]) is None\n       True\n       >>> crt([3, 6], [2, 5])\n       (5, 6)\n\n    Note: the order of gf_crt's arguments is reversed relative to crt,\n    and that solve_congruence takes residue, modulus pairs.\n\n    Programmer's note: rather than checking that all pairs of moduli share\n    no GCD (an O(n**2) test) and rather than factoring all moduli and seeing\n    that there is no factor in common, a check that the result gives the\n    indicated residuals is performed -- an O(n) operation.\n\n    See Also\n    ========\n\n    solve_congruence\n    sympy.polys.galoistools.gf_crt : low level crt routine used by this routine\n    "
    if check:
        m = list(map(as_int, m))
        v = list(map(as_int, v))
    result = gf_crt(v, m, ZZ)
    mm = prod(m)
    if check:
        if not all((v % m == result % m for (v, m) in zip(v, m))):
            result = solve_congruence(*list(zip(v, m)), check=False, symmetric=symmetric)
            if result is None:
                return result
            (result, mm) = result
    if symmetric:
        return (int(symmetric_residue(result, mm)), int(mm))
    return (int(result), int(mm))

def crt1(m):
    if False:
        print('Hello World!')
    'First part of Chinese Remainder Theorem, for multiple application.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.modular import crt, crt1, crt2\n    >>> m = [99, 97, 95]\n    >>> v = [49, 76, 65]\n\n    The following two codes have the same result.\n\n    >>> crt(m, v)\n    (639985, 912285)\n\n    >>> mm, e, s = crt1(m)\n    >>> crt2(m, v, mm, e, s)\n    (639985, 912285)\n\n    However, it is faster when we want to fix ``m`` and\n    compute for multiple ``v``, i.e. the following cases:\n\n    >>> mm, e, s = crt1(m)\n    >>> vs = [[52, 21, 37], [19, 46, 76]]\n    >>> for v in vs:\n    ...     print(crt2(m, v, mm, e, s))\n    (397042, 912285)\n    (803206, 912285)\n\n    See Also\n    ========\n\n    sympy.polys.galoistools.gf_crt1 : low level crt routine used by this routine\n    sympy.ntheory.modular.crt\n    sympy.ntheory.modular.crt2\n\n    '
    return gf_crt1(m, ZZ)

def crt2(m, v, mm, e, s, symmetric=False):
    if False:
        for i in range(10):
            print('nop')
    'Second part of Chinese Remainder Theorem, for multiple application.\n\n    See ``crt1`` for usage.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.modular import crt1, crt2\n    >>> mm, e, s = crt1([18, 42, 6])\n    >>> crt2([18, 42, 6], [0, 0, 0], mm, e, s)\n    (0, 4536)\n\n    See Also\n    ========\n\n    sympy.polys.galoistools.gf_crt2 : low level crt routine used by this routine\n    sympy.ntheory.modular.crt\n    sympy.ntheory.modular.crt1\n\n    '
    result = gf_crt2(v, m, mm, e, s, ZZ)
    if symmetric:
        return (int(symmetric_residue(result, mm)), int(mm))
    return (int(result), int(mm))

def solve_congruence(*remainder_modulus_pairs, **hint):
    if False:
        print('Hello World!')
    'Compute the integer ``n`` that has the residual ``ai`` when it is\n    divided by ``mi`` where the ``ai`` and ``mi`` are given as pairs to\n    this function: ((a1, m1), (a2, m2), ...). If there is no solution,\n    return None. Otherwise return ``n`` and its modulus.\n\n    The ``mi`` values need not be co-prime. If it is known that the moduli are\n    not co-prime then the hint ``check`` can be set to False (default=True) and\n    the check for a quicker solution via crt() (valid when the moduli are\n    co-prime) will be skipped.\n\n    If the hint ``symmetric`` is True (default is False), the value of ``n``\n    will be within 1/2 of the modulus, possibly negative.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.modular import solve_congruence\n\n    What number is 2 mod 3, 3 mod 5 and 2 mod 7?\n\n    >>> solve_congruence((2, 3), (3, 5), (2, 7))\n    (23, 105)\n    >>> [23 % m for m in [3, 5, 7]]\n    [2, 3, 2]\n\n    If you prefer to work with all remainder in one list and\n    all moduli in another, send the arguments like this:\n\n    >>> solve_congruence(*zip((2, 3, 2), (3, 5, 7)))\n    (23, 105)\n\n    The moduli need not be co-prime; in this case there may or\n    may not be a solution:\n\n    >>> solve_congruence((2, 3), (4, 6)) is None\n    True\n\n    >>> solve_congruence((2, 3), (5, 6))\n    (5, 6)\n\n    The symmetric flag will make the result be within 1/2 of the modulus:\n\n    >>> solve_congruence((2, 3), (5, 6), symmetric=True)\n    (-1, 6)\n\n    See Also\n    ========\n\n    crt : high level routine implementing the Chinese Remainder Theorem\n\n    '

    def combine(c1, c2):
        if False:
            print('Hello World!')
        'Return the tuple (a, m) which satisfies the requirement\n        that n = a + i*m satisfy n = a1 + j*m1 and n = a2 = k*m2.\n\n        References\n        ==========\n\n        .. [1] https://en.wikipedia.org/wiki/Method_of_successive_substitution\n        '
        (a1, m1) = c1
        (a2, m2) = c2
        (a, b, c) = (m1, a2 - a1, m2)
        g = gcd(a, b, c)
        (a, b, c) = [i // g for i in [a, b, c]]
        if a != 1:
            (g, inv_a, _) = gcdext(a, c)
            if g != 1:
                return None
            b *= inv_a
        (a, m) = (a1 + m1 * b, m1 * c)
        return (a, m)
    rm = remainder_modulus_pairs
    symmetric = hint.get('symmetric', False)
    if hint.get('check', True):
        rm = [(as_int(r), as_int(m)) for (r, m) in rm]
        uniq = {}
        for (r, m) in rm:
            r %= m
            if m in uniq:
                if r != uniq[m]:
                    return None
                continue
            uniq[m] = r
        rm = [(r, m) for (m, r) in uniq.items()]
        del uniq
        if all((isprime(m) for (r, m) in rm)):
            (r, m) = list(zip(*rm))
            return crt(m, r, symmetric=symmetric, check=False)
    rv = (0, 1)
    for rmi in rm:
        rv = combine(rv, rmi)
        if rv is None:
            break
        (n, m) = rv
        n = n % m
    else:
        if symmetric:
            return (symmetric_residue(n, m), m)
        return (n, m)