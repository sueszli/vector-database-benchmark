from __future__ import annotations
from sympy.core.function import Function
from sympy.core.singleton import S
from sympy.external.gmpy import gcd, lcm, invert, sqrt, legendre, jacobi, kronecker, bit_scan1, remove
from sympy.polys import Poly
from sympy.polys.domains import ZZ
from sympy.polys.galoistools import gf_crt1, gf_crt2, linear_congruence, gf_csolve
from .primetest import isprime
from .factor_ import factorint, _perfect_power
from .modular import crt
from sympy.utilities.misc import as_int
from sympy.utilities.iterables import iproduct
from sympy.core.random import _randint, randint
from itertools import product

def n_order(a, n):
    if False:
        while True:
            i = 10
    ' Returns the order of ``a`` modulo ``n``.\n\n    Explanation\n    ===========\n\n    The order of ``a`` modulo ``n`` is the smallest integer\n    ``k`` such that `a^k` leaves a remainder of 1 with ``n``.\n\n    Parameters\n    ==========\n\n    a : integer\n    n : integer, n > 1. a and n should be relatively prime\n\n    Returns\n    =======\n\n    int : the order of ``a`` modulo ``n``\n\n    Raises\n    ======\n\n    ValueError\n        If `n \\le 1` or `\\gcd(a, n) \\neq 1`.\n        If ``a`` or ``n`` is not an integer.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory import n_order\n    >>> n_order(3, 7)\n    6\n    >>> n_order(4, 7)\n    3\n\n    See Also\n    ========\n\n    is_primitive_root\n        We say that ``a`` is a primitive root of ``n``\n        when the order of ``a`` modulo ``n`` equals ``totient(n)``\n\n    '
    (a, n) = (as_int(a), as_int(n))
    if n <= 1:
        raise ValueError('n should be an integer greater than 1')
    a = a % n
    if a == 1:
        return 1
    if gcd(a, n) != 1:
        raise ValueError('The two numbers should be relatively prime')
    a_order = 1
    for (p, e) in factorint(n).items():
        pe = p ** e
        pe_order = (p - 1) * p ** (e - 1)
        factors = factorint(p - 1)
        if e > 1:
            factors[p] = e - 1
        order = 1
        for (px, ex) in factors.items():
            x = pow(a, pe_order // px ** ex, pe)
            while x != 1:
                x = pow(x, px, pe)
                order *= px
        a_order = lcm(a_order, order)
    return int(a_order)

def _primitive_root_prime_iter(p):
    if False:
        while True:
            i = 10
    ' Generates the primitive roots for a prime ``p``.\n\n    Explanation\n    ===========\n\n    The primitive roots generated are not necessarily sorted.\n    However, the first one is the smallest primitive root.\n\n    Find the element whose order is ``p-1`` from the smaller one.\n    If we can find the first primitive root ``g``, we can use the following theorem.\n\n    .. math ::\n        \\operatorname{ord}(g^k) = \\frac{\\operatorname{ord}(g)}{\\gcd(\\operatorname{ord}(g), k)}\n\n    From the assumption that `\\operatorname{ord}(g)=p-1`,\n    it is a necessary and sufficient condition for\n    `\\operatorname{ord}(g^k)=p-1` that `\\gcd(p-1, k)=1`.\n\n    Parameters\n    ==========\n\n    p : odd prime\n\n    Yields\n    ======\n\n    int\n        the primitive roots of ``p``\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.residue_ntheory import _primitive_root_prime_iter\n    >>> sorted(_primitive_root_prime_iter(19))\n    [2, 3, 10, 13, 14, 15]\n\n    References\n    ==========\n\n    .. [1] W. Stein "Elementary Number Theory" (2011), page 44\n\n    '
    if p == 3:
        yield 2
        return
    g_min = 3 if p % 8 in [1, 7] else 2
    if p < 41:
        g = 5 if p == 23 else g_min
    else:
        v = [(p - 1) // i for i in factorint(p - 1).keys()]
        for g in range(g_min, p):
            if all((pow(g, pw, p) != 1 for pw in v)):
                break
    yield g
    for k in range(3, p, 2):
        if gcd(p - 1, k) == 1:
            yield pow(g, k, p)

def _primitive_root_prime_power_iter(p, e):
    if False:
        return 10
    ' Generates the primitive roots of `p^e`.\n\n    Explanation\n    ===========\n\n    Let ``g`` be the primitive root of ``p``.\n    If `g^{p-1} \\not\\equiv 1 \\pmod{p^2}`, then ``g`` is primitive root of `p^e`.\n    Thus, if we find a primitive root ``g`` of ``p``,\n    then `g, g+p, g+2p, \\ldots, g+(p-1)p` are primitive roots of `p^2` except one.\n    That one satisfies `\\hat{g}^{p-1} \\equiv 1 \\pmod{p^2}`.\n    If ``h`` is the primitive root of `p^2`,\n    then `h, h+p^2, h+2p^2, \\ldots, h+(p^{e-2}-1)p^e` are primitive roots of `p^e`.\n\n    Parameters\n    ==========\n\n    p : odd prime\n    e : positive integer\n\n    Yields\n    ======\n\n    int\n        the primitive roots of `p^e`\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.residue_ntheory import _primitive_root_prime_power_iter\n    >>> sorted(_primitive_root_prime_power_iter(5, 2))\n    [2, 3, 8, 12, 13, 17, 22, 23]\n\n    '
    if e == 1:
        yield from _primitive_root_prime_iter(p)
    else:
        p2 = p ** 2
        for g in _primitive_root_prime_iter(p):
            t = (g - pow(g, 2 - p, p2)) % p2
            for k in range(0, p2, p):
                if k != t:
                    yield from (g + k + m for m in range(0, p ** e, p2))

def _primitive_root_prime_power2_iter(p, e):
    if False:
        print('Hello World!')
    ' Generates the primitive roots of `2p^e`.\n\n    Explanation\n    ===========\n\n    If ``g`` is the primitive root of ``p**e``,\n    then the odd one of ``g`` and ``g+p**e`` is the primitive root of ``2*p**e``.\n\n    Parameters\n    ==========\n\n    p : odd prime\n    e : positive integer\n\n    Yields\n    ======\n\n    int\n        the primitive roots of `2p^e`\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.residue_ntheory import _primitive_root_prime_power2_iter\n    >>> sorted(_primitive_root_prime_power2_iter(5, 2))\n    [3, 13, 17, 23, 27, 33, 37, 47]\n\n    '
    for g in _primitive_root_prime_power_iter(p, e):
        if g % 2 == 1:
            yield g
        else:
            yield (g + p ** e)

def primitive_root(p, smallest=True):
    if False:
        i = 10
        return i + 15
    ' Returns a primitive root of ``p`` or None.\n\n    Explanation\n    ===========\n\n    For the definition of primitive root,\n    see the explanation of ``is_primitive_root``.\n\n    The primitive root of ``p`` exist only for\n    `p = 2, 4, q^e, 2q^e` (``q`` is an odd prime).\n    Now, if we know the primitive root of ``q``,\n    we can calculate the primitive root of `q^e`,\n    and if we know the primitive root of `q^e`,\n    we can calculate the primitive root of `2q^e`.\n    When there is no need to find the smallest primitive root,\n    this property can be used to obtain a fast primitive root.\n    On the other hand, when we want the smallest primitive root,\n    we naively determine whether it is a primitive root or not.\n\n    Parameters\n    ==========\n\n    p : integer, p > 1\n    smallest : if True the smallest primitive root is returned or None\n\n    Returns\n    =======\n\n    int | None :\n        If the primitive root exists, return the primitive root of ``p``.\n        If not, return None.\n\n    Raises\n    ======\n\n    ValueError\n        If `p \\le 1` or ``p`` is not an integer.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.residue_ntheory import primitive_root\n    >>> primitive_root(19)\n    2\n    >>> primitive_root(21) is None\n    True\n    >>> primitive_root(50, smallest=False)\n    27\n\n    See Also\n    ========\n\n    is_primitive_root\n\n    References\n    ==========\n\n    .. [1] W. Stein "Elementary Number Theory" (2011), page 44\n    .. [2] P. Hackman "Elementary Number Theory" (2009), Chapter C\n\n    '
    p = as_int(p)
    if p <= 1:
        raise ValueError('p should be an integer greater than 1')
    if p <= 4:
        return p - 1
    p_even = p % 2 == 0
    if not p_even:
        q = p
    elif p % 4:
        q = p // 2
    else:
        return None
    if isprime(q):
        e = 1
    else:
        m = _perfect_power(q, 3)
        if not m:
            return None
        (q, e) = m
        if not isprime(q):
            return None
    if not smallest:
        if p_even:
            return next(_primitive_root_prime_power2_iter(q, e))
        return next(_primitive_root_prime_power_iter(q, e))
    if p_even:
        for i in range(3, p, 2):
            if i % q and is_primitive_root(i, p):
                return i
    g = next(_primitive_root_prime_iter(q))
    if e == 1 or pow(g, q - 1, q ** 2) != 1:
        return g
    for i in range(g + 1, p):
        if i % q and is_primitive_root(i, p):
            return i

def is_primitive_root(a, p):
    if False:
        while True:
            i = 10
    " Returns True if ``a`` is a primitive root of ``p``.\n\n    Explanation\n    ===========\n\n    ``a`` is said to be the primitive root of ``p`` if `\\gcd(a, p) = 1` and\n    `\\phi(p)` is the smallest positive number s.t.\n\n        `a^{\\phi(p)} \\equiv 1 \\pmod{p}`.\n\n    where `\\phi(p)` is Euler's totient function.\n\n    The primitive root of ``p`` exist only for\n    `p = 2, 4, q^e, 2q^e` (``q`` is an odd prime).\n    Hence, if it is not such a ``p``, it returns False.\n    To determine the primitive root, we need to know\n    the prime factorization of ``q-1``.\n    The hardness of the determination depends on this complexity.\n\n    Parameters\n    ==========\n\n    a : integer\n    p : integer, ``p`` > 1. ``a`` and ``p`` should be relatively prime\n\n    Returns\n    =======\n\n    bool : If True, ``a`` is the primitive root of ``p``.\n\n    Raises\n    ======\n\n    ValueError\n        If `p \\le 1` or `\\gcd(a, p) \\neq 1`.\n        If ``a`` or ``p`` is not an integer.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory import is_primitive_root, n_order, totient\n    >>> is_primitive_root(3, 10)\n    True\n    >>> is_primitive_root(9, 10)\n    False\n    >>> n_order(3, 10) == totient(10)\n    True\n    >>> n_order(9, 10) == totient(10)\n    False\n\n    See Also\n    ========\n\n    primitive_root\n\n    "
    (a, p) = (as_int(a), as_int(p))
    if p <= 1:
        raise ValueError('p should be an integer greater than 1')
    a = a % p
    if gcd(a, p) != 1:
        raise ValueError('The two numbers should be relatively prime')
    if p <= 4:
        return a == p - 1
    if p % 2:
        q = p
    elif p % 4:
        q = p // 2
    else:
        return False
    if isprime(q):
        group_order = q - 1
        factors = factorint(q - 1).keys()
    else:
        m = _perfect_power(q, 3)
        if not m:
            return False
        (q, e) = m
        if not isprime(q):
            return False
        group_order = q ** (e - 1) * (q - 1)
        factors = set(factorint(q - 1).keys())
        factors.add(q)
    return all((pow(a, group_order // prime, p) != 1 for prime in factors))

def _sqrt_mod_tonelli_shanks(a, p):
    if False:
        print('Hello World!')
    '\n    Returns the square root in the case of ``p`` prime with ``p == 1 (mod 8)``\n\n    Assume that the root exists.\n\n    Parameters\n    ==========\n\n    a : int\n    p : int\n        prime number. should be ``p % 8 == 1``\n\n    Returns\n    =======\n\n    int : Generally, there are two roots, but only one is returned.\n          Which one is returned is random.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.residue_ntheory import _sqrt_mod_tonelli_shanks\n    >>> _sqrt_mod_tonelli_shanks(2, 17) in [6, 11]\n    True\n\n    References\n    ==========\n\n    .. [1] Carl Pomerance, Richard Crandall, Prime Numbers: A Computational Perspective,\n           2nd Edition (2005), page 101, ISBN:978-0387252827\n\n    '
    s = bit_scan1(p - 1)
    t = p >> s
    if p % 12 == 5:
        d = 3
    elif p % 5 in [2, 3]:
        d = 5
    else:
        while 1:
            d = randint(6, p - 1)
            if jacobi(d, p) == -1:
                break
    A = pow(a, t, p)
    D = pow(d, t, p)
    m = 0
    for i in range(s):
        adm = A * pow(D, m, p) % p
        adm = pow(adm, 2 ** (s - 1 - i), p)
        if adm % p == p - 1:
            m += 2 ** i
    x = pow(a, (t + 1) // 2, p) * pow(D, m // 2, p) % p
    return x

def sqrt_mod(a, p, all_roots=False):
    if False:
        while True:
            i = 10
    '\n    Find a root of ``x**2 = a mod p``.\n\n    Parameters\n    ==========\n\n    a : integer\n    p : positive integer\n    all_roots : if True the list of roots is returned or None\n\n    Notes\n    =====\n\n    If there is no root it is returned None; else the returned root\n    is less or equal to ``p // 2``; in general is not the smallest one.\n    It is returned ``p // 2`` only if it is the only root.\n\n    Use ``all_roots`` only when it is expected that all the roots fit\n    in memory; otherwise use ``sqrt_mod_iter``.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory import sqrt_mod\n    >>> sqrt_mod(11, 43)\n    21\n    >>> sqrt_mod(17, 32, True)\n    [7, 9, 23, 25]\n    '
    if all_roots:
        return sorted(sqrt_mod_iter(a, p))
    p = abs(as_int(p))
    halfp = p // 2
    x = None
    for r in sqrt_mod_iter(a, p):
        if r < halfp:
            return r
        elif r > halfp:
            return p - r
        else:
            x = r
    return x

def sqrt_mod_iter(a, p, domain=int):
    if False:
        print('Hello World!')
    '\n    Iterate over solutions to ``x**2 = a mod p``.\n\n    Parameters\n    ==========\n\n    a : integer\n    p : positive integer\n    domain : integer domain, ``int``, ``ZZ`` or ``Integer``\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.residue_ntheory import sqrt_mod_iter\n    >>> list(sqrt_mod_iter(11, 43))\n    [21, 22]\n\n    See Also\n    ========\n\n    sqrt_mod : Same functionality, but you want a sorted list or only one solution.\n\n    '
    (a, p) = (as_int(a), abs(as_int(p)))
    v = []
    pv = []
    _product = product
    for (px, ex) in factorint(p).items():
        if a % px:
            rx = _sqrt_mod_prime_power(a, px, ex)
        else:
            rx = _sqrt_mod1(a, px, ex)
            _product = iproduct
        if not rx:
            return
        v.append(rx)
        pv.append(px ** ex)
    if len(v) == 1:
        yield from map(domain, v[0])
    else:
        (mm, e, s) = gf_crt1(pv, ZZ)
        for vx in _product(*v):
            yield domain(gf_crt2(vx, pv, mm, e, s, ZZ))

def _sqrt_mod_prime_power(a, p, k):
    if False:
        print('Hello World!')
    '\n    Find the solutions to ``x**2 = a mod p**k`` when ``a % p != 0``.\n    If no solution exists, return ``None``.\n    Solutions are returned in an ascending list.\n\n    Parameters\n    ==========\n\n    a : integer\n    p : prime number\n    k : positive integer\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.residue_ntheory import _sqrt_mod_prime_power\n    >>> _sqrt_mod_prime_power(11, 43, 1)\n    [21, 22]\n\n    References\n    ==========\n\n    .. [1] P. Hackman "Elementary Number Theory" (2009), page 160\n    .. [2] http://www.numbertheory.org/php/squareroot.html\n    .. [3] [Gathen99]_\n    '
    pk = p ** k
    a = a % pk
    if p == 2:
        if a % 8 != 1:
            return None
        if k <= 3:
            return list(range(1, pk, 2))
        r = 1
        for nx in range(3, k):
            if (r ** 2 - a >> nx) % 2:
                r += 1 << nx - 1
        h = 1 << k - 1
        return sorted([r, pk - r, (r + h) % pk, -(r + h) % pk])
    if jacobi(a, p) != 1:
        return None
    if p % 4 == 3:
        res = pow(a, (p + 1) // 4, p)
    elif p % 8 == 5:
        res = pow(a, (p + 3) // 8, p)
        if pow(res, 2, p) != a % p:
            res = res * pow(2, (p - 1) // 4, p) % p
    else:
        res = _sqrt_mod_tonelli_shanks(a, p)
    if k > 1:
        px = p
        for _ in range(k.bit_length() - 1):
            px = px ** 2
            frinv = invert(2 * res, px)
            res = (res - (res ** 2 - a) * frinv) % px
        if k & k - 1:
            frinv = invert(2 * res, pk)
            res = (res - (res ** 2 - a) * frinv) % pk
    return sorted([res, pk - res])

def _sqrt_mod1(a, p, n):
    if False:
        for i in range(10):
            print('nop')
    '\n    Find solution to ``x**2 == a mod p**n`` when ``a % p == 0``.\n    If no solution exists, return ``None``.\n\n    Parameters\n    ==========\n\n    a : integer\n    p : prime number, p must divide a\n    n : positive integer\n\n    References\n    ==========\n\n    .. [1] http://www.numbertheory.org/php/squareroot.html\n    '
    pn = p ** n
    a = a % pn
    if a == 0:
        return range(0, pn, p ** ((n + 1) // 2))
    (a, r) = remove(a, p)
    if r % 2 == 1:
        return None
    res = _sqrt_mod_prime_power(a, p, n - r)
    if res is None:
        return None
    m = r // 2
    return (x for rx in res for x in range(rx * p ** m, pn, p ** (n - m)))

def is_quad_residue(a, p):
    if False:
        print('Hello World!')
    '\n    Returns True if ``a`` (mod ``p``) is in the set of squares mod ``p``,\n    i.e a % p in set([i**2 % p for i in range(p)]).\n\n    Parameters\n    ==========\n\n    a : integer\n    p : positive integer\n\n    Returns\n    =======\n\n    bool : If True, ``x**2 == a (mod p)`` has solution.\n\n    Raises\n    ======\n\n    ValueError\n        If ``a``, ``p`` is not integer.\n        If ``p`` is not positive.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory import is_quad_residue\n    >>> is_quad_residue(21, 100)\n    True\n\n    Indeed, ``pow(39, 2, 100)`` would be 21.\n\n    >>> is_quad_residue(21, 120)\n    False\n\n    That is, for any integer ``x``, ``pow(x, 2, 120)`` is not 21.\n\n    If ``p`` is an odd\n    prime, an iterative method is used to make the determination:\n\n    >>> from sympy.ntheory import is_quad_residue\n    >>> sorted(set([i**2 % 7 for i in range(7)]))\n    [0, 1, 2, 4]\n    >>> [j for j in range(7) if is_quad_residue(j, 7)]\n    [0, 1, 2, 4]\n\n    See Also\n    ========\n\n    legendre_symbol, jacobi_symbol, sqrt_mod\n    '
    (a, p) = (as_int(a), as_int(p))
    if p < 1:
        raise ValueError('p must be > 0')
    a %= p
    if a < 2 or p < 3:
        return True
    t = bit_scan1(p)
    if t:
        a_ = a % (1 << t)
        if a_:
            r = bit_scan1(a_)
            if r % 2 or a_ >> r & 6:
                return False
        p >>= t
        a %= p
        if a < 2 or p < 3:
            return True
    j = jacobi(a, p)
    if j == -1 or isprime(p):
        return j == 1
    for (px, ex) in factorint(p).items():
        if a % px:
            if jacobi(a, px) != 1:
                return False
        else:
            a_ = a % px ** ex
            if a_ == 0:
                continue
            (a_, r) = remove(a_, px)
            if r % 2 or jacobi(a_, px) != 1:
                return False
    return True

def is_nthpow_residue(a, n, m):
    if False:
        i = 10
        return i + 15
    '\n    Returns True if ``x**n == a (mod m)`` has solutions.\n\n    References\n    ==========\n\n    .. [1] P. Hackman "Elementary Number Theory" (2009), page 76\n\n    '
    a = a % m
    (a, n, m) = (as_int(a), as_int(n), as_int(m))
    if m <= 0:
        raise ValueError('m must be > 0')
    if n < 0:
        raise ValueError('n must be >= 0')
    if n == 0:
        if m == 1:
            return False
        return a == 1
    if a == 0:
        return True
    if n == 1:
        return True
    if n == 2:
        return is_quad_residue(a, m)
    return all((_is_nthpow_residue_bign_prime_power(a, n, p, e) for (p, e) in factorint(m).items()))

def _is_nthpow_residue_bign_prime_power(a, n, p, k):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns True if `x^n = a \\pmod{p^k}` has solutions for `n > 2`.\n\n    Parameters\n    ==========\n\n    a : positive integer\n    n : integer, n > 2\n    p : prime number\n    k : positive integer\n\n    '
    while a % p == 0:
        a %= pow(p, k)
        if not a:
            return True
        (a, mu) = remove(a, p)
        if mu % n:
            return False
        k -= mu
    if p != 2:
        f = p ** (k - 1) * (p - 1)
        return pow(a, f // gcd(f, n), pow(p, k)) == 1
    if n & 1:
        return True
    c = min(bit_scan1(n) + 2, k)
    return a % pow(2, c) == 1

def _nthroot_mod1(s, q, p, all_roots):
    if False:
        for i in range(10):
            print('nop')
    '\n    Root of ``x**q = s mod p``, ``p`` prime and ``q`` divides ``p - 1``.\n    Assume that the root exists.\n\n    Parameters\n    ==========\n\n    s : integer\n    q : integer, n > 2. ``q`` divides ``p - 1``.\n    p : prime number\n    all_roots : if False returns the smallest root, else the list of roots\n\n    Returns\n    =======\n\n    list[int] | int :\n        Root of ``x**q = s mod p``. If ``all_roots == True``,\n        returned ascending list. otherwise, returned an int.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.residue_ntheory import _nthroot_mod1\n    >>> _nthroot_mod1(5, 3, 13, False)\n    7\n    >>> _nthroot_mod1(13, 4, 17, True)\n    [3, 5, 12, 14]\n\n    References\n    ==========\n\n    .. [1] A. M. Johnston, A Generalized qth Root Algorithm,\n           ACM-SIAM Symposium on Discrete Algorithms (1999), pp. 929-930\n\n    '
    g = next(_primitive_root_prime_iter(p))
    r = s
    for (qx, ex) in factorint(q).items():
        f = (p - 1) // qx ** ex
        while f % qx == 0:
            f //= qx
        z = f * invert(-f, qx)
        x = (1 + z) // qx
        t = discrete_log(p, pow(r, f, p), pow(g, f * qx, p))
        for _ in range(ex):
            r = pow(r, x, p) * pow(g, -z * t % (p - 1), p) % p
            t //= qx
    res = [r]
    h = pow(g, (p - 1) // q, p)
    hx = r
    for _ in range(q - 1):
        hx = hx * h % p
        res.append(hx)
    if all_roots:
        res.sort()
        return res
    return min(res)

def _nthroot_mod_prime_power(a, n, p, k):
    if False:
        while True:
            i = 10
    ' Root of ``x**n = a mod p**k``.\n\n    Parameters\n    ==========\n\n    a : integer\n    n : integer, n > 2\n    p : prime number\n    k : positive integer\n\n    Returns\n    =======\n\n    list[int] :\n        Ascending list of roots of ``x**n = a mod p**k``.\n        If no solution exists, return ``[]``.\n\n    '
    if not _is_nthpow_residue_bign_prime_power(a, n, p, k):
        return []
    a_mod_p = a % p
    if a_mod_p == 0:
        base_roots = [0]
    elif (p - 1) % n == 0:
        base_roots = _nthroot_mod1(a_mod_p, n, p, all_roots=True)
    else:
        pa = n
        pb = p - 1
        b = 1
        if pa < pb:
            (a_mod_p, pa, b, pb) = (b, pb, a_mod_p, pa)
        while pb:
            (q, pc) = divmod(pa, pb)
            c = pow(b, -q, p) * a_mod_p % p
            (pa, pb) = (pb, pc)
            (a_mod_p, b) = (b, c)
        if pa == 1:
            base_roots = [a_mod_p]
        elif pa == 2:
            base_roots = sqrt_mod(a_mod_p, p, all_roots=True)
        else:
            base_roots = _nthroot_mod1(a_mod_p, pa, p, all_roots=True)
    if k == 1:
        return base_roots
    a %= p ** k
    tot_roots = set()
    for root in base_roots:
        diff = pow(root, n - 1, p) * n % p
        new_base = p
        if diff != 0:
            m_inv = invert(diff, p)
            for _ in range(k - 1):
                new_base *= p
                tmp = pow(root, n, new_base) - a
                tmp *= m_inv
                root = (root - tmp) % new_base
            tot_roots.add(root)
        else:
            roots_in_base = {root}
            for _ in range(k - 1):
                new_base *= p
                new_roots = set()
                for k_ in roots_in_base:
                    if pow(k_, n, new_base) != a % new_base:
                        continue
                    while k_ not in new_roots:
                        new_roots.add(k_)
                        k_ = (k_ + new_base // p) % new_base
                roots_in_base = new_roots
            tot_roots = tot_roots | roots_in_base
    return sorted(tot_roots)

def nthroot_mod(a, n, p, all_roots=False):
    if False:
        return 10
    '\n    Find the solutions to ``x**n = a mod p``.\n\n    Parameters\n    ==========\n\n    a : integer\n    n : positive integer\n    p : positive integer\n    all_roots : if False returns the smallest root, else the list of roots\n\n    Returns\n    =======\n\n        list[int] | int | None :\n            solutions to ``x**n = a mod p``.\n            The table of the output type is:\n\n            ========== ========== ==========\n            all_roots  has roots  Returns\n            ========== ========== ==========\n            True       Yes        list[int]\n            True       No         []\n            False      Yes        int\n            False      No         None\n            ========== ========== ==========\n\n    Raises\n    ======\n\n        ValueError\n            If ``a``, ``n`` or ``p`` is not integer.\n            If ``n`` or ``p`` is not positive.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.residue_ntheory import nthroot_mod\n    >>> nthroot_mod(11, 4, 19)\n    8\n    >>> nthroot_mod(11, 4, 19, True)\n    [8, 11]\n    >>> nthroot_mod(68, 3, 109)\n    23\n\n    References\n    ==========\n\n    .. [1] P. Hackman "Elementary Number Theory" (2009), page 76\n\n    '
    a = a % p
    (a, n, p) = (as_int(a), as_int(n), as_int(p))
    if n < 1:
        raise ValueError('n should be positive')
    if p < 1:
        raise ValueError('p should be positive')
    if n == 1:
        return [a] if all_roots else a
    if n == 2:
        return sqrt_mod(a, p, all_roots)
    base = []
    prime_power = []
    for (q, e) in factorint(p).items():
        tot_roots = _nthroot_mod_prime_power(a, n, q, e)
        if not tot_roots:
            return [] if all_roots else None
        prime_power.append(q ** e)
        base.append(sorted(tot_roots))
    (P, E, S) = gf_crt1(prime_power, ZZ)
    ret = sorted(map(int, {gf_crt2(c, prime_power, P, E, S, ZZ) for c in product(*base)}))
    if all_roots:
        return ret
    if ret:
        return ret[0]

def quadratic_residues(p) -> list[int]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns the list of quadratic residues.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.residue_ntheory import quadratic_residues\n    >>> quadratic_residues(7)\n    [0, 1, 2, 4]\n    '
    p = as_int(p)
    r = {pow(i, 2, p) for i in range(p // 2 + 1)}
    return sorted(r)

def legendre_symbol(a, p):
    if False:
        i = 10
        return i + 15
    '\n    Returns the Legendre symbol `(a / p)`.\n\n    For an integer ``a`` and an odd prime ``p``, the Legendre symbol is\n    defined as\n\n    .. math ::\n        \\genfrac(){}{}{a}{p} = \\begin{cases}\n             0 & \\text{if } p \\text{ divides } a\\\\\n             1 & \\text{if } a \\text{ is a quadratic residue modulo } p\\\\\n            -1 & \\text{if } a \\text{ is a quadratic nonresidue modulo } p\n        \\end{cases}\n\n    Parameters\n    ==========\n\n    a : integer\n    p : odd prime\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory import legendre_symbol\n    >>> [legendre_symbol(i, 7) for i in range(7)]\n    [0, 1, 1, -1, 1, -1, -1]\n    >>> sorted(set([i**2 % 7 for i in range(7)]))\n    [0, 1, 2, 4]\n\n    See Also\n    ========\n\n    is_quad_residue, jacobi_symbol\n\n    '
    (a, p) = (as_int(a), as_int(p))
    if p == 2 or not isprime(p):
        raise ValueError('p should be an odd prime')
    return int(legendre(a, p))

def jacobi_symbol(m, n):
    if False:
        i = 10
        return i + 15
    '\n    Returns the Jacobi symbol `(m / n)`.\n\n    For any integer ``m`` and any positive odd integer ``n`` the Jacobi symbol\n    is defined as the product of the Legendre symbols corresponding to the\n    prime factors of ``n``:\n\n    .. math ::\n        \\genfrac(){}{}{m}{n} =\n            \\genfrac(){}{}{m}{p^{1}}^{\\alpha_1}\n            \\genfrac(){}{}{m}{p^{2}}^{\\alpha_2}\n            ...\n            \\genfrac(){}{}{m}{p^{k}}^{\\alpha_k}\n            \\text{ where } n =\n                p_1^{\\alpha_1}\n                p_2^{\\alpha_2}\n                ...\n                p_k^{\\alpha_k}\n\n    Like the Legendre symbol, if the Jacobi symbol `\\genfrac(){}{}{m}{n} = -1`\n    then ``m`` is a quadratic nonresidue modulo ``n``.\n\n    But, unlike the Legendre symbol, if the Jacobi symbol\n    `\\genfrac(){}{}{m}{n} = 1` then ``m`` may or may not be a quadratic residue\n    modulo ``n``.\n\n    Parameters\n    ==========\n\n    m : integer\n    n : odd positive integer\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory import jacobi_symbol, legendre_symbol\n    >>> from sympy import S\n    >>> jacobi_symbol(45, 77)\n    -1\n    >>> jacobi_symbol(60, 121)\n    1\n\n    The relationship between the ``jacobi_symbol`` and ``legendre_symbol`` can\n    be demonstrated as follows:\n\n    >>> L = legendre_symbol\n    >>> S(45).factors()\n    {3: 2, 5: 1}\n    >>> jacobi_symbol(7, 45) == L(7, 3)**2 * L(7, 5)**1\n    True\n\n    See Also\n    ========\n\n    is_quad_residue, legendre_symbol\n    '
    (m, n) = (as_int(m), as_int(n))
    return int(jacobi(m, n))

def kronecker_symbol(a, n):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns the Kronecker symbol `(a / n)`.\n\n    Parameters\n    ==========\n\n    a : integer\n    n : integer\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.residue_ntheory import kronecker_symbol\n    >>> kronecker_symbol(45, 77)\n    -1\n    >>> kronecker_symbol(13, -120)\n    1\n\n    See Also\n    ========\n\n    jacobi_symbol, legendre_symbol\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Kronecker_symbol\n\n    '
    return int(kronecker(as_int(a), as_int(n)))

class mobius(Function):
    """
    Mobius function maps natural number to {-1, 0, 1}

    It is defined as follows:
        1) `1` if `n = 1`.
        2) `0` if `n` has a squared prime factor.
        3) `(-1)^k` if `n` is a square-free positive integer with `k`
           number of prime factors.

    It is an important multiplicative function in number theory
    and combinatorics.  It has applications in mathematical series,
    algebraic number theory and also physics (Fermion operator has very
    concrete realization with Mobius Function model).

    Parameters
    ==========

    n : positive integer

    Examples
    ========

    >>> from sympy.ntheory import mobius
    >>> mobius(13*7)
    1
    >>> mobius(1)
    1
    >>> mobius(13*7*5)
    -1
    >>> mobius(13**2)
    0

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/M%C3%B6bius_function
    .. [2] Thomas Koshy "Elementary Number Theory with Applications"

    """

    @classmethod
    def eval(cls, n):
        if False:
            while True:
                i = 10
        if n.is_integer:
            if n.is_positive is not True:
                raise ValueError('n should be a positive integer')
        else:
            raise TypeError('n should be an integer')
        if n.is_prime:
            return S.NegativeOne
        elif n is S.One:
            return S.One
        elif n.is_Integer:
            a = factorint(n)
            if any((i > 1 for i in a.values())):
                return S.Zero
            return S.NegativeOne ** len(a)

def _discrete_log_trial_mul(n, a, b, order=None):
    if False:
        i = 10
        return i + 15
    '\n    Trial multiplication algorithm for computing the discrete logarithm of\n    ``a`` to the base ``b`` modulo ``n``.\n\n    The algorithm finds the discrete logarithm using exhaustive search. This\n    naive method is used as fallback algorithm of ``discrete_log`` when the\n    group order is very small.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.residue_ntheory import _discrete_log_trial_mul\n    >>> _discrete_log_trial_mul(41, 15, 7)\n    3\n\n    See Also\n    ========\n\n    discrete_log\n\n    References\n    ==========\n\n    .. [1] "Handbook of applied cryptography", Menezes, A. J., Van, O. P. C., &\n        Vanstone, S. A. (1997).\n    '
    a %= n
    b %= n
    if order is None:
        order = n
    x = 1
    for i in range(order):
        if x == a:
            return i
        x = x * b % n
    raise ValueError('Log does not exist')

def _discrete_log_shanks_steps(n, a, b, order=None):
    if False:
        while True:
            i = 10
    '\n    Baby-step giant-step algorithm for computing the discrete logarithm of\n    ``a`` to the base ``b`` modulo ``n``.\n\n    The algorithm is a time-memory trade-off of the method of exhaustive\n    search. It uses `O(sqrt(m))` memory, where `m` is the group order.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.residue_ntheory import _discrete_log_shanks_steps\n    >>> _discrete_log_shanks_steps(41, 15, 7)\n    3\n\n    See Also\n    ========\n\n    discrete_log\n\n    References\n    ==========\n\n    .. [1] "Handbook of applied cryptography", Menezes, A. J., Van, O. P. C., &\n        Vanstone, S. A. (1997).\n    '
    a %= n
    b %= n
    if order is None:
        order = n_order(b, n)
    m = sqrt(order) + 1
    T = {}
    x = 1
    for i in range(m):
        T[x] = i
        x = x * b % n
    z = pow(b, -m, n)
    x = a
    for i in range(m):
        if x in T:
            return i * m + T[x]
        x = x * z % n
    raise ValueError('Log does not exist')

def _discrete_log_pollard_rho(n, a, b, order=None, retries=10, rseed=None):
    if False:
        print('Hello World!')
    '\n    Pollard\'s Rho algorithm for computing the discrete logarithm of ``a`` to\n    the base ``b`` modulo ``n``.\n\n    It is a randomized algorithm with the same expected running time as\n    ``_discrete_log_shanks_steps``, but requires a negligible amount of memory.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.residue_ntheory import _discrete_log_pollard_rho\n    >>> _discrete_log_pollard_rho(227, 3**7, 3)\n    7\n\n    See Also\n    ========\n\n    discrete_log\n\n    References\n    ==========\n\n    .. [1] "Handbook of applied cryptography", Menezes, A. J., Van, O. P. C., &\n        Vanstone, S. A. (1997).\n    '
    a %= n
    b %= n
    if order is None:
        order = n_order(b, n)
    randint = _randint(rseed)
    for i in range(retries):
        aa = randint(1, order - 1)
        ba = randint(1, order - 1)
        xa = pow(b, aa, n) * pow(a, ba, n) % n
        c = xa % 3
        if c == 0:
            xb = a * xa % n
            ab = aa
            bb = (ba + 1) % order
        elif c == 1:
            xb = xa * xa % n
            ab = (aa + aa) % order
            bb = (ba + ba) % order
        else:
            xb = b * xa % n
            ab = (aa + 1) % order
            bb = ba
        for j in range(order):
            c = xa % 3
            if c == 0:
                xa = a * xa % n
                ba = (ba + 1) % order
            elif c == 1:
                xa = xa * xa % n
                aa = (aa + aa) % order
                ba = (ba + ba) % order
            else:
                xa = b * xa % n
                aa = (aa + 1) % order
            c = xb % 3
            if c == 0:
                xb = a * xb % n
                bb = (bb + 1) % order
            elif c == 1:
                xb = xb * xb % n
                ab = (ab + ab) % order
                bb = (bb + bb) % order
            else:
                xb = b * xb % n
                ab = (ab + 1) % order
            c = xb % 3
            if c == 0:
                xb = a * xb % n
                bb = (bb + 1) % order
            elif c == 1:
                xb = xb * xb % n
                ab = (ab + ab) % order
                bb = (bb + bb) % order
            else:
                xb = b * xb % n
                ab = (ab + 1) % order
            if xa == xb:
                r = (ba - bb) % order
                try:
                    e = invert(r, order) * (ab - aa) % order
                    if (pow(b, e, n) - a) % n == 0:
                        return e
                except ZeroDivisionError:
                    pass
                break
    raise ValueError("Pollard's Rho failed to find logarithm")

def _discrete_log_pohlig_hellman(n, a, b, order=None):
    if False:
        while True:
            i = 10
    '\n    Pohlig-Hellman algorithm for computing the discrete logarithm of ``a`` to\n    the base ``b`` modulo ``n``.\n\n    In order to compute the discrete logarithm, the algorithm takes advantage\n    of the factorization of the group order. It is more efficient when the\n    group order factors into many small primes.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.residue_ntheory import _discrete_log_pohlig_hellman\n    >>> _discrete_log_pohlig_hellman(251, 210, 71)\n    197\n\n    See Also\n    ========\n\n    discrete_log\n\n    References\n    ==========\n\n    .. [1] "Handbook of applied cryptography", Menezes, A. J., Van, O. P. C., &\n        Vanstone, S. A. (1997).\n    '
    from .modular import crt
    a %= n
    b %= n
    if order is None:
        order = n_order(b, n)
    f = factorint(order)
    l = [0] * len(f)
    for (i, (pi, ri)) in enumerate(f.items()):
        for j in range(ri):
            aj = pow(a * pow(b, -l[i], n), order // pi ** (j + 1), n)
            bj = pow(b, order // pi, n)
            cj = discrete_log(n, aj, bj, pi, True)
            l[i] += cj * pi ** j
    (d, _) = crt([pi ** ri for (pi, ri) in f.items()], l)
    return d

def discrete_log(n, a, b, order=None, prime_order=None):
    if False:
        while True:
            i = 10
    '\n    Compute the discrete logarithm of ``a`` to the base ``b`` modulo ``n``.\n\n    This is a recursive function to reduce the discrete logarithm problem in\n    cyclic groups of composite order to the problem in cyclic groups of prime\n    order.\n\n    It employs different algorithms depending on the problem (subgroup order\n    size, prime order or not):\n\n        * Trial multiplication\n        * Baby-step giant-step\n        * Pollard\'s Rho\n        * Pohlig-Hellman\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory import discrete_log\n    >>> discrete_log(41, 15, 7)\n    3\n\n    References\n    ==========\n\n    .. [1] https://mathworld.wolfram.com/DiscreteLogarithm.html\n    .. [2] "Handbook of applied cryptography", Menezes, A. J., Van, O. P. C., &\n        Vanstone, S. A. (1997).\n\n    '
    (n, a, b) = (as_int(n), as_int(a), as_int(b))
    if order is None:
        order = n_order(b, n)
    if prime_order is None:
        prime_order = isprime(order)
    if order < 1000:
        return _discrete_log_trial_mul(n, a, b, order)
    elif prime_order:
        if order < 1000000000000:
            return _discrete_log_shanks_steps(n, a, b, order)
        return _discrete_log_pollard_rho(n, a, b, order)
    return _discrete_log_pohlig_hellman(n, a, b, order)

def quadratic_congruence(a, b, c, n):
    if False:
        for i in range(10):
            print('nop')
    '\n    Find the solutions to `a x^2 + b x + c \\equiv 0 \\pmod{n}`.\n\n    Parameters\n    ==========\n\n    a : int\n    b : int\n    c : int\n    n : int\n        A positive integer.\n\n    Returns\n    =======\n\n    list[int] :\n        A sorted list of solutions. If no solution exists, ``[]``.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.residue_ntheory import quadratic_congruence\n    >>> quadratic_congruence(2, 5, 3, 7) # 2x^2 + 5x + 3 = 0 (mod 7)\n    [2, 6]\n    >>> quadratic_congruence(8, 6, 4, 15) # No solution\n    []\n\n    See Also\n    ========\n\n    polynomial_congruence : Solve the polynomial congruence\n\n    '
    a = as_int(a)
    b = as_int(b)
    c = as_int(c)
    n = as_int(n)
    if n <= 1:
        raise ValueError('n should be an integer greater than 1')
    a %= n
    b %= n
    c %= n
    if a == 0:
        return linear_congruence(b, -c, n)
    if n == 2:
        roots = []
        if c == 0:
            roots.append(0)
        if (b + c) % 2:
            roots.append(1)
        return roots
    if gcd(2 * a, n) == 1:
        inv_a = invert(a, n)
        b *= inv_a
        c *= inv_a
        if b % 2:
            b += n
        b >>= 1
        return sorted(((i - b) % n for i in sqrt_mod_iter(b ** 2 - c, n)))
    res = set()
    for i in sqrt_mod_iter(b ** 2 - 4 * a * c, 4 * a * n):
        res.update((j % n for j in linear_congruence(2 * a, i - b, 4 * a * n)))
    return sorted(res)

def _valid_expr(expr):
    if False:
        while True:
            i = 10
    '\n    return coefficients of expr if it is a univariate polynomial\n    with integer coefficients else raise a ValueError.\n    '
    if not expr.is_polynomial():
        raise ValueError('The expression should be a polynomial')
    polynomial = Poly(expr)
    if not polynomial.is_univariate:
        raise ValueError('The expression should be univariate')
    if not polynomial.domain == ZZ:
        raise ValueError('The expression should should have integer coefficients')
    return polynomial.all_coeffs()

def polynomial_congruence(expr, m):
    if False:
        print('Hello World!')
    '\n    Find the solutions to a polynomial congruence equation modulo m.\n\n    Parameters\n    ==========\n\n    expr : integer coefficient polynomial\n    m : positive integer\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory import polynomial_congruence\n    >>> from sympy.abc import x\n    >>> expr = x**6 - 2*x**5 -35\n    >>> polynomial_congruence(expr, 6125)\n    [3257]\n\n    See Also\n    ========\n\n    sympy.polys.galoistools.gf_csolve : low level solving routine used by this routine\n\n    '
    coefficients = _valid_expr(expr)
    coefficients = [num % m for num in coefficients]
    rank = len(coefficients)
    if rank == 3:
        return quadratic_congruence(*coefficients, m)
    if rank == 2:
        return quadratic_congruence(0, *coefficients, m)
    if coefficients[0] == 1 and 1 + coefficients[-1] == sum(coefficients):
        return nthroot_mod(-coefficients[-1], rank - 1, m, True)
    return gf_csolve(coefficients, m)

def binomial_mod(n, m, k):
    if False:
        while True:
            i = 10
    "Compute ``binomial(n, m) % k``.\n\n    Explanation\n    ===========\n\n    Returns ``binomial(n, m) % k`` using a generalization of Lucas'\n    Theorem for prime powers given by Granville [1]_, in conjunction with\n    the Chinese Remainder Theorem.  The residue for each prime power\n    is calculated in time O(log^2(n) + q^4*log(n)log(p) + q^4*p*log^3(p)).\n\n    Parameters\n    ==========\n\n    n : an integer\n    m : an integer\n    k : a positive integer\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.residue_ntheory import binomial_mod\n    >>> binomial_mod(10, 2, 6)  # binomial(10, 2) = 45\n    3\n    >>> binomial_mod(17, 9, 10)  # binomial(17, 9) = 24310\n    0\n\n    References\n    ==========\n\n    .. [1] Binomial coefficients modulo prime powers, Andrew Granville,\n        Available: https://web.archive.org/web/20170202003812/http://www.dms.umontreal.ca/~andrew/PDF/BinCoeff.pdf\n    "
    if k < 1:
        raise ValueError('k is required to be positive')
    if n < 0 or m < 0 or m > n:
        return 0
    factorisation = factorint(k)
    residues = [_binomial_mod_prime_power(n, m, p, e) for (p, e) in factorisation.items()]
    return crt([p ** pw for (p, pw) in factorisation.items()], residues, check=False)[0]

def _binomial_mod_prime_power(n, m, p, q):
    if False:
        i = 10
        return i + 15
    'Compute ``binomial(n, m) % p**q`` for a prime ``p``.\n\n    Parameters\n    ==========\n\n    n : positive integer\n    m : a nonnegative integer\n    p : a prime\n    q : a positive integer (the prime exponent)\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.residue_ntheory import _binomial_mod_prime_power\n    >>> _binomial_mod_prime_power(10, 2, 3, 2)  # binomial(10, 2) = 45\n    0\n    >>> _binomial_mod_prime_power(17, 9, 2, 4)  # binomial(17, 9) = 24310\n    6\n\n    References\n    ==========\n\n    .. [1] Binomial coefficients modulo prime powers, Andrew Granville,\n        Available: https://web.archive.org/web/20170202003812/http://www.dms.umontreal.ca/~andrew/PDF/BinCoeff.pdf\n    '
    modulo = pow(p, q)

    def up_factorial(u):
        if False:
            i = 10
            return i + 15
        'Compute (u*p)!_p modulo p^q.'
        r = q // 2
        fac = prod = 1
        if r == 1 and p == 2 or 2 * r + 1 in (p, p * p):
            if q % 2 == 1:
                r += 1
            (modulo, div) = (pow(p, 2 * r), pow(p, 2 * r - q))
        else:
            (modulo, div) = (pow(p, 2 * r + 1), pow(p, 2 * r + 1 - q))
        for j in range(1, r + 1):
            for mul in range((j - 1) * p + 1, j * p):
                fac *= mul
                fac %= modulo
            bj_ = bj(u, j, r)
            prod *= pow(fac, bj_, modulo)
            prod %= modulo
        if p == 2:
            sm = u // 2
            for j in range(1, r + 1):
                sm += j // 2 * bj(u, j, r)
            if sm % 2 == 1:
                prod *= -1
        prod %= modulo // div
        return prod % modulo

    def bj(u, j, r):
        if False:
            return 10
        'Compute the exponent of (j*p)!_p in the calculation of (u*p)!_p.'
        prod = u
        for i in range(1, r + 1):
            if i != j:
                prod *= u * u - i * i
        for i in range(1, r + 1):
            if i != j:
                prod //= j * j - i * i
        return prod // j

    def up_plus_v_binom(u, v):
        if False:
            while True:
                i = 10
        'Compute binomial(u*p + v, v)_p modulo p^q.'
        prod = div = 1
        for i in range(1, v + 1):
            div *= i
            div %= modulo
        div = invert(div, modulo)
        for j in range(1, q):
            b = div
            for v_ in range(j * p + 1, j * p + v + 1):
                b *= v_
                b %= modulo
            aj = u
            for i in range(1, q):
                if i != j:
                    aj *= u - i
            for i in range(1, q):
                if i != j:
                    aj //= j - i
            aj //= j
            prod *= pow(b, aj, modulo)
            prod %= modulo
        return prod
    factorials = [1]

    def factorial(v):
        if False:
            i = 10
            return i + 15
        'Compute v! modulo p^q.'
        if len(factorials) <= v:
            for i in range(len(factorials), v + 1):
                factorials.append(factorials[-1] * i % modulo)
        return factorials[v]

    def factorial_p(n):
        if False:
            print('Hello World!')
        'Compute n!_p modulo p^q.'
        (u, v) = divmod(n, p)
        return factorial(v) * up_factorial(u) * up_plus_v_binom(u, v) % modulo
    prod = 1
    (Nj, Mj, Rj) = (n, m, n - m)
    e0 = carry = eq_1 = j = 0
    while Nj:
        numerator = factorial_p(Nj % modulo)
        denominator = factorial_p(Mj % modulo) * factorial_p(Rj % modulo) % modulo
        (Nj, (Mj, mj), (Rj, rj)) = (Nj // p, divmod(Mj, p), divmod(Rj, p))
        carry = (mj + rj + carry) // p
        e0 += carry
        if j >= q - 1:
            eq_1 += carry
        prod *= numerator * invert(denominator, modulo)
        prod %= modulo
        j += 1
    mul = pow(1 if p == 2 and q >= 3 else -1, eq_1, modulo)
    return pow(p, e0, modulo) * mul * prod % modulo