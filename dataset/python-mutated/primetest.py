"""
Primality testing

"""
from itertools import count
from sympy.core.sympify import sympify
from sympy.external.gmpy import gmpy as _gmpy, gcd, jacobi, is_square as gmpy_is_square, bit_scan1, is_fermat_prp, is_euler_prp, is_selfridge_prp, is_strong_selfridge_prp, is_strong_bpsw_prp
from sympy.external.ntheory import _lucas_sequence
from sympy.utilities.misc import as_int, filldedent
MERSENNE_PRIME_EXPONENTS = (2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203, 2281, 3217, 4253, 4423, 9689, 9941, 11213, 19937, 21701, 23209, 44497, 86243, 110503, 132049, 216091, 756839, 859433, 1257787, 1398269, 2976221, 3021377, 6972593, 13466917, 20996011, 24036583, 25964951, 30402457, 32582657, 37156667, 42643801, 43112609, 57885161, 74207281, 77232917, 82589933)

def is_fermat_pseudoprime(n, a):
    if False:
        i = 10
        return i + 15
    'Returns True if ``n`` is prime or is an odd composite integer that\n    is coprime to ``a`` and satisfy the modular arithmetic congruence relation:\n\n    .. math ::\n        a^{n-1} \\equiv 1 \\pmod{n}\n\n    (where mod refers to the modulo operation).\n\n    Parameters\n    ==========\n\n    n : Integer\n        ``n`` is a positive integer.\n    a : Integer\n        ``a`` is a positive integer.\n        ``a`` and ``n`` should be relatively prime.\n\n    Returns\n    =======\n\n    bool : If ``n`` is prime, it always returns ``True``.\n           The composite number that returns ``True`` is called an Fermat pseudoprime.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.primetest import is_fermat_pseudoprime\n    >>> from sympy.ntheory.factor_ import isprime\n    >>> for n in range(1, 1000):\n    ...     if is_fermat_pseudoprime(n, 2) and not isprime(n):\n    ...         print(n)\n    341\n    561\n    645\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Fermat_pseudoprime\n    '
    (n, a) = (as_int(n), as_int(a))
    if a == 1:
        return n == 2 or bool(n % 2)
    return is_fermat_prp(n, a)

def is_euler_pseudoprime(n, a):
    if False:
        while True:
            i = 10
    'Returns True if ``n`` is prime or is an odd composite integer that\n    is coprime to ``a`` and satisfy the modular arithmetic congruence relation:\n\n    .. math ::\n        a^{(n-1)/2} \\equiv \\pm 1 \\pmod{n}\n\n    (where mod refers to the modulo operation).\n\n    Parameters\n    ==========\n\n    n : Integer\n        ``n`` is a positive integer.\n    a : Integer\n        ``a`` is a positive integer.\n        ``a`` and ``n`` should be relatively prime.\n\n    Returns\n    =======\n\n    bool : If ``n`` is prime, it always returns ``True``.\n           The composite number that returns ``True`` is called an Euler pseudoprime.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.primetest import is_euler_pseudoprime\n    >>> from sympy.ntheory.factor_ import isprime\n    >>> for n in range(1, 1000):\n    ...     if is_euler_pseudoprime(n, 2) and not isprime(n):\n    ...         print(n)\n    341\n    561\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Euler_pseudoprime\n    '
    (n, a) = (as_int(n), as_int(a))
    if a < 1:
        raise ValueError('a should be an integer greater than 0')
    if n < 1:
        raise ValueError('n should be an integer greater than 0')
    if n == 1:
        return False
    if a == 1:
        return n == 2 or bool(n % 2)
    if n % 2 == 0:
        return n == 2
    if gcd(n, a) != 1:
        raise ValueError('The two numbers should be relatively prime')
    return pow(a, (n - 1) // 2, n) in [1, n - 1]

def is_euler_jacobi_pseudoprime(n, a):
    if False:
        i = 10
        return i + 15
    'Returns True if ``n`` is prime or is an odd composite integer that\n    is coprime to ``a`` and satisfy the modular arithmetic congruence relation:\n\n    .. math ::\n        a^{(n-1)/2} \\equiv \\left(\\frac{a}{n}\\right) \\pmod{n}\n\n    (where mod refers to the modulo operation).\n\n    Parameters\n    ==========\n\n    n : Integer\n        ``n`` is a positive integer.\n    a : Integer\n        ``a`` is a positive integer.\n        ``a`` and ``n`` should be relatively prime.\n\n    Returns\n    =======\n\n    bool : If ``n`` is prime, it always returns ``True``.\n           The composite number that returns ``True`` is called an Euler-Jacobi pseudoprime.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.primetest import is_euler_jacobi_pseudoprime\n    >>> from sympy.ntheory.factor_ import isprime\n    >>> for n in range(1, 1000):\n    ...     if is_euler_jacobi_pseudoprime(n, 2) and not isprime(n):\n    ...         print(n)\n    561\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Euler%E2%80%93Jacobi_pseudoprime\n    '
    (n, a) = (as_int(n), as_int(a))
    if a == 1:
        return n == 2 or bool(n % 2)
    return is_euler_prp(n, a)

def is_square(n, prep=True):
    if False:
        return 10
    'Return True if n == a * a for some integer a, else False.\n    If n is suspected of *not* being a square then this is a\n    quick method of confirming that it is not.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.primetest import is_square\n    >>> is_square(25)\n    True\n    >>> is_square(2)\n    False\n\n    References\n    ==========\n\n    .. [1]  https://mersenneforum.org/showpost.php?p=110896\n\n    See Also\n    ========\n    sympy.core.intfunc.isqrt\n    '
    if prep:
        n = as_int(n)
        if n < 0:
            return False
        if n in (0, 1):
            return True
    return gmpy_is_square(n)

def _test(n, base, s, t):
    if False:
        while True:
            i = 10
    'Miller-Rabin strong pseudoprime test for one base.\n    Return False if n is definitely composite, True if n is\n    probably prime, with a probability greater than 3/4.\n\n    '
    b = pow(base, t, n)
    if b == 1 or b == n - 1:
        return True
    for _ in range(s - 1):
        b = pow(b, 2, n)
        if b == n - 1:
            return True
        if b == 1:
            return False
    return False

def mr(n, bases):
    if False:
        return 10
    'Perform a Miller-Rabin strong pseudoprime test on n using a\n    given list of bases/witnesses.\n\n    References\n    ==========\n\n    .. [1] Richard Crandall & Carl Pomerance (2005), "Prime Numbers:\n           A Computational Perspective", Springer, 2nd edition, 135-138\n\n    A list of thresholds and the bases they require are here:\n    https://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test#Deterministic_variants\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.primetest import mr\n    >>> mr(1373651, [2, 3])\n    False\n    >>> mr(479001599, [31, 73])\n    True\n\n    '
    from sympy.polys.domains import ZZ
    n = as_int(n)
    if n < 2:
        return False
    s = bit_scan1(n - 1)
    t = n >> s
    for base in bases:
        if base >= n:
            base %= n
        if base >= 2:
            base = ZZ(base)
            if not _test(n, base, s, t):
                return False
    return True

def _lucas_extrastrong_params(n):
    if False:
        return 10
    'Calculates the "extra strong" parameters (D, P, Q) for n.\n\n    Parameters\n    ==========\n\n    n : int\n        positive odd integer\n\n    Returns\n    =======\n\n    D, P, Q: "extra strong" parameters.\n             ``(0, 0, 0)`` if we find a nontrivial divisor of ``n``.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.primetest import _lucas_extrastrong_params\n    >>> _lucas_extrastrong_params(101)\n    (12, 4, 1)\n    >>> _lucas_extrastrong_params(15)\n    (0, 0, 0)\n\n    References\n    ==========\n    .. [1] OEIS A217719: Extra Strong Lucas Pseudoprimes\n           https://oeis.org/A217719\n    .. [2] https://en.wikipedia.org/wiki/Lucas_pseudoprime\n\n    '
    for P in count(3):
        D = P ** 2 - 4
        j = jacobi(D, n)
        if j == -1:
            return (D, P, 1)
        elif j == 0 and D % n:
            return (0, 0, 0)

def is_lucas_prp(n):
    if False:
        while True:
            i = 10
    'Standard Lucas compositeness test with Selfridge parameters.  Returns\n    False if n is definitely composite, and True if n is a Lucas probable\n    prime.\n\n    This is typically used in combination with the Miller-Rabin test.\n\n    References\n    ==========\n    .. [1] Robert Baillie, Samuel S. Wagstaff, Lucas Pseudoprimes,\n           Math. Comp. Vol 35, Number 152 (1980), pp. 1391-1417,\n           https://doi.org/10.1090%2FS0025-5718-1980-0583518-6\n           http://mpqs.free.fr/LucasPseudoprimes.pdf\n    .. [2] OEIS A217120: Lucas Pseudoprimes\n           https://oeis.org/A217120\n    .. [3] https://en.wikipedia.org/wiki/Lucas_pseudoprime\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.primetest import isprime, is_lucas_prp\n    >>> for i in range(10000):\n    ...     if is_lucas_prp(i) and not isprime(i):\n    ...         print(i)\n    323\n    377\n    1159\n    1829\n    3827\n    5459\n    5777\n    9071\n    9179\n    '
    n = as_int(n)
    if n < 2:
        return False
    return is_selfridge_prp(n)

def is_strong_lucas_prp(n):
    if False:
        return 10
    'Strong Lucas compositeness test with Selfridge parameters.  Returns\n    False if n is definitely composite, and True if n is a strong Lucas\n    probable prime.\n\n    This is often used in combination with the Miller-Rabin test, and\n    in particular, when combined with M-R base 2 creates the strong BPSW test.\n\n    References\n    ==========\n    .. [1] Robert Baillie, Samuel S. Wagstaff, Lucas Pseudoprimes,\n           Math. Comp. Vol 35, Number 152 (1980), pp. 1391-1417,\n           https://doi.org/10.1090%2FS0025-5718-1980-0583518-6\n           http://mpqs.free.fr/LucasPseudoprimes.pdf\n    .. [2] OEIS A217255: Strong Lucas Pseudoprimes\n           https://oeis.org/A217255\n    .. [3] https://en.wikipedia.org/wiki/Lucas_pseudoprime\n    .. [4] https://en.wikipedia.org/wiki/Baillie-PSW_primality_test\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.primetest import isprime, is_strong_lucas_prp\n    >>> for i in range(20000):\n    ...     if is_strong_lucas_prp(i) and not isprime(i):\n    ...        print(i)\n    5459\n    5777\n    10877\n    16109\n    18971\n    '
    n = as_int(n)
    if n < 2:
        return False
    return is_strong_selfridge_prp(n)

def is_extra_strong_lucas_prp(n):
    if False:
        i = 10
        return i + 15
    'Extra Strong Lucas compositeness test.  Returns False if n is\n    definitely composite, and True if n is an "extra strong" Lucas probable\n    prime.\n\n    The parameters are selected using P = 3, Q = 1, then incrementing P until\n    (D|n) == -1.  The test itself is as defined in [1]_, from the\n    Mo and Jones preprint.  The parameter selection and test are the same as\n    used in OEIS A217719, Perl\'s Math::Prime::Util, and the Lucas pseudoprime\n    page on Wikipedia.\n\n    It is 20-50% faster than the strong test.\n\n    Because of the different parameters selected, there is no relationship\n    between the strong Lucas pseudoprimes and extra strong Lucas pseudoprimes.\n    In particular, one is not a subset of the other.\n\n    References\n    ==========\n    .. [1] Jon Grantham, Frobenius Pseudoprimes,\n           Math. Comp. Vol 70, Number 234 (2001), pp. 873-891,\n           https://doi.org/10.1090%2FS0025-5718-00-01197-2\n    .. [2] OEIS A217719: Extra Strong Lucas Pseudoprimes\n           https://oeis.org/A217719\n    .. [3] https://en.wikipedia.org/wiki/Lucas_pseudoprime\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.primetest import isprime, is_extra_strong_lucas_prp\n    >>> for i in range(20000):\n    ...     if is_extra_strong_lucas_prp(i) and not isprime(i):\n    ...        print(i)\n    989\n    3239\n    5777\n    10877\n    '
    n = as_int(n)
    if n == 2:
        return True
    if n < 2 or n % 2 == 0:
        return False
    if gmpy_is_square(n):
        return False
    (D, P, Q) = _lucas_extrastrong_params(n)
    if D == 0:
        return False
    s = bit_scan1(n + 1)
    k = n + 1 >> s
    (U, V, _) = _lucas_sequence(n, P, Q, k)
    if U == 0 and (V == 2 or V == n - 2):
        return True
    for _ in range(1, s):
        if V == 0:
            return True
        V = (V * V - 2) % n
    return False

def proth_test(n):
    if False:
        for i in range(10):
            print('nop')
    ' Test if the Proth number `n = k2^m + 1` is prime. where k is a positive odd number and `2^m > k`.\n\n    Parameters\n    ==========\n\n    n : Integer\n        ``n`` is Proth number\n\n    Returns\n    =======\n\n    bool : If ``True``, then ``n`` is the Proth prime\n\n    Raises\n    ======\n\n    ValueError\n        If ``n`` is not Proth number.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.primetest import proth_test\n    >>> proth_test(41)\n    True\n    >>> proth_test(57)\n    False\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Proth_prime\n\n    '
    n = as_int(n)
    if n < 3:
        raise ValueError('n is not Proth number')
    m = bit_scan1(n - 1)
    k = n >> m
    if m < k.bit_length():
        raise ValueError('n is not Proth number')
    if n % 3 == 0:
        return n == 3
    if k % 3:
        return pow(3, n >> 1, n) == n - 1
    if gmpy_is_square(n):
        return False
    for a in range(5, n):
        j = jacobi(a, n)
        if j == -1:
            return pow(a, n >> 1, n) == n - 1
        if j == 0:
            return False

def _lucas_lehmer_primality_test(p):
    if False:
        print('Hello World!')
    ' Test if the Mersenne number `M_p = 2^p-1` is prime.\n\n    Parameters\n    ==========\n\n    p : int\n        ``p`` is an odd prime number\n\n    Returns\n    =======\n\n    bool : If ``True``, then `M_p` is the Mersenne prime\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.primetest import _lucas_lehmer_primality_test\n    >>> _lucas_lehmer_primality_test(5) # 2**5 - 1 = 31 is prime\n    True\n    >>> _lucas_lehmer_primality_test(11) # 2**11 - 1 = 2047 is not prime\n    False\n\n    See Also\n    ========\n\n    is_mersenne_prime\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Lucas%E2%80%93Lehmer_primality_test\n\n    '
    v = 4
    m = 2 ** p - 1
    for _ in range(p - 2):
        v = pow(v, 2, m) - 2
    return v == 0

def is_mersenne_prime(n):
    if False:
        return 10
    'Returns True if  ``n`` is a Mersenne prime, else False.\n\n    A Mersenne prime is a prime number having the form `2^i - 1`.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory.factor_ import is_mersenne_prime\n    >>> is_mersenne_prime(6)\n    False\n    >>> is_mersenne_prime(127)\n    True\n\n    References\n    ==========\n\n    .. [1] https://mathworld.wolfram.com/MersennePrime.html\n\n    '
    n = as_int(n)
    if n < 1:
        return False
    if n & n + 1:
        return False
    p = n.bit_length()
    if p in MERSENNE_PRIME_EXPONENTS:
        return True
    if p < 65000000 or not isprime(p):
        return False
    result = _lucas_lehmer_primality_test(p)
    if result:
        raise ValueError(filldedent("\n            This Mersenne Prime, 2^%s - 1, should\n            be added to SymPy's known values." % p))
    return result

def isprime(n):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if n is a prime number (True) or not (False). For n < 2^64 the\n    answer is definitive; larger n values have a small probability of actually\n    being pseudoprimes.\n\n    Negative numbers (e.g. -2) are not considered prime.\n\n    The first step is looking for trivial factors, which if found enables\n    a quick return.  Next, if the sieve is large enough, use bisection search\n    on the sieve.  For small numbers, a set of deterministic Miller-Rabin\n    tests are performed with bases that are known to have no counterexamples\n    in their range.  Finally if the number is larger than 2^64, a strong\n    BPSW test is performed.  While this is a probable prime test and we\n    believe counterexamples exist, there are no known counterexamples.\n\n    Examples\n    ========\n\n    >>> from sympy.ntheory import isprime\n    >>> isprime(13)\n    True\n    >>> isprime(13.0)  # limited precision\n    False\n    >>> isprime(15)\n    False\n\n    Notes\n    =====\n\n    This routine is intended only for integer input, not numerical\n    expressions which may represent numbers. Floats are also\n    rejected as input because they represent numbers of limited\n    precision. While it is tempting to permit 7.0 to represent an\n    integer there are errors that may "pass silently" if this is\n    allowed:\n\n    >>> from sympy import Float, S\n    >>> int(1e3) == 1e3 == 10**3\n    True\n    >>> int(1e23) == 1e23\n    True\n    >>> int(1e23) == 10**23\n    False\n\n    >>> near_int = 1 + S(1)/10**19\n    >>> near_int == int(near_int)\n    False\n    >>> n = Float(near_int, 10)  # truncated by precision\n    >>> n % 1 == 0\n    True\n    >>> n = Float(near_int, 20)\n    >>> n % 1 == 0\n    False\n\n    See Also\n    ========\n\n    sympy.ntheory.generate.primerange : Generates all primes in a given range\n    sympy.ntheory.generate.primepi : Return the number of primes less than or equal to n\n    sympy.ntheory.generate.prime : Return the nth prime\n\n    References\n    ==========\n    - https://en.wikipedia.org/wiki/Strong_pseudoprime\n    - "Lucas Pseudoprimes", Baillie and Wagstaff, 1980.\n      http://mpqs.free.fr/LucasPseudoprimes.pdf\n    - https://en.wikipedia.org/wiki/Baillie-PSW_primality_test\n    '
    try:
        n = as_int(n)
    except ValueError:
        return False
    if n in [2, 3, 5]:
        return True
    if n < 2 or n % 2 == 0 or n % 3 == 0 or (n % 5 == 0):
        return False
    if n < 49:
        return True
    if n % 7 == 0 or n % 11 == 0 or n % 13 == 0 or (n % 17 == 0) or (n % 19 == 0) or (n % 23 == 0) or (n % 29 == 0) or (n % 31 == 0) or (n % 37 == 0) or (n % 41 == 0) or (n % 43 == 0) or (n % 47 == 0):
        return False
    if n < 2809:
        return True
    if n < 65077:
        return pow(2, n >> 1, n) in [1, n - 1] and n not in [8321, 31621, 42799, 49141, 49981]
    from sympy.ntheory.generate import sieve as s
    if n <= s._list[-1]:
        (l, u) = s.search(n)
        return l == u
    if _gmpy is not None:
        return is_strong_bpsw_prp(n)
    if n < 341531:
        return mr(n, [9345883071009581737])
    if n < 885594169:
        return mr(n, [725270293939359937, 3569819667048198375])
    if n < 350269456337:
        return mr(n, [4230279247111683200, 14694767155120705706, 16641139526367750375])
    if n < 55245642489451:
        return mr(n, [2, 141889084524735, 1199124725622454117, 11096072698276303650])
    if n < 7999252175582851:
        return mr(n, [2, 4130806001517, 149795463772692060, 186635894390467037, 3967304179347715805])
    if n < 585226005592931977:
        return mr(n, [2, 123635709730000, 9233062284813009, 43835965440333360, 761179012939631437, 1263739024124850375])
    if n < 18446744073709551616:
        return mr(n, [2, 325, 9375, 28178, 450775, 9780504, 1795265022])
    if n < 318665857834031151167461:
        return mr(n, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37])
    if n < 3317044064679887385961981:
        return mr(n, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41])
    return is_strong_bpsw_prp(n)

def is_gaussian_prime(num):
    if False:
        i = 10
        return i + 15
    'Test if num is a Gaussian prime number.\n\n    References\n    ==========\n\n    .. [1] https://oeis.org/wiki/Gaussian_primes\n    '
    num = sympify(num)
    (a, b) = num.as_real_imag()
    a = as_int(a, strict=False)
    b = as_int(b, strict=False)
    if a == 0:
        b = abs(b)
        return isprime(b) and b % 4 == 3
    elif b == 0:
        a = abs(a)
        return isprime(a) and a % 4 == 3
    return isprime(a ** 2 + b ** 2)