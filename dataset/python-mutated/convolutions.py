"""
Convolution (using **FFT**, **NTT**, **FWHT**), Subset Convolution,
Covering Product, Intersecting Product
"""
from sympy.core import S, sympify, Rational
from sympy.core.function import expand_mul
from sympy.discrete.transforms import fft, ifft, ntt, intt, fwht, ifwht, mobius_transform, inverse_mobius_transform
from sympy.external.gmpy import MPZ, lcm
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import as_int

def convolution(a, b, cycle=0, dps=None, prime=None, dyadic=None, subset=None):
    if False:
        print('Hello World!')
    "\n    Performs convolution by determining the type of desired\n    convolution using hints.\n\n    Exactly one of ``dps``, ``prime``, ``dyadic``, ``subset`` arguments\n    should be specified explicitly for identifying the type of convolution,\n    and the argument ``cycle`` can be specified optionally.\n\n    For the default arguments, linear convolution is performed using **FFT**.\n\n    Parameters\n    ==========\n\n    a, b : iterables\n        The sequences for which convolution is performed.\n    cycle : Integer\n        Specifies the length for doing cyclic convolution.\n    dps : Integer\n        Specifies the number of decimal digits for precision for\n        performing **FFT** on the sequence.\n    prime : Integer\n        Prime modulus of the form `(m 2^k + 1)` to be used for\n        performing **NTT** on the sequence.\n    dyadic : bool\n        Identifies the convolution type as dyadic (*bitwise-XOR*)\n        convolution, which is performed using **FWHT**.\n    subset : bool\n        Identifies the convolution type as subset convolution.\n\n    Examples\n    ========\n\n    >>> from sympy import convolution, symbols, S, I\n    >>> u, v, w, x, y, z = symbols('u v w x y z')\n\n    >>> convolution([1 + 2*I, 4 + 3*I], [S(5)/4, 6], dps=3)\n    [1.25 + 2.5*I, 11.0 + 15.8*I, 24.0 + 18.0*I]\n    >>> convolution([1, 2, 3], [4, 5, 6], cycle=3)\n    [31, 31, 28]\n\n    >>> convolution([111, 777], [888, 444], prime=19*2**10 + 1)\n    [1283, 19351, 14219]\n    >>> convolution([111, 777], [888, 444], prime=19*2**10 + 1, cycle=2)\n    [15502, 19351]\n\n    >>> convolution([u, v], [x, y, z], dyadic=True)\n    [u*x + v*y, u*y + v*x, u*z, v*z]\n    >>> convolution([u, v], [x, y, z], dyadic=True, cycle=2)\n    [u*x + u*z + v*y, u*y + v*x + v*z]\n\n    >>> convolution([u, v, w], [x, y, z], subset=True)\n    [u*x, u*y + v*x, u*z + w*x, v*z + w*y]\n    >>> convolution([u, v, w], [x, y, z], subset=True, cycle=3)\n    [u*x + v*z + w*y, u*y + v*x, u*z + w*x]\n\n    "
    c = as_int(cycle)
    if c < 0:
        raise ValueError('The length for cyclic convolution must be non-negative')
    dyadic = True if dyadic else None
    subset = True if subset else None
    if sum((x is not None for x in (prime, dps, dyadic, subset))) > 1:
        raise TypeError('Ambiguity in determining the type of convolution')
    if prime is not None:
        ls = convolution_ntt(a, b, prime=prime)
        return ls if not c else [sum(ls[i::c]) % prime for i in range(c)]
    if dyadic:
        ls = convolution_fwht(a, b)
    elif subset:
        ls = convolution_subset(a, b)
    else:

        def loop(a):
            if False:
                return 10
            dens = []
            for i in a:
                if isinstance(i, Rational) and i.q - 1:
                    dens.append(i.q)
                elif not isinstance(i, int):
                    return
            if dens:
                l = lcm(*dens)
                return ([i * l if type(i) is int else i.p * (l // i.q) for i in a], l)
            return (a, 1)
        ls = None
        da = loop(a)
        if da is not None:
            db = loop(b)
            if db is not None:
                ((ia, ma), (ib, mb)) = (da, db)
                den = ma * mb
                ls = convolution_int(ia, ib)
                if den != 1:
                    ls = [Rational(i, den) for i in ls]
        if ls is None:
            ls = convolution_fft(a, b, dps)
    return ls if not c else [sum(ls[i::c]) for i in range(c)]

def convolution_fft(a, b, dps=None):
    if False:
        print('Hello World!')
    '\n    Performs linear convolution using Fast Fourier Transform.\n\n    Parameters\n    ==========\n\n    a, b : iterables\n        The sequences for which convolution is performed.\n    dps : Integer\n        Specifies the number of decimal digits for precision.\n\n    Examples\n    ========\n\n    >>> from sympy import S, I\n    >>> from sympy.discrete.convolutions import convolution_fft\n\n    >>> convolution_fft([2, 3], [4, 5])\n    [8, 22, 15]\n    >>> convolution_fft([2, 5], [6, 7, 3])\n    [12, 44, 41, 15]\n    >>> convolution_fft([1 + 2*I, 4 + 3*I], [S(5)/4, 6])\n    [5/4 + 5*I/2, 11 + 63*I/4, 24 + 18*I]\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Convolution_theorem\n    .. [2] https://en.wikipedia.org/wiki/Discrete_Fourier_transform_(general%29\n\n    '
    (a, b) = (a[:], b[:])
    n = m = len(a) + len(b) - 1
    if n > 0 and n & n - 1:
        n = 2 ** n.bit_length()
    a += [S.Zero] * (n - len(a))
    b += [S.Zero] * (n - len(b))
    (a, b) = (fft(a, dps), fft(b, dps))
    a = [expand_mul(x * y) for (x, y) in zip(a, b)]
    a = ifft(a, dps)[:m]
    return a

def convolution_ntt(a, b, prime):
    if False:
        while True:
            i = 10
    '\n    Performs linear convolution using Number Theoretic Transform.\n\n    Parameters\n    ==========\n\n    a, b : iterables\n        The sequences for which convolution is performed.\n    prime : Integer\n        Prime modulus of the form `(m 2^k + 1)` to be used for performing\n        **NTT** on the sequence.\n\n    Examples\n    ========\n\n    >>> from sympy.discrete.convolutions import convolution_ntt\n    >>> convolution_ntt([2, 3], [4, 5], prime=19*2**10 + 1)\n    [8, 22, 15]\n    >>> convolution_ntt([2, 5], [6, 7, 3], prime=19*2**10 + 1)\n    [12, 44, 41, 15]\n    >>> convolution_ntt([333, 555], [222, 666], prime=19*2**10 + 1)\n    [15555, 14219, 19404]\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Convolution_theorem\n    .. [2] https://en.wikipedia.org/wiki/Discrete_Fourier_transform_(general%29\n\n    '
    (a, b, p) = (a[:], b[:], as_int(prime))
    n = m = len(a) + len(b) - 1
    if n > 0 and n & n - 1:
        n = 2 ** n.bit_length()
    a += [0] * (n - len(a))
    b += [0] * (n - len(b))
    (a, b) = (ntt(a, p), ntt(b, p))
    a = [x * y % p for (x, y) in zip(a, b)]
    a = intt(a, p)[:m]
    return a

def convolution_fwht(a, b):
    if False:
        print('Hello World!')
    "\n    Performs dyadic (*bitwise-XOR*) convolution using Fast Walsh Hadamard\n    Transform.\n\n    The convolution is automatically padded to the right with zeros, as the\n    *radix-2 FWHT* requires the number of sample points to be a power of 2.\n\n    Parameters\n    ==========\n\n    a, b : iterables\n        The sequences for which convolution is performed.\n\n    Examples\n    ========\n\n    >>> from sympy import symbols, S, I\n    >>> from sympy.discrete.convolutions import convolution_fwht\n\n    >>> u, v, x, y = symbols('u v x y')\n    >>> convolution_fwht([u, v], [x, y])\n    [u*x + v*y, u*y + v*x]\n\n    >>> convolution_fwht([2, 3], [4, 5])\n    [23, 22]\n    >>> convolution_fwht([2, 5 + 4*I, 7], [6*I, 7, 3 + 4*I])\n    [56 + 68*I, -10 + 30*I, 6 + 50*I, 48 + 32*I]\n\n    >>> convolution_fwht([S(33)/7, S(55)/6, S(7)/4], [S(2)/3, 5])\n    [2057/42, 1870/63, 7/6, 35/4]\n\n    References\n    ==========\n\n    .. [1] https://www.radioeng.cz/fulltexts/2002/02_03_40_42.pdf\n    .. [2] https://en.wikipedia.org/wiki/Hadamard_transform\n\n    "
    if not a or not b:
        return []
    (a, b) = (a[:], b[:])
    n = max(len(a), len(b))
    if n & n - 1:
        n = 2 ** n.bit_length()
    a += [S.Zero] * (n - len(a))
    b += [S.Zero] * (n - len(b))
    (a, b) = (fwht(a), fwht(b))
    a = [expand_mul(x * y) for (x, y) in zip(a, b)]
    a = ifwht(a)
    return a

def convolution_subset(a, b):
    if False:
        while True:
            i = 10
    "\n    Performs Subset Convolution of given sequences.\n\n    The indices of each argument, considered as bit strings, correspond to\n    subsets of a finite set.\n\n    The sequence is automatically padded to the right with zeros, as the\n    definition of subset based on bitmasks (indices) requires the size of\n    sequence to be a power of 2.\n\n    Parameters\n    ==========\n\n    a, b : iterables\n        The sequences for which convolution is performed.\n\n    Examples\n    ========\n\n    >>> from sympy import symbols, S\n    >>> from sympy.discrete.convolutions import convolution_subset\n    >>> u, v, x, y, z = symbols('u v x y z')\n\n    >>> convolution_subset([u, v], [x, y])\n    [u*x, u*y + v*x]\n    >>> convolution_subset([u, v, x], [y, z])\n    [u*y, u*z + v*y, x*y, x*z]\n\n    >>> convolution_subset([1, S(2)/3], [3, 4])\n    [3, 6]\n    >>> convolution_subset([1, 3, S(5)/7], [7])\n    [7, 21, 5, 0]\n\n    References\n    ==========\n\n    .. [1] https://people.csail.mit.edu/rrw/presentations/subset-conv.pdf\n\n    "
    if not a or not b:
        return []
    if not iterable(a) or not iterable(b):
        raise TypeError('Expected a sequence of coefficients for convolution')
    a = [sympify(arg) for arg in a]
    b = [sympify(arg) for arg in b]
    n = max(len(a), len(b))
    if n & n - 1:
        n = 2 ** n.bit_length()
    a += [S.Zero] * (n - len(a))
    b += [S.Zero] * (n - len(b))
    c = [S.Zero] * n
    for mask in range(n):
        smask = mask
        while smask > 0:
            c[mask] += expand_mul(a[smask] * b[mask ^ smask])
            smask = smask - 1 & mask
        c[mask] += expand_mul(a[smask] * b[mask ^ smask])
    return c

def covering_product(a, b):
    if False:
        i = 10
        return i + 15
    "\n    Returns the covering product of given sequences.\n\n    The indices of each argument, considered as bit strings, correspond to\n    subsets of a finite set.\n\n    The covering product of given sequences is a sequence which contains\n    the sum of products of the elements of the given sequences grouped by\n    the *bitwise-OR* of the corresponding indices.\n\n    The sequence is automatically padded to the right with zeros, as the\n    definition of subset based on bitmasks (indices) requires the size of\n    sequence to be a power of 2.\n\n    Parameters\n    ==========\n\n    a, b : iterables\n        The sequences for which covering product is to be obtained.\n\n    Examples\n    ========\n\n    >>> from sympy import symbols, S, I, covering_product\n    >>> u, v, x, y, z = symbols('u v x y z')\n\n    >>> covering_product([u, v], [x, y])\n    [u*x, u*y + v*x + v*y]\n    >>> covering_product([u, v, x], [y, z])\n    [u*y, u*z + v*y + v*z, x*y, x*z]\n\n    >>> covering_product([1, S(2)/3], [3, 4 + 5*I])\n    [3, 26/3 + 25*I/3]\n    >>> covering_product([1, 3, S(5)/7], [7, 8])\n    [7, 53, 5, 40/7]\n\n    References\n    ==========\n\n    .. [1] https://people.csail.mit.edu/rrw/presentations/subset-conv.pdf\n\n    "
    if not a or not b:
        return []
    (a, b) = (a[:], b[:])
    n = max(len(a), len(b))
    if n & n - 1:
        n = 2 ** n.bit_length()
    a += [S.Zero] * (n - len(a))
    b += [S.Zero] * (n - len(b))
    (a, b) = (mobius_transform(a), mobius_transform(b))
    a = [expand_mul(x * y) for (x, y) in zip(a, b)]
    a = inverse_mobius_transform(a)
    return a

def intersecting_product(a, b):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns the intersecting product of given sequences.\n\n    The indices of each argument, considered as bit strings, correspond to\n    subsets of a finite set.\n\n    The intersecting product of given sequences is the sequence which\n    contains the sum of products of the elements of the given sequences\n    grouped by the *bitwise-AND* of the corresponding indices.\n\n    The sequence is automatically padded to the right with zeros, as the\n    definition of subset based on bitmasks (indices) requires the size of\n    sequence to be a power of 2.\n\n    Parameters\n    ==========\n\n    a, b : iterables\n        The sequences for which intersecting product is to be obtained.\n\n    Examples\n    ========\n\n    >>> from sympy import symbols, S, I, intersecting_product\n    >>> u, v, x, y, z = symbols('u v x y z')\n\n    >>> intersecting_product([u, v], [x, y])\n    [u*x + u*y + v*x, v*y]\n    >>> intersecting_product([u, v, x], [y, z])\n    [u*y + u*z + v*y + x*y + x*z, v*z, 0, 0]\n\n    >>> intersecting_product([1, S(2)/3], [3, 4 + 5*I])\n    [9 + 5*I, 8/3 + 10*I/3]\n    >>> intersecting_product([1, 3, S(5)/7], [7, 8])\n    [327/7, 24, 0, 0]\n\n    References\n    ==========\n\n    .. [1] https://people.csail.mit.edu/rrw/presentations/subset-conv.pdf\n\n    "
    if not a or not b:
        return []
    (a, b) = (a[:], b[:])
    n = max(len(a), len(b))
    if n & n - 1:
        n = 2 ** n.bit_length()
    a += [S.Zero] * (n - len(a))
    b += [S.Zero] * (n - len(b))
    (a, b) = (mobius_transform(a, subset=False), mobius_transform(b, subset=False))
    a = [expand_mul(x * y) for (x, y) in zip(a, b)]
    a = inverse_mobius_transform(a, subset=False)
    return a

def convolution_int(a, b):
    if False:
        for i in range(10):
            print('nop')
    'Return the convolution of two sequences as a list.\n\n    The iterables must consist solely of integers.\n\n    Parameters\n    ==========\n\n    a, b : Sequence\n        The sequences for which convolution is performed.\n\n    Explanation\n    ===========\n\n    This function performs the convolution of ``a`` and ``b`` by packing\n    each into a single integer, multiplying them together, and then\n    unpacking the result from the product.  The intuition behind this is\n    that if we evaluate some polynomial [1]:\n\n    .. math ::\n        1156x^6 + 3808x^5 + 8440x^4 + 14856x^3 + 16164x^2 + 14040x + 8100\n\n    at say $x = 10^5$ we obtain $1156038080844014856161641404008100$.\n    Note we can read of the coefficients for each term every five digits.\n    If the $x$ we chose to evaluate at is large enough, the same will hold\n    for the product.\n\n    The idea now is since big integer multiplication in libraries such\n    as GMP is highly optimised, this will be reasonably fast.\n\n    Examples\n    ========\n\n    >>> from sympy.discrete.convolutions import convolution_int\n\n    >>> convolution_int([2, 3], [4, 5])\n    [8, 22, 15]\n    >>> convolution_int([1, 1, -1], [1, 1])\n    [1, 2, 0, -1]\n\n    References\n    ==========\n\n    .. [1] Fateman, Richard J.\n           Can you save time in multiplying polynomials by encoding them as integers?\n           University of California, Berkeley, California (2004).\n           https://people.eecs.berkeley.edu/~fateman/papers/polysbyGMP.pdf\n    '
    B = max((abs(c) for c in a)) * max((abs(c) for c in b)) * (1 + min(len(a) - 1, len(b) - 1))
    (x, power) = (MPZ(1), 0)
    while x <= 2 * B:
        x <<= 1
        power += 1

    def to_integer(poly):
        if False:
            return 10
        (n, mul) = (MPZ(0), 0)
        for c in reversed(poly):
            if c and (not mul):
                mul = -1 if c < 0 else 1
            n <<= power
            n += mul * int(c)
        return (mul, n)
    ((a_mul, a_packed), (b_mul, b_packed)) = (to_integer(a), to_integer(b))
    result = a_packed * b_packed
    mul = a_mul * b_mul
    (mask, half, borrow, poly) = (x - 1, x >> 1, 0, [])
    while result or borrow:
        coeff = (result & mask) + borrow
        result >>= power
        borrow = coeff >= half
        poly.append(mul * int(coeff if coeff < half else coeff - x))
    return poly or [0]