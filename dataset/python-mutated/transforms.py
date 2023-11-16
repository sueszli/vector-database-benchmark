"""
Discrete Fourier Transform, Number Theoretic Transform,
Walsh Hadamard Transform, Mobius Transform
"""
from sympy.core import S, Symbol, sympify
from sympy.core.function import expand_mul
from sympy.core.numbers import pi, I
from sympy.functions.elementary.trigonometric import sin, cos
from sympy.ntheory import isprime, primitive_root
from sympy.utilities.iterables import ibin, iterable
from sympy.utilities.misc import as_int

def _fourier_transform(seq, dps, inverse=False):
    if False:
        print('Hello World!')
    'Utility function for the Discrete Fourier Transform'
    if not iterable(seq):
        raise TypeError('Expected a sequence of numeric coefficients for Fourier Transform')
    a = [sympify(arg) for arg in seq]
    if any((x.has(Symbol) for x in a)):
        raise ValueError('Expected non-symbolic coefficients')
    n = len(a)
    if n < 2:
        return a
    b = n.bit_length() - 1
    if n & n - 1:
        b += 1
        n = 2 ** b
    a += [S.Zero] * (n - len(a))
    for i in range(1, n):
        j = int(ibin(i, b, str=True)[::-1], 2)
        if i < j:
            (a[i], a[j]) = (a[j], a[i])
    ang = -2 * pi / n if inverse else 2 * pi / n
    if dps is not None:
        ang = ang.evalf(dps + 2)
    w = [cos(ang * i) + I * sin(ang * i) for i in range(n // 2)]
    h = 2
    while h <= n:
        (hf, ut) = (h // 2, n // h)
        for i in range(0, n, h):
            for j in range(hf):
                (u, v) = (a[i + j], expand_mul(a[i + j + hf] * w[ut * j]))
                (a[i + j], a[i + j + hf]) = (u + v, u - v)
        h *= 2
    if inverse:
        a = [(x / n).evalf(dps) for x in a] if dps is not None else [x / n for x in a]
    return a

def fft(seq, dps=None):
    if False:
        i = 10
        return i + 15
    '\n    Performs the Discrete Fourier Transform (**DFT**) in the complex domain.\n\n    The sequence is automatically padded to the right with zeros, as the\n    *radix-2 FFT* requires the number of sample points to be a power of 2.\n\n    This method should be used with default arguments only for short sequences\n    as the complexity of expressions increases with the size of the sequence.\n\n    Parameters\n    ==========\n\n    seq : iterable\n        The sequence on which **DFT** is to be applied.\n    dps : Integer\n        Specifies the number of decimal digits for precision.\n\n    Examples\n    ========\n\n    >>> from sympy import fft, ifft\n\n    >>> fft([1, 2, 3, 4])\n    [10, -2 - 2*I, -2, -2 + 2*I]\n    >>> ifft(_)\n    [1, 2, 3, 4]\n\n    >>> ifft([1, 2, 3, 4])\n    [5/2, -1/2 + I/2, -1/2, -1/2 - I/2]\n    >>> fft(_)\n    [1, 2, 3, 4]\n\n    >>> ifft([1, 7, 3, 4], dps=15)\n    [3.75, -0.5 - 0.75*I, -1.75, -0.5 + 0.75*I]\n    >>> fft(_)\n    [1.0, 7.0, 3.0, 4.0]\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm\n    .. [2] https://mathworld.wolfram.com/FastFourierTransform.html\n\n    '
    return _fourier_transform(seq, dps=dps)

def ifft(seq, dps=None):
    if False:
        i = 10
        return i + 15
    return _fourier_transform(seq, dps=dps, inverse=True)
ifft.__doc__ = fft.__doc__

def _number_theoretic_transform(seq, prime, inverse=False):
    if False:
        for i in range(10):
            print('nop')
    'Utility function for the Number Theoretic Transform'
    if not iterable(seq):
        raise TypeError('Expected a sequence of integer coefficients for Number Theoretic Transform')
    p = as_int(prime)
    if not isprime(p):
        raise ValueError('Expected prime modulus for Number Theoretic Transform')
    a = [as_int(x) % p for x in seq]
    n = len(a)
    if n < 1:
        return a
    b = n.bit_length() - 1
    if n & n - 1:
        b += 1
        n = 2 ** b
    if (p - 1) % n:
        raise ValueError('Expected prime modulus of the form (m*2**k + 1)')
    a += [0] * (n - len(a))
    for i in range(1, n):
        j = int(ibin(i, b, str=True)[::-1], 2)
        if i < j:
            (a[i], a[j]) = (a[j], a[i])
    pr = primitive_root(p)
    rt = pow(pr, (p - 1) // n, p)
    if inverse:
        rt = pow(rt, p - 2, p)
    w = [1] * (n // 2)
    for i in range(1, n // 2):
        w[i] = w[i - 1] * rt % p
    h = 2
    while h <= n:
        (hf, ut) = (h // 2, n // h)
        for i in range(0, n, h):
            for j in range(hf):
                (u, v) = (a[i + j], a[i + j + hf] * w[ut * j])
                (a[i + j], a[i + j + hf]) = ((u + v) % p, (u - v) % p)
        h *= 2
    if inverse:
        rv = pow(n, p - 2, p)
        a = [x * rv % p for x in a]
    return a

def ntt(seq, prime):
    if False:
        for i in range(10):
            print('nop')
    '\n    Performs the Number Theoretic Transform (**NTT**), which specializes the\n    Discrete Fourier Transform (**DFT**) over quotient ring `Z/pZ` for prime\n    `p` instead of complex numbers `C`.\n\n    The sequence is automatically padded to the right with zeros, as the\n    *radix-2 NTT* requires the number of sample points to be a power of 2.\n\n    Parameters\n    ==========\n\n    seq : iterable\n        The sequence on which **DFT** is to be applied.\n    prime : Integer\n        Prime modulus of the form `(m 2^k + 1)` to be used for performing\n        **NTT** on the sequence.\n\n    Examples\n    ========\n\n    >>> from sympy import ntt, intt\n    >>> ntt([1, 2, 3, 4], prime=3*2**8 + 1)\n    [10, 643, 767, 122]\n    >>> intt(_, 3*2**8 + 1)\n    [1, 2, 3, 4]\n    >>> intt([1, 2, 3, 4], prime=3*2**8 + 1)\n    [387, 415, 384, 353]\n    >>> ntt(_, prime=3*2**8 + 1)\n    [1, 2, 3, 4]\n\n    References\n    ==========\n\n    .. [1] http://www.apfloat.org/ntt.html\n    .. [2] https://mathworld.wolfram.com/NumberTheoreticTransform.html\n    .. [3] https://en.wikipedia.org/wiki/Discrete_Fourier_transform_(general%29\n\n    '
    return _number_theoretic_transform(seq, prime=prime)

def intt(seq, prime):
    if False:
        return 10
    return _number_theoretic_transform(seq, prime=prime, inverse=True)
intt.__doc__ = ntt.__doc__

def _walsh_hadamard_transform(seq, inverse=False):
    if False:
        while True:
            i = 10
    'Utility function for the Walsh Hadamard Transform'
    if not iterable(seq):
        raise TypeError('Expected a sequence of coefficients for Walsh Hadamard Transform')
    a = [sympify(arg) for arg in seq]
    n = len(a)
    if n < 2:
        return a
    if n & n - 1:
        n = 2 ** n.bit_length()
    a += [S.Zero] * (n - len(a))
    h = 2
    while h <= n:
        hf = h // 2
        for i in range(0, n, h):
            for j in range(hf):
                (u, v) = (a[i + j], a[i + j + hf])
                (a[i + j], a[i + j + hf]) = (u + v, u - v)
        h *= 2
    if inverse:
        a = [x / n for x in a]
    return a

def fwht(seq):
    if False:
        return 10
    '\n    Performs the Walsh Hadamard Transform (**WHT**), and uses Hadamard\n    ordering for the sequence.\n\n    The sequence is automatically padded to the right with zeros, as the\n    *radix-2 FWHT* requires the number of sample points to be a power of 2.\n\n    Parameters\n    ==========\n\n    seq : iterable\n        The sequence on which WHT is to be applied.\n\n    Examples\n    ========\n\n    >>> from sympy import fwht, ifwht\n    >>> fwht([4, 2, 2, 0, 0, 2, -2, 0])\n    [8, 0, 8, 0, 8, 8, 0, 0]\n    >>> ifwht(_)\n    [4, 2, 2, 0, 0, 2, -2, 0]\n\n    >>> ifwht([19, -1, 11, -9, -7, 13, -15, 5])\n    [2, 0, 4, 0, 3, 10, 0, 0]\n    >>> fwht(_)\n    [19, -1, 11, -9, -7, 13, -15, 5]\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Hadamard_transform\n    .. [2] https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform\n\n    '
    return _walsh_hadamard_transform(seq)

def ifwht(seq):
    if False:
        i = 10
        return i + 15
    return _walsh_hadamard_transform(seq, inverse=True)
ifwht.__doc__ = fwht.__doc__

def _mobius_transform(seq, sgn, subset):
    if False:
        return 10
    "Utility function for performing Mobius Transform using\n    Yate's Dynamic Programming method"
    if not iterable(seq):
        raise TypeError('Expected a sequence of coefficients')
    a = [sympify(arg) for arg in seq]
    n = len(a)
    if n < 2:
        return a
    if n & n - 1:
        n = 2 ** n.bit_length()
    a += [S.Zero] * (n - len(a))
    if subset:
        i = 1
        while i < n:
            for j in range(n):
                if j & i:
                    a[j] += sgn * a[j ^ i]
            i *= 2
    else:
        i = 1
        while i < n:
            for j in range(n):
                if j & i:
                    continue
                a[j] += sgn * a[j ^ i]
            i *= 2
    return a

def mobius_transform(seq, subset=True):
    if False:
        print('Hello World!')
    "\n    Performs the Mobius Transform for subset lattice with indices of\n    sequence as bitmasks.\n\n    The indices of each argument, considered as bit strings, correspond\n    to subsets of a finite set.\n\n    The sequence is automatically padded to the right with zeros, as the\n    definition of subset/superset based on bitmasks (indices) requires\n    the size of sequence to be a power of 2.\n\n    Parameters\n    ==========\n\n    seq : iterable\n        The sequence on which Mobius Transform is to be applied.\n    subset : bool\n        Specifies if Mobius Transform is applied by enumerating subsets\n        or supersets of the given set.\n\n    Examples\n    ========\n\n    >>> from sympy import symbols\n    >>> from sympy import mobius_transform, inverse_mobius_transform\n    >>> x, y, z = symbols('x y z')\n\n    >>> mobius_transform([x, y, z])\n    [x, x + y, x + z, x + y + z]\n    >>> inverse_mobius_transform(_)\n    [x, y, z, 0]\n\n    >>> mobius_transform([x, y, z], subset=False)\n    [x + y + z, y, z, 0]\n    >>> inverse_mobius_transform(_, subset=False)\n    [x, y, z, 0]\n\n    >>> mobius_transform([1, 2, 3, 4])\n    [1, 3, 4, 10]\n    >>> inverse_mobius_transform(_)\n    [1, 2, 3, 4]\n    >>> mobius_transform([1, 2, 3, 4], subset=False)\n    [10, 6, 7, 4]\n    >>> inverse_mobius_transform(_, subset=False)\n    [1, 2, 3, 4]\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/M%C3%B6bius_inversion_formula\n    .. [2] https://people.csail.mit.edu/rrw/presentations/subset-conv.pdf\n    .. [3] https://arxiv.org/pdf/1211.0189.pdf\n\n    "
    return _mobius_transform(seq, sgn=+1, subset=subset)

def inverse_mobius_transform(seq, subset=True):
    if False:
        while True:
            i = 10
    return _mobius_transform(seq, sgn=-1, subset=subset)
inverse_mobius_transform.__doc__ = mobius_transform.__doc__