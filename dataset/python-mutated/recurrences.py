"""
Recurrences
"""
from sympy.core import S, sympify
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import as_int

def linrec(coeffs, init, n):
    if False:
        while True:
            i = 10
    '\n    Evaluation of univariate linear recurrences of homogeneous type\n    having coefficients independent of the recurrence variable.\n\n    Parameters\n    ==========\n\n    coeffs : iterable\n        Coefficients of the recurrence\n    init : iterable\n        Initial values of the recurrence\n    n : Integer\n        Point of evaluation for the recurrence\n\n    Notes\n    =====\n\n    Let `y(n)` be the recurrence of given type, ``c`` be the sequence\n    of coefficients, ``b`` be the sequence of initial/base values of the\n    recurrence and ``k`` (equal to ``len(c)``) be the order of recurrence.\n    Then,\n\n    .. math :: y(n) = \\begin{cases} b_n & 0 \\le n < k \\\\\n        c_0 y(n-1) + c_1 y(n-2) + \\cdots + c_{k-1} y(n-k) & n \\ge k\n        \\end{cases}\n\n    Let `x_0, x_1, \\ldots, x_n` be a sequence and consider the transformation\n    that maps each polynomial `f(x)` to `T(f(x))` where each power `x^i` is\n    replaced by the corresponding value `x_i`. The sequence is then a solution\n    of the recurrence if and only if `T(x^i p(x)) = 0` for each `i \\ge 0` where\n    `p(x) = x^k - c_0 x^(k-1) - \\cdots - c_{k-1}` is the characteristic\n    polynomial.\n\n    Then `T(f(x)p(x)) = 0` for each polynomial `f(x)` (as it is a linear\n    combination of powers `x^i`). Now, if `x^n` is congruent to\n    `g(x) = a_0 x^0 + a_1 x^1 + \\cdots + a_{k-1} x^{k-1}` modulo `p(x)`, then\n    `T(x^n) = x_n` is equal to\n    `T(g(x)) = a_0 x_0 + a_1 x_1 + \\cdots + a_{k-1} x_{k-1}`.\n\n    Computation of `x^n`,\n    given `x^k = c_0 x^{k-1} + c_1 x^{k-2} + \\cdots + c_{k-1}`\n    is performed using exponentiation by squaring (refer to [1_]) with\n    an additional reduction step performed to retain only first `k` powers\n    of `x` in the representation of `x^n`.\n\n    Examples\n    ========\n\n    >>> from sympy.discrete.recurrences import linrec\n    >>> from sympy.abc import x, y, z\n\n    >>> linrec(coeffs=[1, 1], init=[0, 1], n=10)\n    55\n\n    >>> linrec(coeffs=[1, 1], init=[x, y], n=10)\n    34*x + 55*y\n\n    >>> linrec(coeffs=[x, y], init=[0, 1], n=5)\n    x**2*y + x*(x**3 + 2*x*y) + y**2\n\n    >>> linrec(coeffs=[1, 2, 3, 0, 0, 4], init=[x, y, z], n=16)\n    13576*x + 5676*y + 2356*z\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Exponentiation_by_squaring\n    .. [2] https://en.wikipedia.org/w/index.php?title=Modular_exponentiation&section=6#Matrices\n\n    See Also\n    ========\n\n    sympy.polys.agca.extensions.ExtensionElement.__pow__\n\n    '
    if not coeffs:
        return S.Zero
    if not iterable(coeffs):
        raise TypeError('Expected a sequence of coefficients for the recurrence')
    if not iterable(init):
        raise TypeError('Expected a sequence of values for the initialization of the recurrence')
    n = as_int(n)
    if n < 0:
        raise ValueError('Point of evaluation of recurrence must be a non-negative integer')
    c = [sympify(arg) for arg in coeffs]
    b = [sympify(arg) for arg in init]
    k = len(c)
    if len(b) > k:
        raise TypeError('Count of initial values should not exceed the order of the recurrence')
    else:
        b += [S.Zero] * (k - len(b))
    if n < k:
        return b[n]
    terms = [u * v for (u, v) in zip(linrec_coeffs(c, n), b)]
    return sum(terms[:-1], terms[-1])

def linrec_coeffs(c, n):
    if False:
        print('Hello World!')
    "\n    Compute the coefficients of n'th term in linear recursion\n    sequence defined by c.\n\n    `x^k = c_0 x^{k-1} + c_1 x^{k-2} + \\cdots + c_{k-1}`.\n\n    It computes the coefficients by using binary exponentiation.\n    This function is used by `linrec` and `_eval_pow_by_cayley`.\n\n    Parameters\n    ==========\n\n    c = coefficients of the divisor polynomial\n    n = exponent of x, so dividend is x^n\n\n    "
    k = len(c)

    def _square_and_reduce(u, offset):
        if False:
            return 10
        w = [S.Zero] * (2 * len(u) - 1 + offset)
        for (i, p) in enumerate(u):
            for (j, q) in enumerate(u):
                w[offset + i + j] += p * q
        for j in range(len(w) - 1, k - 1, -1):
            for i in range(k):
                w[j - i - 1] += w[j] * c[i]
        return w[:k]

    def _final_coeffs(n):
        if False:
            print('Hello World!')
        if n < k:
            return [S.Zero] * n + [S.One] + [S.Zero] * (k - n - 1)
        else:
            return _square_and_reduce(_final_coeffs(n // 2), n % 2)
    return _final_coeffs(n)