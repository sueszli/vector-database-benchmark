"""Various algorithms for helping identifying numbers and sequences."""
from sympy.concrete.products import Product, product
from sympy.core import Function, S
from sympy.core.add import Add
from sympy.core.numbers import Integer, Rational
from sympy.core.symbol import Symbol, symbols
from sympy.core.sympify import sympify
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.integers import floor
from sympy.integrals.integrals import integrate
from sympy.polys.polyfuncs import rational_interpolate as rinterp
from sympy.polys.polytools import lcm
from sympy.simplify.radsimp import denom
from sympy.utilities import public

@public
def find_simple_recurrence_vector(l):
    if False:
        print('Hello World!')
    '\n    This function is used internally by other functions from the\n    sympy.concrete.guess module. While most users may want to rather use the\n    function find_simple_recurrence when looking for recurrence relations\n    among rational numbers, the current function may still be useful when\n    some post-processing has to be done.\n\n    Explanation\n    ===========\n\n    The function returns a vector of length n when a recurrence relation of\n    order n is detected in the sequence of rational numbers v.\n\n    If the returned vector has a length 1, then the returned value is always\n    the list [0], which means that no relation has been found.\n\n    While the functions is intended to be used with rational numbers, it should\n    work for other kinds of real numbers except for some cases involving\n    quadratic numbers; for that reason it should be used with some caution when\n    the argument is not a list of rational numbers.\n\n    Examples\n    ========\n\n    >>> from sympy.concrete.guess import find_simple_recurrence_vector\n    >>> from sympy import fibonacci\n    >>> find_simple_recurrence_vector([fibonacci(k) for k in range(12)])\n    [1, -1, -1]\n\n    See Also\n    ========\n\n    See the function sympy.concrete.guess.find_simple_recurrence which is more\n    user-friendly.\n\n    '
    q1 = [0]
    q2 = [1]
    (b, z) = (0, len(l) >> 1)
    while len(q2) <= z:
        while l[b] == 0:
            b += 1
            if b == len(l):
                c = 1
                for x in q2:
                    c = lcm(c, denom(x))
                if q2[0] * c < 0:
                    c = -c
                for k in range(len(q2)):
                    q2[k] = int(q2[k] * c)
                return q2
        a = S.One / l[b]
        m = [a]
        for k in range(b + 1, len(l)):
            m.append(-sum((l[j + 1] * m[b - j - 1] for j in range(b, k))) * a)
        (l, m) = (m, [0] * max(len(q2), b + len(q1)))
        for (k, q) in enumerate(q2):
            m[k] = a * q
        for (k, q) in enumerate(q1):
            m[k + b] += q
        while m[-1] == 0:
            m.pop()
        (q1, q2, b) = (q2, m, 1)
    return [0]

@public
def find_simple_recurrence(v, A=Function('a'), N=Symbol('n')):
    if False:
        print('Hello World!')
    "\n    Detects and returns a recurrence relation from a sequence of several integer\n    (or rational) terms. The name of the function in the returned expression is\n    'a' by default; the main variable is 'n' by default. The smallest index in\n    the returned expression is always n (and never n-1, n-2, etc.).\n\n    Examples\n    ========\n\n    >>> from sympy.concrete.guess import find_simple_recurrence\n    >>> from sympy import fibonacci\n    >>> find_simple_recurrence([fibonacci(k) for k in range(12)])\n    -a(n) - a(n + 1) + a(n + 2)\n\n    >>> from sympy import Function, Symbol\n    >>> a = [1, 1, 1]\n    >>> for k in range(15): a.append(5*a[-1]-3*a[-2]+8*a[-3])\n    >>> find_simple_recurrence(a, A=Function('f'), N=Symbol('i'))\n    -8*f(i) + 3*f(i + 1) - 5*f(i + 2) + f(i + 3)\n\n    "
    p = find_simple_recurrence_vector(v)
    n = len(p)
    if n <= 1:
        return S.Zero
    return Add(*[A(N + n - 1 - k) * p[k] for k in range(n)])

@public
def rationalize(x, maxcoeff=10000):
    if False:
        print('Hello World!')
    '\n    Helps identifying a rational number from a float (or mpmath.mpf) value by\n    using a continued fraction. The algorithm stops as soon as a large partial\n    quotient is detected (greater than 10000 by default).\n\n    Examples\n    ========\n\n    >>> from sympy.concrete.guess import rationalize\n    >>> from mpmath import cos, pi\n    >>> rationalize(cos(pi/3))\n    1/2\n\n    >>> from mpmath import mpf\n    >>> rationalize(mpf("0.333333333333333"))\n    1/3\n\n    While the function is rather intended to help \'identifying\' rational\n    values, it may be used in some cases for approximating real numbers.\n    (Though other functions may be more relevant in that case.)\n\n    >>> rationalize(pi, maxcoeff = 250)\n    355/113\n\n    See Also\n    ========\n\n    Several other methods can approximate a real number as a rational, like:\n\n      * fractions.Fraction.from_decimal\n      * fractions.Fraction.from_float\n      * mpmath.identify\n      * mpmath.pslq by using the following syntax: mpmath.pslq([x, 1])\n      * mpmath.findpoly by using the following syntax: mpmath.findpoly(x, 1)\n      * sympy.simplify.nsimplify (which is a more general function)\n\n    The main difference between the current function and all these variants is\n    that control focuses on magnitude of partial quotients here rather than on\n    global precision of the approximation. If the real is "known to be" a\n    rational number, the current function should be able to detect it correctly\n    with the default settings even when denominator is great (unless its\n    expansion contains unusually big partial quotients) which may occur\n    when studying sequences of increasing numbers. If the user cares more\n    on getting simple fractions, other methods may be more convenient.\n\n    '
    (p0, p1) = (0, 1)
    (q0, q1) = (1, 0)
    a = floor(x)
    while a < maxcoeff or q1 == 0:
        p = a * p1 + p0
        q = a * q1 + q0
        (p0, p1) = (p1, p)
        (q0, q1) = (q1, q)
        if x == a:
            break
        x = 1 / (x - a)
        a = floor(x)
    return sympify(p) / q

@public
def guess_generating_function_rational(v, X=Symbol('x')):
    if False:
        print('Hello World!')
    '\n    Tries to "guess" a rational generating function for a sequence of rational\n    numbers v.\n\n    Examples\n    ========\n\n    >>> from sympy.concrete.guess import guess_generating_function_rational\n    >>> from sympy import fibonacci\n    >>> l = [fibonacci(k) for k in range(5,15)]\n    >>> guess_generating_function_rational(l)\n    (3*x + 5)/(-x**2 - x + 1)\n\n    See Also\n    ========\n\n    sympy.series.approximants\n    mpmath.pade\n\n    '
    q = find_simple_recurrence_vector(v)
    n = len(q)
    if n <= 1:
        return None
    p = [sum((v[i - k] * q[k] for k in range(min(i + 1, n)))) for i in range(len(v) >> 1)]
    return sum((p[k] * X ** k for k in range(len(p)))) / sum((q[k] * X ** k for k in range(n)))

@public
def guess_generating_function(v, X=Symbol('x'), types=['all'], maxsqrtn=2):
    if False:
        i = 10
        return i + 15
    '\n    Tries to "guess" a generating function for a sequence of rational numbers v.\n    Only a few patterns are implemented yet.\n\n    Explanation\n    ===========\n\n    The function returns a dictionary where keys are the name of a given type of\n    generating function. Six types are currently implemented:\n\n         type  |  formal definition\n        -------+----------------------------------------------------------------\n        ogf    | f(x) = Sum(            a_k * x^k       ,  k: 0..infinity )\n        egf    | f(x) = Sum(            a_k * x^k / k!  ,  k: 0..infinity )\n        lgf    | f(x) = Sum( (-1)^(k+1) a_k * x^k / k   ,  k: 1..infinity )\n               |        (with initial index being hold as 1 rather than 0)\n        hlgf   | f(x) = Sum(            a_k * x^k / k   ,  k: 1..infinity )\n               |        (with initial index being hold as 1 rather than 0)\n        lgdogf | f(x) = derivate( log(Sum( a_k * x^k, k: 0..infinity )), x)\n        lgdegf | f(x) = derivate( log(Sum( a_k * x^k / k!, k: 0..infinity )), x)\n\n    In order to spare time, the user can select only some types of generating\n    functions (default being [\'all\']). While forgetting to use a list in the\n    case of a single type may seem to work most of the time as in: types=\'ogf\'\n    this (convenient) syntax may lead to unexpected extra results in some cases.\n\n    Discarding a type when calling the function does not mean that the type will\n    not be present in the returned dictionary; it only means that no extra\n    computation will be performed for that type, but the function may still add\n    it in the result when it can be easily converted from another type.\n\n    Two generating functions (lgdogf and lgdegf) are not even computed if the\n    initial term of the sequence is 0; it may be useful in that case to try\n    again after having removed the leading zeros.\n\n    Examples\n    ========\n\n    >>> from sympy.concrete.guess import guess_generating_function as ggf\n    >>> ggf([k+1 for k in range(12)], types=[\'ogf\', \'lgf\', \'hlgf\'])\n    {\'hlgf\': 1/(1 - x), \'lgf\': 1/(x + 1), \'ogf\': 1/(x**2 - 2*x + 1)}\n\n    >>> from sympy import sympify\n    >>> l = sympify("[3/2, 11/2, 0, -121/2, -363/2, 121]")\n    >>> ggf(l)\n    {\'ogf\': (x + 3/2)/(11*x**2 - 3*x + 1)}\n\n    >>> from sympy import fibonacci\n    >>> ggf([fibonacci(k) for k in range(5, 15)], types=[\'ogf\'])\n    {\'ogf\': (3*x + 5)/(-x**2 - x + 1)}\n\n    >>> from sympy import factorial\n    >>> ggf([factorial(k) for k in range(12)], types=[\'ogf\', \'egf\', \'lgf\'])\n    {\'egf\': 1/(1 - x)}\n\n    >>> ggf([k+1 for k in range(12)], types=[\'egf\'])\n    {\'egf\': (x + 1)*exp(x), \'lgdegf\': (x + 2)/(x + 1)}\n\n    N-th root of a rational function can also be detected (below is an example\n    coming from the sequence A108626 from https://oeis.org).\n    The greatest n-th root to be tested is specified as maxsqrtn (default 2).\n\n    >>> ggf([1, 2, 5, 14, 41, 124, 383, 1200, 3799, 12122, 38919])[\'ogf\']\n    sqrt(1/(x**4 + 2*x**2 - 4*x + 1))\n\n    References\n    ==========\n\n    .. [1] "Concrete Mathematics", R.L. Graham, D.E. Knuth, O. Patashnik\n    .. [2] https://oeis.org/wiki/Generating_functions\n\n    '
    if 'all' in types:
        types = ('ogf', 'egf', 'lgf', 'hlgf', 'lgdogf', 'lgdegf')
    result = {}
    if 'ogf' in types:
        t = [1] + [0] * (len(v) - 1)
        for d in range(max(1, maxsqrtn)):
            t = [sum((t[n - i] * v[i] for i in range(n + 1))) for n in range(len(v))]
            g = guess_generating_function_rational(t, X=X)
            if g:
                result['ogf'] = g ** Rational(1, d + 1)
                break
    if 'egf' in types:
        (w, f) = ([], S.One)
        for (i, k) in enumerate(v):
            f *= i if i else 1
            w.append(k / f)
        t = [1] + [0] * (len(w) - 1)
        for d in range(max(1, maxsqrtn)):
            t = [sum((t[n - i] * w[i] for i in range(n + 1))) for n in range(len(w))]
            g = guess_generating_function_rational(t, X=X)
            if g:
                result['egf'] = g ** Rational(1, d + 1)
                break
    if 'lgf' in types:
        (w, f) = ([], S.NegativeOne)
        for (i, k) in enumerate(v):
            f = -f
            w.append(f * k / Integer(i + 1))
        t = [1] + [0] * (len(w) - 1)
        for d in range(max(1, maxsqrtn)):
            t = [sum((t[n - i] * w[i] for i in range(n + 1))) for n in range(len(w))]
            g = guess_generating_function_rational(t, X=X)
            if g:
                result['lgf'] = g ** Rational(1, d + 1)
                break
    if 'hlgf' in types:
        w = []
        for (i, k) in enumerate(v):
            w.append(k / Integer(i + 1))
        t = [1] + [0] * (len(w) - 1)
        for d in range(max(1, maxsqrtn)):
            t = [sum((t[n - i] * w[i] for i in range(n + 1))) for n in range(len(w))]
            g = guess_generating_function_rational(t, X=X)
            if g:
                result['hlgf'] = g ** Rational(1, d + 1)
                break
    if v[0] != 0 and ('lgdogf' in types or ('ogf' in types and 'ogf' not in result)):
        (a, w) = (sympify(v[0]), [])
        for n in range(len(v) - 1):
            w.append((v[n + 1] * (n + 1) - sum((w[-i - 1] * v[i + 1] for i in range(n)))) / a)
        t = [1] + [0] * (len(w) - 1)
        for d in range(max(1, maxsqrtn)):
            t = [sum((t[n - i] * w[i] for i in range(n + 1))) for n in range(len(w))]
            g = guess_generating_function_rational(t, X=X)
            if g:
                result['lgdogf'] = g ** Rational(1, d + 1)
                if 'ogf' not in result:
                    result['ogf'] = exp(integrate(result['lgdogf'], X))
                break
    if v[0] != 0 and ('lgdegf' in types or ('egf' in types and 'egf' not in result)):
        (z, f) = ([], S.One)
        for (i, k) in enumerate(v):
            f *= i if i else 1
            z.append(k / f)
        (a, w) = (z[0], [])
        for n in range(len(z) - 1):
            w.append((z[n + 1] * (n + 1) - sum((w[-i - 1] * z[i + 1] for i in range(n)))) / a)
        t = [1] + [0] * (len(w) - 1)
        for d in range(max(1, maxsqrtn)):
            t = [sum((t[n - i] * w[i] for i in range(n + 1))) for n in range(len(w))]
            g = guess_generating_function_rational(t, X=X)
            if g:
                result['lgdegf'] = g ** Rational(1, d + 1)
                if 'egf' not in result:
                    result['egf'] = exp(integrate(result['lgdegf'], X))
                break
    return result

@public
def guess(l, all=False, evaluate=True, niter=2, variables=None):
    if False:
        i = 10
        return i + 15
    '\n    This function is adapted from the Rate.m package for Mathematica\n    written by Christian Krattenthaler.\n    It tries to guess a formula from a given sequence of rational numbers.\n\n    Explanation\n    ===========\n\n    In order to speed up the process, the \'all\' variable is set to False by\n    default, stopping the computation as some results are returned during an\n    iteration; the variable can be set to True if more iterations are needed\n    (other formulas may be found; however they may be equivalent to the first\n    ones).\n\n    Another option is the \'evaluate\' variable (default is True); setting it\n    to False will leave the involved products unevaluated.\n\n    By default, the number of iterations is set to 2 but a greater value (up\n    to len(l)-1) can be specified with the optional \'niter\' variable.\n    More and more convoluted results are found when the order of the\n    iteration gets higher:\n\n      * first iteration returns polynomial or rational functions;\n      * second iteration returns products of rising factorials and their\n        inverses;\n      * third iteration returns products of products of rising factorials\n        and their inverses;\n      * etc.\n\n    The returned formulas contain symbols i0, i1, i2, ... where the main\n    variables is i0 (and auxiliary variables are i1, i2, ...). A list of\n    other symbols can be provided in the \'variables\' option; the length of\n    the least should be the value of \'niter\' (more is acceptable but only\n    the first symbols will be used); in this case, the main variable will be\n    the first symbol in the list.\n\n    Examples\n    ========\n\n    >>> from sympy.concrete.guess import guess\n    >>> guess([1,2,6,24,120], evaluate=False)\n    [Product(i1 + 1, (i1, 1, i0 - 1))]\n\n    >>> from sympy import symbols\n    >>> r = guess([1,2,7,42,429,7436,218348,10850216], niter=4)\n    >>> i0 = symbols("i0")\n    >>> [r[0].subs(i0,n).doit() for n in range(1,10)]\n    [1, 2, 7, 42, 429, 7436, 218348, 10850216, 911835460]\n    '
    if any((a == 0 for a in l[:-1])):
        return []
    N = len(l)
    niter = min(N - 1, niter)
    myprod = product if evaluate else Product
    g = []
    res = []
    if variables is None:
        symb = symbols('i:' + str(niter))
    else:
        symb = variables
    for (k, s) in enumerate(symb):
        g.append(l)
        (n, r) = (len(l), [])
        for i in range(n - 2 - 1, -1, -1):
            ri = rinterp(enumerate(g[k][:-1], start=1), i, X=s)
            if denom(ri).subs({s: n}) != 0 and ri.subs({s: n}) - g[k][-1] == 0 and (ri not in r):
                r.append(ri)
        if r:
            for i in range(k - 1, -1, -1):
                r = [g[i][0] * myprod(v, (symb[i + 1], 1, symb[i] - 1)) for v in r]
            if not all:
                return r
            res += r
        l = [Rational(l[i + 1], l[i]) for i in range(N - k - 1)]
    return res