"""
Convergence acceleration / extrapolation methods for series and
sequences.

References:
Carl M. Bender & Steven A. Orszag, "Advanced Mathematical Methods for
Scientists and Engineers: Asymptotic Methods and Perturbation Theory",
Springer 1999. (Shanks transformation: pp. 368-375, Richardson
extrapolation: pp. 375-377.)
"""
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.functions.combinatorial.factorials import factorial

def richardson(A, k, n, N):
    if False:
        print('Hello World!')
    '\n    Calculate an approximation for lim k->oo A(k) using Richardson\n    extrapolation with the terms A(n), A(n+1), ..., A(n+N+1).\n    Choosing N ~= 2*n often gives good results.\n\n    Examples\n    ========\n\n    A simple example is to calculate exp(1) using the limit definition.\n    This limit converges slowly; n = 100 only produces two accurate\n    digits:\n\n        >>> from sympy.abc import n\n        >>> e = (1 + 1/n)**n\n        >>> print(round(e.subs(n, 100).evalf(), 10))\n        2.7048138294\n\n    Richardson extrapolation with 11 appropriately chosen terms gives\n    a value that is accurate to the indicated precision:\n\n        >>> from sympy import E\n        >>> from sympy.series.acceleration import richardson\n        >>> print(round(richardson(e, n, 10, 20).evalf(), 10))\n        2.7182818285\n        >>> print(round(E.evalf(), 10))\n        2.7182818285\n\n    Another useful application is to speed up convergence of series.\n    Computing 100 terms of the zeta(2) series 1/k**2 yields only\n    two accurate digits:\n\n        >>> from sympy.abc import k, n\n        >>> from sympy import Sum\n        >>> A = Sum(k**-2, (k, 1, n))\n        >>> print(round(A.subs(n, 100).evalf(), 10))\n        1.6349839002\n\n    Richardson extrapolation performs much better:\n\n        >>> from sympy import pi\n        >>> print(round(richardson(A, n, 10, 20).evalf(), 10))\n        1.6449340668\n        >>> print(round(((pi**2)/6).evalf(), 10))     # Exact value\n        1.6449340668\n\n    '
    s = S.Zero
    for j in range(0, N + 1):
        s += A.subs(k, Integer(n + j)).doit() * (n + j) ** N * S.NegativeOne ** (j + N) / (factorial(j) * factorial(N - j))
    return s

def shanks(A, k, n, m=1):
    if False:
        for i in range(10):
            print('nop')
    '\n    Calculate an approximation for lim k->oo A(k) using the n-term Shanks\n    transformation S(A)(n). With m > 1, calculate the m-fold recursive\n    Shanks transformation S(S(...S(A)...))(n).\n\n    The Shanks transformation is useful for summing Taylor series that\n    converge slowly near a pole or singularity, e.g. for log(2):\n\n        >>> from sympy.abc import k, n\n        >>> from sympy import Sum, Integer\n        >>> from sympy.series.acceleration import shanks\n        >>> A = Sum(Integer(-1)**(k+1) / k, (k, 1, n))\n        >>> print(round(A.subs(n, 100).doit().evalf(), 10))\n        0.6881721793\n        >>> print(round(shanks(A, n, 25).evalf(), 10))\n        0.6931396564\n        >>> print(round(shanks(A, n, 25, 5).evalf(), 10))\n        0.6931471806\n\n    The correct value is 0.6931471805599453094172321215.\n    '
    table = [A.subs(k, Integer(j)).doit() for j in range(n + m + 2)]
    table2 = table[:]
    for i in range(1, m + 1):
        for j in range(i, n + m + 1):
            (x, y, z) = (table[j - 1], table[j], table[j + 1])
            table2[j] = (z * x - y ** 2) / (z + x - 2 * y)
        table = table2[:]
    return table[n]