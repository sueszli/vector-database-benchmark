def finite_diff(expression, variable, increment=1):
    if False:
        while True:
            i = 10
    "\n    Takes as input a polynomial expression and the variable used to construct\n    it and returns the difference between function's value when the input is\n    incremented to 1 and the original function value. If you want an increment\n    other than one supply it as a third argument.\n\n    Examples\n    ========\n\n    >>> from sympy.abc import x, y, z\n    >>> from sympy.series.kauers import finite_diff\n    >>> finite_diff(x**2, x)\n    2*x + 1\n    >>> finite_diff(y**3 + 2*y**2 + 3*y + 4, y)\n    3*y**2 + 7*y + 6\n    >>> finite_diff(x**2 + 3*x + 8, x, 2)\n    4*x + 10\n    >>> finite_diff(z**3 + 8*z, z, 3)\n    9*z**2 + 27*z + 51\n    "
    expression = expression.expand()
    expression2 = expression.subs(variable, variable + increment)
    expression2 = expression2.expand()
    return expression2 - expression

def finite_diff_kauers(sum):
    if False:
        while True:
            i = 10
    '\n    Takes as input a Sum instance and returns the difference between the sum\n    with the upper index incremented by 1 and the original sum. For example,\n    if S(n) is a sum, then finite_diff_kauers will return S(n + 1) - S(n).\n\n    Examples\n    ========\n\n    >>> from sympy.series.kauers import finite_diff_kauers\n    >>> from sympy import Sum\n    >>> from sympy.abc import x, y, m, n, k\n    >>> finite_diff_kauers(Sum(k, (k, 1, n)))\n    n + 1\n    >>> finite_diff_kauers(Sum(1/k, (k, 1, n)))\n    1/(n + 1)\n    >>> finite_diff_kauers(Sum((x*y**2), (x, 1, n), (y, 1, m)))\n    (m + 1)**2*(n + 1)\n    >>> finite_diff_kauers(Sum((x*y), (x, 1, m), (y, 1, n)))\n    (m + 1)*(n + 1)\n    '
    function = sum.function
    for l in sum.limits:
        function = function.subs(l[0], l[-1] + 1)
    return function