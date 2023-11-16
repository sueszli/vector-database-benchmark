from sympy.functions import SingularityFunction, DiracDelta
from sympy.integrals import integrate

def singularityintegrate(f, x):
    if False:
        while True:
            i = 10
    "\n    This function handles the indefinite integrations of Singularity functions.\n    The ``integrate`` function calls this function internally whenever an\n    instance of SingularityFunction is passed as argument.\n\n    Explanation\n    ===========\n\n    The idea for integration is the following:\n\n    - If we are dealing with a SingularityFunction expression,\n      i.e. ``SingularityFunction(x, a, n)``, we just return\n      ``SingularityFunction(x, a, n + 1)/(n + 1)`` if ``n >= 0`` and\n      ``SingularityFunction(x, a, n + 1)`` if ``n < 0``.\n\n    - If the node is a multiplication or power node having a\n      SingularityFunction term we rewrite the whole expression in terms of\n      Heaviside and DiracDelta and then integrate the output. Lastly, we\n      rewrite the output of integration back in terms of SingularityFunction.\n\n    - If none of the above case arises, we return None.\n\n    Examples\n    ========\n\n    >>> from sympy.integrals.singularityfunctions import singularityintegrate\n    >>> from sympy import SingularityFunction, symbols, Function\n    >>> x, a, n, y = symbols('x a n y')\n    >>> f = Function('f')\n    >>> singularityintegrate(SingularityFunction(x, a, 3), x)\n    SingularityFunction(x, a, 4)/4\n    >>> singularityintegrate(5*SingularityFunction(x, 5, -2), x)\n    5*SingularityFunction(x, 5, -1)\n    >>> singularityintegrate(6*SingularityFunction(x, 5, -1), x)\n    6*SingularityFunction(x, 5, 0)\n    >>> singularityintegrate(x*SingularityFunction(x, 0, -1), x)\n    0\n    >>> singularityintegrate(SingularityFunction(x, 1, -1) * f(x), x)\n    f(1)*SingularityFunction(x, 1, 0)\n\n    "
    if not f.has(SingularityFunction):
        return None
    if isinstance(f, SingularityFunction):
        (x, a, n) = f.args
        if n.is_positive or n.is_zero:
            return SingularityFunction(x, a, n + 1) / (n + 1)
        elif n in (-1, -2):
            return SingularityFunction(x, a, n + 1)
    if f.is_Mul or f.is_Pow:
        expr = f.rewrite(DiracDelta)
        expr = integrate(expr, x)
        return expr.rewrite(SingularityFunction)
    return None