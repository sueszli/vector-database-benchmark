"""Tools for arithmetic error propagation."""
from itertools import repeat, combinations
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp
from sympy.simplify.simplify import simplify
from sympy.stats.symbolic_probability import RandomSymbol, Variance, Covariance
from sympy.stats.rv import is_random
_arg0_or_var = lambda var: var.args[0] if len(var.args) > 0 else var

def variance_prop(expr, consts=(), include_covar=False):
    if False:
        return 10
    "Symbolically propagates variance (`\\sigma^2`) for expressions.\n    This is computed as as seen in [1]_.\n\n    Parameters\n    ==========\n\n    expr : Expr\n        A SymPy expression to compute the variance for.\n    consts : sequence of Symbols, optional\n        Represents symbols that are known constants in the expr,\n        and thus have zero variance. All symbols not in consts are\n        assumed to be variant.\n    include_covar : bool, optional\n        Flag for whether or not to include covariances, default=False.\n\n    Returns\n    =======\n\n    var_expr : Expr\n        An expression for the total variance of the expr.\n        The variance for the original symbols (e.g. x) are represented\n        via instance of the Variance symbol (e.g. Variance(x)).\n\n    Examples\n    ========\n\n    >>> from sympy import symbols, exp\n    >>> from sympy.stats.error_prop import variance_prop\n    >>> x, y = symbols('x y')\n\n    >>> variance_prop(x + y)\n    Variance(x) + Variance(y)\n\n    >>> variance_prop(x * y)\n    x**2*Variance(y) + y**2*Variance(x)\n\n    >>> variance_prop(exp(2*x))\n    4*exp(4*x)*Variance(x)\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Propagation_of_uncertainty\n\n    "
    args = expr.args
    if len(args) == 0:
        if expr in consts:
            return S.Zero
        elif is_random(expr):
            return Variance(expr).doit()
        elif isinstance(expr, Symbol):
            return Variance(RandomSymbol(expr)).doit()
        else:
            return S.Zero
    nargs = len(args)
    var_args = list(map(variance_prop, args, repeat(consts, nargs), repeat(include_covar, nargs)))
    if isinstance(expr, Add):
        var_expr = Add(*var_args)
        if include_covar:
            terms = [2 * Covariance(_arg0_or_var(x), _arg0_or_var(y)).expand() for (x, y) in combinations(var_args, 2)]
            var_expr += Add(*terms)
    elif isinstance(expr, Mul):
        terms = [v / a ** 2 for (a, v) in zip(args, var_args)]
        var_expr = simplify(expr ** 2 * Add(*terms))
        if include_covar:
            terms = [2 * Covariance(_arg0_or_var(x), _arg0_or_var(y)).expand() / (a * b) for ((a, b), (x, y)) in zip(combinations(args, 2), combinations(var_args, 2))]
            var_expr += Add(*terms)
    elif isinstance(expr, Pow):
        b = args[1]
        v = var_args[0] * (expr * b / args[0]) ** 2
        var_expr = simplify(v)
    elif isinstance(expr, exp):
        var_expr = simplify(var_args[0] * expr ** 2)
    else:
        var_expr = Variance(expr)
    return var_expr