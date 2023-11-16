from collections import defaultdict
from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core import Basic, S, Add, Mul, Pow, Symbol, sympify, expand_func, Function, Dummy, Expr, factor_terms, expand_power_exp, Eq
from sympy.core.exprtools import factor_nc
from sympy.core.parameters import global_parameters
from sympy.core.function import expand_log, count_ops, _mexpand, nfloat, expand_mul, expand
from sympy.core.numbers import Float, I, pi, Rational
from sympy.core.relational import Relational
from sympy.core.rules import Transform
from sympy.core.sorting import ordered
from sympy.core.sympify import _sympify
from sympy.core.traversal import bottom_up as _bottom_up, walk as _walk
from sympy.functions import gamma, exp, sqrt, log, exp_polar, re
from sympy.functions.combinatorial.factorials import CombinatorialFunction
from sympy.functions.elementary.complexes import unpolarify, Abs, sign
from sympy.functions.elementary.exponential import ExpBase
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.piecewise import Piecewise, piecewise_fold, piecewise_simplify
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.functions.special.bessel import BesselBase, besselj, besseli, besselk, bessely, jn
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.integrals.integrals import Integral
from sympy.matrices.expressions import MatrixExpr, MatAdd, MatMul, MatPow, MatrixSymbol
from sympy.polys import together, cancel, factor
from sympy.polys.numberfields.minpoly import _is_sum_surds, _minimal_polynomial_sq
from sympy.simplify.combsimp import combsimp
from sympy.simplify.cse_opts import sub_pre, sub_post
from sympy.simplify.hyperexpand import hyperexpand
from sympy.simplify.powsimp import powsimp
from sympy.simplify.radsimp import radsimp, fraction, collect_abs
from sympy.simplify.sqrtdenest import sqrtdenest
from sympy.simplify.trigsimp import trigsimp, exptrigsimp
from sympy.utilities.decorator import deprecated
from sympy.utilities.iterables import has_variety, sift, subsets, iterable
from sympy.utilities.misc import as_int
import mpmath

def separatevars(expr, symbols=[], dict=False, force=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Separates variables in an expression, if possible.  By\n    default, it separates with respect to all symbols in an\n    expression and collects constant coefficients that are\n    independent of symbols.\n\n    Explanation\n    ===========\n\n    If ``dict=True`` then the separated terms will be returned\n    in a dictionary keyed to their corresponding symbols.\n    By default, all symbols in the expression will appear as\n    keys; if symbols are provided, then all those symbols will\n    be used as keys, and any terms in the expression containing\n    other symbols or non-symbols will be returned keyed to the\n    string 'coeff'. (Passing None for symbols will return the\n    expression in a dictionary keyed to 'coeff'.)\n\n    If ``force=True``, then bases of powers will be separated regardless\n    of assumptions on the symbols involved.\n\n    Notes\n    =====\n\n    The order of the factors is determined by Mul, so that the\n    separated expressions may not necessarily be grouped together.\n\n    Although factoring is necessary to separate variables in some\n    expressions, it is not necessary in all cases, so one should not\n    count on the returned factors being factored.\n\n    Examples\n    ========\n\n    >>> from sympy.abc import x, y, z, alpha\n    >>> from sympy import separatevars, sin\n    >>> separatevars((x*y)**y)\n    (x*y)**y\n    >>> separatevars((x*y)**y, force=True)\n    x**y*y**y\n\n    >>> e = 2*x**2*z*sin(y)+2*z*x**2\n    >>> separatevars(e)\n    2*x**2*z*(sin(y) + 1)\n    >>> separatevars(e, symbols=(x, y), dict=True)\n    {'coeff': 2*z, x: x**2, y: sin(y) + 1}\n    >>> separatevars(e, [x, y, alpha], dict=True)\n    {'coeff': 2*z, alpha: 1, x: x**2, y: sin(y) + 1}\n\n    If the expression is not really separable, or is only partially\n    separable, separatevars will do the best it can to separate it\n    by using factoring.\n\n    >>> separatevars(x + x*y - 3*x**2)\n    -x*(3*x - y - 1)\n\n    If the expression is not separable then expr is returned unchanged\n    or (if dict=True) then None is returned.\n\n    >>> eq = 2*x + y*sin(x)\n    >>> separatevars(eq) == eq\n    True\n    >>> separatevars(2*x + y*sin(x), symbols=(x, y), dict=True) is None\n    True\n\n    "
    expr = sympify(expr)
    if dict:
        return _separatevars_dict(_separatevars(expr, force), symbols)
    else:
        return _separatevars(expr, force)

def _separatevars(expr, force):
    if False:
        print('Hello World!')
    if isinstance(expr, Abs):
        arg = expr.args[0]
        if arg.is_Mul and (not arg.is_number):
            s = separatevars(arg, dict=True, force=force)
            if s is not None:
                return Mul(*map(expr.func, s.values()))
            else:
                return expr
    if len(expr.free_symbols) < 2:
        return expr
    if expr.is_Mul:
        args = list(expr.args)
        changed = False
        for (i, a) in enumerate(args):
            args[i] = separatevars(a, force)
            changed = changed or args[i] != a
        if changed:
            expr = expr.func(*args)
        return expr
    if expr.is_Pow and expr.base != S.Exp1:
        expr = Pow(separatevars(expr.base, force=force), expr.exp)
    expr = expr.expand(mul=False, multinomial=False, force=force)
    (_expr, reps) = posify(expr) if force else (expr, {})
    expr = factor(_expr).subs(reps)
    if not expr.is_Add:
        return expr
    args = list(expr.args)
    commonc = args[0].args_cnc(cset=True, warn=False)[0]
    for i in args[1:]:
        commonc &= i.args_cnc(cset=True, warn=False)[0]
    commonc = Mul(*commonc)
    commonc = commonc.as_coeff_Mul()[1]
    commonc_set = commonc.args_cnc(cset=True, warn=False)[0]
    for (i, a) in enumerate(args):
        (c, nc) = a.args_cnc(cset=True, warn=False)
        c = c - commonc_set
        args[i] = Mul(*c) * Mul(*nc)
    nonsepar = Add(*args)
    if len(nonsepar.free_symbols) > 1:
        _expr = nonsepar
        (_expr, reps) = posify(_expr) if force else (_expr, {})
        _expr = factor(_expr).subs(reps)
        if not _expr.is_Add:
            nonsepar = _expr
    return commonc * nonsepar

def _separatevars_dict(expr, symbols):
    if False:
        while True:
            i = 10
    if symbols:
        if not all((t.is_Atom for t in symbols)):
            raise ValueError('symbols must be Atoms.')
        symbols = list(symbols)
    elif symbols is None:
        return {'coeff': expr}
    else:
        symbols = list(expr.free_symbols)
        if not symbols:
            return None
    ret = {i: [] for i in symbols + ['coeff']}
    for i in Mul.make_args(expr):
        expsym = i.free_symbols
        intersection = set(symbols).intersection(expsym)
        if len(intersection) > 1:
            return None
        if len(intersection) == 0:
            ret['coeff'].append(i)
        else:
            ret[intersection.pop()].append(i)
    for (k, v) in ret.items():
        ret[k] = Mul(*v)
    return ret

def posify(eq):
    if False:
        print('Hello World!')
    "Return ``eq`` (with generic symbols made positive) and a\n    dictionary containing the mapping between the old and new\n    symbols.\n\n    Explanation\n    ===========\n\n    Any symbol that has positive=None will be replaced with a positive dummy\n    symbol having the same name. This replacement will allow more symbolic\n    processing of expressions, especially those involving powers and\n    logarithms.\n\n    A dictionary that can be sent to subs to restore ``eq`` to its original\n    symbols is also returned.\n\n    >>> from sympy import posify, Symbol, log, solve\n    >>> from sympy.abc import x\n    >>> posify(x + Symbol('p', positive=True) + Symbol('n', negative=True))\n    (_x + n + p, {_x: x})\n\n    >>> eq = 1/x\n    >>> log(eq).expand()\n    log(1/x)\n    >>> log(posify(eq)[0]).expand()\n    -log(_x)\n    >>> p, rep = posify(eq)\n    >>> log(p).expand().subs(rep)\n    -log(x)\n\n    It is possible to apply the same transformations to an iterable\n    of expressions:\n\n    >>> eq = x**2 - 4\n    >>> solve(eq, x)\n    [-2, 2]\n    >>> eq_x, reps = posify([eq, x]); eq_x\n    [_x**2 - 4, _x]\n    >>> solve(*eq_x)\n    [2]\n    "
    eq = sympify(eq)
    if iterable(eq):
        f = type(eq)
        eq = list(eq)
        syms = set()
        for e in eq:
            syms = syms.union(e.atoms(Symbol))
        reps = {}
        for s in syms:
            reps.update({v: k for (k, v) in posify(s)[1].items()})
        for (i, e) in enumerate(eq):
            eq[i] = e.subs(reps)
        return (f(eq), {r: s for (s, r) in reps.items()})
    reps = {s: Dummy(s.name, positive=True, **s.assumptions0) for s in eq.free_symbols if s.is_positive is None}
    eq = eq.subs(reps)
    return (eq, {r: s for (s, r) in reps.items()})

def hypersimp(f, k):
    if False:
        i = 10
        return i + 15
    'Given combinatorial term f(k) simplify its consecutive term ratio\n       i.e. f(k+1)/f(k).  The input term can be composed of functions and\n       integer sequences which have equivalent representation in terms\n       of gamma special function.\n\n       Explanation\n       ===========\n\n       The algorithm performs three basic steps:\n\n       1. Rewrite all functions in terms of gamma, if possible.\n\n       2. Rewrite all occurrences of gamma in terms of products\n          of gamma and rising factorial with integer,  absolute\n          constant exponent.\n\n       3. Perform simplification of nested fractions, powers\n          and if the resulting expression is a quotient of\n          polynomials, reduce their total degree.\n\n       If f(k) is hypergeometric then as result we arrive with a\n       quotient of polynomials of minimal degree. Otherwise None\n       is returned.\n\n       For more information on the implemented algorithm refer to:\n\n       1. W. Koepf, Algorithms for m-fold Hypergeometric Summation,\n          Journal of Symbolic Computation (1995) 20, 399-417\n    '
    f = sympify(f)
    g = f.subs(k, k + 1) / f
    g = g.rewrite(gamma)
    if g.has(Piecewise):
        g = piecewise_fold(g)
        g = g.args[-1][0]
    g = expand_func(g)
    g = powsimp(g, deep=True, combine='exp')
    if g.is_rational_function(k):
        return simplify(g, ratio=S.Infinity)
    else:
        return None

def hypersimilar(f, g, k):
    if False:
        i = 10
        return i + 15
    '\n    Returns True if ``f`` and ``g`` are hyper-similar.\n\n    Explanation\n    ===========\n\n    Similarity in hypergeometric sense means that a quotient of\n    f(k) and g(k) is a rational function in ``k``. This procedure\n    is useful in solving recurrence relations.\n\n    For more information see hypersimp().\n\n    '
    (f, g) = list(map(sympify, (f, g)))
    h = (f / g).rewrite(gamma)
    h = h.expand(func=True, basic=False)
    return h.is_rational_function(k)

def signsimp(expr, evaluate=None):
    if False:
        for i in range(10):
            print('nop')
    "Make all Add sub-expressions canonical wrt sign.\n\n    Explanation\n    ===========\n\n    If an Add subexpression, ``a``, can have a sign extracted,\n    as determined by could_extract_minus_sign, it is replaced\n    with Mul(-1, a, evaluate=False). This allows signs to be\n    extracted from powers and products.\n\n    Examples\n    ========\n\n    >>> from sympy import signsimp, exp, symbols\n    >>> from sympy.abc import x, y\n    >>> i = symbols('i', odd=True)\n    >>> n = -1 + 1/x\n    >>> n/x/(-n)**2 - 1/n/x\n    (-1 + 1/x)/(x*(1 - 1/x)**2) - 1/(x*(-1 + 1/x))\n    >>> signsimp(_)\n    0\n    >>> x*n + x*-n\n    x*(-1 + 1/x) + x*(1 - 1/x)\n    >>> signsimp(_)\n    0\n\n    Since powers automatically handle leading signs\n\n    >>> (-2)**i\n    -2**i\n\n    signsimp can be used to put the base of a power with an integer\n    exponent into canonical form:\n\n    >>> n**i\n    (-1 + 1/x)**i\n\n    By default, signsimp does not leave behind any hollow simplification:\n    if making an Add canonical wrt sign didn't change the expression, the\n    original Add is restored. If this is not desired then the keyword\n    ``evaluate`` can be set to False:\n\n    >>> e = exp(y - x)\n    >>> signsimp(e) == e\n    True\n    >>> signsimp(e, evaluate=False)\n    exp(-(x - y))\n\n    "
    if evaluate is None:
        evaluate = global_parameters.evaluate
    expr = sympify(expr)
    if not isinstance(expr, (Expr, Relational)) or expr.is_Atom:
        return expr
    e = expr.replace(lambda x: x.is_Mul and --x != x, lambda x: --x)
    e = sub_post(sub_pre(e))
    if not isinstance(e, (Expr, Relational)) or e.is_Atom:
        return e
    if e.is_Add:
        rv = e.func(*[signsimp(a) for a in e.args])
        if not evaluate and isinstance(rv, Add) and rv.could_extract_minus_sign():
            return Mul(S.NegativeOne, -rv, evaluate=False)
        return rv
    if evaluate:
        e = e.replace(lambda x: x.is_Mul and --x != x, lambda x: --x)
    return e

def simplify(expr, ratio=1.7, measure=count_ops, rational=False, inverse=False, doit=True, **kwargs):
    if False:
        print('Hello World!')
    'Simplifies the given expression.\n\n    Explanation\n    ===========\n\n    Simplification is not a well defined term and the exact strategies\n    this function tries can change in the future versions of SymPy. If\n    your algorithm relies on "simplification" (whatever it is), try to\n    determine what you need exactly  -  is it powsimp()?, radsimp()?,\n    together()?, logcombine()?, or something else? And use this particular\n    function directly, because those are well defined and thus your algorithm\n    will be robust.\n\n    Nonetheless, especially for interactive use, or when you do not know\n    anything about the structure of the expression, simplify() tries to apply\n    intelligent heuristics to make the input expression "simpler".  For\n    example:\n\n    >>> from sympy import simplify, cos, sin\n    >>> from sympy.abc import x, y\n    >>> a = (x + x**2)/(x*sin(y)**2 + x*cos(y)**2)\n    >>> a\n    (x**2 + x)/(x*sin(y)**2 + x*cos(y)**2)\n    >>> simplify(a)\n    x + 1\n\n    Note that we could have obtained the same result by using specific\n    simplification functions:\n\n    >>> from sympy import trigsimp, cancel\n    >>> trigsimp(a)\n    (x**2 + x)/x\n    >>> cancel(_)\n    x + 1\n\n    In some cases, applying :func:`simplify` may actually result in some more\n    complicated expression. The default ``ratio=1.7`` prevents more extreme\n    cases: if (result length)/(input length) > ratio, then input is returned\n    unmodified.  The ``measure`` parameter lets you specify the function used\n    to determine how complex an expression is.  The function should take a\n    single argument as an expression and return a number such that if\n    expression ``a`` is more complex than expression ``b``, then\n    ``measure(a) > measure(b)``.  The default measure function is\n    :func:`~.count_ops`, which returns the total number of operations in the\n    expression.\n\n    For example, if ``ratio=1``, ``simplify`` output cannot be longer\n    than input.\n\n    ::\n\n        >>> from sympy import sqrt, simplify, count_ops, oo\n        >>> root = 1/(sqrt(2)+3)\n\n    Since ``simplify(root)`` would result in a slightly longer expression,\n    root is returned unchanged instead::\n\n       >>> simplify(root, ratio=1) == root\n       True\n\n    If ``ratio=oo``, simplify will be applied anyway::\n\n        >>> count_ops(simplify(root, ratio=oo)) > count_ops(root)\n        True\n\n    Note that the shortest expression is not necessary the simplest, so\n    setting ``ratio`` to 1 may not be a good idea.\n    Heuristically, the default value ``ratio=1.7`` seems like a reasonable\n    choice.\n\n    You can easily define your own measure function based on what you feel\n    should represent the "size" or "complexity" of the input expression.  Note\n    that some choices, such as ``lambda expr: len(str(expr))`` may appear to be\n    good metrics, but have other problems (in this case, the measure function\n    may slow down simplify too much for very large expressions).  If you do not\n    know what a good metric would be, the default, ``count_ops``, is a good\n    one.\n\n    For example:\n\n    >>> from sympy import symbols, log\n    >>> a, b = symbols(\'a b\', positive=True)\n    >>> g = log(a) + log(b) + log(a)*log(1/b)\n    >>> h = simplify(g)\n    >>> h\n    log(a*b**(1 - log(a)))\n    >>> count_ops(g)\n    8\n    >>> count_ops(h)\n    5\n\n    So you can see that ``h`` is simpler than ``g`` using the count_ops metric.\n    However, we may not like how ``simplify`` (in this case, using\n    ``logcombine``) has created the ``b**(log(1/a) + 1)`` term.  A simple way\n    to reduce this would be to give more weight to powers as operations in\n    ``count_ops``.  We can do this by using the ``visual=True`` option:\n\n    >>> print(count_ops(g, visual=True))\n    2*ADD + DIV + 4*LOG + MUL\n    >>> print(count_ops(h, visual=True))\n    2*LOG + MUL + POW + SUB\n\n    >>> from sympy import Symbol, S\n    >>> def my_measure(expr):\n    ...     POW = Symbol(\'POW\')\n    ...     # Discourage powers by giving POW a weight of 10\n    ...     count = count_ops(expr, visual=True).subs(POW, 10)\n    ...     # Every other operation gets a weight of 1 (the default)\n    ...     count = count.replace(Symbol, type(S.One))\n    ...     return count\n    >>> my_measure(g)\n    8\n    >>> my_measure(h)\n    14\n    >>> 15./8 > 1.7 # 1.7 is the default ratio\n    True\n    >>> simplify(g, measure=my_measure)\n    -log(a)*log(b) + log(a) + log(b)\n\n    Note that because ``simplify()`` internally tries many different\n    simplification strategies and then compares them using the measure\n    function, we get a completely different result that is still different\n    from the input expression by doing this.\n\n    If ``rational=True``, Floats will be recast as Rationals before simplification.\n    If ``rational=None``, Floats will be recast as Rationals but the result will\n    be recast as Floats. If rational=False(default) then nothing will be done\n    to the Floats.\n\n    If ``inverse=True``, it will be assumed that a composition of inverse\n    functions, such as sin and asin, can be cancelled in any order.\n    For example, ``asin(sin(x))`` will yield ``x`` without checking whether\n    x belongs to the set where this relation is true. The default is\n    False.\n\n    Note that ``simplify()`` automatically calls ``doit()`` on the final\n    expression. You can avoid this behavior by passing ``doit=False`` as\n    an argument.\n\n    Also, it should be noted that simplifying a boolean expression is not\n    well defined. If the expression prefers automatic evaluation (such as\n    :obj:`~.Eq()` or :obj:`~.Or()`), simplification will return ``True`` or\n    ``False`` if truth value can be determined. If the expression is not\n    evaluated by default (such as :obj:`~.Predicate()`), simplification will\n    not reduce it and you should use :func:`~.refine()` or :func:`~.ask()`\n    function. This inconsistency will be resolved in future version.\n\n    See Also\n    ========\n\n    sympy.assumptions.refine.refine : Simplification using assumptions.\n    sympy.assumptions.ask.ask : Query for boolean expressions using assumptions.\n    '

    def shorter(*choices):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the choice that has the fewest ops. In case of a tie,\n        the expression listed first is selected.\n        '
        if not has_variety(choices):
            return choices[0]
        return min(choices, key=measure)

    def done(e):
        if False:
            for i in range(10):
                print('nop')
        rv = e.doit() if doit else e
        return shorter(rv, collect_abs(rv))
    expr = sympify(expr, rational=rational)
    kwargs = {'ratio': kwargs.get('ratio', ratio), 'measure': kwargs.get('measure', measure), 'rational': kwargs.get('rational', rational), 'inverse': kwargs.get('inverse', inverse), 'doit': kwargs.get('doit', doit)}
    if isinstance(expr, Expr) and expr.is_zero:
        return S.Zero if not expr.is_Number else expr
    _eval_simplify = getattr(expr, '_eval_simplify', None)
    if _eval_simplify is not None:
        return _eval_simplify(**kwargs)
    original_expr = expr = collect_abs(signsimp(expr))
    if not isinstance(expr, Basic) or not expr.args:
        return expr
    if inverse and expr.has(Function):
        expr = inversecombine(expr)
        if not expr.args:
            return expr
    handled = (Add, Mul, Pow, ExpBase)
    expr = expr.replace(lambda x: isinstance(x, Expr) and x.args and (not isinstance(x, handled)), lambda x: x.func(*[simplify(i, **kwargs) for i in x.args]), simultaneous=False)
    if not isinstance(expr, handled):
        return done(expr)
    if not expr.is_commutative:
        expr = nc_simplify(expr)
    floats = False
    if rational is not False and expr.has(Float):
        floats = True
        expr = nsimplify(expr, rational=True)
    expr = _bottom_up(expr, lambda w: getattr(w, 'normal', lambda : w)())
    expr = Mul(*powsimp(expr).as_content_primitive())
    _e = cancel(expr)
    expr1 = shorter(_e, _mexpand(_e).cancel())
    expr2 = shorter(together(expr, deep=True), together(expr1, deep=True))
    if ratio is S.Infinity:
        expr = expr2
    else:
        expr = shorter(expr2, expr1, expr)
    if not isinstance(expr, Basic):
        return expr
    expr = factor_terms(expr, sign=False)
    if expr.has(sign):
        expr = expr.rewrite(Abs)
    if expr.has(Piecewise):
        expr = piecewise_fold(expr)
        expr = done(expr)
        if expr.has(Piecewise):
            expr = piecewise_fold(expr)
            if expr.has(KroneckerDelta):
                expr = kroneckersimp(expr)
            if expr.has(Piecewise):
                expr = piecewise_simplify(expr, deep=True, doit=False)
                if expr.has(Piecewise):
                    expr = shorter(expr, factor_terms(expr))
                    return expr
    expr = hyperexpand(expr)
    if expr.has(KroneckerDelta):
        expr = kroneckersimp(expr)
    if expr.has(BesselBase):
        expr = besselsimp(expr)
    if expr.has(TrigonometricFunction, HyperbolicFunction):
        expr = trigsimp(expr, deep=True)
    if expr.has(log):
        expr = shorter(expand_log(expr, deep=True), logcombine(expr))
    if expr.has(CombinatorialFunction, gamma):
        expr = combsimp(expr)
    if expr.has(Sum):
        expr = sum_simplify(expr, **kwargs)
    if expr.has(Integral):
        expr = expr.xreplace({i: factor_terms(i) for i in expr.atoms(Integral)})
    if expr.has(Product):
        expr = product_simplify(expr, **kwargs)
    from sympy.physics.units import Quantity
    if expr.has(Quantity):
        from sympy.physics.units.util import quantity_simplify
        expr = quantity_simplify(expr)
    short = shorter(powsimp(expr, combine='exp', deep=True), powsimp(expr), expr)
    short = shorter(short, cancel(short))
    short = shorter(short, factor_terms(short), expand_power_exp(expand_mul(short)))
    if short.has(TrigonometricFunction, HyperbolicFunction, ExpBase, exp):
        short = exptrigsimp(short)
    hollow_mul = Transform(lambda x: Mul(*x.args), lambda x: x.is_Mul and len(x.args) == 2 and x.args[0].is_Number and x.args[1].is_Add and x.is_commutative)
    expr = short.xreplace(hollow_mul)
    (numer, denom) = expr.as_numer_denom()
    if denom.is_Add:
        (n, d) = fraction(radsimp(1 / denom, symbolic=False, max_terms=1))
        if n is not S.One:
            expr = (numer * n).expand() / d
    if expr.could_extract_minus_sign():
        (n, d) = fraction(expr)
        if d != 0:
            expr = signsimp(-n / -d)
    if measure(expr) > ratio * measure(original_expr):
        expr = original_expr
    if floats and rational is None:
        expr = nfloat(expr, exponent=False)
    return done(expr)

def sum_simplify(s, **kwargs):
    if False:
        print('Hello World!')
    'Main function for Sum simplification'
    if not isinstance(s, Add):
        s = s.xreplace({a: sum_simplify(a, **kwargs) for a in s.atoms(Add) if a.has(Sum)})
    s = expand(s)
    if not isinstance(s, Add):
        return s
    terms = s.args
    s_t = []
    o_t = []
    for term in terms:
        (sum_terms, other) = sift(Mul.make_args(term), lambda i: isinstance(i, Sum), binary=True)
        if not sum_terms:
            o_t.append(term)
            continue
        other = [Mul(*other)]
        s_t.append(Mul(*other + [s._eval_simplify(**kwargs) for s in sum_terms]))
    result = Add(sum_combine(s_t), *o_t)
    return result

def sum_combine(s_t):
    if False:
        i = 10
        return i + 15
    "Helper function for Sum simplification\n\n       Attempts to simplify a list of sums, by combining limits / sum function's\n       returns the simplified sum\n    "
    used = [False] * len(s_t)
    for method in range(2):
        for (i, s_term1) in enumerate(s_t):
            if not used[i]:
                for (j, s_term2) in enumerate(s_t):
                    if not used[j] and i != j:
                        temp = sum_add(s_term1, s_term2, method)
                        if isinstance(temp, (Sum, Mul)):
                            s_t[i] = temp
                            s_term1 = s_t[i]
                            used[j] = True
    result = S.Zero
    for (i, s_term) in enumerate(s_t):
        if not used[i]:
            result = Add(result, s_term)
    return result

def factor_sum(self, limits=None, radical=False, clear=False, fraction=False, sign=True):
    if False:
        return 10
    'Return Sum with constant factors extracted.\n\n    If ``limits`` is specified then ``self`` is the summand; the other\n    keywords are passed to ``factor_terms``.\n\n    Examples\n    ========\n\n    >>> from sympy import Sum\n    >>> from sympy.abc import x, y\n    >>> from sympy.simplify.simplify import factor_sum\n    >>> s = Sum(x*y, (x, 1, 3))\n    >>> factor_sum(s)\n    y*Sum(x, (x, 1, 3))\n    >>> factor_sum(s.function, s.limits)\n    y*Sum(x, (x, 1, 3))\n    '
    kwargs = {'radical': radical, 'clear': clear, 'fraction': fraction, 'sign': sign}
    expr = Sum(self, *limits) if limits else self
    return factor_terms(expr, **kwargs)

def sum_add(self, other, method=0):
    if False:
        return 10
    'Helper function for Sum simplification'

    def __refactor(val):
        if False:
            for i in range(10):
                print('nop')
        args = Mul.make_args(val)
        sumv = next((x for x in args if isinstance(x, Sum)))
        constant = Mul(*[x for x in args if x != sumv])
        return Sum(constant * sumv.function, *sumv.limits)
    if isinstance(self, Mul):
        rself = __refactor(self)
    else:
        rself = self
    if isinstance(other, Mul):
        rother = __refactor(other)
    else:
        rother = other
    if type(rself) is type(rother):
        if method == 0:
            if rself.limits == rother.limits:
                return factor_sum(Sum(rself.function + rother.function, *rself.limits))
        elif method == 1:
            if simplify(rself.function - rother.function) == 0:
                if len(rself.limits) == len(rother.limits) == 1:
                    i = rself.limits[0][0]
                    x1 = rself.limits[0][1]
                    y1 = rself.limits[0][2]
                    j = rother.limits[0][0]
                    x2 = rother.limits[0][1]
                    y2 = rother.limits[0][2]
                    if i == j:
                        if x2 == y1 + 1:
                            return factor_sum(Sum(rself.function, (i, x1, y2)))
                        elif x1 == y2 + 1:
                            return factor_sum(Sum(rself.function, (i, x2, y1)))
    return Add(self, other)

def product_simplify(s, **kwargs):
    if False:
        return 10
    'Main function for Product simplification'
    terms = Mul.make_args(s)
    p_t = []
    o_t = []
    deep = kwargs.get('deep', True)
    for term in terms:
        if isinstance(term, Product):
            if deep:
                p_t.append(Product(term.function.simplify(**kwargs), *term.limits))
            else:
                p_t.append(term)
        else:
            o_t.append(term)
    used = [False] * len(p_t)
    for method in range(2):
        for (i, p_term1) in enumerate(p_t):
            if not used[i]:
                for (j, p_term2) in enumerate(p_t):
                    if not used[j] and i != j:
                        tmp_prod = product_mul(p_term1, p_term2, method)
                        if isinstance(tmp_prod, Product):
                            p_t[i] = tmp_prod
                            used[j] = True
    result = Mul(*o_t)
    for (i, p_term) in enumerate(p_t):
        if not used[i]:
            result = Mul(result, p_term)
    return result

def product_mul(self, other, method=0):
    if False:
        for i in range(10):
            print('nop')
    'Helper function for Product simplification'
    if type(self) is type(other):
        if method == 0:
            if self.limits == other.limits:
                return Product(self.function * other.function, *self.limits)
        elif method == 1:
            if simplify(self.function - other.function) == 0:
                if len(self.limits) == len(other.limits) == 1:
                    i = self.limits[0][0]
                    x1 = self.limits[0][1]
                    y1 = self.limits[0][2]
                    j = other.limits[0][0]
                    x2 = other.limits[0][1]
                    y2 = other.limits[0][2]
                    if i == j:
                        if x2 == y1 + 1:
                            return Product(self.function, (i, x1, y2))
                        elif x1 == y2 + 1:
                            return Product(self.function, (i, x2, y1))
    return Mul(self, other)

def _nthroot_solve(p, n, prec):
    if False:
        print('Hello World!')
    '\n     helper function for ``nthroot``\n     It denests ``p**Rational(1, n)`` using its minimal polynomial\n    '
    from sympy.solvers import solve
    while n % 2 == 0:
        p = sqrtdenest(sqrt(p))
        n = n // 2
    if n == 1:
        return p
    pn = p ** Rational(1, n)
    x = Symbol('x')
    f = _minimal_polynomial_sq(p, n, x)
    if f is None:
        return None
    sols = solve(f, x)
    for sol in sols:
        if abs(sol - pn).n() < 1.0 / 10 ** prec:
            sol = sqrtdenest(sol)
            if _mexpand(sol ** n) == p:
                return sol

def logcombine(expr, force=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Takes logarithms and combines them using the following rules:\n\n    - log(x) + log(y) == log(x*y) if both are positive\n    - a*log(x) == log(x**a) if x is positive and a is real\n\n    If ``force`` is ``True`` then the assumptions above will be assumed to hold if\n    there is no assumption already in place on a quantity. For example, if\n    ``a`` is imaginary or the argument negative, force will not perform a\n    combination but if ``a`` is a symbol with no assumptions the change will\n    take place.\n\n    Examples\n    ========\n\n    >>> from sympy import Symbol, symbols, log, logcombine, I\n    >>> from sympy.abc import a, x, y, z\n    >>> logcombine(a*log(x) + log(y) - log(z))\n    a*log(x) + log(y) - log(z)\n    >>> logcombine(a*log(x) + log(y) - log(z), force=True)\n    log(x**a*y/z)\n    >>> x,y,z = symbols('x,y,z', positive=True)\n    >>> a = Symbol('a', real=True)\n    >>> logcombine(a*log(x) + log(y) - log(z))\n    log(x**a*y/z)\n\n    The transformation is limited to factors and/or terms that\n    contain logs, so the result depends on the initial state of\n    expansion:\n\n    >>> eq = (2 + 3*I)*log(x)\n    >>> logcombine(eq, force=True) == eq\n    True\n    >>> logcombine(eq.expand(), force=True)\n    log(x**2) + I*log(x**3)\n\n    See Also\n    ========\n\n    posify: replace all symbols with symbols having positive assumptions\n    sympy.core.function.expand_log: expand the logarithms of products\n        and powers; the opposite of logcombine\n\n    "

    def f(rv):
        if False:
            return 10
        if not (rv.is_Add or rv.is_Mul):
            return rv

        def gooda(a):
            if False:
                while True:
                    i = 10
            return a is not S.NegativeOne and (a.is_extended_real or (force and a.is_extended_real is not False))

        def goodlog(l):
            if False:
                while True:
                    i = 10
            a = l.args[0]
            return a.is_positive or (force and a.is_nonpositive is not False)
        other = []
        logs = []
        log1 = defaultdict(list)
        for a in Add.make_args(rv):
            if isinstance(a, log) and goodlog(a):
                log1[()].append(([], a))
            elif not a.is_Mul:
                other.append(a)
            else:
                ot = []
                co = []
                lo = []
                for ai in a.args:
                    if ai.is_Rational and ai < 0:
                        ot.append(S.NegativeOne)
                        co.append(-ai)
                    elif isinstance(ai, log) and goodlog(ai):
                        lo.append(ai)
                    elif gooda(ai):
                        co.append(ai)
                    else:
                        ot.append(ai)
                if len(lo) > 1:
                    logs.append((ot, co, lo))
                elif lo:
                    log1[tuple(ot)].append((co, lo[0]))
                else:
                    other.append(a)
        if len(other) == 1 and isinstance(other[0], log):
            log1[()].append(([], other.pop()))
        if not logs and all((len(log1[k]) == 1 and log1[k][0] == [] for k in log1)):
            return rv
        for (o, e, l) in logs:
            l = list(ordered(l))
            e = log(l.pop(0).args[0] ** Mul(*e))
            while l:
                li = l.pop(0)
                e = log(li.args[0] ** e)
            (c, l) = (Mul(*o), e)
            if isinstance(l, log):
                log1[c,].append(([], l))
            else:
                other.append(c * l)
        for k in list(log1.keys()):
            log1[Mul(*k)] = log(logcombine(Mul(*[l.args[0] ** Mul(*c) for (c, l) in log1.pop(k)]), force=force), evaluate=False)
        for k in ordered(list(log1.keys())):
            if k not in log1:
                continue
            if -k in log1:
                (num, den) = (k, -k)
                if num.count_ops() > den.count_ops():
                    (num, den) = (den, num)
                other.append(num * log(log1.pop(num).args[0] / log1.pop(den).args[0], evaluate=False))
            else:
                other.append(k * log1.pop(k))
        return Add(*other)
    return _bottom_up(expr, f)

def inversecombine(expr):
    if False:
        i = 10
        return i + 15
    'Simplify the composition of a function and its inverse.\n\n    Explanation\n    ===========\n\n    No attention is paid to whether the inverse is a left inverse or a\n    right inverse; thus, the result will in general not be equivalent\n    to the original expression.\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.simplify import inversecombine\n    >>> from sympy import asin, sin, log, exp\n    >>> from sympy.abc import x\n    >>> inversecombine(asin(sin(x)))\n    x\n    >>> inversecombine(2*log(exp(3*x)))\n    6*x\n    '

    def f(rv):
        if False:
            while True:
                i = 10
        if isinstance(rv, log):
            if isinstance(rv.args[0], exp) or (rv.args[0].is_Pow and rv.args[0].base == S.Exp1):
                rv = rv.args[0].exp
        elif rv.is_Function and hasattr(rv, 'inverse'):
            if len(rv.args) == 1 and len(rv.args[0].args) == 1 and isinstance(rv.args[0], rv.inverse(argindex=1)):
                rv = rv.args[0].args[0]
        if rv.is_Pow and rv.base == S.Exp1:
            if isinstance(rv.exp, log):
                rv = rv.exp.args[0]
        return rv
    return _bottom_up(expr, f)

def kroneckersimp(expr):
    if False:
        while True:
            i = 10
    '\n    Simplify expressions with KroneckerDelta.\n\n    The only simplification currently attempted is to identify multiplicative cancellation:\n\n    Examples\n    ========\n\n    >>> from sympy import KroneckerDelta, kroneckersimp\n    >>> from sympy.abc import i\n    >>> kroneckersimp(1 + KroneckerDelta(0, i) * KroneckerDelta(1, i))\n    1\n    '

    def args_cancel(args1, args2):
        if False:
            while True:
                i = 10
        for i1 in range(2):
            for i2 in range(2):
                a1 = args1[i1]
                a2 = args2[i2]
                a3 = args1[(i1 + 1) % 2]
                a4 = args2[(i2 + 1) % 2]
                if Eq(a1, a2) is S.true and Eq(a3, a4) is S.false:
                    return True
        return False

    def cancel_kronecker_mul(m):
        if False:
            print('Hello World!')
        args = m.args
        deltas = [a for a in args if isinstance(a, KroneckerDelta)]
        for (delta1, delta2) in subsets(deltas, 2):
            args1 = delta1.args
            args2 = delta2.args
            if args_cancel(args1, args2):
                return S.Zero * m
        return m
    if not expr.has(KroneckerDelta):
        return expr
    if expr.has(Piecewise):
        expr = expr.rewrite(KroneckerDelta)
    newexpr = expr
    expr = None
    while newexpr != expr:
        expr = newexpr
        newexpr = expr.replace(lambda e: isinstance(e, Mul), cancel_kronecker_mul)
    return expr

def besselsimp(expr):
    if False:
        return 10
    '\n    Simplify bessel-type functions.\n\n    Explanation\n    ===========\n\n    This routine tries to simplify bessel-type functions. Currently it only\n    works on the Bessel J and I functions, however. It works by looking at all\n    such functions in turn, and eliminating factors of "I" and "-1" (actually\n    their polar equivalents) in front of the argument. Then, functions of\n    half-integer order are rewritten using strigonometric functions and\n    functions of integer order (> 1) are rewritten using functions\n    of low order.  Finally, if the expression was changed, compute\n    factorization of the result with factor().\n\n    >>> from sympy import besselj, besseli, besselsimp, polar_lift, I, S\n    >>> from sympy.abc import z, nu\n    >>> besselsimp(besselj(nu, z*polar_lift(-1)))\n    exp(I*pi*nu)*besselj(nu, z)\n    >>> besselsimp(besseli(nu, z*polar_lift(-I)))\n    exp(-I*pi*nu/2)*besselj(nu, z)\n    >>> besselsimp(besseli(S(-1)/2, z))\n    sqrt(2)*cosh(z)/(sqrt(pi)*sqrt(z))\n    >>> besselsimp(z*besseli(0, z) + z*(besseli(2, z))/2 + besseli(1, z))\n    3*z*besseli(0, z)/2\n    '

    def replacer(fro, to, factors):
        if False:
            print('Hello World!')
        factors = set(factors)

        def repl(nu, z):
            if False:
                return 10
            if factors.intersection(Mul.make_args(z)):
                return to(nu, z)
            return fro(nu, z)
        return repl

    def torewrite(fro, to):
        if False:
            return 10

        def tofunc(nu, z):
            if False:
                for i in range(10):
                    print('nop')
            return fro(nu, z).rewrite(to)
        return tofunc

    def tominus(fro):
        if False:
            for i in range(10):
                print('nop')

        def tofunc(nu, z):
            if False:
                while True:
                    i = 10
            return exp(I * pi * nu) * fro(nu, exp_polar(-I * pi) * z)
        return tofunc
    orig_expr = expr
    ifactors = [I, exp_polar(I * pi / 2), exp_polar(-I * pi / 2)]
    expr = expr.replace(besselj, replacer(besselj, torewrite(besselj, besseli), ifactors))
    expr = expr.replace(besseli, replacer(besseli, torewrite(besseli, besselj), ifactors))
    minusfactors = [-1, exp_polar(I * pi)]
    expr = expr.replace(besselj, replacer(besselj, tominus(besselj), minusfactors))
    expr = expr.replace(besseli, replacer(besseli, tominus(besseli), minusfactors))
    z0 = Dummy('z')

    def expander(fro):
        if False:
            while True:
                i = 10

        def repl(nu, z):
            if False:
                i = 10
                return i + 15
            if nu % 1 == S.Half:
                return simplify(trigsimp(unpolarify(fro(nu, z0).rewrite(besselj).rewrite(jn).expand(func=True)).subs(z0, z)))
            elif nu.is_Integer and nu > 1:
                return fro(nu, z).expand(func=True)
            return fro(nu, z)
        return repl
    expr = expr.replace(besselj, expander(besselj))
    expr = expr.replace(bessely, expander(bessely))
    expr = expr.replace(besseli, expander(besseli))
    expr = expr.replace(besselk, expander(besselk))

    def _bessel_simp_recursion(expr):
        if False:
            while True:
                i = 10

        def _use_recursion(bessel, expr):
            if False:
                for i in range(10):
                    print('nop')
            while True:
                bessels = expr.find(lambda x: isinstance(x, bessel))
                try:
                    for ba in sorted(bessels, key=lambda x: re(x.args[0])):
                        (a, x) = ba.args
                        bap1 = bessel(a + 1, x)
                        bap2 = bessel(a + 2, x)
                        if expr.has(bap1) and expr.has(bap2):
                            expr = expr.subs(ba, 2 * (a + 1) / x * bap1 - bap2)
                            break
                    else:
                        return expr
                except (ValueError, TypeError):
                    return expr
        if expr.has(besselj):
            expr = _use_recursion(besselj, expr)
        if expr.has(bessely):
            expr = _use_recursion(bessely, expr)
        return expr
    expr = _bessel_simp_recursion(expr)
    if expr != orig_expr:
        expr = expr.factor()
    return expr

def nthroot(expr, n, max_len=4, prec=15):
    if False:
        return 10
    '\n    Compute a real nth-root of a sum of surds.\n\n    Parameters\n    ==========\n\n    expr : sum of surds\n    n : integer\n    max_len : maximum number of surds passed as constants to ``nsimplify``\n\n    Algorithm\n    =========\n\n    First ``nsimplify`` is used to get a candidate root; if it is not a\n    root the minimal polynomial is computed; the answer is one of its\n    roots.\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.simplify import nthroot\n    >>> from sympy import sqrt\n    >>> nthroot(90 + 34*sqrt(7), 3)\n    sqrt(7) + 3\n\n    '
    expr = sympify(expr)
    n = sympify(n)
    p = expr ** Rational(1, n)
    if not n.is_integer:
        return p
    if not _is_sum_surds(expr):
        return p
    surds = []
    coeff_muls = [x.as_coeff_Mul() for x in expr.args]
    for (x, y) in coeff_muls:
        if not x.is_rational:
            return p
        if y is S.One:
            continue
        if not (y.is_Pow and y.exp == S.Half and y.base.is_integer):
            return p
        surds.append(y)
    surds.sort()
    surds = surds[:max_len]
    if expr < 0 and n % 2 == 1:
        p = (-expr) ** Rational(1, n)
        a = nsimplify(p, constants=surds)
        res = a if _mexpand(a ** n) == _mexpand(-expr) else p
        return -res
    a = nsimplify(p, constants=surds)
    if _mexpand(a) is not _mexpand(p) and _mexpand(a ** n) == _mexpand(expr):
        return _mexpand(a)
    expr = _nthroot_solve(expr, n, prec)
    if expr is None:
        return p
    return expr

def nsimplify(expr, constants=(), tolerance=None, full=False, rational=None, rational_conversion='base10'):
    if False:
        return 10
    "\n    Find a simple representation for a number or, if there are free symbols or\n    if ``rational=True``, then replace Floats with their Rational equivalents. If\n    no change is made and rational is not False then Floats will at least be\n    converted to Rationals.\n\n    Explanation\n    ===========\n\n    For numerical expressions, a simple formula that numerically matches the\n    given numerical expression is sought (and the input should be possible\n    to evalf to a precision of at least 30 digits).\n\n    Optionally, a list of (rationally independent) constants to\n    include in the formula may be given.\n\n    A lower tolerance may be set to find less exact matches. If no tolerance\n    is given then the least precise value will set the tolerance (e.g. Floats\n    default to 15 digits of precision, so would be tolerance=10**-15).\n\n    With ``full=True``, a more extensive search is performed\n    (this is useful to find simpler numbers when the tolerance\n    is set low).\n\n    When converting to rational, if rational_conversion='base10' (the default), then\n    convert floats to rationals using their base-10 (string) representation.\n    When rational_conversion='exact' it uses the exact, base-2 representation.\n\n    Examples\n    ========\n\n    >>> from sympy import nsimplify, sqrt, GoldenRatio, exp, I, pi\n    >>> nsimplify(4/(1+sqrt(5)), [GoldenRatio])\n    -2 + 2*GoldenRatio\n    >>> nsimplify((1/(exp(3*pi*I/5)+1)))\n    1/2 - I*sqrt(sqrt(5)/10 + 1/4)\n    >>> nsimplify(I**I, [pi])\n    exp(-pi/2)\n    >>> nsimplify(pi, tolerance=0.01)\n    22/7\n\n    >>> nsimplify(0.333333333333333, rational=True, rational_conversion='exact')\n    6004799503160655/18014398509481984\n    >>> nsimplify(0.333333333333333, rational=True)\n    1/3\n\n    See Also\n    ========\n\n    sympy.core.function.nfloat\n\n    "
    try:
        return sympify(as_int(expr))
    except (TypeError, ValueError):
        pass
    expr = sympify(expr).xreplace({Float('inf'): S.Infinity, Float('-inf'): S.NegativeInfinity})
    if expr is S.Infinity or expr is S.NegativeInfinity:
        return expr
    if rational or expr.free_symbols:
        return _real_to_rational(expr, tolerance, rational_conversion)
    if tolerance is None:
        tolerance = 10 ** (-min([15] + [mpmath.libmp.libmpf.prec_to_dps(n._prec) for n in expr.atoms(Float)]))
    prec = 30
    bprec = int(prec * 3.33)
    constants_dict = {}
    for constant in constants:
        constant = sympify(constant)
        v = constant.evalf(prec)
        if not v.is_Float:
            raise ValueError('constants must be real-valued')
        constants_dict[str(constant)] = v._to_mpmath(bprec)
    exprval = expr.evalf(prec, chop=True)
    (re, im) = exprval.as_real_imag()
    if not (re.is_Number and im.is_Number):
        return expr

    def nsimplify_real(x):
        if False:
            for i in range(10):
                print('nop')
        orig = mpmath.mp.dps
        xv = x._to_mpmath(bprec)
        try:
            if not (tolerance or full):
                mpmath.mp.dps = 15
                rat = mpmath.pslq([xv, 1])
                if rat is not None:
                    return Rational(-int(rat[1]), int(rat[0]))
            mpmath.mp.dps = prec
            newexpr = mpmath.identify(xv, constants=constants_dict, tol=tolerance, full=full)
            if not newexpr:
                raise ValueError
            if full:
                newexpr = newexpr[0]
            expr = sympify(newexpr)
            if x and (not expr):
                raise ValueError
            if expr.is_finite is False and xv not in [mpmath.inf, mpmath.ninf]:
                raise ValueError
            return expr
        finally:
            mpmath.mp.dps = orig
    try:
        if re:
            re = nsimplify_real(re)
        if im:
            im = nsimplify_real(im)
    except ValueError:
        if rational is None:
            return _real_to_rational(expr, rational_conversion=rational_conversion)
        return expr
    rv = re + im * S.ImaginaryUnit
    if rv != expr or rational is False:
        return rv
    return _real_to_rational(expr, rational_conversion=rational_conversion)

def _real_to_rational(expr, tolerance=None, rational_conversion='base10'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Replace all reals in expr with rationals.\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.simplify import _real_to_rational\n    >>> from sympy.abc import x\n\n    >>> _real_to_rational(.76 + .1*x**.5)\n    sqrt(x)/10 + 19/25\n\n    If rational_conversion='base10', this uses the base-10 string. If\n    rational_conversion='exact', the exact, base-2 representation is used.\n\n    >>> _real_to_rational(0.333333333333333, rational_conversion='exact')\n    6004799503160655/18014398509481984\n    >>> _real_to_rational(0.333333333333333)\n    1/3\n\n    "
    expr = _sympify(expr)
    inf = Float('inf')
    p = expr
    reps = {}
    reduce_num = None
    if tolerance is not None and tolerance < 1:
        reduce_num = ceiling(1 / tolerance)
    for fl in p.atoms(Float):
        key = fl
        if reduce_num is not None:
            r = Rational(fl).limit_denominator(reduce_num)
        elif tolerance is not None and tolerance >= 1 and (fl.is_Integer is False):
            r = Rational(tolerance * round(fl / tolerance)).limit_denominator(int(tolerance))
        else:
            if rational_conversion == 'exact':
                r = Rational(fl)
                reps[key] = r
                continue
            elif rational_conversion != 'base10':
                raise ValueError("rational_conversion must be 'base10' or 'exact'")
            r = nsimplify(fl, rational=False)
            if fl and (not r):
                r = Rational(fl)
            elif not r.is_Rational:
                if fl in (inf, -inf):
                    r = S.ComplexInfinity
                elif fl < 0:
                    fl = -fl
                    d = Pow(10, int(mpmath.log(fl) / mpmath.log(10)))
                    r = -Rational(str(fl / d)) * d
                elif fl > 0:
                    d = Pow(10, int(mpmath.log(fl) / mpmath.log(10)))
                    r = Rational(str(fl / d)) * d
                else:
                    r = S.Zero
        reps[key] = r
    return p.subs(reps, simultaneous=True)

def clear_coefficients(expr, rhs=S.Zero):
    if False:
        for i in range(10):
            print('nop')
    "Return `p, r` where `p` is the expression obtained when Rational\n    additive and multiplicative coefficients of `expr` have been stripped\n    away in a naive fashion (i.e. without simplification). The operations\n    needed to remove the coefficients will be applied to `rhs` and returned\n    as `r`.\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.simplify import clear_coefficients\n    >>> from sympy.abc import x, y\n    >>> from sympy import Dummy\n    >>> expr = 4*y*(6*x + 3)\n    >>> clear_coefficients(expr - 2)\n    (y*(2*x + 1), 1/6)\n\n    When solving 2 or more expressions like `expr = a`,\n    `expr = b`, etc..., it is advantageous to provide a Dummy symbol\n    for `rhs` and  simply replace it with `a`, `b`, etc... in `r`.\n\n    >>> rhs = Dummy('rhs')\n    >>> clear_coefficients(expr, rhs)\n    (y*(2*x + 1), _rhs/12)\n    >>> _[1].subs(rhs, 2)\n    1/6\n    "
    was = None
    free = expr.free_symbols
    if expr.is_Rational:
        return (S.Zero, rhs - expr)
    while expr and was != expr:
        was = expr
        (m, expr) = expr.as_content_primitive() if free else factor_terms(expr).as_coeff_Mul(rational=True)
        rhs /= m
        (c, expr) = expr.as_coeff_Add(rational=True)
        rhs -= c
    expr = signsimp(expr, evaluate=False)
    if expr.could_extract_minus_sign():
        expr = -expr
        rhs = -rhs
    return (expr, rhs)

def nc_simplify(expr, deep=True):
    if False:
        i = 10
        return i + 15
    '\n    Simplify a non-commutative expression composed of multiplication\n    and raising to a power by grouping repeated subterms into one power.\n    Priority is given to simplifications that give the fewest number\n    of arguments in the end (for example, in a*b*a*b*c*a*b*c simplifying\n    to (a*b)**2*c*a*b*c gives 5 arguments while a*b*(a*b*c)**2 has 3).\n    If ``expr`` is a sum of such terms, the sum of the simplified terms\n    is returned.\n\n    Keyword argument ``deep`` controls whether or not subexpressions\n    nested deeper inside the main expression are simplified. See examples\n    below. Setting `deep` to `False` can save time on nested expressions\n    that do not need simplifying on all levels.\n\n    Examples\n    ========\n\n    >>> from sympy import symbols\n    >>> from sympy.simplify.simplify import nc_simplify\n    >>> a, b, c = symbols("a b c", commutative=False)\n    >>> nc_simplify(a*b*a*b*c*a*b*c)\n    a*b*(a*b*c)**2\n    >>> expr = a**2*b*a**4*b*a**4\n    >>> nc_simplify(expr)\n    a**2*(b*a**4)**2\n    >>> nc_simplify(a*b*a*b*c**2*(a*b)**2*c**2)\n    ((a*b)**2*c**2)**2\n    >>> nc_simplify(a*b*a*b + 2*a*c*a**2*c*a**2*c*a)\n    (a*b)**2 + 2*(a*c*a)**3\n    >>> nc_simplify(b**-1*a**-1*(a*b)**2)\n    a*b\n    >>> nc_simplify(a**-1*b**-1*c*a)\n    (b*a)**(-1)*c*a\n    >>> expr = (a*b*a*b)**2*a*c*a*c\n    >>> nc_simplify(expr)\n    (a*b)**4*(a*c)**2\n    >>> nc_simplify(expr, deep=False)\n    (a*b*a*b)**2*(a*c)**2\n\n    '
    if isinstance(expr, MatrixExpr):
        expr = expr.doit(inv_expand=False)
        (_Add, _Mul, _Pow, _Symbol) = (MatAdd, MatMul, MatPow, MatrixSymbol)
    else:
        (_Add, _Mul, _Pow, _Symbol) = (Add, Mul, Pow, Symbol)

    def _overlaps(args):
        if False:
            print('Hello World!')
        m = [[[1, 0] if a == args[0] else [0] for a in args[1:]]]
        for i in range(1, len(args)):
            overlaps = []
            j = 0
            for j in range(len(args) - i - 1):
                overlap = []
                for v in m[i - 1][j + 1]:
                    if j + i + 1 + v < len(args) and args[i] == args[j + i + 1 + v]:
                        overlap.append(v + 1)
                overlap += [0]
                overlaps.append(overlap)
            m.append(overlaps)
        return m

    def _reduce_inverses(_args):
        if False:
            for i in range(10):
                print('nop')
        inv_tot = 0
        inverses = []
        args = []
        for arg in _args:
            if isinstance(arg, _Pow) and arg.args[1].is_extended_negative:
                inverses = [arg ** (-1)] + inverses
                inv_tot += 1
            else:
                if len(inverses) == 1:
                    args.append(inverses[0] ** (-1))
                elif len(inverses) > 1:
                    args.append(_Pow(_Mul(*inverses), -1))
                    inv_tot -= len(inverses) - 1
                inverses = []
                args.append(arg)
        if inverses:
            args.append(_Pow(_Mul(*inverses), -1))
            inv_tot -= len(inverses) - 1
        return (inv_tot, tuple(args))

    def get_score(s):
        if False:
            print('Hello World!')
        if isinstance(s, _Pow):
            return get_score(s.args[0])
        elif isinstance(s, (_Add, _Mul)):
            return sum([get_score(a) for a in s.args])
        return 1

    def compare(s, alt_s):
        if False:
            return 10
        if s != alt_s and get_score(alt_s) < get_score(s):
            return alt_s
        return s
    if not isinstance(expr, (_Add, _Mul, _Pow)) or expr.is_commutative:
        return expr
    args = expr.args[:]
    if isinstance(expr, _Pow):
        if deep:
            return _Pow(nc_simplify(args[0]), args[1]).doit()
        else:
            return expr
    elif isinstance(expr, _Add):
        return _Add(*[nc_simplify(a, deep=deep) for a in args]).doit()
    else:
        (c_args, args) = expr.args_cnc()
        com_coeff = Mul(*c_args)
        if com_coeff != 1:
            return com_coeff * nc_simplify(expr / com_coeff, deep=deep)
    (inv_tot, args) = _reduce_inverses(args)
    invert = False
    if inv_tot > len(args) / 2:
        invert = True
        args = [a ** (-1) for a in args[::-1]]
    if deep:
        args = tuple((nc_simplify(a) for a in args))
    m = _overlaps(args)
    simps = {}
    post = 1
    pre = 1
    max_simp_coeff = 0
    simp = None
    for i in range(1, len(args)):
        simp_coeff = 0
        l = 0
        p = 0
        if i < len(args) - 1:
            rep = m[i][0]
        start = i
        end = i + 1
        if i == len(args) - 1 or rep == [0]:
            if isinstance(args[i], _Pow) and (not isinstance(args[i].args[0], _Symbol)):
                subterm = args[i].args[0].args
                l = len(subterm)
                if args[i - l:i] == subterm:
                    p += 1
                    start -= l
                if args[i + 1:i + 1 + l] == subterm:
                    p += 1
                    end += l
            if p:
                p += args[i].args[1]
            else:
                continue
        else:
            l = rep[0]
            start -= l - 1
            subterm = args[start:end]
            p = 2
            end += l
        if subterm in simps and simps[subterm] >= start:
            continue
        while end < len(args):
            if l in m[end - 1][0]:
                p += 1
                end += l
            elif isinstance(args[end], _Pow) and args[end].args[0].args == subterm:
                p += args[end].args[1]
                end += 1
            else:
                break
        pre_exp = 0
        pre_arg = 1
        if start - l >= 0 and args[start - l + 1:start] == subterm[1:]:
            if isinstance(subterm[0], _Pow):
                pre_arg = subterm[0].args[0]
                exp = subterm[0].args[1]
            else:
                pre_arg = subterm[0]
                exp = 1
            if isinstance(args[start - l], _Pow) and args[start - l].args[0] == pre_arg:
                pre_exp = args[start - l].args[1] - exp
                start -= l
                p += 1
            elif args[start - l] == pre_arg:
                pre_exp = 1 - exp
                start -= l
                p += 1
        post_exp = 0
        post_arg = 1
        if end + l - 1 < len(args) and args[end:end + l - 1] == subterm[:-1]:
            if isinstance(subterm[-1], _Pow):
                post_arg = subterm[-1].args[0]
                exp = subterm[-1].args[1]
            else:
                post_arg = subterm[-1]
                exp = 1
            if isinstance(args[end + l - 1], _Pow) and args[end + l - 1].args[0] == post_arg:
                post_exp = args[end + l - 1].args[1] - exp
                end += l
                p += 1
            elif args[end + l - 1] == post_arg:
                post_exp = 1 - exp
                end += l
                p += 1
        if post_exp and exp % 2 == 0 and (start > 0):
            exp = exp / 2
            _pre_exp = 1
            _post_exp = 1
            if isinstance(args[start - 1], _Pow) and args[start - 1].args[0] == post_arg:
                _post_exp = post_exp + exp
                _pre_exp = args[start - 1].args[1] - exp
            elif args[start - 1] == post_arg:
                _post_exp = post_exp + exp
                _pre_exp = 1 - exp
            if _pre_exp == 0 or _post_exp == 0:
                if not pre_exp:
                    start -= 1
                post_exp = _post_exp
                pre_exp = _pre_exp
                pre_arg = post_arg
                subterm = (post_arg ** exp,) + subterm[:-1] + (post_arg ** exp,)
        simp_coeff += end - start
        if post_exp:
            simp_coeff -= 1
        if pre_exp:
            simp_coeff -= 1
        simps[subterm] = end
        if simp_coeff > max_simp_coeff:
            max_simp_coeff = simp_coeff
            simp = (start, _Mul(*subterm), p, end, l)
            pre = pre_arg ** pre_exp
            post = post_arg ** post_exp
    if simp:
        subterm = _Pow(nc_simplify(simp[1], deep=deep), simp[2])
        pre = nc_simplify(_Mul(*args[:simp[0]]) * pre, deep=deep)
        post = post * nc_simplify(_Mul(*args[simp[3]:]), deep=deep)
        simp = pre * subterm * post
        if pre != 1 or post != 1:
            simp = nc_simplify(simp, deep=False)
    else:
        simp = _Mul(*args)
    if invert:
        simp = _Pow(simp, -1)
    if not isinstance(expr, MatrixExpr):
        f_expr = factor_nc(expr)
        if f_expr != expr:
            alt_simp = nc_simplify(f_expr, deep=deep)
            simp = compare(simp, alt_simp)
    else:
        simp = simp.doit(inv_expand=False)
    return simp

def dotprodsimp(expr, withsimp=False):
    if False:
        i = 10
        return i + 15
    'Simplification for a sum of products targeted at the kind of blowup that\n    occurs during summation of products. Intended to reduce expression blowup\n    during matrix multiplication or other similar operations. Only works with\n    algebraic expressions and does not recurse into non.\n\n    Parameters\n    ==========\n\n    withsimp : bool, optional\n        Specifies whether a flag should be returned along with the expression\n        to indicate roughly whether simplification was successful. It is used\n        in ``MatrixArithmetic._eval_pow_by_recursion`` to avoid attempting to\n        simplify an expression repetitively which does not simplify.\n    '

    def count_ops_alg(expr):
        if False:
            for i in range(10):
                print('nop')
        'Optimized count algebraic operations with no recursion into\n        non-algebraic args that ``core.function.count_ops`` does. Also returns\n        whether rational functions may be present according to negative\n        exponents of powers or non-number fractions.\n\n        Returns\n        =======\n\n        ops, ratfunc : int, bool\n            ``ops`` is the number of algebraic operations starting at the top\n            level expression (not recursing into non-alg children). ``ratfunc``\n            specifies whether the expression MAY contain rational functions\n            which ``cancel`` MIGHT optimize.\n        '
        ops = 0
        args = [expr]
        ratfunc = False
        while args:
            a = args.pop()
            if not isinstance(a, Basic):
                continue
            if a.is_Rational:
                if a is not S.One:
                    ops += bool(a.p < 0) + bool(a.q != 1)
            elif a.is_Mul:
                if a.could_extract_minus_sign():
                    ops += 1
                    if a.args[0] is S.NegativeOne:
                        a = a.as_two_terms()[1]
                    else:
                        a = -a
                (n, d) = fraction(a)
                if n.is_Integer:
                    ops += 1 + bool(n < 0)
                    args.append(d)
                elif d is not S.One:
                    if not d.is_Integer:
                        args.append(d)
                        ratfunc = True
                    ops += 1
                    args.append(n)
                else:
                    ops += len(a.args) - 1
                    args.extend(a.args)
            elif a.is_Add:
                laargs = len(a.args)
                negs = 0
                for ai in a.args:
                    if ai.could_extract_minus_sign():
                        negs += 1
                        ai = -ai
                    args.append(ai)
                ops += laargs - (negs != laargs)
            elif a.is_Pow:
                ops += 1
                args.append(a.base)
                if not ratfunc:
                    ratfunc = a.exp.is_negative is not False
        return (ops, ratfunc)

    def nonalg_subs_dummies(expr, dummies):
        if False:
            while True:
                i = 10
        'Substitute dummy variables for non-algebraic expressions to avoid\n        evaluation of non-algebraic terms that ``polys.polytools.cancel`` does.\n        '
        if not expr.args:
            return expr
        if expr.is_Add or expr.is_Mul or expr.is_Pow:
            args = None
            for (i, a) in enumerate(expr.args):
                c = nonalg_subs_dummies(a, dummies)
                if c is a:
                    continue
                if args is None:
                    args = list(expr.args)
                args[i] = c
            if args is None:
                return expr
            return expr.func(*args)
        return dummies.setdefault(expr, Dummy())
    simplified = False
    if isinstance(expr, Basic) and (expr.is_Add or expr.is_Mul or expr.is_Pow):
        expr2 = expr.expand(deep=True, modulus=None, power_base=False, power_exp=False, mul=True, log=False, multinomial=True, basic=False)
        if expr2 != expr:
            expr = expr2
            simplified = True
        (exprops, ratfunc) = count_ops_alg(expr)
        if exprops >= 6:
            if ratfunc:
                dummies = {}
                expr2 = nonalg_subs_dummies(expr, dummies)
                if expr2 is expr or count_ops_alg(expr2)[0] >= 6:
                    expr3 = cancel(expr2)
                    if expr3 != expr2:
                        expr = expr3.subs([(d, e) for (e, d) in dummies.items()])
                        simplified = True
        elif exprops == 5 and expr.is_Add and expr.args[0].is_Mul and expr.args[1].is_Mul and expr.args[0].args[-1].is_Pow and expr.args[1].args[-1].is_Pow and (expr.args[0].args[-1].exp is S.NegativeOne) and (expr.args[1].args[-1].exp is S.NegativeOne):
            expr2 = together(expr)
            expr2ops = count_ops_alg(expr2)[0]
            if expr2ops < exprops:
                expr = expr2
                simplified = True
        else:
            simplified = True
    return (expr, simplified) if withsimp else expr
bottom_up = deprecated('\n    Using bottom_up from the sympy.simplify.simplify submodule is\n    deprecated.\n\n    Instead, use bottom_up from the top-level sympy namespace, like\n\n        sympy.bottom_up\n    ', deprecated_since_version='1.10', active_deprecations_target='deprecated-traversal-functions-moved')(_bottom_up)
walk = deprecated('\n    Using walk from the sympy.simplify.simplify submodule is\n    deprecated.\n\n    Instead, use walk from sympy.core.traversal.walk\n    ', deprecated_since_version='1.10', active_deprecations_target='deprecated-traversal-functions-moved')(_walk)