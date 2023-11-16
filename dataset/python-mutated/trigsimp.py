from collections import defaultdict
from functools import reduce
from sympy.core import sympify, Basic, S, Expr, factor_terms, Mul, Add, bottom_up
from sympy.core.cache import cacheit
from sympy.core.function import count_ops, _mexpand, FunctionClass, expand, expand_mul, _coeff_isneg, Derivative
from sympy.core.numbers import I, Integer
from sympy.core.intfunc import igcd
from sympy.core.sorting import _nodes
from sympy.core.symbol import Dummy, symbols, Wild
from sympy.external.gmpy import SYMPY_INTS
from sympy.functions import sin, cos, exp, cosh, tanh, sinh, tan, cot, coth
from sympy.functions import atan2
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.polys import Poly, factor, cancel, parallel_poly_from_expr
from sympy.polys.domains import ZZ
from sympy.polys.polyerrors import PolificationFailed
from sympy.polys.polytools import groebner
from sympy.simplify.cse_main import cse
from sympy.strategies.core import identity
from sympy.strategies.tree import greedy
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import debug

def trigsimp_groebner(expr, hints=[], quick=False, order='grlex', polynomial=False):
    if False:
        while True:
            i = 10
    '\n    Simplify trigonometric expressions using a groebner basis algorithm.\n\n    Explanation\n    ===========\n\n    This routine takes a fraction involving trigonometric or hyperbolic\n    expressions, and tries to simplify it. The primary metric is the\n    total degree. Some attempts are made to choose the simplest possible\n    expression of the minimal degree, but this is non-rigorous, and also\n    very slow (see the ``quick=True`` option).\n\n    If ``polynomial`` is set to True, instead of simplifying numerator and\n    denominator together, this function just brings numerator and denominator\n    into a canonical form. This is much faster, but has potentially worse\n    results. However, if the input is a polynomial, then the result is\n    guaranteed to be an equivalent polynomial of minimal degree.\n\n    The most important option is hints. Its entries can be any of the\n    following:\n\n    - a natural number\n    - a function\n    - an iterable of the form (func, var1, var2, ...)\n    - anything else, interpreted as a generator\n\n    A number is used to indicate that the search space should be increased.\n    A function is used to indicate that said function is likely to occur in a\n    simplified expression.\n    An iterable is used indicate that func(var1 + var2 + ...) is likely to\n    occur in a simplified .\n    An additional generator also indicates that it is likely to occur.\n    (See examples below).\n\n    This routine carries out various computationally intensive algorithms.\n    The option ``quick=True`` can be used to suppress one particularly slow\n    step (at the expense of potentially more complicated results, but never at\n    the expense of increased total degree).\n\n    Examples\n    ========\n\n    >>> from sympy.abc import x, y\n    >>> from sympy import sin, tan, cos, sinh, cosh, tanh\n    >>> from sympy.simplify.trigsimp import trigsimp_groebner\n\n    Suppose you want to simplify ``sin(x)*cos(x)``. Naively, nothing happens:\n\n    >>> ex = sin(x)*cos(x)\n    >>> trigsimp_groebner(ex)\n    sin(x)*cos(x)\n\n    This is because ``trigsimp_groebner`` only looks for a simplification\n    involving just ``sin(x)`` and ``cos(x)``. You can tell it to also try\n    ``2*x`` by passing ``hints=[2]``:\n\n    >>> trigsimp_groebner(ex, hints=[2])\n    sin(2*x)/2\n    >>> trigsimp_groebner(sin(x)**2 - cos(x)**2, hints=[2])\n    -cos(2*x)\n\n    Increasing the search space this way can quickly become expensive. A much\n    faster way is to give a specific expression that is likely to occur:\n\n    >>> trigsimp_groebner(ex, hints=[sin(2*x)])\n    sin(2*x)/2\n\n    Hyperbolic expressions are similarly supported:\n\n    >>> trigsimp_groebner(sinh(2*x)/sinh(x))\n    2*cosh(x)\n\n    Note how no hints had to be passed, since the expression already involved\n    ``2*x``.\n\n    The tangent function is also supported. You can either pass ``tan`` in the\n    hints, to indicate that tan should be tried whenever cosine or sine are,\n    or you can pass a specific generator:\n\n    >>> trigsimp_groebner(sin(x)/cos(x), hints=[tan])\n    tan(x)\n    >>> trigsimp_groebner(sinh(x)/cosh(x), hints=[tanh(x)])\n    tanh(x)\n\n    Finally, you can use the iterable form to suggest that angle sum formulae\n    should be tried:\n\n    >>> ex = (tan(x) + tan(y))/(1 - tan(x)*tan(y))\n    >>> trigsimp_groebner(ex, hints=[(tan, x, y)])\n    tan(x + y)\n    '

    def parse_hints(hints):
        if False:
            print('Hello World!')
        'Split hints into (n, funcs, iterables, gens).'
        n = 1
        (funcs, iterables, gens) = ([], [], [])
        for e in hints:
            if isinstance(e, (SYMPY_INTS, Integer)):
                n = e
            elif isinstance(e, FunctionClass):
                funcs.append(e)
            elif iterable(e):
                iterables.append((e[0], e[1:]))
                gens.extend(parallel_poly_from_expr([e[0](x) for x in e[1:]] + [e[0](Add(*e[1:]))])[1].gens)
            else:
                gens.append(e)
        return (n, funcs, iterables, gens)

    def build_ideal(x, terms):
        if False:
            for i in range(10):
                print('nop')
        '\n        Build generators for our ideal. ``Terms`` is an iterable with elements of\n        the form (fn, coeff), indicating that we have a generator fn(coeff*x).\n\n        If any of the terms is trigonometric, sin(x) and cos(x) are guaranteed\n        to appear in terms. Similarly for hyperbolic functions. For tan(n*x),\n        sin(n*x) and cos(n*x) are guaranteed.\n        '
        I = []
        y = Dummy('y')
        for (fn, coeff) in terms:
            for (c, s, t, rel) in ([cos, sin, tan, cos(x) ** 2 + sin(x) ** 2 - 1], [cosh, sinh, tanh, cosh(x) ** 2 - sinh(x) ** 2 - 1]):
                if coeff == 1 and fn in [c, s]:
                    I.append(rel)
                elif fn == t:
                    I.append(t(coeff * x) * c(coeff * x) - s(coeff * x))
                elif fn in [c, s]:
                    cn = fn(coeff * y).expand(trig=True).subs(y, x)
                    I.append(fn(coeff * x) - cn)
        return list(set(I))

    def analyse_gens(gens, hints):
        if False:
            print('Hello World!')
        '\n        Analyse the generators ``gens``, using the hints ``hints``.\n\n        The meaning of ``hints`` is described in the main docstring.\n        Return a new list of generators, and also the ideal we should\n        work with.\n        '
        (n, funcs, iterables, extragens) = parse_hints(hints)
        debug('n=%s   funcs: %s   iterables: %s    extragens: %s', (funcs, iterables, extragens))
        gens = list(gens)
        gens.extend(extragens)
        funcs = list(set(funcs))
        iterables = list(set(iterables))
        gens = list(set(gens))
        allfuncs = {sin, cos, tan, sinh, cosh, tanh}
        trigterms = [(g.args[0].as_coeff_mul(), g.func) for g in gens if g.func in allfuncs]
        freegens = [g for g in gens if g.func not in allfuncs]
        newgens = []
        trigdict = {}
        for ((coeff, var), fn) in trigterms:
            trigdict.setdefault(var, []).append((coeff, fn))
        res = []
        for (key, val) in trigdict.items():
            fns = [x[1] for x in val]
            val = [x[0] for x in val]
            gcd = reduce(igcd, val)
            terms = [(fn, v / gcd) for (fn, v) in zip(fns, val)]
            fs = set(funcs + fns)
            for (c, s, t) in ([cos, sin, tan], [cosh, sinh, tanh]):
                if any((x in fs for x in (c, s, t))):
                    fs.add(c)
                    fs.add(s)
            for fn in fs:
                for k in range(1, n + 1):
                    terms.append((fn, k))
            extra = []
            for (fn, v) in terms:
                if fn == tan:
                    extra.append((sin, v))
                    extra.append((cos, v))
                if fn in [sin, cos] and tan in fs:
                    extra.append((tan, v))
                if fn == tanh:
                    extra.append((sinh, v))
                    extra.append((cosh, v))
                if fn in [sinh, cosh] and tanh in fs:
                    extra.append((tanh, v))
            terms.extend(extra)
            x = gcd * Mul(*key)
            r = build_ideal(x, terms)
            res.extend(r)
            newgens.extend({fn(v * x) for (fn, v) in terms})
        for (fn, args) in iterables:
            if fn == tan:
                iterables.extend([(sin, args), (cos, args)])
            elif fn == tanh:
                iterables.extend([(sinh, args), (cosh, args)])
            else:
                dummys = symbols('d:%i' % len(args), cls=Dummy)
                expr = fn(Add(*dummys)).expand(trig=True).subs(list(zip(dummys, args)))
                res.append(fn(Add(*args)) - expr)
        if myI in gens:
            res.append(myI ** 2 + 1)
            freegens.remove(myI)
            newgens.append(myI)
        return (res, freegens, newgens)
    myI = Dummy('I')
    expr = expr.subs(S.ImaginaryUnit, myI)
    subs = [(myI, S.ImaginaryUnit)]
    (num, denom) = cancel(expr).as_numer_denom()
    try:
        ((pnum, pdenom), opt) = parallel_poly_from_expr([num, denom])
    except PolificationFailed:
        return expr
    debug('initial gens:', opt.gens)
    (ideal, freegens, gens) = analyse_gens(opt.gens, hints)
    debug('ideal:', ideal)
    debug('new gens:', gens, ' -- len', len(gens))
    debug('free gens:', freegens, ' -- len', len(gens))
    if not gens:
        return expr
    G = groebner(ideal, order=order, gens=gens, domain=ZZ)
    debug('groebner basis:', list(G), ' -- len', len(G))
    from sympy.simplify.ratsimp import ratsimpmodprime
    if freegens and pdenom.has_only_gens(*set(gens).intersection(pdenom.gens)):
        num = Poly(num, gens=gens + freegens).eject(*gens)
        res = []
        for (monom, coeff) in num.terms():
            ourgens = set(parallel_poly_from_expr([coeff, denom])[1].gens)
            changed = True
            while changed:
                changed = False
                for p in ideal:
                    p = Poly(p)
                    if not ourgens.issuperset(p.gens) and (not p.has_only_gens(*set(p.gens).difference(ourgens))):
                        changed = True
                        ourgens.update(p.exclude().gens)
            realgens = [x for x in gens if x in ourgens]
            ourG = [g.as_expr() for g in G.polys if g.has_only_gens(*ourgens.intersection(g.gens))]
            res.append(Mul(*[a ** b for (a, b) in zip(freegens, monom)]) * ratsimpmodprime(coeff / denom, ourG, order=order, gens=realgens, quick=quick, domain=ZZ, polynomial=polynomial).subs(subs))
        return Add(*res)
        return Add(*[Mul(*[a ** b for (a, b) in zip(freegens, monom)]) * ratsimpmodprime(coeff / denom, list(G), order=order, gens=gens, quick=quick, domain=ZZ) for (monom, coeff) in num.terms()])
    else:
        return ratsimpmodprime(expr, list(G), order=order, gens=freegens + gens, quick=quick, domain=ZZ, polynomial=polynomial).subs(subs)
_trigs = (TrigonometricFunction, HyperbolicFunction)

def _trigsimp_inverse(rv):
    if False:
        i = 10
        return i + 15

    def check_args(x, y):
        if False:
            for i in range(10):
                print('nop')
        try:
            return x.args[0] == y.args[0]
        except IndexError:
            return False

    def f(rv):
        if False:
            print('Hello World!')
        g = getattr(rv, 'inverse', None)
        if g is not None and isinstance(rv.args[0], g()) and isinstance(g()(1), TrigonometricFunction):
            return rv.args[0].args[0]
        if isinstance(rv, atan2):
            (y, x) = rv.args
            if _coeff_isneg(y):
                return -f(atan2(-y, x))
            elif _coeff_isneg(x):
                return S.Pi - f(atan2(y, -x))
            if check_args(x, y):
                if isinstance(y, sin) and isinstance(x, cos):
                    return x.args[0]
                if isinstance(y, cos) and isinstance(x, sin):
                    return S.Pi / 2 - x.args[0]
        return rv
    return bottom_up(rv, f)

def trigsimp(expr, inverse=False, **opts):
    if False:
        i = 10
        return i + 15
    "Returns a reduced expression by using known trig identities.\n\n    Parameters\n    ==========\n\n    inverse : bool, optional\n        If ``inverse=True``, it will be assumed that a composition of inverse\n        functions, such as sin and asin, can be cancelled in any order.\n        For example, ``asin(sin(x))`` will yield ``x`` without checking whether\n        x belongs to the set where this relation is true. The default is False.\n        Default : True\n\n    method : string, optional\n        Specifies the method to use. Valid choices are:\n\n        - ``'matching'``, default\n        - ``'groebner'``\n        - ``'combined'``\n        - ``'fu'``\n        - ``'old'``\n\n        If ``'matching'``, simplify the expression recursively by targeting\n        common patterns. If ``'groebner'``, apply an experimental groebner\n        basis algorithm. In this case further options are forwarded to\n        ``trigsimp_groebner``, please refer to\n        its docstring. If ``'combined'``, it first runs the groebner basis\n        algorithm with small default parameters, then runs the ``'matching'``\n        algorithm. If ``'fu'``, run the collection of trigonometric\n        transformations described by Fu, et al. (see the\n        :py:func:`~sympy.simplify.fu.fu` docstring). If ``'old'``, the original\n        SymPy trig simplification function is run.\n    opts :\n        Optional keyword arguments passed to the method. See each method's\n        function docstring for details.\n\n    Examples\n    ========\n\n    >>> from sympy import trigsimp, sin, cos, log\n    >>> from sympy.abc import x\n    >>> e = 2*sin(x)**2 + 2*cos(x)**2\n    >>> trigsimp(e)\n    2\n\n    Simplification occurs wherever trigonometric functions are located.\n\n    >>> trigsimp(log(e))\n    log(2)\n\n    Using ``method='groebner'`` (or ``method='combined'``) might lead to\n    greater simplification.\n\n    The old trigsimp routine can be accessed as with method ``method='old'``.\n\n    >>> from sympy import coth, tanh\n    >>> t = 3*tanh(x)**7 - 2/coth(x)**7\n    >>> trigsimp(t, method='old') == t\n    True\n    >>> trigsimp(t)\n    tanh(x)**7\n\n    "
    from sympy.simplify.fu import fu
    expr = sympify(expr)
    _eval_trigsimp = getattr(expr, '_eval_trigsimp', None)
    if _eval_trigsimp is not None:
        return _eval_trigsimp(**opts)
    old = opts.pop('old', False)
    if not old:
        opts.pop('deep', None)
        opts.pop('recursive', None)
        method = opts.pop('method', 'matching')
    else:
        method = 'old'

    def groebnersimp(ex, **opts):
        if False:
            while True:
                i = 10

        def traverse(e):
            if False:
                return 10
            if e.is_Atom:
                return e
            args = [traverse(x) for x in e.args]
            if e.is_Function or e.is_Pow:
                args = [trigsimp_groebner(x, **opts) for x in args]
            return e.func(*args)
        new = traverse(ex)
        if not isinstance(new, Expr):
            return new
        return trigsimp_groebner(new, **opts)
    trigsimpfunc = {'fu': lambda x: fu(x, **opts), 'matching': lambda x: futrig(x), 'groebner': lambda x: groebnersimp(x, **opts), 'combined': lambda x: futrig(groebnersimp(x, polynomial=True, hints=[2, tan])), 'old': lambda x: trigsimp_old(x, **opts)}[method]
    expr_simplified = trigsimpfunc(expr)
    if inverse:
        expr_simplified = _trigsimp_inverse(expr_simplified)
    return expr_simplified

def exptrigsimp(expr):
    if False:
        i = 10
        return i + 15
    '\n    Simplifies exponential / trigonometric / hyperbolic functions.\n\n    Examples\n    ========\n\n    >>> from sympy import exptrigsimp, exp, cosh, sinh\n    >>> from sympy.abc import z\n\n    >>> exptrigsimp(exp(z) + exp(-z))\n    2*cosh(z)\n    >>> exptrigsimp(cosh(z) - sinh(z))\n    exp(-z)\n    '
    from sympy.simplify.fu import hyper_as_trig, TR2i

    def exp_trig(e):
        if False:
            return 10
        choices = [e]
        if e.has(*_trigs):
            choices.append(e.rewrite(exp))
        choices.append(e.rewrite(cos))
        return min(*choices, key=count_ops)
    newexpr = bottom_up(expr, exp_trig)

    def f(rv):
        if False:
            i = 10
            return i + 15
        if not rv.is_Mul:
            return rv
        (commutative_part, noncommutative_part) = rv.args_cnc()
        if len(noncommutative_part) > 1:
            return f(Mul(*commutative_part)) * Mul(*noncommutative_part)
        rvd = rv.as_powers_dict()
        newd = rvd.copy()

        def signlog(expr, sign=S.One):
            if False:
                for i in range(10):
                    print('nop')
            if expr is S.Exp1:
                return (sign, S.One)
            elif isinstance(expr, exp) or (expr.is_Pow and expr.base == S.Exp1):
                return (sign, expr.exp)
            elif sign is S.One:
                return signlog(-expr, sign=-S.One)
            else:
                return (None, None)
        ee = rvd[S.Exp1]
        for k in rvd:
            if k.is_Add and len(k.args) == 2:
                c = k.args[0]
                (sign, x) = signlog(k.args[1] / c)
                if not x:
                    continue
                m = rvd[k]
                newd[k] -= m
                if ee == -x * m / 2:
                    newd[S.Exp1] -= ee
                    ee = 0
                    if sign == 1:
                        newd[2 * c * cosh(x / 2)] += m
                    else:
                        newd[-2 * c * sinh(x / 2)] += m
                elif newd[1 - sign * S.Exp1 ** x] == -m:
                    del newd[1 - sign * S.Exp1 ** x]
                    if sign == 1:
                        newd[-c / tanh(x / 2)] += m
                    else:
                        newd[-c * tanh(x / 2)] += m
                else:
                    newd[1 + sign * S.Exp1 ** x] += m
                    newd[c] += m
        return Mul(*[k ** newd[k] for k in newd])
    newexpr = bottom_up(newexpr, f)
    if newexpr.has(HyperbolicFunction):
        (e, f) = hyper_as_trig(newexpr)
        newexpr = f(TR2i(e))
    if newexpr.has(TrigonometricFunction):
        newexpr = TR2i(newexpr)
    if not (newexpr.has(I) and (not expr.has(I))):
        expr = newexpr
    return expr

def trigsimp_old(expr, *, first=True, **opts):
    if False:
        for i in range(10):
            print('nop')
    '\n    Reduces expression by using known trig identities.\n\n    Notes\n    =====\n\n    deep:\n    - Apply trigsimp inside all objects with arguments\n\n    recursive:\n    - Use common subexpression elimination (cse()) and apply\n    trigsimp recursively (this is quite expensive if the\n    expression is large)\n\n    method:\n    - Determine the method to use. Valid choices are \'matching\' (default),\n    \'groebner\', \'combined\', \'fu\' and \'futrig\'. If \'matching\', simplify the\n    expression recursively by pattern matching. If \'groebner\', apply an\n    experimental groebner basis algorithm. In this case further options\n    are forwarded to ``trigsimp_groebner``, please refer to its docstring.\n    If \'combined\', first run the groebner basis algorithm with small\n    default parameters, then run the \'matching\' algorithm. \'fu\' runs the\n    collection of trigonometric transformations described by Fu, et al.\n    (see the `fu` docstring) while `futrig` runs a subset of Fu-transforms\n    that mimic the behavior of `trigsimp`.\n\n    compare:\n    - show input and output from `trigsimp` and `futrig` when different,\n    but returns the `trigsimp` value.\n\n    Examples\n    ========\n\n    >>> from sympy import trigsimp, sin, cos, log, cot\n    >>> from sympy.abc import x\n    >>> e = 2*sin(x)**2 + 2*cos(x)**2\n    >>> trigsimp(e, old=True)\n    2\n    >>> trigsimp(log(e), old=True)\n    log(2*sin(x)**2 + 2*cos(x)**2)\n    >>> trigsimp(log(e), deep=True, old=True)\n    log(2)\n\n    Using `method="groebner"` (or `"combined"`) can sometimes lead to a lot\n    more simplification:\n\n    >>> e = (-sin(x) + 1)/cos(x) + cos(x)/(-sin(x) + 1)\n    >>> trigsimp(e, old=True)\n    (1 - sin(x))/cos(x) + cos(x)/(1 - sin(x))\n    >>> trigsimp(e, method="groebner", old=True)\n    2/cos(x)\n\n    >>> trigsimp(1/cot(x)**2, compare=True, old=True)\n          futrig: tan(x)**2\n    cot(x)**(-2)\n\n    '
    old = expr
    if first:
        if not expr.has(*_trigs):
            return expr
        trigsyms = set().union(*[t.free_symbols for t in expr.atoms(*_trigs)])
        if len(trigsyms) > 1:
            from sympy.simplify.simplify import separatevars
            d = separatevars(expr)
            if d.is_Mul:
                d = separatevars(d, dict=True) or d
            if isinstance(d, dict):
                expr = 1
                for (k, v) in d.items():
                    was = v
                    v = expand_mul(v)
                    opts['first'] = False
                    vnew = trigsimp(v, **opts)
                    if vnew == v:
                        vnew = was
                    expr *= vnew
                old = expr
            elif d.is_Add:
                for s in trigsyms:
                    (r, e) = expr.as_independent(s)
                    if r:
                        opts['first'] = False
                        expr = r + trigsimp(e, **opts)
                        if not expr.is_Add:
                            break
                old = expr
    recursive = opts.pop('recursive', False)
    deep = opts.pop('deep', False)
    method = opts.pop('method', 'matching')

    def groebnersimp(ex, deep, **opts):
        if False:
            for i in range(10):
                print('nop')

        def traverse(e):
            if False:
                print('Hello World!')
            if e.is_Atom:
                return e
            args = [traverse(x) for x in e.args]
            if e.is_Function or e.is_Pow:
                args = [trigsimp_groebner(x, **opts) for x in args]
            return e.func(*args)
        if deep:
            ex = traverse(ex)
        return trigsimp_groebner(ex, **opts)
    trigsimpfunc = {'matching': lambda x, d: _trigsimp(x, d), 'groebner': lambda x, d: groebnersimp(x, d, **opts), 'combined': lambda x, d: _trigsimp(groebnersimp(x, d, polynomial=True, hints=[2, tan]), d)}[method]
    if recursive:
        (w, g) = cse(expr)
        g = trigsimpfunc(g[0], deep)
        for sub in reversed(w):
            g = g.subs(sub[0], sub[1])
            g = trigsimpfunc(g, deep)
        result = g
    else:
        result = trigsimpfunc(expr, deep)
    if opts.get('compare', False):
        f = futrig(old)
        if f != result:
            print('\tfutrig:', f)
    return result

def _dotrig(a, b):
    if False:
        print('Hello World!')
    'Helper to tell whether ``a`` and ``b`` have the same sorts\n    of symbols in them -- no need to test hyperbolic patterns against\n    expressions that have no hyperbolics in them.'
    return a.func == b.func and (a.has(TrigonometricFunction) and b.has(TrigonometricFunction) or (a.has(HyperbolicFunction) and b.has(HyperbolicFunction)))
_trigpat = None

def _trigpats():
    if False:
        while True:
            i = 10
    global _trigpat
    (a, b, c) = symbols('a b c', cls=Wild)
    d = Wild('d', commutative=False)
    matchers_division = ((a * sin(b) ** c / cos(b) ** c, a * tan(b) ** c, sin(b), cos(b)), (a * tan(b) ** c * cos(b) ** c, a * sin(b) ** c, sin(b), cos(b)), (a * cot(b) ** c * sin(b) ** c, a * cos(b) ** c, sin(b), cos(b)), (a * tan(b) ** c / sin(b) ** c, a / cos(b) ** c, sin(b), cos(b)), (a * cot(b) ** c / cos(b) ** c, a / sin(b) ** c, sin(b), cos(b)), (a * cot(b) ** c * tan(b) ** c, a, sin(b), cos(b)), (a * (cos(b) + 1) ** c * (cos(b) - 1) ** c, a * (-sin(b) ** 2) ** c, cos(b) + 1, cos(b) - 1), (a * (sin(b) + 1) ** c * (sin(b) - 1) ** c, a * (-cos(b) ** 2) ** c, sin(b) + 1, sin(b) - 1), (a * sinh(b) ** c / cosh(b) ** c, a * tanh(b) ** c, S.One, S.One), (a * tanh(b) ** c * cosh(b) ** c, a * sinh(b) ** c, S.One, S.One), (a * coth(b) ** c * sinh(b) ** c, a * cosh(b) ** c, S.One, S.One), (a * tanh(b) ** c / sinh(b) ** c, a / cosh(b) ** c, S.One, S.One), (a * coth(b) ** c / cosh(b) ** c, a / sinh(b) ** c, S.One, S.One), (a * coth(b) ** c * tanh(b) ** c, a, S.One, S.One), (c * (tanh(a) + tanh(b)) / (1 + tanh(a) * tanh(b)), tanh(a + b) * c, S.One, S.One))
    matchers_add = ((c * sin(a) * cos(b) + c * cos(a) * sin(b) + d, sin(a + b) * c + d), (c * cos(a) * cos(b) - c * sin(a) * sin(b) + d, cos(a + b) * c + d), (c * sin(a) * cos(b) - c * cos(a) * sin(b) + d, sin(a - b) * c + d), (c * cos(a) * cos(b) + c * sin(a) * sin(b) + d, cos(a - b) * c + d), (c * sinh(a) * cosh(b) + c * sinh(b) * cosh(a) + d, sinh(a + b) * c + d), (c * cosh(a) * cosh(b) + c * sinh(a) * sinh(b) + d, cosh(a + b) * c + d))
    matchers_identity = ((a * sin(b) ** 2, a - a * cos(b) ** 2), (a * tan(b) ** 2, a * (1 / cos(b)) ** 2 - a), (a * cot(b) ** 2, a * (1 / sin(b)) ** 2 - a), (a * sin(b + c), a * (sin(b) * cos(c) + sin(c) * cos(b))), (a * cos(b + c), a * (cos(b) * cos(c) - sin(b) * sin(c))), (a * tan(b + c), a * ((tan(b) + tan(c)) / (1 - tan(b) * tan(c)))), (a * sinh(b) ** 2, a * cosh(b) ** 2 - a), (a * tanh(b) ** 2, a - a * (1 / cosh(b)) ** 2), (a * coth(b) ** 2, a + a * (1 / sinh(b)) ** 2), (a * sinh(b + c), a * (sinh(b) * cosh(c) + sinh(c) * cosh(b))), (a * cosh(b + c), a * (cosh(b) * cosh(c) + sinh(b) * sinh(c))), (a * tanh(b + c), a * ((tanh(b) + tanh(c)) / (1 + tanh(b) * tanh(c)))))
    artifacts = ((a - a * cos(b) ** 2 + c, a * sin(b) ** 2 + c, cos), (a - a * (1 / cos(b)) ** 2 + c, -a * tan(b) ** 2 + c, cos), (a - a * (1 / sin(b)) ** 2 + c, -a * cot(b) ** 2 + c, sin), (a - a * cosh(b) ** 2 + c, -a * sinh(b) ** 2 + c, cosh), (a - a * (1 / cosh(b)) ** 2 + c, a * tanh(b) ** 2 + c, cosh), (a + a * (1 / sinh(b)) ** 2 + c, a * coth(b) ** 2 + c, sinh), (a * d - a * d * cos(b) ** 2 + c, a * d * sin(b) ** 2 + c, cos), (a * d - a * d * (1 / cos(b)) ** 2 + c, -a * d * tan(b) ** 2 + c, cos), (a * d - a * d * (1 / sin(b)) ** 2 + c, -a * d * cot(b) ** 2 + c, sin), (a * d - a * d * cosh(b) ** 2 + c, -a * d * sinh(b) ** 2 + c, cosh), (a * d - a * d * (1 / cosh(b)) ** 2 + c, a * d * tanh(b) ** 2 + c, cosh), (a * d + a * d * (1 / sinh(b)) ** 2 + c, a * d * coth(b) ** 2 + c, sinh))
    _trigpat = (a, b, c, d, matchers_division, matchers_add, matchers_identity, artifacts)
    return _trigpat

def _replace_mul_fpowxgpow(expr, f, g, rexp, h, rexph):
    if False:
        while True:
            i = 10
    'Helper for _match_div_rewrite.\n\n    Replace f(b_)**c_*g(b_)**(rexp(c_)) with h(b)**rexph(c) if f(b_)\n    and g(b_) are both positive or if c_ is an integer.\n    '
    fargs = defaultdict(int)
    gargs = defaultdict(int)
    args = []
    for x in expr.args:
        if x.is_Pow or x.func in (f, g):
            (b, e) = x.as_base_exp()
            if b.is_positive or e.is_integer:
                if b.func == f:
                    fargs[b.args[0]] += e
                    continue
                elif b.func == g:
                    gargs[b.args[0]] += e
                    continue
        args.append(x)
    common = set(fargs) & set(gargs)
    hit = False
    while common:
        key = common.pop()
        fe = fargs.pop(key)
        ge = gargs.pop(key)
        if fe == rexp(ge):
            args.append(h(key) ** rexph(fe))
            hit = True
        else:
            fargs[key] = fe
            gargs[key] = ge
    if not hit:
        return expr
    while fargs:
        (key, e) = fargs.popitem()
        args.append(f(key) ** e)
    while gargs:
        (key, e) = gargs.popitem()
        args.append(g(key) ** e)
    return Mul(*args)
_idn = lambda x: x
_midn = lambda x: -x
_one = lambda x: S.One

def _match_div_rewrite(expr, i):
    if False:
        for i in range(10):
            print('nop')
    'helper for __trigsimp'
    if i == 0:
        expr = _replace_mul_fpowxgpow(expr, sin, cos, _midn, tan, _idn)
    elif i == 1:
        expr = _replace_mul_fpowxgpow(expr, tan, cos, _idn, sin, _idn)
    elif i == 2:
        expr = _replace_mul_fpowxgpow(expr, cot, sin, _idn, cos, _idn)
    elif i == 3:
        expr = _replace_mul_fpowxgpow(expr, tan, sin, _midn, cos, _midn)
    elif i == 4:
        expr = _replace_mul_fpowxgpow(expr, cot, cos, _midn, sin, _midn)
    elif i == 5:
        expr = _replace_mul_fpowxgpow(expr, cot, tan, _idn, _one, _idn)
    elif i == 8:
        expr = _replace_mul_fpowxgpow(expr, sinh, cosh, _midn, tanh, _idn)
    elif i == 9:
        expr = _replace_mul_fpowxgpow(expr, tanh, cosh, _idn, sinh, _idn)
    elif i == 10:
        expr = _replace_mul_fpowxgpow(expr, coth, sinh, _idn, cosh, _idn)
    elif i == 11:
        expr = _replace_mul_fpowxgpow(expr, tanh, sinh, _midn, cosh, _midn)
    elif i == 12:
        expr = _replace_mul_fpowxgpow(expr, coth, cosh, _midn, sinh, _midn)
    elif i == 13:
        expr = _replace_mul_fpowxgpow(expr, coth, tanh, _idn, _one, _idn)
    else:
        return None
    return expr

def _trigsimp(expr, deep=False):
    if False:
        i = 10
        return i + 15
    if expr.has(*_trigs):
        return __trigsimp(expr, deep)
    return expr

@cacheit
def __trigsimp(expr, deep=False):
    if False:
        i = 10
        return i + 15
    'recursive helper for trigsimp'
    from sympy.simplify.fu import TR10i
    if _trigpat is None:
        _trigpats()
    (a, b, c, d, matchers_division, matchers_add, matchers_identity, artifacts) = _trigpat
    if expr.is_Mul:
        if not expr.is_commutative:
            (com, nc) = expr.args_cnc()
            expr = _trigsimp(Mul._from_args(com), deep) * Mul._from_args(nc)
        else:
            for (i, (pattern, simp, ok1, ok2)) in enumerate(matchers_division):
                if not _dotrig(expr, pattern):
                    continue
                newexpr = _match_div_rewrite(expr, i)
                if newexpr is not None:
                    if newexpr != expr:
                        expr = newexpr
                        break
                    else:
                        continue
                res = expr.match(pattern)
                if res and res.get(c, 0):
                    if not res[c].is_integer:
                        ok = ok1.subs(res)
                        if not ok.is_positive:
                            continue
                        ok = ok2.subs(res)
                        if not ok.is_positive:
                            continue
                    if any((w.args[0] == res[b] for w in res[a].atoms(TrigonometricFunction, HyperbolicFunction))):
                        continue
                    expr = simp.subs(res)
                    break
    if expr.is_Add:
        args = []
        for term in expr.args:
            if not term.is_commutative:
                (com, nc) = term.args_cnc()
                nc = Mul._from_args(nc)
                term = Mul._from_args(com)
            else:
                nc = S.One
            term = _trigsimp(term, deep)
            for (pattern, result) in matchers_identity:
                res = term.match(pattern)
                if res is not None:
                    term = result.subs(res)
                    break
            args.append(term * nc)
        if args != expr.args:
            expr = Add(*args)
            expr = min(expr, expand(expr), key=count_ops)
        if expr.is_Add:
            for (pattern, result) in matchers_add:
                if not _dotrig(expr, pattern):
                    continue
                expr = TR10i(expr)
                if expr.has(HyperbolicFunction):
                    res = expr.match(pattern)
                    if res is None or not (a in res and b in res) or any((w.args[0] in (res[a], res[b]) for w in res[d].atoms(TrigonometricFunction, HyperbolicFunction))):
                        continue
                    expr = result.subs(res)
                    break
        for (pattern, result, ex) in artifacts:
            if not _dotrig(expr, pattern):
                continue
            a_t = Wild('a', exclude=[ex])
            pattern = pattern.subs(a, a_t)
            result = result.subs(a, a_t)
            m = expr.match(pattern)
            was = None
            while m and was != expr:
                was = expr
                if m[a_t] == 0 or -m[a_t] in m[c].args or m[a_t] + m[c] == 0:
                    break
                if d in m and m[a_t] * m[d] + m[c] == 0:
                    break
                expr = result.subs(m)
                m = expr.match(pattern)
                m.setdefault(c, S.Zero)
    elif expr.is_Mul or expr.is_Pow or (deep and expr.args):
        expr = expr.func(*[_trigsimp(a, deep) for a in expr.args])
    try:
        if not expr.has(*_trigs):
            raise TypeError
        e = expr.atoms(exp)
        new = expr.rewrite(exp, deep=deep)
        if new == e:
            raise TypeError
        fnew = factor(new)
        if fnew != new:
            new = sorted([new, factor(new)], key=count_ops)[0]
        if not new.atoms(exp) - e:
            expr = new
    except TypeError:
        pass
    return expr

def futrig(e, *, hyper=True, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Return simplified ``e`` using Fu-like transformations.\n    This is not the "Fu" algorithm. This is called by default\n    from ``trigsimp``. By default, hyperbolics subexpressions\n    will be simplified, but this can be disabled by setting\n    ``hyper=False``.\n\n    Examples\n    ========\n\n    >>> from sympy import trigsimp, tan, sinh, tanh\n    >>> from sympy.simplify.trigsimp import futrig\n    >>> from sympy.abc import x\n    >>> trigsimp(1/tan(x)**2)\n    tan(x)**(-2)\n\n    >>> futrig(sinh(x)/tanh(x))\n    cosh(x)\n\n    '
    from sympy.simplify.fu import hyper_as_trig
    e = sympify(e)
    if not isinstance(e, Basic):
        return e
    if not e.args:
        return e
    old = e
    e = bottom_up(e, _futrig)
    if hyper and e.has(HyperbolicFunction):
        (e, f) = hyper_as_trig(e)
        e = f(bottom_up(e, _futrig))
    if e != old and e.is_Mul and e.args[0].is_Rational:
        e = Mul(*e.as_coeff_Mul())
    return e

def _futrig(e):
    if False:
        return 10
    'Helper for futrig.'
    from sympy.simplify.fu import TR1, TR2, TR3, TR2i, TR10, L, TR10i, TR8, TR6, TR15, TR16, TR111, TR5, TRmorrie, TR11, _TR11, TR14, TR22, TR12
    if not e.has(TrigonometricFunction):
        return e
    if e.is_Mul:
        (coeff, e) = e.as_independent(TrigonometricFunction)
    else:
        coeff = None
    Lops = lambda x: (L(x), x.count_ops(), _nodes(x), len(x.args), x.is_Add)
    trigs = lambda x: x.has(TrigonometricFunction)
    tree = [identity, (TR3, TR1, TR12, lambda x: _eapply(factor, x, trigs), TR2, [identity, lambda x: _eapply(_mexpand, x, trigs)], TR2i, lambda x: _eapply(lambda i: factor(i.normal()), x, trigs), TR14, TR5, TR10, TR11, _TR11, TR6, lambda x: _eapply(factor, x, trigs), TR14, [identity, lambda x: _eapply(_mexpand, x, trigs)], TR10i, TRmorrie, [identity, TR8], [identity, lambda x: TR2i(TR2(x))], [lambda x: _eapply(expand_mul, TR5(x), trigs), lambda x: _eapply(expand_mul, TR15(x), trigs)], [lambda x: _eapply(expand_mul, TR6(x), trigs), lambda x: _eapply(expand_mul, TR16(x), trigs)], TR111, [identity, TR2i], [identity, lambda x: _eapply(expand_mul, TR22(x), trigs)], TR1, TR2, TR2i, [identity, lambda x: _eapply(factor_terms, TR12(x), trigs)])]
    e = greedy(tree, objective=Lops)(e)
    if coeff is not None:
        e = coeff * e
    return e

def _is_Expr(e):
    if False:
        while True:
            i = 10
    '_eapply helper to tell whether ``e`` and all its args\n    are Exprs.'
    if isinstance(e, Derivative):
        return _is_Expr(e.expr)
    if not isinstance(e, Expr):
        return False
    return all((_is_Expr(i) for i in e.args))

def _eapply(func, e, cond=None):
    if False:
        for i in range(10):
            print('nop')
    'Apply ``func`` to ``e`` if all args are Exprs else only\n    apply it to those args that *are* Exprs.'
    if not isinstance(e, Expr):
        return e
    if _is_Expr(e) or not e.args:
        return func(e)
    return e.func(*[_eapply(func, ei) if cond is None or cond(ei) else ei for ei in e.args])