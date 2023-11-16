"""
This module contains functions to:

    - solve a single equation for a single variable, in any domain either real or complex.

    - solve a single transcendental equation for a single variable in any domain either real or complex.
      (currently supports solving in real domain only)

    - solve a system of linear equations with N variables and M equations.

    - solve a system of Non Linear Equations with N variables and M equations
"""
from sympy.core.sympify import sympify
from sympy.core import S, Pow, Dummy, pi, Expr, Wild, Mul, Add, Basic
from sympy.core.containers import Tuple
from sympy.core.function import Lambda, expand_complex, AppliedUndef, expand_log, _mexpand, expand_trig, nfloat
from sympy.core.mod import Mod
from sympy.core.numbers import I, Number, Rational, oo
from sympy.core.intfunc import integer_log
from sympy.core.relational import Eq, Ne, Relational
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Symbol, _uniquely_named_symbol
from sympy.core.sympify import _sympify
from sympy.external.gmpy import gcd as number_gcd, lcm as number_lcm
from sympy.polys.matrices.linsolve import _linear_eq_to_dict
from sympy.polys.polyroots import UnsolvableFactorError
from sympy.simplify.simplify import simplify, fraction, trigsimp, nsimplify
from sympy.simplify import powdenest, logcombine
from sympy.functions import log, tan, cot, sin, cos, sec, csc, exp, acos, asin, acsc, asec, piecewise_fold, Piecewise
from sympy.functions.elementary.complexes import Abs, arg, re, im
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.miscellaneous import real_root
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.logic.boolalg import And, BooleanTrue
from sympy.sets import FiniteSet, imageset, Interval, Intersection, Union, ConditionSet, ImageSet, Complement, Contains
from sympy.sets.sets import Set, ProductSet
from sympy.matrices import zeros, Matrix, MatrixBase
from sympy.ntheory import totient
from sympy.ntheory.factor_ import divisors
from sympy.ntheory.residue_ntheory import discrete_log, nthroot_mod
from sympy.polys import roots, Poly, degree, together, PolynomialError, RootOf, factor, lcm, gcd
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polytools import invert, groebner, poly
from sympy.polys.solvers import sympy_eqs_to_ring, solve_lin_sys, PolyNonlinearError
from sympy.polys.matrices.linsolve import _linsolve
from sympy.solvers.solvers import checksol, denoms, unrad, _simple_dens, recast_to_symbols
from sympy.solvers.polysys import solve_poly_system
from sympy.utilities import filldedent
from sympy.utilities.iterables import numbered_symbols, has_dups, is_sequence, iterable
from sympy.calculus.util import periodicity, continuous_domain, function_range
from types import GeneratorType

class NonlinearError(ValueError):
    """Raised when unexpectedly encountering nonlinear equations"""
    pass
_rc = (Dummy('R', real=True), Dummy('C', complex=True))

def _masked(f, *atoms):
    if False:
        for i in range(10):
            print('nop')
    'Return ``f``, with all objects given by ``atoms`` replaced with\n    Dummy symbols, ``d``, and the list of replacements, ``(d, e)``,\n    where ``e`` is an object of type given by ``atoms`` in which\n    any other instances of atoms have been recursively replaced with\n    Dummy symbols, too. The tuples are ordered so that if they are\n    applied in sequence, the origin ``f`` will be restored.\n\n    Examples\n    ========\n\n    >>> from sympy import cos\n    >>> from sympy.abc import x\n    >>> from sympy.solvers.solveset import _masked\n\n    >>> f = cos(cos(x) + 1)\n    >>> f, reps = _masked(cos(1 + cos(x)), cos)\n    >>> f\n    _a1\n    >>> reps\n    [(_a1, cos(_a0 + 1)), (_a0, cos(x))]\n    >>> for d, e in reps:\n    ...     f = f.xreplace({d: e})\n    >>> f\n    cos(cos(x) + 1)\n    '
    sym = numbered_symbols('a', cls=Dummy, real=True)
    mask = []
    for a in ordered(f.atoms(*atoms)):
        for i in mask:
            a = a.replace(*i)
        mask.append((a, next(sym)))
    for (i, (o, n)) in enumerate(mask):
        f = f.replace(o, n)
        mask[i] = (n, o)
    mask = list(reversed(mask))
    return (f, mask)

def _invert(f_x, y, x, domain=S.Complexes):
    if False:
        while True:
            i = 10
    '\n    Reduce the complex valued equation $f(x) = y$ to a set of equations\n\n    $$\\left\\{g(x) = h_1(y),\\  g(x) = h_2(y),\\ \\dots,\\  g(x) = h_n(y) \\right\\}$$\n\n    where $g(x)$ is a simpler function than $f(x)$.  The return value is a tuple\n    $(g(x), \\mathrm{set}_h)$, where $g(x)$ is a function of $x$ and $\\mathrm{set}_h$ is\n    the set of function $\\left\\{h_1(y), h_2(y), \\dots, h_n(y)\\right\\}$.\n    Here, $y$ is not necessarily a symbol.\n\n    $\\mathrm{set}_h$ contains the functions, along with the information\n    about the domain in which they are valid, through set\n    operations. For instance, if :math:`y = |x| - n` is inverted\n    in the real domain, then $\\mathrm{set}_h$ is not simply\n    $\\{-n, n\\}$ as the nature of `n` is unknown; rather, it is:\n\n    $$ \\left(\\left[0, \\infty\\right) \\cap \\left\\{n\\right\\}\\right) \\cup\n                       \\left(\\left(-\\infty, 0\\right] \\cap \\left\\{- n\\right\\}\\right)$$\n\n    By default, the complex domain is used which means that inverting even\n    seemingly simple functions like $\\exp(x)$ will give very different\n    results from those obtained in the real domain.\n    (In the case of $\\exp(x)$, the inversion via $\\log$ is multi-valued\n    in the complex domain, having infinitely many branches.)\n\n    If you are working with real values only (or you are not sure which\n    function to use) you should probably set the domain to\n    ``S.Reals`` (or use ``invert_real`` which does that automatically).\n\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.solveset import invert_complex, invert_real\n    >>> from sympy.abc import x, y\n    >>> from sympy import exp\n\n    When does exp(x) == y?\n\n    >>> invert_complex(exp(x), y, x)\n    (x, ImageSet(Lambda(_n, I*(2*_n*pi + arg(y)) + log(Abs(y))), Integers))\n    >>> invert_real(exp(x), y, x)\n    (x, Intersection({log(y)}, Reals))\n\n    When does exp(x) == 1?\n\n    >>> invert_complex(exp(x), 1, x)\n    (x, ImageSet(Lambda(_n, 2*_n*I*pi), Integers))\n    >>> invert_real(exp(x), 1, x)\n    (x, {0})\n\n    See Also\n    ========\n    invert_real, invert_complex\n    '
    x = sympify(x)
    if not x.is_Symbol:
        raise ValueError('x must be a symbol')
    f_x = sympify(f_x)
    if x not in f_x.free_symbols:
        raise ValueError("Inverse of constant function doesn't exist")
    y = sympify(y)
    if x in y.free_symbols:
        raise ValueError('y should be independent of x ')
    if domain.is_subset(S.Reals):
        (x1, s) = _invert_real(f_x, FiniteSet(y), x)
    else:
        (x1, s) = _invert_complex(f_x, FiniteSet(y), x)
    if not isinstance(s, FiniteSet) or x1 != x:
        return (x1, s)
    if domain is S.Complexes:
        return (x1, s)
    else:
        return (x1, s.intersection(domain))
invert_complex = _invert

def invert_real(f_x, y, x):
    if False:
        for i in range(10):
            print('nop')
    '\n    Inverts a real-valued function. Same as :func:`invert_complex`, but sets\n    the domain to ``S.Reals`` before inverting.\n    '
    return _invert(f_x, y, x, S.Reals)

def _invert_real(f, g_ys, symbol):
    if False:
        for i in range(10):
            print('nop')
    'Helper function for _invert.'
    if f == symbol or g_ys is S.EmptySet:
        return (f, g_ys)
    n = Dummy('n', real=True)
    if isinstance(f, exp) or (f.is_Pow and f.base == S.Exp1):
        return _invert_real(f.exp, imageset(Lambda(n, log(n)), g_ys), symbol)
    if hasattr(f, 'inverse') and f.inverse() is not None and (not isinstance(f, (TrigonometricFunction, HyperbolicFunction))):
        if len(f.args) > 1:
            raise ValueError('Only functions with one argument are supported.')
        return _invert_real(f.args[0], imageset(Lambda(n, f.inverse()(n)), g_ys), symbol)
    if isinstance(f, Abs):
        return _invert_abs(f.args[0], g_ys, symbol)
    if f.is_Add:
        (g, h) = f.as_independent(symbol)
        if g is not S.Zero:
            return _invert_real(h, imageset(Lambda(n, n - g), g_ys), symbol)
    if f.is_Mul:
        (g, h) = f.as_independent(symbol)
        if g is not S.One:
            return _invert_real(h, imageset(Lambda(n, n / g), g_ys), symbol)
    if f.is_Pow:
        (base, expo) = f.args
        base_has_sym = base.has(symbol)
        expo_has_sym = expo.has(symbol)
        if not expo_has_sym:
            if expo.is_rational:
                (num, den) = expo.as_numer_denom()
                if den % 2 == 0 and num % 2 == 1 and (den.is_zero is False):
                    root = Lambda(n, real_root(n, expo))
                    g_ys_pos = g_ys & Interval(0, oo)
                    res = imageset(root, g_ys_pos)
                    (_inv, _set) = _invert_real(base, res, symbol)
                    return (_inv, _set)
                if den % 2 == 1:
                    root = Lambda(n, real_root(n, expo))
                    res = imageset(root, g_ys)
                    if num % 2 == 0:
                        neg_res = imageset(Lambda(n, -n), res)
                        return _invert_real(base, res + neg_res, symbol)
                    if num % 2 == 1:
                        return _invert_real(base, res, symbol)
            elif expo.is_irrational:
                root = Lambda(n, real_root(n, expo))
                g_ys_pos = g_ys & Interval(0, oo)
                res = imageset(root, g_ys_pos)
                return _invert_real(base, res, symbol)
            else:
                pass
        if not base_has_sym:
            rhs = g_ys.args[0]
            if base.is_positive:
                return _invert_real(expo, imageset(Lambda(n, log(n, base, evaluate=False)), g_ys), symbol)
            elif base.is_negative:
                (s, b) = integer_log(rhs, base)
                if b:
                    return _invert_real(expo, FiniteSet(s), symbol)
                else:
                    return (expo, S.EmptySet)
            elif base.is_zero:
                one = Eq(rhs, 1)
                if one == S.true:
                    return _invert_real(expo, FiniteSet(0), symbol)
                elif one == S.false:
                    return (expo, S.EmptySet)
    if isinstance(f, TrigonometricFunction):
        if isinstance(g_ys, FiniteSet):

            def inv(trig):
                if False:
                    for i in range(10):
                        print('nop')
                if isinstance(trig, (sin, csc)):
                    F = asin if isinstance(trig, sin) else acsc
                    return (lambda a: 2 * n * pi + F(a), lambda a: 2 * n * pi + pi - F(a))
                if isinstance(trig, (cos, sec)):
                    F = acos if isinstance(trig, cos) else asec
                    return (lambda a: 2 * n * pi + F(a), lambda a: 2 * n * pi - F(a))
                if isinstance(trig, (tan, cot)):
                    return (lambda a: n * pi + trig.inverse()(a),)
            n = Dummy('n', integer=True)
            invs = S.EmptySet
            for L in inv(f):
                invs += Union(*[imageset(Lambda(n, L(g)), S.Integers) for g in g_ys])
            return _invert_real(f.args[0], invs, symbol)
    return (f, g_ys)

def _invert_complex(f, g_ys, symbol):
    if False:
        return 10
    'Helper function for _invert.'
    if f == symbol or g_ys is S.EmptySet:
        return (f, g_ys)
    n = Dummy('n')
    if f.is_Add:
        (g, h) = f.as_independent(symbol)
        if g is not S.Zero:
            return _invert_complex(h, imageset(Lambda(n, n - g), g_ys), symbol)
    if f.is_Mul:
        (g, h) = f.as_independent(symbol)
        if g is not S.One:
            if g in {S.NegativeInfinity, S.ComplexInfinity, S.Infinity}:
                return (h, S.EmptySet)
            return _invert_complex(h, imageset(Lambda(n, n / g), g_ys), symbol)
    if f.is_Pow:
        (base, expo) = f.args
        if expo.is_Rational and g_ys == FiniteSet(0):
            if expo.is_positive:
                return _invert_complex(base, g_ys, symbol)
    if hasattr(f, 'inverse') and f.inverse() is not None and (not isinstance(f, TrigonometricFunction)) and (not isinstance(f, HyperbolicFunction)) and (not isinstance(f, exp)):
        if len(f.args) > 1:
            raise ValueError('Only functions with one argument are supported.')
        return _invert_complex(f.args[0], imageset(Lambda(n, f.inverse()(n)), g_ys), symbol)
    if isinstance(f, exp) or (f.is_Pow and f.base == S.Exp1):
        if isinstance(g_ys, ImageSet):
            g_ys_expr = g_ys.lamda.expr
            g_ys_vars = g_ys.lamda.variables
            k = Dummy('k{}'.format(len(g_ys_vars)))
            g_ys_vars_1 = (k,) + g_ys_vars
            exp_invs = Union(*[imageset(Lambda((g_ys_vars_1,), I * (2 * k * pi + arg(g_ys_expr)) + log(Abs(g_ys_expr))), S.Integers ** len(g_ys_vars_1))])
            return _invert_complex(f.exp, exp_invs, symbol)
        elif isinstance(g_ys, FiniteSet):
            exp_invs = Union(*[imageset(Lambda(n, I * (2 * n * pi + arg(g_y)) + log(Abs(g_y))), S.Integers) for g_y in g_ys if g_y != 0])
            return _invert_complex(f.exp, exp_invs, symbol)
    return (f, g_ys)

def _invert_abs(f, g_ys, symbol):
    if False:
        print('Hello World!')
    'Helper function for inverting absolute value functions.\n\n    Returns the complete result of inverting an absolute value\n    function along with the conditions which must also be satisfied.\n\n    If it is certain that all these conditions are met, a :class:`~.FiniteSet`\n    of all possible solutions is returned. If any condition cannot be\n    satisfied, an :class:`~.EmptySet` is returned. Otherwise, a\n    :class:`~.ConditionSet` of the solutions, with all the required conditions\n    specified, is returned.\n\n    '
    if not g_ys.is_FiniteSet:
        pos = Intersection(g_ys, Interval(0, S.Infinity))
        parg = _invert_real(f, pos, symbol)
        narg = _invert_real(-f, pos, symbol)
        if parg[0] != narg[0]:
            raise NotImplementedError
        return (parg[0], Union(narg[1], parg[1]))
    unknown = []
    for a in g_ys.args:
        ok = a.is_nonnegative if a.is_Number else a.is_positive
        if ok is None:
            unknown.append(a)
        elif not ok:
            return (symbol, S.EmptySet)
    if unknown:
        conditions = And(*[Contains(i, Interval(0, oo)) for i in unknown])
    else:
        conditions = True
    n = Dummy('n', real=True)
    (g_x, values) = _invert_real(f, Union(imageset(Lambda(n, n), g_ys), imageset(Lambda(n, -n), g_ys)), symbol)
    return (g_x, ConditionSet(g_x, conditions, values))

def domain_check(f, symbol, p):
    if False:
        print('Hello World!')
    'Returns False if point p is infinite or any subexpression of f\n    is infinite or becomes so after replacing symbol with p. If none of\n    these conditions is met then True will be returned.\n\n    Examples\n    ========\n\n    >>> from sympy import Mul, oo\n    >>> from sympy.abc import x\n    >>> from sympy.solvers.solveset import domain_check\n    >>> g = 1/(1 + (1/(x + 1))**2)\n    >>> domain_check(g, x, -1)\n    False\n    >>> domain_check(x**2, x, 0)\n    True\n    >>> domain_check(1/x, x, oo)\n    False\n\n    * The function relies on the assumption that the original form\n      of the equation has not been changed by automatic simplification.\n\n    >>> domain_check(x/x, x, 0) # x/x is automatically simplified to 1\n    True\n\n    * To deal with automatic evaluations use evaluate=False:\n\n    >>> domain_check(Mul(x, 1/x, evaluate=False), x, 0)\n    False\n    '
    (f, p) = (sympify(f), sympify(p))
    if p.is_infinite:
        return False
    return _domain_check(f, symbol, p)

def _domain_check(f, symbol, p):
    if False:
        for i in range(10):
            print('nop')
    if f.is_Atom and f.is_finite:
        return True
    elif f.subs(symbol, p).is_infinite:
        return False
    elif isinstance(f, Piecewise):
        for (expr, cond) in f.args:
            condsubs = cond.subs(symbol, p)
            if condsubs is S.false:
                continue
            elif condsubs is S.true:
                return _domain_check(expr, symbol, p)
            else:
                return True
    else:
        return all((_domain_check(g, symbol, p) for g in f.args))

def _is_finite_with_finite_vars(f, domain=S.Complexes):
    if False:
        i = 10
        return i + 15
    '\n    Return True if the given expression is finite. For symbols that\n    do not assign a value for `complex` and/or `real`, the domain will\n    be used to assign a value; symbols that do not assign a value\n    for `finite` will be made finite. All other assumptions are\n    left unmodified.\n    '

    def assumptions(s):
        if False:
            i = 10
            return i + 15
        A = s.assumptions0
        A.setdefault('finite', A.get('finite', True))
        if domain.is_subset(S.Reals):
            A.setdefault('real', True)
        else:
            A.setdefault('complex', True)
        return A
    reps = {s: Dummy(**assumptions(s)) for s in f.free_symbols}
    return f.xreplace(reps).is_finite

def _is_function_class_equation(func_class, f, symbol):
    if False:
        for i in range(10):
            print('nop')
    ' Tests whether the equation is an equation of the given function class.\n\n    The given equation belongs to the given function class if it is\n    comprised of functions of the function class which are multiplied by\n    or added to expressions independent of the symbol. In addition, the\n    arguments of all such functions must be linear in the symbol as well.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.solveset import _is_function_class_equation\n    >>> from sympy import tan, sin, tanh, sinh, exp\n    >>> from sympy.abc import x\n    >>> from sympy.functions.elementary.trigonometric import TrigonometricFunction\n    >>> from sympy.functions.elementary.hyperbolic import HyperbolicFunction\n    >>> _is_function_class_equation(TrigonometricFunction, exp(x) + tan(x), x)\n    False\n    >>> _is_function_class_equation(TrigonometricFunction, tan(x) + sin(x), x)\n    True\n    >>> _is_function_class_equation(TrigonometricFunction, tan(x**2), x)\n    False\n    >>> _is_function_class_equation(TrigonometricFunction, tan(x + 2), x)\n    True\n    >>> _is_function_class_equation(HyperbolicFunction, tanh(x) + sinh(x), x)\n    True\n    '
    if f.is_Mul or f.is_Add:
        return all((_is_function_class_equation(func_class, arg, symbol) for arg in f.args))
    if f.is_Pow:
        if not f.exp.has(symbol):
            return _is_function_class_equation(func_class, f.base, symbol)
        else:
            return False
    if not f.has(symbol):
        return True
    if isinstance(f, func_class):
        try:
            g = Poly(f.args[0], symbol)
            return g.degree() <= 1
        except PolynomialError:
            return False
    else:
        return False

def _solve_as_rational(f, symbol, domain):
    if False:
        print('Hello World!')
    ' solve rational functions'
    f = together(_mexpand(f, recursive=True), deep=True)
    (g, h) = fraction(f)
    if not h.has(symbol):
        try:
            return _solve_as_poly(g, symbol, domain)
        except NotImplementedError:
            return ConditionSet(symbol, Eq(f, 0), domain)
        except CoercionFailed:
            return S.EmptySet
    else:
        valid_solns = _solveset(g, symbol, domain)
        invalid_solns = _solveset(h, symbol, domain)
        return valid_solns - invalid_solns

class _SolveTrig1Error(Exception):
    """Raised when _solve_trig1 heuristics do not apply"""

def _solve_trig(f, symbol, domain):
    if False:
        for i in range(10):
            print('nop')
    'Function to call other helpers to solve trigonometric equations '
    sol = None
    try:
        sol = _solve_trig1(f, symbol, domain)
    except _SolveTrig1Error:
        try:
            sol = _solve_trig2(f, symbol, domain)
        except ValueError:
            raise NotImplementedError(filldedent('\n                Solution to this kind of trigonometric equations\n                is yet to be implemented'))
    return sol

def _solve_trig1(f, symbol, domain):
    if False:
        return 10
    "Primary solver for trigonometric and hyperbolic equations\n\n    Returns either the solution set as a ConditionSet (auto-evaluated to a\n    union of ImageSets if no variables besides 'symbol' are involved) or\n    raises _SolveTrig1Error if f == 0 cannot be solved.\n\n    Notes\n    =====\n    Algorithm:\n    1. Do a change of variable x -> mu*x in arguments to trigonometric and\n    hyperbolic functions, in order to reduce them to small integers. (This\n    step is crucial to keep the degrees of the polynomials of step 4 low.)\n    2. Rewrite trigonometric/hyperbolic functions as exponentials.\n    3. Proceed to a 2nd change of variable, replacing exp(I*x) or exp(x) by y.\n    4. Solve the resulting rational equation.\n    5. Use invert_complex or invert_real to return to the original variable.\n    6. If the coefficients of 'symbol' were symbolic in nature, add the\n    necessary consistency conditions in a ConditionSet.\n\n    "
    x = Dummy('x')
    if _is_function_class_equation(HyperbolicFunction, f, symbol):
        cov = exp(x)
        inverter = invert_real if domain.is_subset(S.Reals) else invert_complex
    else:
        cov = exp(I * x)
        inverter = invert_complex
    f = trigsimp(f)
    f_original = f
    trig_functions = f.atoms(TrigonometricFunction, HyperbolicFunction)
    trig_arguments = [e.args[0] for e in trig_functions]
    if not any((a.has(symbol) for a in trig_arguments)):
        return solveset(f_original, symbol, domain)
    denominators = []
    numerators = []
    for ar in trig_arguments:
        try:
            poly_ar = Poly(ar, symbol)
        except PolynomialError:
            raise _SolveTrig1Error('trig argument is not a polynomial')
        if poly_ar.degree() > 1:
            raise _SolveTrig1Error('degree of variable must not exceed one')
        if poly_ar.degree() == 0:
            continue
        c = poly_ar.all_coeffs()[0]
        numerators.append(fraction(c)[0])
        denominators.append(fraction(c)[1])
    mu = lcm(denominators) / gcd(numerators)
    f = f.subs(symbol, mu * x)
    f = f.rewrite(exp)
    f = together(f)
    (g, h) = fraction(f)
    y = Dummy('y')
    (g, h) = (g.expand(), h.expand())
    (g, h) = (g.subs(cov, y), h.subs(cov, y))
    if g.has(x) or h.has(x):
        raise _SolveTrig1Error('change of variable not possible')
    solns = solveset_complex(g, y) - solveset_complex(h, y)
    if isinstance(solns, ConditionSet):
        raise _SolveTrig1Error('polynomial has ConditionSet solution')
    if isinstance(solns, FiniteSet):
        if any((isinstance(s, RootOf) for s in solns)):
            raise _SolveTrig1Error('polynomial results in RootOf object')
        cov = cov.subs(x, symbol / mu)
        result = Union(*[inverter(cov, s, symbol)[1] for s in solns])
        if mu.has(Symbol):
            syms = mu.atoms(Symbol)
            (munum, muden) = fraction(mu)
            condnum = munum.as_independent(*syms, as_Add=False)[1]
            condden = muden.as_independent(*syms, as_Add=False)[1]
            cond = And(Ne(condnum, 0), Ne(condden, 0))
        else:
            cond = True
        if domain is S.Complexes:
            return ConditionSet(symbol, cond, result)
        else:
            return ConditionSet(symbol, cond, Intersection(result, domain))
    elif solns is S.EmptySet:
        return S.EmptySet
    else:
        raise _SolveTrig1Error('polynomial solutions must form FiniteSet')

def _solve_trig2(f, symbol, domain):
    if False:
        for i in range(10):
            print('nop')
    'Secondary helper to solve trigonometric equations,\n    called when first helper fails '
    f = trigsimp(f)
    f_original = f
    trig_functions = f.atoms(sin, cos, tan, sec, cot, csc)
    trig_arguments = [e.args[0] for e in trig_functions]
    denominators = []
    numerators = []
    if not trig_functions:
        return ConditionSet(symbol, Eq(f_original, 0), domain)
    for ar in trig_arguments:
        try:
            poly_ar = Poly(ar, symbol)
        except PolynomialError:
            raise ValueError('give up, we cannot solve if this is not a polynomial in x')
        if poly_ar.degree() > 1:
            raise ValueError('degree of variable inside polynomial should not exceed one')
        if poly_ar.degree() == 0:
            continue
        c = poly_ar.all_coeffs()[0]
        try:
            numerators.append(Rational(c).p)
            denominators.append(Rational(c).q)
        except TypeError:
            return ConditionSet(symbol, Eq(f_original, 0), domain)
    x = Dummy('x')
    mu = Rational(2) * number_lcm(*denominators) / number_gcd(*numerators)
    f = f.subs(symbol, mu * x)
    f = f.rewrite(tan)
    f = expand_trig(f)
    f = together(f)
    (g, h) = fraction(f)
    y = Dummy('y')
    (g, h) = (g.expand(), h.expand())
    (g, h) = (g.subs(tan(x), y), h.subs(tan(x), y))
    if g.has(x) or h.has(x):
        return ConditionSet(symbol, Eq(f_original, 0), domain)
    solns = solveset(g, y, S.Reals) - solveset(h, y, S.Reals)
    if isinstance(solns, FiniteSet):
        result = Union(*[invert_real(tan(symbol / mu), s, symbol)[1] for s in solns])
        dsol = invert_real(tan(symbol / mu), oo, symbol)[1]
        if degree(h) > degree(g):
            result = Union(result, dsol)
        return Intersection(result, domain)
    elif solns is S.EmptySet:
        return S.EmptySet
    else:
        return ConditionSet(symbol, Eq(f_original, 0), S.Reals)

def _solve_as_poly(f, symbol, domain=S.Complexes):
    if False:
        i = 10
        return i + 15
    '\n    Solve the equation using polynomial techniques if it already is a\n    polynomial equation or, with a change of variables, can be made so.\n    '
    result = None
    if f.is_polynomial(symbol):
        solns = roots(f, symbol, cubics=True, quartics=True, quintics=True, domain='EX')
        num_roots = sum(solns.values())
        if degree(f, symbol) <= num_roots:
            result = FiniteSet(*solns.keys())
        else:
            poly = Poly(f, symbol)
            solns = poly.all_roots()
            if poly.degree() <= len(solns):
                result = FiniteSet(*solns)
            else:
                result = ConditionSet(symbol, Eq(f, 0), domain)
    else:
        poly = Poly(f)
        if poly is None:
            result = ConditionSet(symbol, Eq(f, 0), domain)
        gens = [g for g in poly.gens if g.has(symbol)]
        if len(gens) == 1:
            poly = Poly(poly, gens[0])
            gen = poly.gen
            deg = poly.degree()
            poly = Poly(poly.as_expr(), poly.gen, composite=True)
            poly_solns = FiniteSet(*roots(poly, cubics=True, quartics=True, quintics=True).keys())
            if len(poly_solns) < deg:
                result = ConditionSet(symbol, Eq(f, 0), domain)
            if gen != symbol:
                y = Dummy('y')
                inverter = invert_real if domain.is_subset(S.Reals) else invert_complex
                (lhs, rhs_s) = inverter(gen, y, symbol)
                if lhs == symbol:
                    result = Union(*[rhs_s.subs(y, s) for s in poly_solns])
                    if isinstance(result, FiniteSet) and isinstance(gen, Pow) and gen.base.is_Rational:
                        result = FiniteSet(*[expand_log(i) for i in result])
                else:
                    result = ConditionSet(symbol, Eq(f, 0), domain)
        else:
            result = ConditionSet(symbol, Eq(f, 0), domain)
    if result is not None:
        if isinstance(result, FiniteSet):
            if all((s.atoms(Symbol, AppliedUndef) == set() and (not isinstance(s, RootOf)) for s in result)):
                s = Dummy('s')
                result = imageset(Lambda(s, expand_complex(s)), result)
        if isinstance(result, FiniteSet) and domain != S.Complexes:
            result = result.intersection(domain)
        return result
    else:
        return ConditionSet(symbol, Eq(f, 0), domain)

def _solve_radical(f, unradf, symbol, solveset_solver):
    if False:
        while True:
            i = 10
    ' Helper function to solve equations with radicals '
    res = unradf
    (eq, cov) = res if res else (f, [])
    if not cov:
        result = solveset_solver(eq, symbol) - Union(*[solveset_solver(g, symbol) for g in denoms(f, symbol)])
    else:
        (y, yeq) = cov
        if not solveset_solver(y - I, y):
            yreal = Dummy('yreal', real=True)
            yeq = yeq.xreplace({y: yreal})
            eq = eq.xreplace({y: yreal})
            y = yreal
        g_y_s = solveset_solver(yeq, symbol)
        f_y_sols = solveset_solver(eq, y)
        result = Union(*[imageset(Lambda(y, g_y), f_y_sols) for g_y in g_y_s])

    def check_finiteset(solutions):
        if False:
            print('Hello World!')
        f_set = []
        c_set = []
        for s in solutions:
            if checksol(f, symbol, s):
                f_set.append(s)
            else:
                c_set.append(s)
        return FiniteSet(*f_set) + ConditionSet(symbol, Eq(f, 0), FiniteSet(*c_set))

    def check_set(solutions):
        if False:
            for i in range(10):
                print('nop')
        if solutions is S.EmptySet:
            return solutions
        elif isinstance(solutions, ConditionSet):
            return solutions
        elif isinstance(solutions, FiniteSet):
            return check_finiteset(solutions)
        elif isinstance(solutions, Complement):
            (A, B) = solutions.args
            return Complement(check_set(A), B)
        elif isinstance(solutions, Union):
            return Union(*[check_set(s) for s in solutions.args])
        else:
            return solutions
    solution_set = check_set(result)
    return solution_set

def _solve_abs(f, symbol, domain):
    if False:
        return 10
    ' Helper function to solve equation involving absolute value function '
    if not domain.is_subset(S.Reals):
        raise ValueError(filldedent('\n            Absolute values cannot be inverted in the\n            complex domain.'))
    (p, q, r) = (Wild('p'), Wild('q'), Wild('r'))
    pattern_match = f.match(p * Abs(q) + r) or {}
    (f_p, f_q, f_r) = [pattern_match.get(i, S.Zero) for i in (p, q, r)]
    if not (f_p.is_zero or f_q.is_zero):
        domain = continuous_domain(f_q, symbol, domain)
        from .inequalities import solve_univariate_inequality
        q_pos_cond = solve_univariate_inequality(f_q >= 0, symbol, relational=False, domain=domain, continuous=True)
        q_neg_cond = q_pos_cond.complement(domain)
        sols_q_pos = solveset_real(f_p * f_q + f_r, symbol).intersect(q_pos_cond)
        sols_q_neg = solveset_real(f_p * -f_q + f_r, symbol).intersect(q_neg_cond)
        return Union(sols_q_pos, sols_q_neg)
    else:
        return ConditionSet(symbol, Eq(f, 0), domain)

def solve_decomposition(f, symbol, domain):
    if False:
        print('Hello World!')
    '\n    Function to solve equations via the principle of "Decomposition\n    and Rewriting".\n\n    Examples\n    ========\n    >>> from sympy import exp, sin, Symbol, pprint, S\n    >>> from sympy.solvers.solveset import solve_decomposition as sd\n    >>> x = Symbol(\'x\')\n    >>> f1 = exp(2*x) - 3*exp(x) + 2\n    >>> sd(f1, x, S.Reals)\n    {0, log(2)}\n    >>> f2 = sin(x)**2 + 2*sin(x) + 1\n    >>> pprint(sd(f2, x, S.Reals), use_unicode=False)\n              3*pi\n    {2*n*pi + ---- | n in Integers}\n               2\n    >>> f3 = sin(x + 2)\n    >>> pprint(sd(f3, x, S.Reals), use_unicode=False)\n    {2*n*pi - 2 | n in Integers} U {2*n*pi - 2 + pi | n in Integers}\n\n    '
    from sympy.solvers.decompogen import decompogen
    g_s = decompogen(f, symbol)
    y_s = FiniteSet(0)
    for g in g_s:
        frange = function_range(g, symbol, domain)
        y_s = Intersection(frange, y_s)
        result = S.EmptySet
        if isinstance(y_s, FiniteSet):
            for y in y_s:
                solutions = solveset(Eq(g, y), symbol, domain)
                if not isinstance(solutions, ConditionSet):
                    result += solutions
        else:
            if isinstance(y_s, ImageSet):
                iter_iset = (y_s,)
            elif isinstance(y_s, Union):
                iter_iset = y_s.args
            elif y_s is S.EmptySet:
                return S.EmptySet
            for iset in iter_iset:
                new_solutions = solveset(Eq(iset.lamda.expr, g), symbol, domain)
                dummy_var = tuple(iset.lamda.expr.free_symbols)[0]
                (base_set,) = iset.base_sets
                if isinstance(new_solutions, FiniteSet):
                    new_exprs = new_solutions
                elif isinstance(new_solutions, Intersection):
                    if isinstance(new_solutions.args[1], FiniteSet):
                        new_exprs = new_solutions.args[1]
                for new_expr in new_exprs:
                    result += ImageSet(Lambda(dummy_var, new_expr), base_set)
        if result is S.EmptySet:
            return ConditionSet(symbol, Eq(f, 0), domain)
        y_s = result
    return y_s

def _solveset(f, symbol, domain, _check=False):
    if False:
        i = 10
        return i + 15
    "Helper for solveset to return a result from an expression\n    that has already been sympify'ed and is known to contain the\n    given symbol."
    from sympy.simplify.simplify import signsimp
    if isinstance(f, BooleanTrue):
        return domain
    orig_f = f
    invert_trig = False
    if f.is_Mul:
        (coeff, f) = f.as_independent(symbol, as_Add=False)
        if coeff in {S.ComplexInfinity, S.NegativeInfinity, S.Infinity}:
            f = together(orig_f)
    elif f.is_Add:
        (a, h) = f.as_independent(symbol)
        (m, h) = h.as_independent(symbol, as_Add=False)
        if m not in {S.ComplexInfinity, S.Zero, S.Infinity, S.NegativeInfinity}:
            f = a / m + h
        if isinstance(h, TrigonometricFunction) and (a and a.is_number and a.is_real and domain.is_subset(S.Reals)):
            invert_trig = True
    solver = lambda f, x, domain=domain: _solveset(f, x, domain)
    inverter = lambda f, rhs, symbol: _invert(f, rhs, symbol, domain)
    result = S.EmptySet
    if f.expand().is_zero:
        return domain
    elif not f.has(symbol):
        return S.EmptySet
    elif f.is_Mul and all((_is_finite_with_finite_vars(m, domain) for m in f.args)):
        result = Union(*[solver(m, symbol) for m in f.args])
    elif not invert_trig and (_is_function_class_equation(TrigonometricFunction, f, symbol) or _is_function_class_equation(HyperbolicFunction, f, symbol)):
        result = _solve_trig(f, symbol, domain)
    elif isinstance(f, arg):
        a = f.args[0]
        result = Intersection(_solveset(re(a) > 0, symbol, domain), _solveset(im(a), symbol, domain))
    elif f.is_Piecewise:
        expr_set_pairs = f.as_expr_set_pairs(domain)
        for (expr, in_set) in expr_set_pairs:
            if in_set.is_Relational:
                in_set = in_set.as_set()
            solns = solver(expr, symbol, in_set)
            result += solns
    elif isinstance(f, Eq):
        result = solver(Add(f.lhs, -f.rhs, evaluate=False), symbol, domain)
    elif f.is_Relational:
        from .inequalities import solve_univariate_inequality
        try:
            result = solve_univariate_inequality(f, symbol, domain=domain, relational=False)
        except NotImplementedError:
            result = ConditionSet(symbol, f, domain)
        return result
    elif _is_modular(f, symbol):
        result = _solve_modular(f, symbol, domain)
    else:
        (lhs, rhs_s) = inverter(f, 0, symbol)
        if lhs == symbol:
            if isinstance(rhs_s, FiniteSet):
                rhs_s = FiniteSet(*[Mul(*signsimp(i).as_content_primitive()) for i in rhs_s])
            result = rhs_s
        elif isinstance(rhs_s, FiniteSet):
            for equation in [lhs - rhs for rhs in rhs_s]:
                if equation == f:
                    u = unrad(f, symbol)
                    if u:
                        result += _solve_radical(equation, u, symbol, solver)
                    elif equation.has(Abs):
                        result += _solve_abs(f, symbol, domain)
                    else:
                        result_rational = _solve_as_rational(equation, symbol, domain)
                        if not isinstance(result_rational, ConditionSet):
                            result += result_rational
                        else:
                            t_result = _transolve(equation, symbol, domain)
                            if isinstance(t_result, ConditionSet):
                                factored = equation.factor()
                                if factored.is_Mul and equation != factored:
                                    (_, dep) = factored.as_independent(symbol)
                                    if not dep.is_Add:
                                        t_results = []
                                        for fac in Mul.make_args(factored):
                                            if fac.has(symbol):
                                                t_results.append(solver(fac, symbol))
                                        t_result = Union(*t_results)
                            result += t_result
                else:
                    result += solver(equation, symbol)
        elif rhs_s is not S.EmptySet:
            result = ConditionSet(symbol, Eq(f, 0), domain)
    if isinstance(result, ConditionSet):
        if isinstance(f, Expr):
            (num, den) = f.as_numer_denom()
            if den.has(symbol):
                _result = _solveset(num, symbol, domain)
                if not isinstance(_result, ConditionSet):
                    singularities = _solveset(den, symbol, domain)
                    result = _result - singularities
    if _check:
        if isinstance(result, ConditionSet):
            return result
        if isinstance(orig_f, Expr):
            fx = orig_f.as_independent(symbol, as_Add=True)[1]
            fx = fx.as_independent(symbol, as_Add=False)[1]
        else:
            fx = orig_f
        if isinstance(result, FiniteSet):
            result = FiniteSet(*[s for s in result if isinstance(s, RootOf) or domain_check(fx, symbol, s)])
    return result

def _is_modular(f, symbol):
    if False:
        i = 10
        return i + 15
    "\n    Helper function to check below mentioned types of modular equations.\n    ``A - Mod(B, C) = 0``\n\n    A -> This can or cannot be a function of symbol.\n    B -> This is surely a function of symbol.\n    C -> It is an integer.\n\n    Parameters\n    ==========\n\n    f : Expr\n        The equation to be checked.\n\n    symbol : Symbol\n        The concerned variable for which the equation is to be checked.\n\n    Examples\n    ========\n\n    >>> from sympy import symbols, exp, Mod\n    >>> from sympy.solvers.solveset import _is_modular as check\n    >>> x, y = symbols('x y')\n    >>> check(Mod(x, 3) - 1, x)\n    True\n    >>> check(Mod(x, 3) - 1, y)\n    False\n    >>> check(Mod(x, 3)**2 - 5, x)\n    False\n    >>> check(Mod(x, 3)**2 - y, x)\n    False\n    >>> check(exp(Mod(x, 3)) - 1, x)\n    False\n    >>> check(Mod(3, y) - 1, y)\n    False\n    "
    if not f.has(Mod):
        return False
    modterms = list(f.atoms(Mod))
    return len(modterms) == 1 and modterms[0].args[0].has(symbol) and modterms[0].args[1].is_integer and any((isinstance(term, Mod) for term in list(_term_factors(f))))

def _invert_modular(modterm, rhs, n, symbol):
    if False:
        for i in range(10):
            print('nop')
    "\n    Helper function to invert modular equation.\n    ``Mod(a, m) - rhs = 0``\n\n    Generally it is inverted as (a, ImageSet(Lambda(n, m*n + rhs), S.Integers)).\n    More simplified form will be returned if possible.\n\n    If it is not invertible then (modterm, rhs) is returned.\n\n    The following cases arise while inverting equation ``Mod(a, m) - rhs = 0``:\n\n    1. If a is symbol then  m*n + rhs is the required solution.\n\n    2. If a is an instance of ``Add`` then we try to find two symbol independent\n       parts of a and the symbol independent part gets transferred to the other\n       side and again the ``_invert_modular`` is called on the symbol\n       dependent part.\n\n    3. If a is an instance of ``Mul`` then same as we done in ``Add`` we separate\n       out the symbol dependent and symbol independent parts and transfer the\n       symbol independent part to the rhs with the help of invert and again the\n       ``_invert_modular`` is called on the symbol dependent part.\n\n    4. If a is an instance of ``Pow`` then two cases arise as following:\n\n        - If a is of type (symbol_indep)**(symbol_dep) then the remainder is\n          evaluated with the help of discrete_log function and then the least\n          period is being found out with the help of totient function.\n          period*n + remainder is the required solution in this case.\n          For reference: (https://en.wikipedia.org/wiki/Euler's_theorem)\n\n        - If a is of type (symbol_dep)**(symbol_indep) then we try to find all\n          primitive solutions list with the help of nthroot_mod function.\n          m*n + rem is the general solution where rem belongs to solutions list\n          from nthroot_mod function.\n\n    Parameters\n    ==========\n\n    modterm, rhs : Expr\n        The modular equation to be inverted, ``modterm - rhs = 0``\n\n    symbol : Symbol\n        The variable in the equation to be inverted.\n\n    n : Dummy\n        Dummy variable for output g_n.\n\n    Returns\n    =======\n\n    A tuple (f_x, g_n) is being returned where f_x is modular independent function\n    of symbol and g_n being set of values f_x can have.\n\n    Examples\n    ========\n\n    >>> from sympy import symbols, exp, Mod, Dummy, S\n    >>> from sympy.solvers.solveset import _invert_modular as invert_modular\n    >>> x, y = symbols('x y')\n    >>> n = Dummy('n')\n    >>> invert_modular(Mod(exp(x), 7), S(5), n, x)\n    (Mod(exp(x), 7), 5)\n    >>> invert_modular(Mod(x, 7), S(5), n, x)\n    (x, ImageSet(Lambda(_n, 7*_n + 5), Integers))\n    >>> invert_modular(Mod(3*x + 8, 7), S(5), n, x)\n    (x, ImageSet(Lambda(_n, 7*_n + 6), Integers))\n    >>> invert_modular(Mod(x**4, 7), S(5), n, x)\n    (x, EmptySet)\n    >>> invert_modular(Mod(2**(x**2 + x + 1), 7), S(2), n, x)\n    (x**2 + x + 1, ImageSet(Lambda(_n, 3*_n + 1), Naturals0))\n\n    "
    (a, m) = modterm.args
    if rhs.is_integer is False:
        return (symbol, S.EmptySet)
    if rhs.is_real is False or any((term.is_real is False for term in list(_term_factors(a)))):
        return (modterm, rhs)
    if abs(rhs) >= abs(m):
        return (symbol, S.EmptySet)
    if a == symbol:
        return (symbol, ImageSet(Lambda(n, m * n + rhs), S.Integers))
    if a.is_Add:
        (g, h) = a.as_independent(symbol)
        if g is not S.Zero:
            x_indep_term = rhs - Mod(g, m)
            return _invert_modular(Mod(h, m), Mod(x_indep_term, m), n, symbol)
    if a.is_Mul:
        (g, h) = a.as_independent(symbol)
        if g is not S.One:
            x_indep_term = rhs * invert(g, m)
            return _invert_modular(Mod(h, m), Mod(x_indep_term, m), n, symbol)
    if a.is_Pow:
        (base, expo) = a.args
        if expo.has(symbol) and (not base.has(symbol)):
            if not m.is_Integer and rhs.is_Integer and a.base.is_Integer:
                return (modterm, rhs)
            mdiv = m.p // number_gcd(m.p, rhs.p)
            try:
                remainder = discrete_log(mdiv, rhs.p, a.base.p)
            except ValueError:
                return (modterm, rhs)
            period = totient(m)
            for p in divisors(period):
                if pow(a.base, p, m / number_gcd(m.p, a.base.p)) == 1:
                    period = p
                    break
            return (expo, ImageSet(Lambda(n, period * n + remainder), S.Naturals0))
        elif base.has(symbol) and (not expo.has(symbol)):
            try:
                remainder_list = nthroot_mod(rhs, expo, m, all_roots=True)
                if remainder_list == []:
                    return (symbol, S.EmptySet)
            except (ValueError, NotImplementedError):
                return (modterm, rhs)
            g_n = S.EmptySet
            for rem in remainder_list:
                g_n += ImageSet(Lambda(n, m * n + rem), S.Integers)
            return (base, g_n)
    return (modterm, rhs)

def _solve_modular(f, symbol, domain):
    if False:
        i = 10
        return i + 15
    "\n    Helper function for solving modular equations of type ``A - Mod(B, C) = 0``,\n    where A can or cannot be a function of symbol, B is surely a function of\n    symbol and C is an integer.\n\n    Currently ``_solve_modular`` is only able to solve cases\n    where A is not a function of symbol.\n\n    Parameters\n    ==========\n\n    f : Expr\n        The modular equation to be solved, ``f = 0``\n\n    symbol : Symbol\n        The variable in the equation to be solved.\n\n    domain : Set\n        A set over which the equation is solved. It has to be a subset of\n        Integers.\n\n    Returns\n    =======\n\n    A set of integer solutions satisfying the given modular equation.\n    A ``ConditionSet`` if the equation is unsolvable.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.solveset import _solve_modular as solve_modulo\n    >>> from sympy import S, Symbol, sin, Intersection, Interval, Mod\n    >>> x = Symbol('x')\n    >>> solve_modulo(Mod(5*x - 8, 7) - 3, x, S.Integers)\n    ImageSet(Lambda(_n, 7*_n + 5), Integers)\n    >>> solve_modulo(Mod(5*x - 8, 7) - 3, x, S.Reals)  # domain should be subset of integers.\n    ConditionSet(x, Eq(Mod(5*x + 6, 7) - 3, 0), Reals)\n    >>> solve_modulo(-7 + Mod(x, 5), x, S.Integers)\n    EmptySet\n    >>> solve_modulo(Mod(12**x, 21) - 18, x, S.Integers)\n    ImageSet(Lambda(_n, 6*_n + 2), Naturals0)\n    >>> solve_modulo(Mod(sin(x), 7) - 3, x, S.Integers) # not solvable\n    ConditionSet(x, Eq(Mod(sin(x), 7) - 3, 0), Integers)\n    >>> solve_modulo(3 - Mod(x, 5), x, Intersection(S.Integers, Interval(0, 100)))\n    Intersection(ImageSet(Lambda(_n, 5*_n + 3), Integers), Range(0, 101, 1))\n    "
    unsolved_result = ConditionSet(symbol, Eq(f, 0), domain)
    modterm = list(f.atoms(Mod))[0]
    rhs = -S.One * f.subs(modterm, S.Zero)
    if f.as_coefficients_dict()[modterm].is_negative:
        rhs *= -S.One
    if not domain.is_subset(S.Integers):
        return unsolved_result
    if rhs.has(symbol):
        return unsolved_result
    n = Dummy('n', integer=True)
    (f_x, g_n) = _invert_modular(modterm, rhs, n, symbol)
    if f_x == modterm and g_n == rhs:
        return unsolved_result
    if f_x == symbol:
        if domain is not S.Integers:
            return domain.intersect(g_n)
        return g_n
    if isinstance(g_n, ImageSet):
        lamda_expr = g_n.lamda.expr
        lamda_vars = g_n.lamda.variables
        base_sets = g_n.base_sets
        sol_set = _solveset(f_x - lamda_expr, symbol, S.Integers)
        if isinstance(sol_set, FiniteSet):
            tmp_sol = S.EmptySet
            for sol in sol_set:
                tmp_sol += ImageSet(Lambda(lamda_vars, sol), *base_sets)
            sol_set = tmp_sol
        else:
            sol_set = ImageSet(Lambda(lamda_vars, sol_set), *base_sets)
        return domain.intersect(sol_set)
    return unsolved_result

def _term_factors(f):
    if False:
        return 10
    "\n    Iterator to get the factors of all terms present\n    in the given equation.\n\n    Parameters\n    ==========\n    f : Expr\n        Equation that needs to be addressed\n\n    Returns\n    =======\n    Factors of all terms present in the equation.\n\n    Examples\n    ========\n\n    >>> from sympy import symbols\n    >>> from sympy.solvers.solveset import _term_factors\n    >>> x = symbols('x')\n    >>> list(_term_factors(-2 - x**2 + x*(x + 1)))\n    [-2, -1, x**2, x, x + 1]\n    "
    for add_arg in Add.make_args(f):
        yield from Mul.make_args(add_arg)

def _solve_exponential(lhs, rhs, symbol, domain):
    if False:
        return 10
    "\n    Helper function for solving (supported) exponential equations.\n\n    Exponential equations are the sum of (currently) at most\n    two terms with one or both of them having a power with a\n    symbol-dependent exponent.\n\n    For example\n\n    .. math:: 5^{2x + 3} - 5^{3x - 1}\n\n    .. math:: 4^{5 - 9x} - e^{2 - x}\n\n    Parameters\n    ==========\n\n    lhs, rhs : Expr\n        The exponential equation to be solved, `lhs = rhs`\n\n    symbol : Symbol\n        The variable in which the equation is solved\n\n    domain : Set\n        A set over which the equation is solved.\n\n    Returns\n    =======\n\n    A set of solutions satisfying the given equation.\n    A ``ConditionSet`` if the equation is unsolvable or\n    if the assumptions are not properly defined, in that case\n    a different style of ``ConditionSet`` is returned having the\n    solution(s) of the equation with the desired assumptions.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.solveset import _solve_exponential as solve_expo\n    >>> from sympy import symbols, S\n    >>> x = symbols('x', real=True)\n    >>> a, b = symbols('a b')\n    >>> solve_expo(2**x + 3**x - 5**x, 0, x, S.Reals)  # not solvable\n    ConditionSet(x, Eq(2**x + 3**x - 5**x, 0), Reals)\n    >>> solve_expo(a**x - b**x, 0, x, S.Reals)  # solvable but incorrect assumptions\n    ConditionSet(x, (a > 0) & (b > 0), {0})\n    >>> solve_expo(3**(2*x) - 2**(x + 3), 0, x, S.Reals)\n    {-3*log(2)/(-2*log(3) + log(2))}\n    >>> solve_expo(2**x - 4**x, 0, x, S.Reals)\n    {0}\n\n    * Proof of correctness of the method\n\n    The logarithm function is the inverse of the exponential function.\n    The defining relation between exponentiation and logarithm is:\n\n    .. math:: {\\log_b x} = y \\enspace if \\enspace b^y = x\n\n    Therefore if we are given an equation with exponent terms, we can\n    convert every term to its corresponding logarithmic form. This is\n    achieved by taking logarithms and expanding the equation using\n    logarithmic identities so that it can easily be handled by ``solveset``.\n\n    For example:\n\n    .. math:: 3^{2x} = 2^{x + 3}\n\n    Taking log both sides will reduce the equation to\n\n    .. math:: (2x)\\log(3) = (x + 3)\\log(2)\n\n    This form can be easily handed by ``solveset``.\n    "
    unsolved_result = ConditionSet(symbol, Eq(lhs - rhs, 0), domain)
    newlhs = powdenest(lhs)
    if lhs != newlhs:
        neweq = factor(newlhs - rhs)
        if neweq != lhs - rhs:
            return _solveset(neweq, symbol, domain)
    if not (isinstance(lhs, Add) and len(lhs.args) == 2):
        return unsolved_result
    if rhs != 0:
        return unsolved_result
    (a, b) = list(ordered(lhs.args))
    a_term = a.as_independent(symbol)[1]
    b_term = b.as_independent(symbol)[1]
    (a_base, a_exp) = a_term.as_base_exp()
    (b_base, b_exp) = b_term.as_base_exp()
    if domain.is_subset(S.Reals):
        conditions = And(a_base > 0, b_base > 0, Eq(im(a_exp), 0), Eq(im(b_exp), 0))
    else:
        conditions = And(Ne(a_base, 0), Ne(b_base, 0))
    (L, R) = (expand_log(log(i), force=True) for i in (a, -b))
    solutions = _solveset(L - R, symbol, domain)
    return ConditionSet(symbol, conditions, solutions)

def _is_exponential(f, symbol):
    if False:
        while True:
            i = 10
    "\n    Return ``True`` if one or more terms contain ``symbol`` only in\n    exponents, else ``False``.\n\n    Parameters\n    ==========\n\n    f : Expr\n        The equation to be checked\n\n    symbol : Symbol\n        The variable in which the equation is checked\n\n    Examples\n    ========\n\n    >>> from sympy import symbols, cos, exp\n    >>> from sympy.solvers.solveset import _is_exponential as check\n    >>> x, y = symbols('x y')\n    >>> check(y, y)\n    False\n    >>> check(x**y - 1, y)\n    True\n    >>> check(x**y*2**y - 1, y)\n    True\n    >>> check(exp(x + 3) + 3**x, x)\n    True\n    >>> check(cos(2**x), x)\n    False\n\n    * Philosophy behind the helper\n\n    The function extracts each term of the equation and checks if it is\n    of exponential form w.r.t ``symbol``.\n    "
    rv = False
    for expr_arg in _term_factors(f):
        if symbol not in expr_arg.free_symbols:
            continue
        if isinstance(expr_arg, Pow) and symbol not in expr_arg.base.free_symbols or isinstance(expr_arg, exp):
            rv = True
        else:
            return False
    return rv

def _solve_logarithm(lhs, rhs, symbol, domain):
    if False:
        while True:
            i = 10
    "\n    Helper to solve logarithmic equations which are reducible\n    to a single instance of `\\log`.\n\n    Logarithmic equations are (currently) the equations that contains\n    `\\log` terms which can be reduced to a single `\\log` term or\n    a constant using various logarithmic identities.\n\n    For example:\n\n    .. math:: \\log(x) + \\log(x - 4)\n\n    can be reduced to:\n\n    .. math:: \\log(x(x - 4))\n\n    Parameters\n    ==========\n\n    lhs, rhs : Expr\n        The logarithmic equation to be solved, `lhs = rhs`\n\n    symbol : Symbol\n        The variable in which the equation is solved\n\n    domain : Set\n        A set over which the equation is solved.\n\n    Returns\n    =======\n\n    A set of solutions satisfying the given equation.\n    A ``ConditionSet`` if the equation is unsolvable.\n\n    Examples\n    ========\n\n    >>> from sympy import symbols, log, S\n    >>> from sympy.solvers.solveset import _solve_logarithm as solve_log\n    >>> x = symbols('x')\n    >>> f = log(x - 3) + log(x + 3)\n    >>> solve_log(f, 0, x, S.Reals)\n    {-sqrt(10), sqrt(10)}\n\n    * Proof of correctness\n\n    A logarithm is another way to write exponent and is defined by\n\n    .. math:: {\\log_b x} = y \\enspace if \\enspace b^y = x\n\n    When one side of the equation contains a single logarithm, the\n    equation can be solved by rewriting the equation as an equivalent\n    exponential equation as defined above. But if one side contains\n    more than one logarithm, we need to use the properties of logarithm\n    to condense it into a single logarithm.\n\n    Take for example\n\n    .. math:: \\log(2x) - 15 = 0\n\n    contains single logarithm, therefore we can directly rewrite it to\n    exponential form as\n\n    .. math:: x = \\frac{e^{15}}{2}\n\n    But if the equation has more than one logarithm as\n\n    .. math:: \\log(x - 3) + \\log(x + 3) = 0\n\n    we use logarithmic identities to convert it into a reduced form\n\n    Using,\n\n    .. math:: \\log(a) + \\log(b) = \\log(ab)\n\n    the equation becomes,\n\n    .. math:: \\log((x - 3)(x + 3))\n\n    This equation contains one logarithm and can be solved by rewriting\n    to exponents.\n    "
    new_lhs = logcombine(lhs, force=True)
    new_f = new_lhs - rhs
    return _solveset(new_f, symbol, domain)

def _is_logarithmic(f, symbol):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return ``True`` if the equation is in the form\n    `a\\log(f(x)) + b\\log(g(x)) + ... + c` else ``False``.\n\n    Parameters\n    ==========\n\n    f : Expr\n        The equation to be checked\n\n    symbol : Symbol\n        The variable in which the equation is checked\n\n    Returns\n    =======\n\n    ``True`` if the equation is logarithmic otherwise ``False``.\n\n    Examples\n    ========\n\n    >>> from sympy import symbols, tan, log\n    >>> from sympy.solvers.solveset import _is_logarithmic as check\n    >>> x, y = symbols('x y')\n    >>> check(log(x + 2) - log(x + 3), x)\n    True\n    >>> check(tan(log(2*x)), x)\n    False\n    >>> check(x*log(x), x)\n    False\n    >>> check(x + log(x), x)\n    False\n    >>> check(y + log(x), x)\n    True\n\n    * Philosophy behind the helper\n\n    The function extracts each term and checks whether it is\n    logarithmic w.r.t ``symbol``.\n    "
    rv = False
    for term in Add.make_args(f):
        saw_log = False
        for term_arg in Mul.make_args(term):
            if symbol not in term_arg.free_symbols:
                continue
            if isinstance(term_arg, log):
                if saw_log:
                    return False
                saw_log = True
            else:
                return False
        if saw_log:
            rv = True
    return rv

def _is_lambert(f, symbol):
    if False:
        i = 10
        return i + 15
    "\n    If this returns ``False`` then the Lambert solver (``_solve_lambert``) will not be called.\n\n    Explanation\n    ===========\n\n    Quick check for cases that the Lambert solver might be able to handle.\n\n    1. Equations containing more than two operands and `symbol`s involving any of\n       `Pow`, `exp`, `HyperbolicFunction`,`TrigonometricFunction`, `log` terms.\n\n    2. In `Pow`, `exp` the exponent should have `symbol` whereas for\n       `HyperbolicFunction`,`TrigonometricFunction`, `log` should contain `symbol`.\n\n    3. For `HyperbolicFunction`,`TrigonometricFunction` the number of trigonometric functions in\n       equation should be less than number of symbols. (since `A*cos(x) + B*sin(x) - c`\n       is not the Lambert type).\n\n    Some forms of lambert equations are:\n        1. X**X = C\n        2. X*(B*log(X) + D)**A = C\n        3. A*log(B*X + A) + d*X = C\n        4. (B*X + A)*exp(d*X + g) = C\n        5. g*exp(B*X + h) - B*X = C\n        6. A*D**(E*X + g) - B*X = C\n        7. A*cos(X) + B*sin(X) - D*X = C\n        8. A*cosh(X) + B*sinh(X) - D*X = C\n\n    Where X is any variable,\n          A, B, C, D, E are any constants,\n          g, h are linear functions or log terms.\n\n    Parameters\n    ==========\n\n    f : Expr\n        The equation to be checked\n\n    symbol : Symbol\n        The variable in which the equation is checked\n\n    Returns\n    =======\n\n    If this returns ``False`` then the Lambert solver (``_solve_lambert``) will not be called.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.solveset import _is_lambert\n    >>> from sympy import symbols, cosh, sinh, log\n    >>> x = symbols('x')\n\n    >>> _is_lambert(3*log(x) - x*log(3), x)\n    True\n    >>> _is_lambert(log(log(x - 3)) + log(x-3), x)\n    True\n    >>> _is_lambert(cosh(x) - sinh(x), x)\n    False\n    >>> _is_lambert((x**2 - 2*x + 1).subs(x, (log(x) + 3*x)**2 - 1), x)\n    True\n\n    See Also\n    ========\n\n    _solve_lambert\n\n    "
    term_factors = list(_term_factors(f.expand()))
    no_of_symbols = len([arg for arg in term_factors if arg.has(symbol)])
    no_of_trig = len([arg for arg in term_factors if arg.has(HyperbolicFunction, TrigonometricFunction)])
    if f.is_Add and no_of_symbols >= 2:
        lambert_funcs = (log, HyperbolicFunction, TrigonometricFunction)
        if any((isinstance(arg, lambert_funcs) for arg in term_factors if arg.has(symbol))):
            if no_of_trig < no_of_symbols:
                return True
        elif any((isinstance(arg, (Pow, exp)) for arg in term_factors if arg.as_base_exp()[1].has(symbol))):
            return True
    return False

def _transolve(f, symbol, domain):
    if False:
        print('Hello World!')
    "\n    Function to solve transcendental equations. It is a helper to\n    ``solveset`` and should be used internally. ``_transolve``\n    currently supports the following class of equations:\n\n        - Exponential equations\n        - Logarithmic equations\n\n    Parameters\n    ==========\n\n    f : Any transcendental equation that needs to be solved.\n        This needs to be an expression, which is assumed\n        to be equal to ``0``.\n\n    symbol : The variable for which the equation is solved.\n        This needs to be of class ``Symbol``.\n\n    domain : A set over which the equation is solved.\n        This needs to be of class ``Set``.\n\n    Returns\n    =======\n\n    Set\n        A set of values for ``symbol`` for which ``f`` is equal to\n        zero. An ``EmptySet`` is returned if ``f`` does not have solutions\n        in respective domain. A ``ConditionSet`` is returned as unsolved\n        object if algorithms to evaluate complete solution are not\n        yet implemented.\n\n    How to use ``_transolve``\n    =========================\n\n    ``_transolve`` should not be used as an independent function, because\n    it assumes that the equation (``f``) and the ``symbol`` comes from\n    ``solveset`` and might have undergone a few modification(s).\n    To use ``_transolve`` as an independent function the equation (``f``)\n    and the ``symbol`` should be passed as they would have been by\n    ``solveset``.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.solveset import _transolve as transolve\n    >>> from sympy.solvers.solvers import _tsolve as tsolve\n    >>> from sympy import symbols, S, pprint\n    >>> x = symbols('x', real=True) # assumption added\n    >>> transolve(5**(x - 3) - 3**(2*x + 1), x, S.Reals)\n    {-(log(3) + 3*log(5))/(-log(5) + 2*log(3))}\n\n    How ``_transolve`` works\n    ========================\n\n    ``_transolve`` uses two types of helper functions to solve equations\n    of a particular class:\n\n    Identifying helpers: To determine whether a given equation\n    belongs to a certain class of equation or not. Returns either\n    ``True`` or ``False``.\n\n    Solving helpers: Once an equation is identified, a corresponding\n    helper either solves the equation or returns a form of the equation\n    that ``solveset`` might better be able to handle.\n\n    * Philosophy behind the module\n\n    The purpose of ``_transolve`` is to take equations which are not\n    already polynomial in their generator(s) and to either recast them\n    as such through a valid transformation or to solve them outright.\n    A pair of helper functions for each class of supported\n    transcendental functions are employed for this purpose. One\n    identifies the transcendental form of an equation and the other\n    either solves it or recasts it into a tractable form that can be\n    solved by  ``solveset``.\n    For example, an equation in the form `ab^{f(x)} - cd^{g(x)} = 0`\n    can be transformed to\n    `\\log(a) + f(x)\\log(b) - \\log(c) - g(x)\\log(d) = 0`\n    (under certain assumptions) and this can be solved with ``solveset``\n    if `f(x)` and `g(x)` are in polynomial form.\n\n    How ``_transolve`` is better than ``_tsolve``\n    =============================================\n\n    1) Better output\n\n    ``_transolve`` provides expressions in a more simplified form.\n\n    Consider a simple exponential equation\n\n    >>> f = 3**(2*x) - 2**(x + 3)\n    >>> pprint(transolve(f, x, S.Reals), use_unicode=False)\n        -3*log(2)\n    {------------------}\n     -2*log(3) + log(2)\n    >>> pprint(tsolve(f, x), use_unicode=False)\n         /   3     \\\n         | --------|\n         | log(2/9)|\n    [-log\\2         /]\n\n    2) Extensible\n\n    The API of ``_transolve`` is designed such that it is easily\n    extensible, i.e. the code that solves a given class of\n    equations is encapsulated in a helper and not mixed in with\n    the code of ``_transolve`` itself.\n\n    3) Modular\n\n    ``_transolve`` is designed to be modular i.e, for every class of\n    equation a separate helper for identification and solving is\n    implemented. This makes it easy to change or modify any of the\n    method implemented directly in the helpers without interfering\n    with the actual structure of the API.\n\n    4) Faster Computation\n\n    Solving equation via ``_transolve`` is much faster as compared to\n    ``_tsolve``. In ``solve``, attempts are made computing every possibility\n    to get the solutions. This series of attempts makes solving a bit\n    slow. In ``_transolve``, computation begins only after a particular\n    type of equation is identified.\n\n    How to add new class of equations\n    =================================\n\n    Adding a new class of equation solver is a three-step procedure:\n\n    - Identify the type of the equations\n\n      Determine the type of the class of equations to which they belong:\n      it could be of ``Add``, ``Pow``, etc. types. Separate internal functions\n      are used for each type. Write identification and solving helpers\n      and use them from within the routine for the given type of equation\n      (after adding it, if necessary). Something like:\n\n      .. code-block:: python\n\n        def add_type(lhs, rhs, x):\n            ....\n            if _is_exponential(lhs, x):\n                new_eq = _solve_exponential(lhs, rhs, x)\n        ....\n        rhs, lhs = eq.as_independent(x)\n        if lhs.is_Add:\n            result = add_type(lhs, rhs, x)\n\n    - Define the identification helper.\n\n    - Define the solving helper.\n\n    Apart from this, a few other things needs to be taken care while\n    adding an equation solver:\n\n    - Naming conventions:\n      Name of the identification helper should be as\n      ``_is_class`` where class will be the name or abbreviation\n      of the class of equation. The solving helper will be named as\n      ``_solve_class``.\n      For example: for exponential equations it becomes\n      ``_is_exponential`` and ``_solve_expo``.\n    - The identifying helpers should take two input parameters,\n      the equation to be checked and the variable for which a solution\n      is being sought, while solving helpers would require an additional\n      domain parameter.\n    - Be sure to consider corner cases.\n    - Add tests for each helper.\n    - Add a docstring to your helper that describes the method\n      implemented.\n      The documentation of the helpers should identify:\n\n      - the purpose of the helper,\n      - the method used to identify and solve the equation,\n      - a proof of correctness\n      - the return values of the helpers\n    "

    def add_type(lhs, rhs, symbol, domain):
        if False:
            print('Hello World!')
        '\n        Helper for ``_transolve`` to handle equations of\n        ``Add`` type, i.e. equations taking the form as\n        ``a*f(x) + b*g(x) + .... = c``.\n        For example: 4**x + 8**x = 0\n        '
        result = ConditionSet(symbol, Eq(lhs - rhs, 0), domain)
        if _is_exponential(lhs, symbol):
            result = _solve_exponential(lhs, rhs, symbol, domain)
        elif _is_logarithmic(lhs, symbol):
            result = _solve_logarithm(lhs, rhs, symbol, domain)
        return result
    result = ConditionSet(symbol, Eq(f, 0), domain)
    (lhs, rhs_s) = invert_complex(f, 0, symbol, domain)
    if isinstance(rhs_s, FiniteSet):
        assert len(rhs_s.args) == 1
        rhs = rhs_s.args[0]
        if lhs.is_Add:
            result = add_type(lhs, rhs, symbol, domain)
    else:
        result = rhs_s
    return result

def solveset(f, symbol=None, domain=S.Complexes):
    if False:
        print('Hello World!')
    "Solves a given inequality or equation with set as output\n\n    Parameters\n    ==========\n\n    f : Expr or a relational.\n        The target equation or inequality\n    symbol : Symbol\n        The variable for which the equation is solved\n    domain : Set\n        The domain over which the equation is solved\n\n    Returns\n    =======\n\n    Set\n        A set of values for `symbol` for which `f` is True or is equal to\n        zero. An :class:`~.EmptySet` is returned if `f` is False or nonzero.\n        A :class:`~.ConditionSet` is returned as unsolved object if algorithms\n        to evaluate complete solution are not yet implemented.\n\n    ``solveset`` claims to be complete in the solution set that it returns.\n\n    Raises\n    ======\n\n    NotImplementedError\n        The algorithms to solve inequalities in complex domain  are\n        not yet implemented.\n    ValueError\n        The input is not valid.\n    RuntimeError\n        It is a bug, please report to the github issue tracker.\n\n\n    Notes\n    =====\n\n    Python interprets 0 and 1 as False and True, respectively, but\n    in this function they refer to solutions of an expression. So 0 and 1\n    return the domain and EmptySet, respectively, while True and False\n    return the opposite (as they are assumed to be solutions of relational\n    expressions).\n\n\n    See Also\n    ========\n\n    solveset_real: solver for real domain\n    solveset_complex: solver for complex domain\n\n    Examples\n    ========\n\n    >>> from sympy import exp, sin, Symbol, pprint, S, Eq\n    >>> from sympy.solvers.solveset import solveset, solveset_real\n\n    * The default domain is complex. Not specifying a domain will lead\n      to the solving of the equation in the complex domain (and this\n      is not affected by the assumptions on the symbol):\n\n    >>> x = Symbol('x')\n    >>> pprint(solveset(exp(x) - 1, x), use_unicode=False)\n    {2*n*I*pi | n in Integers}\n\n    >>> x = Symbol('x', real=True)\n    >>> pprint(solveset(exp(x) - 1, x), use_unicode=False)\n    {2*n*I*pi | n in Integers}\n\n    * If you want to use ``solveset`` to solve the equation in the\n      real domain, provide a real domain. (Using ``solveset_real``\n      does this automatically.)\n\n    >>> R = S.Reals\n    >>> x = Symbol('x')\n    >>> solveset(exp(x) - 1, x, R)\n    {0}\n    >>> solveset_real(exp(x) - 1, x)\n    {0}\n\n    The solution is unaffected by assumptions on the symbol:\n\n    >>> p = Symbol('p', positive=True)\n    >>> pprint(solveset(p**2 - 4))\n    {-2, 2}\n\n    When a :class:`~.ConditionSet` is returned, symbols with assumptions that\n    would alter the set are replaced with more generic symbols:\n\n    >>> i = Symbol('i', imaginary=True)\n    >>> solveset(Eq(i**2 + i*sin(i), 1), i, domain=S.Reals)\n    ConditionSet(_R, Eq(_R**2 + _R*sin(_R) - 1, 0), Reals)\n\n    * Inequalities can be solved over the real domain only. Use of a complex\n      domain leads to a NotImplementedError.\n\n    >>> solveset(exp(x) > 1, x, R)\n    Interval.open(0, oo)\n\n    "
    f = sympify(f)
    symbol = sympify(symbol)
    if f is S.true:
        return domain
    if f is S.false:
        return S.EmptySet
    if not isinstance(f, (Expr, Relational, Number)):
        raise ValueError('%s is not a valid SymPy expression' % f)
    if not isinstance(symbol, (Expr, Relational)) and symbol is not None:
        raise ValueError('%s is not a valid SymPy symbol' % (symbol,))
    if not isinstance(domain, Set):
        raise ValueError('%s is not a valid domain' % domain)
    free_symbols = f.free_symbols
    if f.has(Piecewise):
        f = piecewise_fold(f)
    if symbol is None and (not free_symbols):
        b = Eq(f, 0)
        if b is S.true:
            return domain
        elif b is S.false:
            return S.EmptySet
        else:
            raise NotImplementedError(filldedent('\n                relationship between value and 0 is unknown: %s' % b))
    if symbol is None:
        if len(free_symbols) == 1:
            symbol = free_symbols.pop()
        elif free_symbols:
            raise ValueError(filldedent('\n                The independent variable must be specified for a\n                multivariate equation.'))
    elif not isinstance(symbol, Symbol):
        (f, s, swap) = recast_to_symbols([f], [symbol])
        return solveset(f[0], s[0], domain).xreplace(swap)
    if symbol not in _rc:
        x = _rc[0] if domain.is_subset(S.Reals) else _rc[1]
        rv = solveset(f.xreplace({symbol: x}), x, domain)
        try:
            _rv = rv.xreplace({x: symbol})
        except TypeError:
            _rv = rv
        if rv.dummy_eq(_rv):
            rv = _rv
        return rv
    (f, mask) = _masked(f, Abs)
    f = f.rewrite(Piecewise)
    for (d, e) in mask:
        e = e.func(e.args[0].rewrite(Piecewise))
        f = f.xreplace({d: e})
    f = piecewise_fold(f)
    return _solveset(f, symbol, domain, _check=True)

def solveset_real(f, symbol):
    if False:
        print('Hello World!')
    return solveset(f, symbol, S.Reals)

def solveset_complex(f, symbol):
    if False:
        return 10
    return solveset(f, symbol, S.Complexes)

def _solveset_multi(eqs, syms, domains):
    if False:
        return 10
    'Basic implementation of a multivariate solveset.\n\n    For internal use (not ready for public consumption)'
    rep = {}
    for (sym, dom) in zip(syms, domains):
        if dom is S.Reals:
            rep[sym] = Symbol(sym.name, real=True)
    eqs = [eq.subs(rep) for eq in eqs]
    syms = [sym.subs(rep) for sym in syms]
    syms = tuple(syms)
    if len(eqs) == 0:
        return ProductSet(*domains)
    if len(syms) == 1:
        sym = syms[0]
        domain = domains[0]
        solsets = [solveset(eq, sym, domain) for eq in eqs]
        solset = Intersection(*solsets)
        return ImageSet(Lambda((sym,), (sym,)), solset).doit()
    eqs = sorted(eqs, key=lambda eq: len(eq.free_symbols & set(syms)))
    for (n, eq) in enumerate(eqs):
        sols = []
        all_handled = True
        for sym in syms:
            if sym not in eq.free_symbols:
                continue
            sol = solveset(eq, sym, domains[syms.index(sym)])
            if isinstance(sol, FiniteSet):
                i = syms.index(sym)
                symsp = syms[:i] + syms[i + 1:]
                domainsp = domains[:i] + domains[i + 1:]
                eqsp = eqs[:n] + eqs[n + 1:]
                for s in sol:
                    eqsp_sub = [eq.subs(sym, s) for eq in eqsp]
                    sol_others = _solveset_multi(eqsp_sub, symsp, domainsp)
                    fun = Lambda((symsp,), symsp[:i] + (s,) + symsp[i:])
                    sols.append(ImageSet(fun, sol_others).doit())
            else:
                all_handled = False
        if all_handled:
            return Union(*sols)

def solvify(f, symbol, domain):
    if False:
        i = 10
        return i + 15
    'Solves an equation using solveset and returns the solution in accordance\n    with the `solve` output API.\n\n    Returns\n    =======\n\n    We classify the output based on the type of solution returned by `solveset`.\n\n    Solution    |    Output\n    ----------------------------------------\n    FiniteSet   | list\n\n    ImageSet,   | list (if `f` is periodic)\n    Union       |\n\n    Union       | list (with FiniteSet)\n\n    EmptySet    | empty list\n\n    Others      | None\n\n\n    Raises\n    ======\n\n    NotImplementedError\n        A ConditionSet is the input.\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.solveset import solvify\n    >>> from sympy.abc import x\n    >>> from sympy import S, tan, sin, exp\n    >>> solvify(x**2 - 9, x, S.Reals)\n    [-3, 3]\n    >>> solvify(sin(x) - 1, x, S.Reals)\n    [pi/2]\n    >>> solvify(tan(x), x, S.Reals)\n    [0]\n    >>> solvify(exp(x) - 1, x, S.Complexes)\n\n    >>> solvify(exp(x) - 1, x, S.Reals)\n    [0]\n\n    '
    solution_set = solveset(f, symbol, domain)
    result = None
    if solution_set is S.EmptySet:
        result = []
    elif isinstance(solution_set, ConditionSet):
        raise NotImplementedError('solveset is unable to solve this equation.')
    elif isinstance(solution_set, FiniteSet):
        result = list(solution_set)
    else:
        period = periodicity(f, symbol)
        if period is not None:
            solutions = S.EmptySet
            iter_solutions = ()
            if isinstance(solution_set, ImageSet):
                iter_solutions = (solution_set,)
            elif isinstance(solution_set, Union):
                if all((isinstance(i, ImageSet) for i in solution_set.args)):
                    iter_solutions = solution_set.args
            for solution in iter_solutions:
                solutions += solution.intersect(Interval(0, period, False, True))
            if isinstance(solutions, FiniteSet):
                result = list(solutions)
        else:
            solution = solution_set.intersect(domain)
            if isinstance(solution, Union):
                if any((isinstance(i, FiniteSet) for i in solution.args)):
                    result = [sol for soln in solution.args for sol in soln.args if isinstance(soln, FiniteSet)]
                else:
                    return None
            elif isinstance(solution, FiniteSet):
                result += solution
    return result

def linear_coeffs(eq, *syms, dict=False):
    if False:
        return 10
    "Return a list whose elements are the coefficients of the\n    corresponding symbols in the sum of terms in  ``eq``.\n    The additive constant is returned as the last element of the\n    list.\n\n    Raises\n    ======\n\n    NonlinearError\n        The equation contains a nonlinear term\n    ValueError\n        duplicate or unordered symbols are passed\n\n    Parameters\n    ==========\n\n    dict - (default False) when True, return coefficients as a\n        dictionary with coefficients keyed to syms that were present;\n        key 1 gives the constant term\n\n    Examples\n    ========\n\n    >>> from sympy.solvers.solveset import linear_coeffs\n    >>> from sympy.abc import x, y, z\n    >>> linear_coeffs(3*x + 2*y - 1, x, y)\n    [3, 2, -1]\n\n    It is not necessary to expand the expression:\n\n        >>> linear_coeffs(x + y*(z*(x*3 + 2) + 3), x)\n        [3*y*z + 1, y*(2*z + 3)]\n\n    When nonlinear is detected, an error will be raised:\n\n        * even if they would cancel after expansion (so the\n        situation does not pass silently past the caller's\n        attention)\n\n        >>> eq = 1/x*(x - 1) + 1/x\n        >>> linear_coeffs(eq.expand(), x)\n        [0, 1]\n        >>> linear_coeffs(eq, x)\n        Traceback (most recent call last):\n        ...\n        NonlinearError:\n        nonlinear in given generators\n\n        * when there are cross terms\n\n        >>> linear_coeffs(x*(y + 1), x, y)\n        Traceback (most recent call last):\n        ...\n        NonlinearError:\n        symbol-dependent cross-terms encountered\n\n        * when there are terms that contain an expression\n        dependent on the symbols that is not linear\n\n        >>> linear_coeffs(x**2, x)\n        Traceback (most recent call last):\n        ...\n        NonlinearError:\n        nonlinear in given generators\n    "
    eq = _sympify(eq)
    if len(syms) == 1 and iterable(syms[0]) and (not isinstance(syms[0], Basic)):
        raise ValueError('expecting unpacked symbols, *syms')
    symset = set(syms)
    if len(symset) != len(syms):
        raise ValueError('duplicate symbols given')
    try:
        (d, c) = _linear_eq_to_dict([eq], symset)
        d = d[0]
        c = c[0]
    except PolyNonlinearError as err:
        raise NonlinearError(str(err))
    if dict:
        if c:
            d[S.One] = c
        return d
    rv = [S.Zero] * (len(syms) + 1)
    rv[-1] = c
    for (i, k) in enumerate(syms):
        if k not in d:
            continue
        rv[i] = d[k]
    return rv

def linear_eq_to_matrix(equations, *symbols):
    if False:
        return 10
    "\n    Converts a given System of Equations into Matrix form.\n    Here `equations` must be a linear system of equations in\n    `symbols`. Element ``M[i, j]`` corresponds to the coefficient\n    of the jth symbol in the ith equation.\n\n    The Matrix form corresponds to the augmented matrix form.\n    For example:\n\n    .. math:: 4x + 2y + 3z  = 1\n    .. math:: 3x +  y +  z  = -6\n    .. math:: 2x + 4y + 9z  = 2\n\n    This system will return $A$ and $b$ as:\n\n    $$ A = \\left[\\begin{array}{ccc}\n        4 & 2 & 3 \\\\\n        3 & 1 & 1 \\\\\n        2 & 4 & 9\n        \\end{array}\\right] \\ \\  b = \\left[\\begin{array}{c}\n        1 \\\\ -6 \\\\ 2\n        \\end{array}\\right] $$\n\n    The only simplification performed is to convert\n    ``Eq(a, b)`` $\\Rightarrow a - b$.\n\n    Raises\n    ======\n\n    NonlinearError\n        The equations contain a nonlinear term.\n    ValueError\n        The symbols are not given or are not unique.\n\n    Examples\n    ========\n\n    >>> from sympy import linear_eq_to_matrix, symbols\n    >>> c, x, y, z = symbols('c, x, y, z')\n\n    The coefficients (numerical or symbolic) of the symbols will\n    be returned as matrices:\n\n        >>> eqns = [c*x + z - 1 - c, y + z, x - y]\n        >>> A, b = linear_eq_to_matrix(eqns, [x, y, z])\n        >>> A\n        Matrix([\n        [c,  0, 1],\n        [0,  1, 1],\n        [1, -1, 0]])\n        >>> b\n        Matrix([\n        [c + 1],\n        [    0],\n        [    0]])\n\n    This routine does not simplify expressions and will raise an error\n    if nonlinearity is encountered:\n\n            >>> eqns = [\n            ...     (x**2 - 3*x)/(x - 3) - 3,\n            ...     y**2 - 3*y - y*(y - 4) + x - 4]\n            >>> linear_eq_to_matrix(eqns, [x, y])\n            Traceback (most recent call last):\n            ...\n            NonlinearError:\n            symbol-dependent term can be ignored using `strict=False`\n\n        Simplifying these equations will discard the removable singularity\n        in the first and reveal the linear structure of the second:\n\n            >>> [e.simplify() for e in eqns]\n            [x - 3, x + y - 4]\n\n        Any such simplification needed to eliminate nonlinear terms must\n        be done *before* calling this routine.\n    "
    if not symbols:
        raise ValueError(filldedent('\n            Symbols must be given, for which coefficients\n            are to be found.\n            '))
    if isinstance(symbols[0], set):
        raise TypeError("Unordered 'set' type is not supported as input for symbols.")
    if hasattr(symbols[0], '__iter__'):
        symbols = symbols[0]
    if has_dups(symbols):
        raise ValueError('Symbols must be unique')
    equations = sympify(equations)
    if isinstance(equations, MatrixBase):
        equations = list(equations)
    elif isinstance(equations, (Expr, Eq)):
        equations = [equations]
    elif not is_sequence(equations):
        raise ValueError(filldedent('\n            Equation(s) must be given as a sequence, Expr,\n            Eq or Matrix.\n            '))
    try:
        (eq, c) = _linear_eq_to_dict(equations, symbols)
    except PolyNonlinearError as err:
        raise NonlinearError(str(err))
    (n, m) = shape = (len(eq), len(symbols))
    ix = dict(zip(symbols, range(m)))
    A = zeros(*shape)
    for (row, d) in enumerate(eq):
        for k in d:
            col = ix[k]
            A[row, col] = d[k]
    b = Matrix(n, 1, [-i for i in c])
    return (A, b)

def linsolve(system, *symbols):
    if False:
        i = 10
        return i + 15
    '\n    Solve system of $N$ linear equations with $M$ variables; both\n    underdetermined and overdetermined systems are supported.\n    The possible number of solutions is zero, one or infinite.\n    Zero solutions throws a ValueError, whereas infinite\n    solutions are represented parametrically in terms of the given\n    symbols. For unique solution a :class:`~.FiniteSet` of ordered tuples\n    is returned.\n\n    All standard input formats are supported:\n    For the given set of equations, the respective input types\n    are given below:\n\n    .. math:: 3x + 2y -   z = 1\n    .. math:: 2x - 2y + 4z = -2\n    .. math:: 2x -   y + 2z = 0\n\n    * Augmented matrix form, ``system`` given below:\n\n    $$ \\text{system} = \\left[{array}{cccc}\n        3 &  2 & -1 &  1\\\\\n        2 & -2 &  4 & -2\\\\\n        2 & -1 &  2 &  0\n        \\end{array}\\right] $$\n\n    ::\n\n        system = Matrix([[3, 2, -1, 1], [2, -2, 4, -2], [2, -1, 2, 0]])\n\n    * List of equations form\n\n    ::\n\n        system  =  [3x + 2y - z - 1, 2x - 2y + 4z + 2, 2x - y + 2z]\n\n    * Input $A$ and $b$ in matrix form (from $Ax = b$) are given as:\n\n    $$ A = \\left[\\begin{array}{ccc}\n        3 &  2 & -1 \\\\\n        2 & -2 &  4 \\\\\n        2 & -1 &  2\n        \\end{array}\\right] \\ \\  b = \\left[\\begin{array}{c}\n        1 \\\\ -2 \\\\ 0\n        \\end{array}\\right] $$\n\n    ::\n\n        A = Matrix([[3, 2, -1], [2, -2, 4], [2, -1, 2]])\n        b = Matrix([[1], [-2], [0]])\n        system = (A, b)\n\n    Symbols can always be passed but are actually only needed\n    when 1) a system of equations is being passed and 2) the\n    system is passed as an underdetermined matrix and one wants\n    to control the name of the free variables in the result.\n    An error is raised if no symbols are used for case 1, but if\n    no symbols are provided for case 2, internally generated symbols\n    will be provided. When providing symbols for case 2, there should\n    be at least as many symbols are there are columns in matrix A.\n\n    The algorithm used here is Gauss-Jordan elimination, which\n    results, after elimination, in a row echelon form matrix.\n\n    Returns\n    =======\n\n    A FiniteSet containing an ordered tuple of values for the\n    unknowns for which the `system` has a solution. (Wrapping\n    the tuple in FiniteSet is used to maintain a consistent\n    output format throughout solveset.)\n\n    Returns EmptySet, if the linear system is inconsistent.\n\n    Raises\n    ======\n\n    ValueError\n        The input is not valid.\n        The symbols are not given.\n\n    Examples\n    ========\n\n    >>> from sympy import Matrix, linsolve, symbols\n    >>> x, y, z = symbols("x, y, z")\n    >>> A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]])\n    >>> b = Matrix([3, 6, 9])\n    >>> A\n    Matrix([\n    [1, 2,  3],\n    [4, 5,  6],\n    [7, 8, 10]])\n    >>> b\n    Matrix([\n    [3],\n    [6],\n    [9]])\n    >>> linsolve((A, b), [x, y, z])\n    {(-1, 2, 0)}\n\n    * Parametric Solution: In case the system is underdetermined, the\n      function will return a parametric solution in terms of the given\n      symbols. Those that are free will be returned unchanged. e.g. in\n      the system below, `z` is returned as the solution for variable z;\n      it can take on any value.\n\n    >>> A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])\n    >>> b = Matrix([3, 6, 9])\n    >>> linsolve((A, b), x, y, z)\n    {(z - 1, 2 - 2*z, z)}\n\n    If no symbols are given, internally generated symbols will be used.\n    The ``tau0`` in the third position indicates (as before) that the third\n    variable -- whatever it is named -- can take on any value:\n\n    >>> linsolve((A, b))\n    {(tau0 - 1, 2 - 2*tau0, tau0)}\n\n    * List of equations as input\n\n    >>> Eqns = [3*x + 2*y - z - 1, 2*x - 2*y + 4*z + 2, - x + y/2 - z]\n    >>> linsolve(Eqns, x, y, z)\n    {(1, -2, -2)}\n\n    * Augmented matrix as input\n\n    >>> aug = Matrix([[2, 1, 3, 1], [2, 6, 8, 3], [6, 8, 18, 5]])\n    >>> aug\n    Matrix([\n    [2, 1,  3, 1],\n    [2, 6,  8, 3],\n    [6, 8, 18, 5]])\n    >>> linsolve(aug, x, y, z)\n    {(3/10, 2/5, 0)}\n\n    * Solve for symbolic coefficients\n\n    >>> a, b, c, d, e, f = symbols(\'a, b, c, d, e, f\')\n    >>> eqns = [a*x + b*y - c, d*x + e*y - f]\n    >>> linsolve(eqns, x, y)\n    {((-b*f + c*e)/(a*e - b*d), (a*f - c*d)/(a*e - b*d))}\n\n    * A degenerate system returns solution as set of given\n      symbols.\n\n    >>> system = Matrix(([0, 0, 0], [0, 0, 0], [0, 0, 0]))\n    >>> linsolve(system, x, y)\n    {(x, y)}\n\n    * For an empty system linsolve returns empty set\n\n    >>> linsolve([], x)\n    EmptySet\n\n    * An error is raised if any nonlinearity is detected, even\n      if it could be removed with expansion\n\n    >>> linsolve([x*(1/x - 1)], x)\n    Traceback (most recent call last):\n    ...\n    NonlinearError: nonlinear term: 1/x\n\n    >>> linsolve([x*(y + 1)], x, y)\n    Traceback (most recent call last):\n    ...\n    NonlinearError: nonlinear cross-term: x*(y + 1)\n\n    >>> linsolve([x**2 - 1], x)\n    Traceback (most recent call last):\n    ...\n    NonlinearError: nonlinear term: x**2\n    '
    if not system:
        return S.EmptySet
    if symbols and hasattr(symbols[0], '__iter__'):
        symbols = symbols[0]
    sym_gen = isinstance(symbols, GeneratorType)
    dup_msg = 'duplicate symbols given'
    b = None
    if hasattr(system, '__iter__'):
        if len(system) == 2 and isinstance(system[0], MatrixBase):
            (A, b) = system
        if not isinstance(system[0], MatrixBase):
            if sym_gen or not symbols:
                raise ValueError(filldedent('\n                    When passing a system of equations, the explicit\n                    symbols for which a solution is being sought must\n                    be given as a sequence, too.\n                '))
            if len(set(symbols)) != len(symbols):
                raise ValueError(dup_msg)
            eqs = system
            eqs = [sympify(eq) for eq in eqs]
            try:
                sol = _linsolve(eqs, symbols)
            except PolyNonlinearError as exc:
                raise NonlinearError(str(exc))
            if sol is None:
                return S.EmptySet
            sol = FiniteSet(Tuple(*(sol.get(sym, sym) for sym in symbols)))
            return sol
    elif isinstance(system, MatrixBase) and (not (symbols and (not isinstance(symbols, GeneratorType)) and isinstance(symbols[0], MatrixBase))):
        (A, b) = (system[:, :-1], system[:, -1:])
    if b is None:
        raise ValueError('Invalid arguments')
    if sym_gen:
        symbols = [next(symbols) for i in range(A.cols)]
        symset = set(symbols)
        if any(symset & (A.free_symbols | b.free_symbols)):
            raise ValueError(filldedent("\n                At least one of the symbols provided\n                already appears in the system to be solved.\n                One way to avoid this is to use Dummy symbols in\n                the generator, e.g. numbered_symbols('%s', cls=Dummy)\n            " % symbols[0].name.rstrip('1234567890')))
        elif len(symset) != len(symbols):
            raise ValueError(dup_msg)
    if not symbols:
        symbols = [Dummy() for _ in range(A.cols)]
        name = _uniquely_named_symbol('tau', (A, b), compare=lambda i: str(i).rstrip('1234567890')).name
        gen = numbered_symbols(name)
    else:
        gen = None
    eqs = []
    rows = A.tolist()
    for (rowi, bi) in zip(rows, b):
        terms = [elem * sym for (elem, sym) in zip(rowi, symbols) if elem]
        terms.append(-bi)
        eqs.append(Add(*terms))
    (eqs, ring) = sympy_eqs_to_ring(eqs, symbols)
    sol = solve_lin_sys(eqs, ring, _raw=False)
    if sol is None:
        return S.EmptySet
    sol = FiniteSet(Tuple(*(sol.get(sym, sym) for sym in symbols)))
    if gen is not None:
        solsym = sol.free_symbols
        rep = {sym: next(gen) for sym in symbols if sym in solsym}
        sol = sol.subs(rep)
    return sol

def _return_conditionset(eqs, symbols):
    if False:
        for i in range(10):
            print('nop')
    eqs = (Eq(lhs, 0) for lhs in eqs)
    condition_set = ConditionSet(Tuple(*symbols), And(*eqs), S.Complexes ** len(symbols))
    return condition_set

def substitution(system, symbols, result=[{}], known_symbols=[], exclude=[], all_symbols=None):
    if False:
        while True:
            i = 10
    "\n    Solves the `system` using substitution method. It is used in\n    :func:`~.nonlinsolve`. This will be called from :func:`~.nonlinsolve` when any\n    equation(s) is non polynomial equation.\n\n    Parameters\n    ==========\n\n    system : list of equations\n        The target system of equations\n    symbols : list of symbols to be solved.\n        The variable(s) for which the system is solved\n    known_symbols : list of solved symbols\n        Values are known for these variable(s)\n    result : An empty list or list of dict\n        If No symbol values is known then empty list otherwise\n        symbol as keys and corresponding value in dict.\n    exclude : Set of expression.\n        Mostly denominator expression(s) of the equations of the system.\n        Final solution should not satisfy these expressions.\n    all_symbols : known_symbols + symbols(unsolved).\n\n    Returns\n    =======\n\n    A FiniteSet of ordered tuple of values of `all_symbols` for which the\n    `system` has solution. Order of values in the tuple is same as symbols\n    present in the parameter `all_symbols`. If parameter `all_symbols` is None\n    then same as symbols present in the parameter `symbols`.\n\n    Please note that general FiniteSet is unordered, the solution returned\n    here is not simply a FiniteSet of solutions, rather it is a FiniteSet of\n    ordered tuple, i.e. the first & only argument to FiniteSet is a tuple of\n    solutions, which is ordered, & hence the returned solution is ordered.\n\n    Also note that solution could also have been returned as an ordered tuple,\n    FiniteSet is just a wrapper `{}` around the tuple. It has no other\n    significance except for the fact it is just used to maintain a consistent\n    output format throughout the solveset.\n\n    Raises\n    ======\n\n    ValueError\n        The input is not valid.\n        The symbols are not given.\n    AttributeError\n        The input symbols are not :class:`~.Symbol` type.\n\n    Examples\n    ========\n\n    >>> from sympy import symbols, substitution\n    >>> x, y = symbols('x, y', real=True)\n    >>> substitution([x + y], [x], [{y: 1}], [y], set([]), [x, y])\n    {(-1, 1)}\n\n    * When you want a soln not satisfying $x + 1 = 0$\n\n    >>> substitution([x + y], [x], [{y: 1}], [y], set([x + 1]), [y, x])\n    EmptySet\n    >>> substitution([x + y], [x], [{y: 1}], [y], set([x - 1]), [y, x])\n    {(1, -1)}\n    >>> substitution([x + y - 1, y - x**2 + 5], [x, y])\n    {(-3, 4), (2, -1)}\n\n    * Returns both real and complex solution\n\n    >>> x, y, z = symbols('x, y, z')\n    >>> from sympy import exp, sin\n    >>> substitution([exp(x) - sin(y), y**2 - 4], [x, y])\n    {(ImageSet(Lambda(_n, I*(2*_n*pi + pi) + log(sin(2))), Integers), -2),\n     (ImageSet(Lambda(_n, 2*_n*I*pi + log(sin(2))), Integers), 2)}\n\n    >>> eqs = [z**2 + exp(2*x) - sin(y), -3 + exp(-y)]\n    >>> substitution(eqs, [y, z])\n    {(-log(3), -sqrt(-exp(2*x) - sin(log(3)))),\n     (-log(3), sqrt(-exp(2*x) - sin(log(3)))),\n     (ImageSet(Lambda(_n, 2*_n*I*pi - log(3)), Integers),\n      ImageSet(Lambda(_n, -sqrt(-exp(2*x) + sin(2*_n*I*pi - log(3)))), Integers)),\n     (ImageSet(Lambda(_n, 2*_n*I*pi - log(3)), Integers),\n      ImageSet(Lambda(_n, sqrt(-exp(2*x) + sin(2*_n*I*pi - log(3)))), Integers))}\n\n    "
    if not system:
        return S.EmptySet
    for (i, e) in enumerate(system):
        if isinstance(e, Eq):
            system[i] = e.lhs - e.rhs
    if not symbols:
        msg = 'Symbols must be given, for which solution of the system is to be found.'
        raise ValueError(filldedent(msg))
    if not is_sequence(symbols):
        msg = 'symbols should be given as a sequence, e.g. a list.Not type %s: %s'
        raise TypeError(filldedent(msg % (type(symbols), symbols)))
    if not getattr(symbols[0], 'is_Symbol', False):
        msg = 'Iterable of symbols must be given as second argument, not type %s: %s'
        raise ValueError(filldedent(msg % (type(symbols[0]), symbols[0])))
    if all_symbols is None:
        all_symbols = symbols
    old_result = result
    complements = {}
    intersections = {}
    total_conditionset = -1
    total_solveset_call = -1

    def _unsolved_syms(eq, sort=False):
        if False:
            i = 10
            return i + 15
        'Returns the unsolved symbol present\n        in the equation `eq`.\n        '
        free = eq.free_symbols
        unsolved = free - set(known_symbols) & set(all_symbols)
        if sort:
            unsolved = list(unsolved)
            unsolved.sort(key=default_sort_key)
        return unsolved
    eqs_in_better_order = list(ordered(system, lambda _: len(_unsolved_syms(_))))

    def add_intersection_complement(result, intersection_dict, complement_dict):
        if False:
            print('Hello World!')
        final_result = []
        for res in result:
            res_copy = res
            for (key_res, value_res) in res.items():
                (intersect_set, complement_set) = (None, None)
                for (key_sym, value_sym) in intersection_dict.items():
                    if key_sym == key_res:
                        intersect_set = value_sym
                for (key_sym, value_sym) in complement_dict.items():
                    if key_sym == key_res:
                        complement_set = value_sym
                if intersect_set or complement_set:
                    new_value = FiniteSet(value_res)
                    if intersect_set and intersect_set != S.Complexes:
                        new_value = Intersection(new_value, intersect_set)
                    if complement_set:
                        new_value = Complement(new_value, complement_set)
                    if new_value is S.EmptySet:
                        res_copy = None
                        break
                    elif new_value.is_FiniteSet and len(new_value) == 1:
                        res_copy[key_res] = set(new_value).pop()
                    else:
                        res_copy[key_res] = new_value
            if res_copy is not None:
                final_result.append(res_copy)
        return final_result

    def _extract_main_soln(sym, sol, soln_imageset):
        if False:
            return 10
        'Separate the Complements, Intersections, ImageSet lambda expr and\n        its base_set. This function returns the unmasked sol from different classes\n        of sets and also returns the appended ImageSet elements in a\n        soln_imageset dict: `{unmasked element: ImageSet}`.\n        '
        if isinstance(sol, ConditionSet):
            sol = sol.base_set
        if isinstance(sol, Complement):
            complements[sym] = sol.args[1]
            sol = sol.args[0]
        if isinstance(sol, Union):
            sol_args = sol.args
            sol = S.EmptySet
            for sol_arg2 in sol_args:
                if isinstance(sol_arg2, FiniteSet):
                    sol += sol_arg2
                else:
                    sol += FiniteSet(sol_arg2)
        if isinstance(sol, Intersection):
            if sol.args[0] not in (S.Reals, S.Complexes):
                intersections[sym] = sol.args[0]
            sol = sol.args[1]
        if isinstance(sol, ImageSet):
            soln_imagest = sol
            expr2 = sol.lamda.expr
            sol = FiniteSet(expr2)
            soln_imageset[expr2] = soln_imagest
        if not isinstance(sol, FiniteSet):
            sol = FiniteSet(sol)
        return (sol, soln_imageset)

    def _check_exclude(rnew, imgset_yes):
        if False:
            while True:
                i = 10
        rnew_ = rnew
        if imgset_yes:
            rnew_copy = rnew.copy()
            dummy_n = imgset_yes[0]
            for (key_res, value_res) in rnew_copy.items():
                rnew_copy[key_res] = value_res.subs(dummy_n, 0)
            rnew_ = rnew_copy
        try:
            satisfy_exclude = any((checksol(d, rnew_) for d in exclude))
        except TypeError:
            satisfy_exclude = None
        return satisfy_exclude

    def _restore_imgset(rnew, original_imageset, newresult):
        if False:
            i = 10
            return i + 15
        restore_sym = set(rnew.keys()) & set(original_imageset.keys())
        for key_sym in restore_sym:
            img = original_imageset[key_sym]
            rnew[key_sym] = img
        if rnew not in newresult:
            newresult.append(rnew)

    def _append_eq(eq, result, res, delete_soln, n=None):
        if False:
            for i in range(10):
                print('nop')
        u = Dummy('u')
        if n:
            eq = eq.subs(n, 0)
        satisfy = eq if eq in (True, False) else checksol(u, u, eq, minimal=True)
        if satisfy is False:
            delete_soln = True
            res = {}
        else:
            result.append(res)
        return (result, res, delete_soln)

    def _append_new_soln(rnew, sym, sol, imgset_yes, soln_imageset, original_imageset, newresult, eq=None):
        if False:
            print('Hello World!')
        'If `rnew` (A dict <symbol: soln>) contains valid soln\n        append it to `newresult` list.\n        `imgset_yes` is (base, dummy_var) if there was imageset in previously\n         calculated result(otherwise empty tuple). `original_imageset` is dict\n         of imageset expr and imageset from this result.\n        `soln_imageset` dict of imageset expr and imageset of new soln.\n        '
        satisfy_exclude = _check_exclude(rnew, imgset_yes)
        delete_soln = False
        if not satisfy_exclude:
            local_n = None
            if imgset_yes:
                local_n = imgset_yes[0]
                base = imgset_yes[1]
                if sym and sol:
                    dummy_list = list(sol.atoms(Dummy))
                    local_n_list = [local_n for i in range(0, len(dummy_list))]
                    dummy_zip = zip(dummy_list, local_n_list)
                    lam = Lambda(local_n, sol.subs(dummy_zip))
                    rnew[sym] = ImageSet(lam, base)
                if eq is not None:
                    (newresult, rnew, delete_soln) = _append_eq(eq, newresult, rnew, delete_soln, local_n)
            elif eq is not None:
                (newresult, rnew, delete_soln) = _append_eq(eq, newresult, rnew, delete_soln)
            elif sol in soln_imageset.keys():
                rnew[sym] = soln_imageset[sol]
                _restore_imgset(rnew, original_imageset, newresult)
            else:
                newresult.append(rnew)
        elif satisfy_exclude:
            delete_soln = True
            rnew = {}
        _restore_imgset(rnew, original_imageset, newresult)
        return (newresult, delete_soln)

    def _new_order_result(result, eq):
        if False:
            while True:
                i = 10
        first_priority = []
        second_priority = []
        for res in result:
            if not any((isinstance(val, ImageSet) for val in res.values())):
                if eq.subs(res) == 0:
                    first_priority.append(res)
                else:
                    second_priority.append(res)
        if first_priority or second_priority:
            return first_priority + second_priority
        return result

    def _solve_using_known_values(result, solver):
        if False:
            print('Hello World!')
        'Solves the system using already known solution\n        (result contains the dict <symbol: value>).\n        solver is :func:`~.solveset_complex` or :func:`~.solveset_real`.\n        '
        soln_imageset = {}
        total_solvest_call = 0
        total_conditionst = 0
        for (index, eq) in enumerate(eqs_in_better_order):
            newresult = []
            original_imageset = {}
            imgset_yes = False
            for res in result:
                got_symbol = set()
                for (k, v) in res.items():
                    if isinstance(v, ImageSet):
                        res[k] = v.lamda.expr
                        original_imageset[k] = v
                        dummy_n = v.lamda.expr.atoms(Dummy).pop()
                        (base,) = v.base_sets
                        imgset_yes = (dummy_n, base)
                    assert not isinstance(v, FiniteSet)
                eq2 = eq.subs(res).expand()
                unsolved_syms = _unsolved_syms(eq2, sort=True)
                if not unsolved_syms:
                    if res:
                        (newresult, delete_res) = _append_new_soln(res, None, None, imgset_yes, soln_imageset, original_imageset, newresult, eq2)
                        if delete_res:
                            result.remove(res)
                    continue
                (depen1, depen2) = eq2.as_independent(*unsolved_syms)
                if (depen1.has(Abs) or depen2.has(Abs)) and solver == solveset_complex:
                    continue
                soln_imageset = {}
                for sym in unsolved_syms:
                    not_solvable = False
                    try:
                        soln = solver(eq2, sym)
                        total_solvest_call += 1
                        soln_new = S.EmptySet
                        if isinstance(soln, Complement):
                            complements[sym] = soln.args[1]
                            soln = soln.args[0]
                        if isinstance(soln, Intersection):
                            if soln.args[0] != Interval(-oo, oo):
                                intersections[sym] = soln.args[0]
                            soln_new += soln.args[1]
                        soln = soln_new if soln_new else soln
                        if index > 0 and solver == solveset_real:
                            if not isinstance(soln, (ImageSet, ConditionSet)):
                                soln += solveset_complex(eq2, sym)
                    except (NotImplementedError, ValueError):
                        continue
                    if isinstance(soln, ConditionSet):
                        if soln.base_set in (S.Reals, S.Complexes):
                            soln = S.EmptySet
                            not_solvable = True
                            total_conditionst += 1
                        else:
                            soln = soln.base_set
                    if soln is not S.EmptySet:
                        (soln, soln_imageset) = _extract_main_soln(sym, soln, soln_imageset)
                    for sol in soln:
                        (sol, soln_imageset) = _extract_main_soln(sym, sol, soln_imageset)
                        sol = set(sol).pop()
                        free = sol.free_symbols
                        if got_symbol and any((ss in free for ss in got_symbol)):
                            continue
                        rnew = res.copy()
                        for (k, v) in res.items():
                            if isinstance(v, Expr) and isinstance(sol, Expr):
                                rnew[k] = v.subs(sym, sol)
                        if sol in soln_imageset.keys():
                            imgst = soln_imageset[sol]
                            rnew[sym] = imgst.lamda(*[0 for i in range(0, len(imgst.lamda.variables))])
                        else:
                            rnew[sym] = sol
                        (newresult, delete_res) = _append_new_soln(rnew, sym, sol, imgset_yes, soln_imageset, original_imageset, newresult)
                        if delete_res:
                            result.remove(res)
                    if not not_solvable:
                        got_symbol.add(sym)
            if newresult:
                result = newresult
        return (result, total_solvest_call, total_conditionst)
    (new_result_real, solve_call1, cnd_call1) = _solve_using_known_values(old_result, solveset_real)
    (new_result_complex, solve_call2, cnd_call2) = _solve_using_known_values(old_result, solveset_complex)
    total_conditionset += cnd_call1 + cnd_call2
    total_solveset_call += solve_call1 + solve_call2
    if total_conditionset == total_solveset_call and total_solveset_call != -1:
        return _return_conditionset(eqs_in_better_order, all_symbols)
    filtered_complex = []
    for i in list(new_result_complex):
        for j in list(new_result_real):
            if i.keys() != j.keys():
                continue
            if all((a.dummy_eq(b) for (a, b) in zip(i.values(), j.values()) if not (isinstance(a, int) and isinstance(b, int)))):
                break
        else:
            filtered_complex.append(i)
    result = new_result_real + filtered_complex
    result_all_variables = []
    result_infinite = []
    for res in result:
        if not res:
            continue
        if len(res) < len(all_symbols):
            solved_symbols = res.keys()
            unsolved = list(filter(lambda x: x not in solved_symbols, all_symbols))
            for unsolved_sym in unsolved:
                res[unsolved_sym] = unsolved_sym
            result_infinite.append(res)
        if res not in result_all_variables:
            result_all_variables.append(res)
    if result_infinite:
        result_all_variables = result_infinite
    if intersections or complements:
        result_all_variables = add_intersection_complement(result_all_variables, intersections, complements)
    result = S.EmptySet
    for r in result_all_variables:
        temp = [r[symb] for symb in all_symbols]
        result += FiniteSet(tuple(temp))
    return result

def _solveset_work(system, symbols):
    if False:
        for i in range(10):
            print('nop')
    soln = solveset(system[0], symbols[0])
    if isinstance(soln, FiniteSet):
        _soln = FiniteSet(*[(s,) for s in soln])
        return _soln
    else:
        return FiniteSet(tuple(FiniteSet(soln)))

def _handle_positive_dimensional(polys, symbols, denominators):
    if False:
        for i in range(10):
            print('nop')
    from sympy.polys.polytools import groebner
    _symbols = list(symbols)
    _symbols.sort(key=default_sort_key)
    basis = groebner(polys, _symbols, polys=True)
    new_system = []
    for poly_eq in basis:
        new_system.append(poly_eq.as_expr())
    result = [{}]
    result = substitution(new_system, symbols, result, [], denominators)
    return result

def _handle_zero_dimensional(polys, symbols, system):
    if False:
        print('Hello World!')
    result = solve_poly_system(polys, *symbols)
    result_update = S.EmptySet
    for res in result:
        dict_sym_value = dict(list(zip(symbols, res)))
        if all((checksol(eq, dict_sym_value) for eq in system)):
            result_update += FiniteSet(res)
    return result_update

def _separate_poly_nonpoly(system, symbols):
    if False:
        return 10
    polys = []
    polys_expr = []
    nonpolys = []
    unrad_changed = []
    denominators = set()
    poly = None
    for eq in system:
        denominators.update(_simple_dens(eq, symbols))
        if isinstance(eq, Eq):
            eq = eq.lhs - eq.rhs
        without_radicals = unrad(simplify(eq), *symbols)
        if without_radicals:
            unrad_changed.append(eq)
            (eq_unrad, cov) = without_radicals
            if not cov:
                eq = eq_unrad
        if isinstance(eq, Expr):
            eq = eq.as_numer_denom()[0]
            poly = eq.as_poly(*symbols, extension=True)
        elif simplify(eq).is_number:
            continue
        if poly is not None:
            polys.append(poly)
            polys_expr.append(poly.as_expr())
        else:
            nonpolys.append(eq)
    return (polys, polys_expr, nonpolys, denominators, unrad_changed)

def _handle_poly(polys, symbols):
    if False:
        i = 10
        return i + 15
    no_information = [{}]
    no_solutions = []
    no_equations = []
    inexact = any((not p.domain.is_Exact for p in polys))
    if inexact:
        polys = [poly(nsimplify(p, rational=True)) for p in polys]
    basis = groebner(polys, symbols, order='grevlex', polys=False)
    if 1 in basis:
        poly_sol = no_solutions
        poly_eqs = no_equations
    elif basis.is_zero_dimensional:
        basis = basis.fglm('lex')
        if inexact:
            basis = [nfloat(p) for p in basis]
        try:
            result = solve_poly_system(basis, *symbols, strict=True)
        except UnsolvableFactorError:
            poly_sol = no_information
            poly_eqs = list(basis)
        else:
            poly_sol = [dict(zip(symbols, res)) for res in result]
            poly_eqs = no_equations
    else:
        poly_sol = no_information
        poly_eqs = list(groebner(polys, symbols, order='lex', polys=False))
        if inexact:
            poly_eqs = [nfloat(p) for p in poly_eqs]
    return (poly_sol, poly_eqs)

def nonlinsolve(system, *symbols):
    if False:
        for i in range(10):
            print('nop')
    "\n    Solve system of $N$ nonlinear equations with $M$ variables, which means both\n    under and overdetermined systems are supported. Positive dimensional\n    system is also supported (A system with infinitely many solutions is said\n    to be positive-dimensional). In a positive dimensional system the solution will\n    be dependent on at least one symbol. Returns both real solution\n    and complex solution (if they exist).\n\n    Parameters\n    ==========\n\n    system : list of equations\n        The target system of equations\n    symbols : list of Symbols\n        symbols should be given as a sequence eg. list\n\n    Returns\n    =======\n\n    A :class:`~.FiniteSet` of ordered tuple of values of `symbols` for which the `system`\n    has solution. Order of values in the tuple is same as symbols present in\n    the parameter `symbols`.\n\n    Please note that general :class:`~.FiniteSet` is unordered, the solution\n    returned here is not simply a :class:`~.FiniteSet` of solutions, rather it\n    is a :class:`~.FiniteSet` of ordered tuple, i.e. the first and only\n    argument to :class:`~.FiniteSet` is a tuple of solutions, which is\n    ordered, and, hence ,the returned solution is ordered.\n\n    Also note that solution could also have been returned as an ordered tuple,\n    FiniteSet is just a wrapper ``{}`` around the tuple. It has no other\n    significance except for the fact it is just used to maintain a consistent\n    output format throughout the solveset.\n\n    For the given set of equations, the respective input types\n    are given below:\n\n    .. math:: xy - 1 = 0\n    .. math:: 4x^2 + y^2 - 5 = 0\n\n    ::\n\n       system  = [x*y - 1, 4*x**2 + y**2 - 5]\n       symbols = [x, y]\n\n    Raises\n    ======\n\n    ValueError\n        The input is not valid.\n        The symbols are not given.\n    AttributeError\n        The input symbols are not `Symbol` type.\n\n    Examples\n    ========\n\n    >>> from sympy import symbols, nonlinsolve\n    >>> x, y, z = symbols('x, y, z', real=True)\n    >>> nonlinsolve([x*y - 1, 4*x**2 + y**2 - 5], [x, y])\n    {(-1, -1), (-1/2, -2), (1/2, 2), (1, 1)}\n\n    1. Positive dimensional system and complements:\n\n    >>> from sympy import pprint\n    >>> from sympy.polys.polytools import is_zero_dimensional\n    >>> a, b, c, d = symbols('a, b, c, d', extended_real=True)\n    >>> eq1 =  a + b + c + d\n    >>> eq2 = a*b + b*c + c*d + d*a\n    >>> eq3 = a*b*c + b*c*d + c*d*a + d*a*b\n    >>> eq4 = a*b*c*d - 1\n    >>> system = [eq1, eq2, eq3, eq4]\n    >>> is_zero_dimensional(system)\n    False\n    >>> pprint(nonlinsolve(system, [a, b, c, d]), use_unicode=False)\n      -1       1               1      -1\n    {(---, -d, -, {d} \\ {0}), (-, -d, ---, {d} \\ {0})}\n       d       d               d       d\n    >>> nonlinsolve([(x+y)**2 - 4, x + y - 2], [x, y])\n    {(2 - y, y)}\n\n    2. If some of the equations are non-polynomial then `nonlinsolve`\n    will call the ``substitution`` function and return real and complex solutions,\n    if present.\n\n    >>> from sympy import exp, sin\n    >>> nonlinsolve([exp(x) - sin(y), y**2 - 4], [x, y])\n    {(ImageSet(Lambda(_n, I*(2*_n*pi + pi) + log(sin(2))), Integers), -2),\n     (ImageSet(Lambda(_n, 2*_n*I*pi + log(sin(2))), Integers), 2)}\n\n    3. If system is non-linear polynomial and zero-dimensional then it\n    returns both solution (real and complex solutions, if present) using\n    :func:`~.solve_poly_system`:\n\n    >>> from sympy import sqrt\n    >>> nonlinsolve([x**2 - 2*y**2 -2, x*y - 2], [x, y])\n    {(-2, -1), (2, 1), (-sqrt(2)*I, sqrt(2)*I), (sqrt(2)*I, -sqrt(2)*I)}\n\n    4. ``nonlinsolve`` can solve some linear (zero or positive dimensional)\n    system (because it uses the :func:`sympy.polys.polytools.groebner` function to get the\n    groebner basis and then uses the ``substitution`` function basis as the\n    new `system`). But it is not recommended to solve linear system using\n    ``nonlinsolve``, because :func:`~.linsolve` is better for general linear systems.\n\n    >>> nonlinsolve([x + 2*y -z - 3, x - y - 4*z + 9, y + z - 4], [x, y, z])\n    {(3*z - 5, 4 - z, z)}\n\n    5. System having polynomial equations and only real solution is\n    solved using :func:`~.solve_poly_system`:\n\n    >>> e1 = sqrt(x**2 + y**2) - 10\n    >>> e2 = sqrt(y**2 + (-x + 10)**2) - 3\n    >>> nonlinsolve((e1, e2), (x, y))\n    {(191/20, -3*sqrt(391)/20), (191/20, 3*sqrt(391)/20)}\n    >>> nonlinsolve([x**2 + 2/y - 2, x + y - 3], [x, y])\n    {(1, 2), (1 - sqrt(5), 2 + sqrt(5)), (1 + sqrt(5), 2 - sqrt(5))}\n    >>> nonlinsolve([x**2 + 2/y - 2, x + y - 3], [y, x])\n    {(2, 1), (2 - sqrt(5), 1 + sqrt(5)), (2 + sqrt(5), 1 - sqrt(5))}\n\n    6. It is better to use symbols instead of trigonometric functions or\n    :class:`~.Function`. For example, replace $\\sin(x)$ with a symbol, replace\n    $f(x)$ with a symbol and so on. Get a solution from ``nonlinsolve`` and then\n    use :func:`~.solveset` to get the value of $x$.\n\n    How nonlinsolve is better than old solver ``_solve_system`` :\n    =============================================================\n\n    1. A positive dimensional system solver: nonlinsolve can return\n    solution for positive dimensional system. It finds the\n    Groebner Basis of the positive dimensional system(calling it as\n    basis) then we can start solving equation(having least number of\n    variable first in the basis) using solveset and substituting that\n    solved solutions into other equation(of basis) to get solution in\n    terms of minimum variables. Here the important thing is how we\n    are substituting the known values and in which equations.\n\n    2. Real and complex solutions: nonlinsolve returns both real\n    and complex solution. If all the equations in the system are polynomial\n    then using :func:`~.solve_poly_system` both real and complex solution is returned.\n    If all the equations in the system are not polynomial equation then goes to\n    ``substitution`` method with this polynomial and non polynomial equation(s),\n    to solve for unsolved variables. Here to solve for particular variable\n    solveset_real and solveset_complex is used. For both real and complex\n    solution ``_solve_using_known_values`` is used inside ``substitution``\n    (``substitution`` will be called when any non-polynomial equation is present).\n    If a solution is valid its general solution is added to the final result.\n\n    3. :class:`~.Complement` and :class:`~.Intersection` will be added:\n    nonlinsolve maintains dict for complements and intersections. If solveset\n    find complements or/and intersections with any interval or set during the\n    execution of ``substitution`` function, then complement or/and\n    intersection for that variable is added before returning final solution.\n\n    "
    if not system:
        return S.EmptySet
    if not symbols:
        msg = 'Symbols must be given, for which solution of the system is to be found.'
        raise ValueError(filldedent(msg))
    if hasattr(symbols[0], '__iter__'):
        symbols = symbols[0]
    if not is_sequence(symbols) or not symbols:
        msg = 'Symbols must be given, for which solution of the system is to be found.'
        raise IndexError(filldedent(msg))
    symbols = list(map(_sympify, symbols))
    (system, symbols, swap) = recast_to_symbols(system, symbols)
    if swap:
        soln = nonlinsolve(system, symbols)
        return FiniteSet(*[tuple((i.xreplace(swap) for i in s)) for s in soln])
    if len(system) == 1 and len(symbols) == 1:
        return _solveset_work(system, symbols)
    (polys, polys_expr, nonpolys, denominators, unrad_changed) = _separate_poly_nonpoly(system, symbols)
    poly_eqs = []
    poly_sol = [{}]
    if polys:
        (poly_sol, poly_eqs) = _handle_poly(polys, symbols)
        if poly_sol and poly_sol[0]:
            poly_syms = set().union(*(eq.free_symbols for eq in polys))
            unrad_syms = set().union(*(eq.free_symbols for eq in unrad_changed))
            if unrad_syms == poly_syms and unrad_changed:
                poly_sol = [sol for sol in poly_sol if checksol(unrad_changed, sol)]
    remaining = poly_eqs + nonpolys
    to_tuple = lambda sol: tuple((sol[s] for s in symbols))
    if not remaining:
        return FiniteSet(*map(to_tuple, poly_sol))
    else:
        subs_res = substitution(remaining, symbols, result=poly_sol, exclude=denominators)
        if not isinstance(subs_res, FiniteSet):
            return subs_res
        if unrad_changed:
            result = [dict(zip(symbols, sol)) for sol in subs_res.args]
            correct_sols = [sol for sol in result if any((isinstance(v, Set) for v in sol)) or checksol(unrad_changed, sol) != False]
            return FiniteSet(*map(to_tuple, correct_sols))
        else:
            return subs_res