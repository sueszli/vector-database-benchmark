"""Tools for manipulating of large commutative expressions. """
from .add import Add
from .mul import Mul, _keep_coeff
from .power import Pow
from .basic import Basic
from .expr import Expr
from .function import expand_power_exp
from .sympify import sympify
from .numbers import Rational, Integer, Number, I, equal_valued
from .singleton import S
from .sorting import default_sort_key, ordered
from .symbol import Dummy
from .traversal import preorder_traversal
from .coreerrors import NonCommutativeExpression
from .containers import Tuple, Dict
from sympy.external.gmpy import SYMPY_INTS
from sympy.utilities.iterables import common_prefix, common_suffix, variations, iterable, is_sequence
from collections import defaultdict
from typing import Tuple as tTuple
_eps = Dummy(positive=True)

def _isnumber(i):
    if False:
        i = 10
        return i + 15
    return isinstance(i, (SYMPY_INTS, float)) or i.is_Number

def _monotonic_sign(self):
    if False:
        return 10
    'Return the value closest to 0 that ``self`` may have if all symbols\n    are signed and the result is uniformly the same sign for all values of symbols.\n    If a symbol is only signed but not known to be an\n    integer or the result is 0 then a symbol representative of the sign of self\n    will be returned. Otherwise, None is returned if a) the sign could be positive\n    or negative or b) self is not in one of the following forms:\n\n    - L(x, y, ...) + A: a function linear in all symbols x, y, ... with an\n      additive constant; if A is zero then the function can be a monomial whose\n      sign is monotonic over the range of the variables, e.g. (x + 1)**3 if x is\n      nonnegative.\n    - A/L(x, y, ...) + B: the inverse of a function linear in all symbols x, y, ...\n      that does not have a sign change from positive to negative for any set\n      of values for the variables.\n    - M(x, y, ...) + A: a monomial M whose factors are all signed and a constant, A.\n    - A/M(x, y, ...) + B: the inverse of a monomial and constants A and B.\n    - P(x): a univariate polynomial\n\n    Examples\n    ========\n\n    >>> from sympy.core.exprtools import _monotonic_sign as F\n    >>> from sympy import Dummy\n    >>> nn = Dummy(integer=True, nonnegative=True)\n    >>> p = Dummy(integer=True, positive=True)\n    >>> p2 = Dummy(integer=True, positive=True)\n    >>> F(nn + 1)\n    1\n    >>> F(p - 1)\n    _nneg\n    >>> F(nn*p + 1)\n    1\n    >>> F(p2*p + 1)\n    2\n    >>> F(nn - 1)  # could be negative, zero or positive\n    '
    if not self.is_extended_real:
        return
    if (-self).is_Symbol:
        rv = _monotonic_sign(-self)
        return rv if rv is None else -rv
    if not self.is_Add and self.as_numer_denom()[1].is_number:
        s = self
        if s.is_prime:
            if s.is_odd:
                return Integer(3)
            else:
                return Integer(2)
        elif s.is_composite:
            if s.is_odd:
                return Integer(9)
            else:
                return Integer(4)
        elif s.is_positive:
            if s.is_even:
                if s.is_prime is False:
                    return Integer(4)
                else:
                    return Integer(2)
            elif s.is_integer:
                return S.One
            else:
                return _eps
        elif s.is_extended_negative:
            if s.is_even:
                return Integer(-2)
            elif s.is_integer:
                return S.NegativeOne
            else:
                return -_eps
        if s.is_zero or s.is_extended_nonpositive or s.is_extended_nonnegative:
            return S.Zero
        return None
    free = self.free_symbols
    if len(free) == 1:
        if self.is_polynomial():
            from sympy.polys.polytools import real_roots
            from sympy.polys.polyroots import roots
            from sympy.polys.polyerrors import PolynomialError
            x = free.pop()
            x0 = _monotonic_sign(x)
            if x0 in (_eps, -_eps):
                x0 = S.Zero
            if x0 is not None:
                d = self.diff(x)
                if d.is_number:
                    currentroots = []
                else:
                    try:
                        currentroots = real_roots(d)
                    except (PolynomialError, NotImplementedError):
                        currentroots = [r for r in roots(d, x) if r.is_extended_real]
                y = self.subs(x, x0)
                if x.is_nonnegative and all(((r - x0).is_nonpositive for r in currentroots)):
                    if y.is_nonnegative and d.is_positive:
                        if y:
                            return y if y.is_positive else Dummy('pos', positive=True)
                        else:
                            return Dummy('nneg', nonnegative=True)
                    if y.is_nonpositive and d.is_negative:
                        if y:
                            return y if y.is_negative else Dummy('neg', negative=True)
                        else:
                            return Dummy('npos', nonpositive=True)
                elif x.is_nonpositive and all(((r - x0).is_nonnegative for r in currentroots)):
                    if y.is_nonnegative and d.is_negative:
                        if y:
                            return Dummy('pos', positive=True)
                        else:
                            return Dummy('nneg', nonnegative=True)
                    if y.is_nonpositive and d.is_positive:
                        if y:
                            return Dummy('neg', negative=True)
                        else:
                            return Dummy('npos', nonpositive=True)
        else:
            (n, d) = self.as_numer_denom()
            den = None
            if n.is_number:
                den = _monotonic_sign(d)
            elif not d.is_number:
                if _monotonic_sign(n) is not None:
                    den = _monotonic_sign(d)
            if den is not None and (den.is_positive or den.is_negative):
                v = n * den
                if v.is_positive:
                    return Dummy('pos', positive=True)
                elif v.is_nonnegative:
                    return Dummy('nneg', nonnegative=True)
                elif v.is_negative:
                    return Dummy('neg', negative=True)
                elif v.is_nonpositive:
                    return Dummy('npos', nonpositive=True)
        return None
    (c, a) = self.as_coeff_Add()
    v = None
    if not a.is_polynomial():
        (n, d) = a.as_numer_denom()
        if not (n.is_number or d.is_number):
            return
        if (a.is_Mul or a.is_Pow) and a.is_rational and all((p.exp.is_Integer for p in a.atoms(Pow) if p.is_Pow)) and (a.is_positive or a.is_negative):
            v = S.One
            for ai in Mul.make_args(a):
                if ai.is_number:
                    v *= ai
                    continue
                reps = {}
                for x in ai.free_symbols:
                    reps[x] = _monotonic_sign(x)
                    if reps[x] is None:
                        return
                v *= ai.subs(reps)
    elif c:
        if not any((p for p in a.atoms(Pow) if not p.is_number)) and (a.is_nonpositive or a.is_nonnegative):
            free = list(a.free_symbols)
            p = {}
            for i in free:
                v = _monotonic_sign(i)
                if v is None:
                    return
                p[i] = v or (_eps if i.is_nonnegative else -_eps)
            v = a.xreplace(p)
    if v is not None:
        rv = v + c
        if v.is_nonnegative and rv.is_positive:
            return rv.subs(_eps, 0)
        if v.is_nonpositive and rv.is_negative:
            return rv.subs(_eps, 0)

def decompose_power(expr: Expr) -> tTuple[Expr, int]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Decompose power into symbolic base and integer exponent.\n\n    Examples\n    ========\n\n    >>> from sympy.core.exprtools import decompose_power\n    >>> from sympy.abc import x, y\n    >>> from sympy import exp\n\n    >>> decompose_power(x)\n    (x, 1)\n    >>> decompose_power(x**2)\n    (x, 2)\n    >>> decompose_power(exp(2*y/3))\n    (exp(y/3), 2)\n\n    '
    (base, exp) = expr.as_base_exp()
    if exp.is_Number:
        if exp.is_Rational:
            if not exp.is_Integer:
                base = Pow(base, Rational(1, exp.q))
            e = exp.p
        else:
            (base, e) = (expr, 1)
    else:
        (exp, tail) = exp.as_coeff_Mul(rational=True)
        if exp is S.NegativeOne:
            (base, e) = (Pow(base, tail), -1)
        elif exp is not S.One:
            tail = _keep_coeff(Rational(1, exp.q), tail)
            (base, e) = (Pow(base, tail), exp.p)
        else:
            (base, e) = (expr, 1)
    return (base, e)

def decompose_power_rat(expr: Expr) -> tTuple[Expr, Rational]:
    if False:
        i = 10
        return i + 15
    '\n    Decompose power into symbolic base and rational exponent;\n    if the exponent is not a Rational, then separate only the\n    integer coefficient.\n\n    Examples\n    ========\n\n    >>> from sympy.core.exprtools import decompose_power_rat\n    >>> from sympy.abc import x\n    >>> from sympy import sqrt, exp\n\n    >>> decompose_power_rat(sqrt(x))\n    (x, 1/2)\n    >>> decompose_power_rat(exp(-3*x/2))\n    (exp(x/2), -3)\n\n    '
    _ = (base, exp) = expr.as_base_exp()
    return _ if exp.is_Rational else decompose_power(expr)

class Factors:
    """Efficient representation of ``f_1*f_2*...*f_n``."""
    __slots__ = ('factors', 'gens')

    def __init__(self, factors=None):
        if False:
            print('Hello World!')
        'Initialize Factors from dict or expr.\n\n        Examples\n        ========\n\n        >>> from sympy.core.exprtools import Factors\n        >>> from sympy.abc import x\n        >>> from sympy import I\n        >>> e = 2*x**3\n        >>> Factors(e)\n        Factors({2: 1, x: 3})\n        >>> Factors(e.as_powers_dict())\n        Factors({2: 1, x: 3})\n        >>> f = _\n        >>> f.factors  # underlying dictionary\n        {2: 1, x: 3}\n        >>> f.gens  # base of each factor\n        frozenset({2, x})\n        >>> Factors(0)\n        Factors({0: 1})\n        >>> Factors(I)\n        Factors({I: 1})\n\n        Notes\n        =====\n\n        Although a dictionary can be passed, only minimal checking is\n        performed: powers of -1 and I are made canonical.\n\n        '
        if isinstance(factors, (SYMPY_INTS, float)):
            factors = S(factors)
        if isinstance(factors, Factors):
            factors = factors.factors.copy()
        elif factors in (None, S.One):
            factors = {}
        elif factors is S.Zero or factors == 0:
            factors = {S.Zero: S.One}
        elif isinstance(factors, Number):
            n = factors
            factors = {}
            if n < 0:
                factors[S.NegativeOne] = S.One
                n = -n
            if n is not S.One:
                if n.is_Float or n.is_Integer or n is S.Infinity:
                    factors[n] = S.One
                elif n.is_Rational:
                    if n.p != 1:
                        factors[Integer(n.p)] = S.One
                    factors[Integer(n.q)] = S.NegativeOne
                else:
                    raise ValueError('Expected Float|Rational|Integer, not %s' % n)
        elif isinstance(factors, Basic) and (not factors.args):
            factors = {factors: S.One}
        elif isinstance(factors, Expr):
            (c, nc) = factors.args_cnc()
            i = c.count(I)
            for _ in range(i):
                c.remove(I)
            factors = dict(Mul._from_args(c).as_powers_dict())
            for f in list(factors.keys()):
                if isinstance(f, Rational) and (not isinstance(f, Integer)):
                    (p, q) = (Integer(f.p), Integer(f.q))
                    factors[p] = (factors[p] if p in factors else S.Zero) + factors[f]
                    factors[q] = (factors[q] if q in factors else S.Zero) - factors[f]
                    factors.pop(f)
            if i:
                factors[I] = factors.get(I, S.Zero) + i
            if nc:
                factors[Mul(*nc, evaluate=False)] = S.One
        else:
            factors = factors.copy()
            handle = [k for k in factors if k is I or k in (-1, 1)]
            if handle:
                i1 = S.One
                for k in handle:
                    if not _isnumber(factors[k]):
                        continue
                    i1 *= k ** factors.pop(k)
                if i1 is not S.One:
                    for a in i1.args if i1.is_Mul else [i1]:
                        if a is S.NegativeOne:
                            factors[a] = S.One
                        elif a is I:
                            factors[I] = S.One
                        elif a.is_Pow:
                            factors[a.base] = factors.get(a.base, S.Zero) + a.exp
                        elif equal_valued(a, 1):
                            factors[a] = S.One
                        elif equal_valued(a, -1):
                            factors[-a] = S.One
                            factors[S.NegativeOne] = S.One
                        else:
                            raise ValueError('unexpected factor in i1: %s' % a)
        self.factors = factors
        keys = getattr(factors, 'keys', None)
        if keys is None:
            raise TypeError('expecting Expr or dictionary')
        self.gens = frozenset(keys())

    def __hash__(self):
        if False:
            print('Hello World!')
        keys = tuple(ordered(self.factors.keys()))
        values = [self.factors[k] for k in keys]
        return hash((keys, values))

    def __repr__(self):
        if False:
            while True:
                i = 10
        return 'Factors({%s})' % ', '.join(['%s: %s' % (k, v) for (k, v) in ordered(self.factors.items())])

    @property
    def is_zero(self):
        if False:
            print('Hello World!')
        '\n        >>> from sympy.core.exprtools import Factors\n        >>> Factors(0).is_zero\n        True\n        '
        f = self.factors
        return len(f) == 1 and S.Zero in f

    @property
    def is_one(self):
        if False:
            i = 10
            return i + 15
        '\n        >>> from sympy.core.exprtools import Factors\n        >>> Factors(1).is_one\n        True\n        '
        return not self.factors

    def as_expr(self):
        if False:
            i = 10
            return i + 15
        'Return the underlying expression.\n\n        Examples\n        ========\n\n        >>> from sympy.core.exprtools import Factors\n        >>> from sympy.abc import x, y\n        >>> Factors((x*y**2).as_powers_dict()).as_expr()\n        x*y**2\n\n        '
        args = []
        for (factor, exp) in self.factors.items():
            if exp != 1:
                if isinstance(exp, Integer):
                    (b, e) = factor.as_base_exp()
                    e = _keep_coeff(exp, e)
                    args.append(b ** e)
                else:
                    args.append(factor ** exp)
            else:
                args.append(factor)
        return Mul(*args)

    def mul(self, other):
        if False:
            print('Hello World!')
        'Return Factors of ``self * other``.\n\n        Examples\n        ========\n\n        >>> from sympy.core.exprtools import Factors\n        >>> from sympy.abc import x, y, z\n        >>> a = Factors((x*y**2).as_powers_dict())\n        >>> b = Factors((x*y/z).as_powers_dict())\n        >>> a.mul(b)\n        Factors({x: 2, y: 3, z: -1})\n        >>> a*b\n        Factors({x: 2, y: 3, z: -1})\n        '
        if not isinstance(other, Factors):
            other = Factors(other)
        if any((f.is_zero for f in (self, other))):
            return Factors(S.Zero)
        factors = dict(self.factors)
        for (factor, exp) in other.factors.items():
            if factor in factors:
                exp = factors[factor] + exp
                if not exp:
                    del factors[factor]
                    continue
            factors[factor] = exp
        return Factors(factors)

    def normal(self, other):
        if False:
            return 10
        'Return ``self`` and ``other`` with ``gcd`` removed from each.\n        The only differences between this and method ``div`` is that this\n        is 1) optimized for the case when there are few factors in common and\n        2) this does not raise an error if ``other`` is zero.\n\n        See Also\n        ========\n        div\n\n        '
        if not isinstance(other, Factors):
            other = Factors(other)
            if other.is_zero:
                return (Factors(), Factors(S.Zero))
            if self.is_zero:
                return (Factors(S.Zero), Factors())
        self_factors = dict(self.factors)
        other_factors = dict(other.factors)
        for (factor, self_exp) in self.factors.items():
            try:
                other_exp = other.factors[factor]
            except KeyError:
                continue
            exp = self_exp - other_exp
            if not exp:
                del self_factors[factor]
                del other_factors[factor]
            elif _isnumber(exp):
                if exp > 0:
                    self_factors[factor] = exp
                    del other_factors[factor]
                else:
                    del self_factors[factor]
                    other_factors[factor] = -exp
            else:
                r = self_exp.extract_additively(other_exp)
                if r is not None:
                    if r:
                        self_factors[factor] = r
                        del other_factors[factor]
                    else:
                        del self_factors[factor]
                        del other_factors[factor]
                else:
                    (sc, sa) = self_exp.as_coeff_Add()
                    if sc:
                        (oc, oa) = other_exp.as_coeff_Add()
                        diff = sc - oc
                        if diff > 0:
                            self_factors[factor] -= oc
                            other_exp = oa
                        elif diff < 0:
                            self_factors[factor] -= sc
                            other_factors[factor] -= sc
                            other_exp = oa - diff
                        else:
                            self_factors[factor] = sa
                            other_exp = oa
                    if other_exp:
                        other_factors[factor] = other_exp
                    else:
                        del other_factors[factor]
        return (Factors(self_factors), Factors(other_factors))

    def div(self, other):
        if False:
            return 10
        'Return ``self`` and ``other`` with ``gcd`` removed from each.\n        This is optimized for the case when there are many factors in common.\n\n        Examples\n        ========\n\n        >>> from sympy.core.exprtools import Factors\n        >>> from sympy.abc import x, y, z\n        >>> from sympy import S\n\n        >>> a = Factors((x*y**2).as_powers_dict())\n        >>> a.div(a)\n        (Factors({}), Factors({}))\n        >>> a.div(x*z)\n        (Factors({y: 2}), Factors({z: 1}))\n\n        The ``/`` operator only gives ``quo``:\n\n        >>> a/x\n        Factors({y: 2})\n\n        Factors treats its factors as though they are all in the numerator, so\n        if you violate this assumption the results will be correct but will\n        not strictly correspond to the numerator and denominator of the ratio:\n\n        >>> a.div(x/z)\n        (Factors({y: 2}), Factors({z: -1}))\n\n        Factors is also naive about bases: it does not attempt any denesting\n        of Rational-base terms, for example the following does not become\n        2**(2*x)/2.\n\n        >>> Factors(2**(2*x + 2)).div(S(8))\n        (Factors({2: 2*x + 2}), Factors({8: 1}))\n\n        factor_terms can clean up such Rational-bases powers:\n\n        >>> from sympy import factor_terms\n        >>> n, d = Factors(2**(2*x + 2)).div(S(8))\n        >>> n.as_expr()/d.as_expr()\n        2**(2*x + 2)/8\n        >>> factor_terms(_)\n        2**(2*x)/2\n\n        '
        (quo, rem) = (dict(self.factors), {})
        if not isinstance(other, Factors):
            other = Factors(other)
            if other.is_zero:
                raise ZeroDivisionError
            if self.is_zero:
                return (Factors(S.Zero), Factors())
        for (factor, exp) in other.factors.items():
            if factor in quo:
                d = quo[factor] - exp
                if _isnumber(d):
                    if d <= 0:
                        del quo[factor]
                    if d >= 0:
                        if d:
                            quo[factor] = d
                        continue
                    exp = -d
                else:
                    r = quo[factor].extract_additively(exp)
                    if r is not None:
                        if r:
                            quo[factor] = r
                        else:
                            del quo[factor]
                    else:
                        other_exp = exp
                        (sc, sa) = quo[factor].as_coeff_Add()
                        if sc:
                            (oc, oa) = other_exp.as_coeff_Add()
                            diff = sc - oc
                            if diff > 0:
                                quo[factor] -= oc
                                other_exp = oa
                            elif diff < 0:
                                quo[factor] -= sc
                                other_exp = oa - diff
                            else:
                                quo[factor] = sa
                                other_exp = oa
                        if other_exp:
                            rem[factor] = other_exp
                        else:
                            assert factor not in rem
                    continue
            rem[factor] = exp
        return (Factors(quo), Factors(rem))

    def quo(self, other):
        if False:
            while True:
                i = 10
        'Return numerator Factor of ``self / other``.\n\n        Examples\n        ========\n\n        >>> from sympy.core.exprtools import Factors\n        >>> from sympy.abc import x, y, z\n        >>> a = Factors((x*y**2).as_powers_dict())\n        >>> b = Factors((x*y/z).as_powers_dict())\n        >>> a.quo(b)  # same as a/b\n        Factors({y: 1})\n        '
        return self.div(other)[0]

    def rem(self, other):
        if False:
            print('Hello World!')
        'Return denominator Factors of ``self / other``.\n\n        Examples\n        ========\n\n        >>> from sympy.core.exprtools import Factors\n        >>> from sympy.abc import x, y, z\n        >>> a = Factors((x*y**2).as_powers_dict())\n        >>> b = Factors((x*y/z).as_powers_dict())\n        >>> a.rem(b)\n        Factors({z: -1})\n        >>> a.rem(a)\n        Factors({})\n        '
        return self.div(other)[1]

    def pow(self, other):
        if False:
            i = 10
            return i + 15
        'Return self raised to a non-negative integer power.\n\n        Examples\n        ========\n\n        >>> from sympy.core.exprtools import Factors\n        >>> from sympy.abc import x, y\n        >>> a = Factors((x*y**2).as_powers_dict())\n        >>> a**2\n        Factors({x: 2, y: 4})\n\n        '
        if isinstance(other, Factors):
            other = other.as_expr()
            if other.is_Integer:
                other = int(other)
        if isinstance(other, SYMPY_INTS) and other >= 0:
            factors = {}
            if other:
                for (factor, exp) in self.factors.items():
                    factors[factor] = exp * other
            return Factors(factors)
        else:
            raise ValueError('expected non-negative integer, got %s' % other)

    def gcd(self, other):
        if False:
            i = 10
            return i + 15
        'Return Factors of ``gcd(self, other)``. The keys are\n        the intersection of factors with the minimum exponent for\n        each factor.\n\n        Examples\n        ========\n\n        >>> from sympy.core.exprtools import Factors\n        >>> from sympy.abc import x, y, z\n        >>> a = Factors((x*y**2).as_powers_dict())\n        >>> b = Factors((x*y/z).as_powers_dict())\n        >>> a.gcd(b)\n        Factors({x: 1, y: 1})\n        '
        if not isinstance(other, Factors):
            other = Factors(other)
            if other.is_zero:
                return Factors(self.factors)
        factors = {}
        for (factor, exp) in self.factors.items():
            (factor, exp) = (sympify(factor), sympify(exp))
            if factor in other.factors:
                lt = (exp - other.factors[factor]).is_negative
                if lt == True:
                    factors[factor] = exp
                elif lt == False:
                    factors[factor] = other.factors[factor]
        return Factors(factors)

    def lcm(self, other):
        if False:
            return 10
        'Return Factors of ``lcm(self, other)`` which are\n        the union of factors with the maximum exponent for\n        each factor.\n\n        Examples\n        ========\n\n        >>> from sympy.core.exprtools import Factors\n        >>> from sympy.abc import x, y, z\n        >>> a = Factors((x*y**2).as_powers_dict())\n        >>> b = Factors((x*y/z).as_powers_dict())\n        >>> a.lcm(b)\n        Factors({x: 1, y: 2, z: -1})\n        '
        if not isinstance(other, Factors):
            other = Factors(other)
            if any((f.is_zero for f in (self, other))):
                return Factors(S.Zero)
        factors = dict(self.factors)
        for (factor, exp) in other.factors.items():
            if factor in factors:
                exp = max(exp, factors[factor])
            factors[factor] = exp
        return Factors(factors)

    def __mul__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self.mul(other)

    def __divmod__(self, other):
        if False:
            print('Hello World!')
        return self.div(other)

    def __truediv__(self, other):
        if False:
            i = 10
            return i + 15
        return self.quo(other)

    def __mod__(self, other):
        if False:
            print('Hello World!')
        return self.rem(other)

    def __pow__(self, other):
        if False:
            i = 10
            return i + 15
        return self.pow(other)

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, Factors):
            other = Factors(other)
        return self.factors == other.factors

    def __ne__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return not self == other

class Term:
    """Efficient representation of ``coeff*(numer/denom)``. """
    __slots__ = ('coeff', 'numer', 'denom')

    def __init__(self, term, numer=None, denom=None):
        if False:
            while True:
                i = 10
        if numer is None and denom is None:
            if not term.is_commutative:
                raise NonCommutativeExpression('commutative expression expected')
            (coeff, factors) = term.as_coeff_mul()
            (numer, denom) = (defaultdict(int), defaultdict(int))
            for factor in factors:
                (base, exp) = decompose_power(factor)
                if base.is_Add:
                    (cont, base) = base.primitive()
                    coeff *= cont ** exp
                if exp > 0:
                    numer[base] += exp
                else:
                    denom[base] += -exp
            numer = Factors(numer)
            denom = Factors(denom)
        else:
            coeff = term
            if numer is None:
                numer = Factors()
            if denom is None:
                denom = Factors()
        self.coeff = coeff
        self.numer = numer
        self.denom = denom

    def __hash__(self):
        if False:
            return 10
        return hash((self.coeff, self.numer, self.denom))

    def __repr__(self):
        if False:
            return 10
        return 'Term(%s, %s, %s)' % (self.coeff, self.numer, self.denom)

    def as_expr(self):
        if False:
            print('Hello World!')
        return self.coeff * (self.numer.as_expr() / self.denom.as_expr())

    def mul(self, other):
        if False:
            return 10
        coeff = self.coeff * other.coeff
        numer = self.numer.mul(other.numer)
        denom = self.denom.mul(other.denom)
        (numer, denom) = numer.normal(denom)
        return Term(coeff, numer, denom)

    def inv(self):
        if False:
            i = 10
            return i + 15
        return Term(1 / self.coeff, self.denom, self.numer)

    def quo(self, other):
        if False:
            while True:
                i = 10
        return self.mul(other.inv())

    def pow(self, other):
        if False:
            print('Hello World!')
        if other < 0:
            return self.inv().pow(-other)
        else:
            return Term(self.coeff ** other, self.numer.pow(other), self.denom.pow(other))

    def gcd(self, other):
        if False:
            return 10
        return Term(self.coeff.gcd(other.coeff), self.numer.gcd(other.numer), self.denom.gcd(other.denom))

    def lcm(self, other):
        if False:
            print('Hello World!')
        return Term(self.coeff.lcm(other.coeff), self.numer.lcm(other.numer), self.denom.lcm(other.denom))

    def __mul__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, Term):
            return self.mul(other)
        else:
            return NotImplemented

    def __truediv__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, Term):
            return self.quo(other)
        else:
            return NotImplemented

    def __pow__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, SYMPY_INTS):
            return self.pow(other)
        else:
            return NotImplemented

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return self.coeff == other.coeff and self.numer == other.numer and (self.denom == other.denom)

    def __ne__(self, other):
        if False:
            return 10
        return not self == other

def _gcd_terms(terms, isprimitive=False, fraction=True):
    if False:
        i = 10
        return i + 15
    'Helper function for :func:`gcd_terms`.\n\n    Parameters\n    ==========\n\n    isprimitive : boolean, optional\n        If ``isprimitive`` is True then the call to primitive\n        for an Add will be skipped. This is useful when the\n        content has already been extracted.\n\n    fraction : boolean, optional\n        If ``fraction`` is True then the expression will appear over a common\n        denominator, the lcm of all term denominators.\n    '
    if isinstance(terms, Basic) and (not isinstance(terms, Tuple)):
        terms = Add.make_args(terms)
    terms = list(map(Term, [t for t in terms if t]))
    if len(terms) == 0:
        return (S.Zero, S.Zero, S.One)
    if len(terms) == 1:
        cont = terms[0].coeff
        numer = terms[0].numer.as_expr()
        denom = terms[0].denom.as_expr()
    else:
        cont = terms[0]
        for term in terms[1:]:
            cont = cont.gcd(term)
        for (i, term) in enumerate(terms):
            terms[i] = term.quo(cont)
        if fraction:
            denom = terms[0].denom
            for term in terms[1:]:
                denom = denom.lcm(term.denom)
            numers = []
            for term in terms:
                numer = term.numer.mul(denom.quo(term.denom))
                numers.append(term.coeff * numer.as_expr())
        else:
            numers = [t.as_expr() for t in terms]
            denom = Term(S.One).numer
        cont = cont.as_expr()
        numer = Add(*numers)
        denom = denom.as_expr()
    if not isprimitive and numer.is_Add:
        (_cont, numer) = numer.primitive()
        cont *= _cont
    return (cont, numer, denom)

def gcd_terms(terms, isprimitive=False, clear=True, fraction=True):
    if False:
        i = 10
        return i + 15
    'Compute the GCD of ``terms`` and put them together.\n\n    Parameters\n    ==========\n\n    terms : Expr\n        Can be an expression or a non-Basic sequence of expressions\n        which will be handled as though they are terms from a sum.\n\n    isprimitive : bool, optional\n        If ``isprimitive`` is True the _gcd_terms will not run the primitive\n        method on the terms.\n\n    clear : bool, optional\n        It controls the removal of integers from the denominator of an Add\n        expression. When True (default), all numerical denominator will be cleared;\n        when False the denominators will be cleared only if all terms had numerical\n        denominators other than 1.\n\n    fraction : bool, optional\n        When True (default), will put the expression over a common\n        denominator.\n\n    Examples\n    ========\n\n    >>> from sympy import gcd_terms\n    >>> from sympy.abc import x, y\n\n    >>> gcd_terms((x + 1)**2*y + (x + 1)*y**2)\n    y*(x + 1)*(x + y + 1)\n    >>> gcd_terms(x/2 + 1)\n    (x + 2)/2\n    >>> gcd_terms(x/2 + 1, clear=False)\n    x/2 + 1\n    >>> gcd_terms(x/2 + y/2, clear=False)\n    (x + y)/2\n    >>> gcd_terms(x/2 + 1/x)\n    (x**2 + 2)/(2*x)\n    >>> gcd_terms(x/2 + 1/x, fraction=False)\n    (x + 2/x)/2\n    >>> gcd_terms(x/2 + 1/x, fraction=False, clear=False)\n    x/2 + 1/x\n\n    >>> gcd_terms(x/2/y + 1/x/y)\n    (x**2 + 2)/(2*x*y)\n    >>> gcd_terms(x/2/y + 1/x/y, clear=False)\n    (x**2/2 + 1)/(x*y)\n    >>> gcd_terms(x/2/y + 1/x/y, clear=False, fraction=False)\n    (x/2 + 1/x)/y\n\n    The ``clear`` flag was ignored in this case because the returned\n    expression was a rational expression, not a simple sum.\n\n    See Also\n    ========\n\n    factor_terms, sympy.polys.polytools.terms_gcd\n\n    '

    def mask(terms):
        if False:
            i = 10
            return i + 15
        'replace nc portions of each term with a unique Dummy symbols\n        and return the replacements to restore them'
        args = [(a, []) if a.is_commutative else a.args_cnc() for a in terms]
        reps = []
        for (i, (c, nc)) in enumerate(args):
            if nc:
                nc = Mul(*nc)
                d = Dummy()
                reps.append((d, nc))
                c.append(d)
                args[i] = Mul(*c)
            else:
                args[i] = c
        return (args, dict(reps))
    isadd = isinstance(terms, Add)
    addlike = isadd or (not isinstance(terms, Basic) and is_sequence(terms, include=set) and (not isinstance(terms, Dict)))
    if addlike:
        if isadd:
            terms = list(terms.args)
        else:
            terms = sympify(terms)
        (terms, reps) = mask(terms)
        (cont, numer, denom) = _gcd_terms(terms, isprimitive, fraction)
        numer = numer.xreplace(reps)
        (coeff, factors) = cont.as_coeff_Mul()
        if not clear:
            (c, _coeff) = coeff.as_coeff_Mul()
            if not c.is_Integer and (not clear) and numer.is_Add:
                (n, d) = c.as_numer_denom()
                _numer = numer / d
                if any((a.as_coeff_Mul()[0].is_Integer for a in _numer.args)):
                    numer = _numer
                    coeff = n * _coeff
        return _keep_coeff(coeff, factors * numer / denom, clear=clear)
    if not isinstance(terms, Basic):
        return terms
    if terms.is_Atom:
        return terms
    if terms.is_Mul:
        (c, args) = terms.as_coeff_mul()
        return _keep_coeff(c, Mul(*[gcd_terms(i, isprimitive, clear, fraction) for i in args]), clear=clear)

    def handle(a):
        if False:
            return 10
        if not isinstance(a, Expr):
            if isinstance(a, Basic):
                if not a.args:
                    return a
                return a.func(*[handle(i) for i in a.args])
            return type(a)([handle(i) for i in a])
        return gcd_terms(a, isprimitive, clear, fraction)
    if isinstance(terms, Dict):
        return Dict(*[(k, handle(v)) for (k, v) in terms.args])
    return terms.func(*[handle(i) for i in terms.args])

def _factor_sum_int(expr, **kwargs):
    if False:
        print('Hello World!')
    'Return Sum or Integral object with factors that are not\n    in the wrt variables removed. In cases where there are additive\n    terms in the function of the object that are independent, the\n    object will be separated into two objects.\n\n    Examples\n    ========\n\n    >>> from sympy import Sum, factor_terms\n    >>> from sympy.abc import x, y\n    >>> factor_terms(Sum(x + y, (x, 1, 3)))\n    y*Sum(1, (x, 1, 3)) + Sum(x, (x, 1, 3))\n    >>> factor_terms(Sum(x*y, (x, 1, 3)))\n    y*Sum(x, (x, 1, 3))\n\n    Notes\n    =====\n\n    If a function in the summand or integrand is replaced\n    with a symbol, then this simplification should not be\n    done or else an incorrect result will be obtained when\n    the symbol is replaced with an expression that depends\n    on the variables of summation/integration:\n\n    >>> eq = Sum(y, (x, 1, 3))\n    >>> factor_terms(eq).subs(y, x).doit()\n    3*x\n    >>> eq.subs(y, x).doit()\n    6\n    '
    result = expr.function
    if result == 0:
        return S.Zero
    limits = expr.limits
    wrt = {i.args[0] for i in limits}
    f = factor_terms(result, **kwargs)
    (i, d) = f.as_independent(*wrt)
    if isinstance(f, Add):
        return i * expr.func(1, *limits) + expr.func(d, *limits)
    else:
        return i * expr.func(d, *limits)

def factor_terms(expr, radical=False, clear=False, fraction=False, sign=True):
    if False:
        print('Hello World!')
    "Remove common factors from terms in all arguments without\n    changing the underlying structure of the expr. No expansion or\n    simplification (and no processing of non-commutatives) is performed.\n\n    Parameters\n    ==========\n\n    radical: bool, optional\n        If radical=True then a radical common to all terms will be factored\n        out of any Add sub-expressions of the expr.\n\n    clear : bool, optional\n        If clear=False (default) then coefficients will not be separated\n        from a single Add if they can be distributed to leave one or more\n        terms with integer coefficients.\n\n    fraction : bool, optional\n        If fraction=True (default is False) then a common denominator will be\n        constructed for the expression.\n\n    sign : bool, optional\n        If sign=True (default) then even if the only factor in common is a -1,\n        it will be factored out of the expression.\n\n    Examples\n    ========\n\n    >>> from sympy import factor_terms, Symbol\n    >>> from sympy.abc import x, y\n    >>> factor_terms(x + x*(2 + 4*y)**3)\n    x*(8*(2*y + 1)**3 + 1)\n    >>> A = Symbol('A', commutative=False)\n    >>> factor_terms(x*A + x*A + x*y*A)\n    x*(y*A + 2*A)\n\n    When ``clear`` is False, a rational will only be factored out of an\n    Add expression if all terms of the Add have coefficients that are\n    fractions:\n\n    >>> factor_terms(x/2 + 1, clear=False)\n    x/2 + 1\n    >>> factor_terms(x/2 + 1, clear=True)\n    (x + 2)/2\n\n    If a -1 is all that can be factored out, to *not* factor it out, the\n    flag ``sign`` must be False:\n\n    >>> factor_terms(-x - y)\n    -(x + y)\n    >>> factor_terms(-x - y, sign=False)\n    -x - y\n    >>> factor_terms(-2*x - 2*y, sign=False)\n    -2*(x + y)\n\n    See Also\n    ========\n\n    gcd_terms, sympy.polys.polytools.terms_gcd\n\n    "

    def do(expr):
        if False:
            for i in range(10):
                print('nop')
        from sympy.concrete.summations import Sum
        from sympy.integrals.integrals import Integral
        is_iterable = iterable(expr)
        if not isinstance(expr, Basic) or expr.is_Atom:
            if is_iterable:
                return type(expr)([do(i) for i in expr])
            return expr
        if expr.is_Pow or expr.is_Function or is_iterable or (not hasattr(expr, 'args_cnc')):
            args = expr.args
            newargs = tuple([do(i) for i in args])
            if newargs == args:
                return expr
            return expr.func(*newargs)
        if isinstance(expr, (Sum, Integral)):
            return _factor_sum_int(expr, radical=radical, clear=clear, fraction=fraction, sign=sign)
        (cont, p) = expr.as_content_primitive(radical=radical, clear=clear)
        if p.is_Add:
            list_args = [do(a) for a in Add.make_args(p)]
            if not any((a.as_coeff_Mul()[0].extract_multiplicatively(-1) is None for a in list_args)):
                cont = -cont
                list_args = [-a for a in list_args]
            special = {}
            for (i, a) in enumerate(list_args):
                (b, e) = a.as_base_exp()
                if e.is_Mul and e != Mul(*e.args):
                    list_args[i] = Dummy()
                    special[list_args[i]] = a
            p = Add._from_args(list_args)
            p = gcd_terms(p, isprimitive=True, clear=clear, fraction=fraction).xreplace(special)
        elif p.args:
            p = p.func(*[do(a) for a in p.args])
        rv = _keep_coeff(cont, p, clear=clear, sign=sign)
        return rv
    expr = sympify(expr)
    return do(expr)

def _mask_nc(eq, name=None):
    if False:
        i = 10
        return i + 15
    "\n    Return ``eq`` with non-commutative objects replaced with Dummy\n    symbols. A dictionary that can be used to restore the original\n    values is returned: if it is None, the expression is noncommutative\n    and cannot be made commutative. The third value returned is a list\n    of any non-commutative symbols that appear in the returned equation.\n\n    Explanation\n    ===========\n\n    All non-commutative objects other than Symbols are replaced with\n    a non-commutative Symbol. Identical objects will be identified\n    by identical symbols.\n\n    If there is only 1 non-commutative object in an expression it will\n    be replaced with a commutative symbol. Otherwise, the non-commutative\n    entities are retained and the calling routine should handle\n    replacements in this case since some care must be taken to keep\n    track of the ordering of symbols when they occur within Muls.\n\n    Parameters\n    ==========\n\n    name : str\n        ``name``, if given, is the name that will be used with numbered Dummy\n        variables that will replace the non-commutative objects and is mainly\n        used for doctesting purposes.\n\n    Examples\n    ========\n\n    >>> from sympy.physics.secondquant import Commutator, NO, F, Fd\n    >>> from sympy import symbols\n    >>> from sympy.core.exprtools import _mask_nc\n    >>> from sympy.abc import x, y\n    >>> A, B, C = symbols('A,B,C', commutative=False)\n\n    One nc-symbol:\n\n    >>> _mask_nc(A**2 - x**2, 'd')\n    (_d0**2 - x**2, {_d0: A}, [])\n\n    Multiple nc-symbols:\n\n    >>> _mask_nc(A**2 - B**2, 'd')\n    (A**2 - B**2, {}, [A, B])\n\n    An nc-object with nc-symbols but no others outside of it:\n\n    >>> _mask_nc(1 + x*Commutator(A, B), 'd')\n    (_d0*x + 1, {_d0: Commutator(A, B)}, [])\n    >>> _mask_nc(NO(Fd(x)*F(y)), 'd')\n    (_d0, {_d0: NO(CreateFermion(x)*AnnihilateFermion(y))}, [])\n\n    Multiple nc-objects:\n\n    >>> eq = x*Commutator(A, B) + x*Commutator(A, C)*Commutator(A, B)\n    >>> _mask_nc(eq, 'd')\n    (x*_d0 + x*_d1*_d0, {_d0: Commutator(A, B), _d1: Commutator(A, C)}, [_d0, _d1])\n\n    Multiple nc-objects and nc-symbols:\n\n    >>> eq = A*Commutator(A, B) + B*Commutator(A, C)\n    >>> _mask_nc(eq, 'd')\n    (A*_d0 + B*_d1, {_d0: Commutator(A, B), _d1: Commutator(A, C)}, [_d0, _d1, A, B])\n\n    "
    name = name or 'mask'

    def numbered_names():
        if False:
            print('Hello World!')
        i = 0
        while True:
            yield (name + str(i))
            i += 1
    names = numbered_names()

    def Dummy(*args, **kwargs):
        if False:
            return 10
        from .symbol import Dummy
        return Dummy(next(names), *args, **kwargs)
    expr = eq
    if expr.is_commutative:
        return (eq, {}, [])
    rep = []
    nc_obj = set()
    nc_syms = set()
    pot = preorder_traversal(expr, keys=default_sort_key)
    for (i, a) in enumerate(pot):
        if any((a == r[0] for r in rep)):
            pot.skip()
        elif not a.is_commutative:
            if a.is_symbol:
                nc_syms.add(a)
                pot.skip()
            elif not (a.is_Add or a.is_Mul or a.is_Pow):
                nc_obj.add(a)
                pot.skip()
    if len(nc_obj) == 1 and (not nc_syms):
        rep.append((nc_obj.pop(), Dummy()))
    elif len(nc_syms) == 1 and (not nc_obj):
        rep.append((nc_syms.pop(), Dummy()))
    nc_obj = sorted(nc_obj, key=default_sort_key)
    for n in nc_obj:
        nc = Dummy(commutative=False)
        rep.append((n, nc))
        nc_syms.add(nc)
    expr = expr.subs(rep)
    nc_syms = list(nc_syms)
    nc_syms.sort(key=default_sort_key)
    return (expr, {v: k for (k, v) in rep}, nc_syms)

def factor_nc(expr):
    if False:
        while True:
            i = 10
    "Return the factored form of ``expr`` while handling non-commutative\n    expressions.\n\n    Examples\n    ========\n\n    >>> from sympy import factor_nc, Symbol\n    >>> from sympy.abc import x\n    >>> A = Symbol('A', commutative=False)\n    >>> B = Symbol('B', commutative=False)\n    >>> factor_nc((x**2 + 2*A*x + A**2).expand())\n    (x + A)**2\n    >>> factor_nc(((x + A)*(x + B)).expand())\n    (x + A)*(x + B)\n    "
    expr = sympify(expr)
    if not isinstance(expr, Expr) or not expr.args:
        return expr
    if not expr.is_Add:
        return expr.func(*[factor_nc(a) for a in expr.args])
    expr = expr.func(*[expand_power_exp(i) for i in expr.args])
    from sympy.polys.polytools import gcd, factor
    (expr, rep, nc_symbols) = _mask_nc(expr)
    if rep:
        return factor(expr).subs(rep)
    else:
        args = [a.args_cnc() for a in Add.make_args(expr)]
        c = g = l = r = S.One
        hit = False
        for (i, a) in enumerate(args):
            if i == 0:
                c = Mul._from_args(a[0])
            elif a[0]:
                c = gcd(c, Mul._from_args(a[0]))
            else:
                c = S.One
        if c is not S.One:
            hit = True
            (c, g) = c.as_coeff_Mul()
            if g is not S.One:
                for (i, (cc, _)) in enumerate(args):
                    cc = list(Mul.make_args(Mul._from_args(list(cc)) / g))
                    args[i][0] = cc
            for (i, (cc, _)) in enumerate(args):
                if cc:
                    cc[0] = cc[0] / c
                else:
                    cc = [1 / c]
                args[i][0] = cc
        for (i, a) in enumerate(args):
            if i == 0:
                n = a[1][:]
            else:
                n = common_prefix(n, a[1])
            if not n:
                if not args[0][1]:
                    break
                (b, e) = args[0][1][0].as_base_exp()
                ok = False
                if e.is_Integer:
                    for t in args:
                        if not t[1]:
                            break
                        (bt, et) = t[1][0].as_base_exp()
                        if et.is_Integer and bt == b:
                            e = min(e, et)
                        else:
                            break
                    else:
                        ok = hit = True
                        l = b ** e
                        il = b ** (-e)
                        for _ in args:
                            _[1][0] = il * _[1][0]
                        break
                if not ok:
                    break
        else:
            hit = True
            lenn = len(n)
            l = Mul(*n)
            for _ in args:
                _[1] = _[1][lenn:]
        for (i, a) in enumerate(args):
            if i == 0:
                n = a[1][:]
            else:
                n = common_suffix(n, a[1])
            if not n:
                if not args[0][1]:
                    break
                (b, e) = args[0][1][-1].as_base_exp()
                ok = False
                if e.is_Integer:
                    for t in args:
                        if not t[1]:
                            break
                        (bt, et) = t[1][-1].as_base_exp()
                        if et.is_Integer and bt == b:
                            e = min(e, et)
                        else:
                            break
                    else:
                        ok = hit = True
                        r = b ** e
                        il = b ** (-e)
                        for _ in args:
                            _[1][-1] = _[1][-1] * il
                        break
                if not ok:
                    break
        else:
            hit = True
            lenn = len(n)
            r = Mul(*n)
            for _ in args:
                _[1] = _[1][:len(_[1]) - lenn]
        if hit:
            mid = Add(*[Mul(*cc) * Mul(*nc) for (cc, nc) in args])
        else:
            mid = expr
        from sympy.simplify.powsimp import powsimp
        rep1 = [(n, Dummy()) for n in sorted(nc_symbols, key=default_sort_key)]
        unrep1 = [(v, k) for (k, v) in rep1]
        unrep1.reverse()
        (new_mid, r2, _) = _mask_nc(mid.subs(rep1))
        new_mid = powsimp(factor(new_mid))
        new_mid = new_mid.subs(r2).subs(unrep1)
        if new_mid.is_Pow:
            return _keep_coeff(c, g * l * new_mid * r)
        if new_mid.is_Mul:

            def _pemexpand(expr):
                if False:
                    for i in range(10):
                        print('nop')
                'Expand with the minimal set of hints necessary to check the result.'
                return expr.expand(deep=True, mul=True, power_exp=True, power_base=False, basic=False, multinomial=True, log=False)
            cfac = []
            ncfac = []
            for f in new_mid.args:
                if f.is_commutative:
                    cfac.append(f)
                else:
                    (b, e) = f.as_base_exp()
                    if e.is_Integer:
                        ncfac.extend([b] * e)
                    else:
                        ncfac.append(f)
            pre_mid = g * Mul(*cfac) * l
            target = _pemexpand(expr / c)
            for s in variations(ncfac, len(ncfac)):
                ok = pre_mid * Mul(*s) * r
                if _pemexpand(ok) == target:
                    return _keep_coeff(c, ok)
        return _keep_coeff(c, g * l * mid * r)