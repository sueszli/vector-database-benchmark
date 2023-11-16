from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef, UndefinedFunction
from sympy.core.mul import Mul
from sympy.core.relational import Equality, Relational
from sympy.core.singleton import S
from sympy.core.symbol import Symbol, Dummy
from sympy.core.sympify import sympify
from sympy.functions.elementary.piecewise import piecewise_fold, Piecewise
from sympy.logic.boolalg import BooleanFunction
from sympy.matrices.matrices import MatrixBase
from sympy.sets.sets import Interval, Set
from sympy.sets.fancysets import Range
from sympy.tensor.indexed import Idx
from sympy.utilities import flatten
from sympy.utilities.iterables import sift, is_sequence
from sympy.utilities.exceptions import sympy_deprecation_warning

def _common_new(cls, function, *symbols, discrete, **assumptions):
    if False:
        while True:
            i = 10
    'Return either a special return value or the tuple,\n    (function, limits, orientation). This code is common to\n    both ExprWithLimits and AddWithLimits.'
    function = sympify(function)
    if isinstance(function, Equality):
        (limits, orientation) = _process_limits(*symbols, discrete=discrete)
        if not (limits and all((len(limit) == 3 for limit in limits))):
            sympy_deprecation_warning('\n                Creating a indefinite integral with an Eq() argument is\n                deprecated.\n\n                This is because indefinite integrals do not preserve equality\n                due to the arbitrary constants. If you want an equality of\n                indefinite integrals, use Eq(Integral(a, x), Integral(b, x))\n                explicitly.\n                ', deprecated_since_version='1.6', active_deprecations_target='deprecated-indefinite-integral-eq', stacklevel=5)
        lhs = function.lhs
        rhs = function.rhs
        return Equality(cls(lhs, *symbols, **assumptions), cls(rhs, *symbols, **assumptions))
    if function is S.NaN:
        return S.NaN
    if symbols:
        (limits, orientation) = _process_limits(*symbols, discrete=discrete)
        for (i, li) in enumerate(limits):
            if len(li) == 4:
                function = function.subs(li[0], li[-1])
                limits[i] = Tuple(*li[:-1])
    else:
        free = function.free_symbols
        if len(free) != 1:
            raise ValueError('specify dummy variables for %s' % function)
        (limits, orientation) = ([Tuple(s) for s in free], 1)
    while cls == type(function):
        limits = list(function.limits) + limits
        function = function.function
    reps = {}
    symbols_of_integration = {i[0] for i in limits}
    for p in function.atoms(Piecewise):
        if not p.has(*symbols_of_integration):
            reps[p] = Dummy()
    function = function.xreplace(reps)
    function = piecewise_fold(function)
    function = function.xreplace({v: k for (k, v) in reps.items()})
    return (function, limits, orientation)

def _process_limits(*symbols, discrete=None):
    if False:
        for i in range(10):
            print('nop')
    'Process the list of symbols and convert them to canonical limits,\n    storing them as Tuple(symbol, lower, upper). The orientation of\n    the function is also returned when the upper limit is missing\n    so (x, 1, None) becomes (x, None, 1) and the orientation is changed.\n    In the case that a limit is specified as (symbol, Range), a list of\n    length 4 may be returned if a change of variables is needed; the\n    expression that should replace the symbol in the expression is\n    the fourth element in the list.\n    '
    limits = []
    orientation = 1
    if discrete is None:
        err_msg = 'discrete must be True or False'
    elif discrete:
        err_msg = 'use Range, not Interval or Relational'
    else:
        err_msg = 'use Interval or Relational, not Range'
    for V in symbols:
        if isinstance(V, (Relational, BooleanFunction)):
            if discrete:
                raise TypeError(err_msg)
            variable = V.atoms(Symbol).pop()
            V = (variable, V.as_set())
        elif isinstance(V, Symbol) or getattr(V, '_diff_wrt', False):
            if isinstance(V, Idx):
                if V.lower is None or V.upper is None:
                    limits.append(Tuple(V))
                else:
                    limits.append(Tuple(V, V.lower, V.upper))
            else:
                limits.append(Tuple(V))
            continue
        if is_sequence(V) and (not isinstance(V, Set)):
            if len(V) == 2 and isinstance(V[1], Set):
                V = list(V)
                if isinstance(V[1], Interval):
                    if discrete:
                        raise TypeError(err_msg)
                    V[1:] = (V[1].inf, V[1].sup)
                elif isinstance(V[1], Range):
                    if not discrete:
                        raise TypeError(err_msg)
                    lo = V[1].inf
                    hi = V[1].sup
                    dx = abs(V[1].step)
                    if dx == 1:
                        V[1:] = [lo, hi]
                    elif lo is not S.NegativeInfinity:
                        V = [V[0]] + [0, (hi - lo) // dx, dx * V[0] + lo]
                    else:
                        V = [V[0]] + [0, S.Infinity, -dx * V[0] + hi]
                else:
                    raise NotImplementedError('expecting Range' if discrete else 'Relational or single Interval')
            V = sympify(flatten(V))
            if isinstance(V[0], (Symbol, Idx)) or getattr(V[0], '_diff_wrt', False):
                newsymbol = V[0]
                if len(V) == 3:
                    if V[2] is None and V[1] is not None:
                        orientation *= -1
                    V = [newsymbol] + [i for i in V[1:] if i is not None]
                lenV = len(V)
                if not isinstance(newsymbol, Idx) or lenV == 3:
                    if lenV == 4:
                        limits.append(Tuple(*V))
                        continue
                    if lenV == 3:
                        if isinstance(newsymbol, Idx):
                            (lo, hi) = (newsymbol.lower, newsymbol.upper)
                            try:
                                if lo is not None and (not bool(V[1] >= lo)):
                                    raise ValueError('Summation will set Idx value too low.')
                            except TypeError:
                                pass
                            try:
                                if hi is not None and (not bool(V[2] <= hi)):
                                    raise ValueError('Summation will set Idx value too high.')
                            except TypeError:
                                pass
                        limits.append(Tuple(*V))
                        continue
                    if lenV == 1 or (lenV == 2 and V[1] is None):
                        limits.append(Tuple(newsymbol))
                        continue
                    elif lenV == 2:
                        limits.append(Tuple(newsymbol, V[1]))
                        continue
        raise ValueError('Invalid limits given: %s' % str(symbols))
    return (limits, orientation)

class ExprWithLimits(Expr):
    __slots__ = ('is_commutative',)

    def __new__(cls, function, *symbols, **assumptions):
        if False:
            return 10
        from sympy.concrete.products import Product
        pre = _common_new(cls, function, *symbols, discrete=issubclass(cls, Product), **assumptions)
        if isinstance(pre, tuple):
            (function, limits, _) = pre
        else:
            return pre
        if any((len(l) != 3 or None in l for l in limits)):
            raise ValueError('ExprWithLimits requires values for lower and upper bounds.')
        obj = Expr.__new__(cls, **assumptions)
        arglist = [function]
        arglist.extend(limits)
        obj._args = tuple(arglist)
        obj.is_commutative = function.is_commutative
        return obj

    @property
    def function(self):
        if False:
            i = 10
            return i + 15
        'Return the function applied across limits.\n\n        Examples\n        ========\n\n        >>> from sympy import Integral\n        >>> from sympy.abc import x\n        >>> Integral(x**2, (x,)).function\n        x**2\n\n        See Also\n        ========\n\n        limits, variables, free_symbols\n        '
        return self._args[0]

    @property
    def kind(self):
        if False:
            while True:
                i = 10
        return self.function.kind

    @property
    def limits(self):
        if False:
            return 10
        'Return the limits of expression.\n\n        Examples\n        ========\n\n        >>> from sympy import Integral\n        >>> from sympy.abc import x, i\n        >>> Integral(x**i, (i, 1, 3)).limits\n        ((i, 1, 3),)\n\n        See Also\n        ========\n\n        function, variables, free_symbols\n        '
        return self._args[1:]

    @property
    def variables(self):
        if False:
            return 10
        'Return a list of the limit variables.\n\n        >>> from sympy import Sum\n        >>> from sympy.abc import x, i\n        >>> Sum(x**i, (i, 1, 3)).variables\n        [i]\n\n        See Also\n        ========\n\n        function, limits, free_symbols\n        as_dummy : Rename dummy variables\n        sympy.integrals.integrals.Integral.transform : Perform mapping on the dummy variable\n        '
        return [l[0] for l in self.limits]

    @property
    def bound_symbols(self):
        if False:
            i = 10
            return i + 15
        'Return only variables that are dummy variables.\n\n        Examples\n        ========\n\n        >>> from sympy import Integral\n        >>> from sympy.abc import x, i, j, k\n        >>> Integral(x**i, (i, 1, 3), (j, 2), k).bound_symbols\n        [i, j]\n\n        See Also\n        ========\n\n        function, limits, free_symbols\n        as_dummy : Rename dummy variables\n        sympy.integrals.integrals.Integral.transform : Perform mapping on the dummy variable\n        '
        return [l[0] for l in self.limits if len(l) != 1]

    @property
    def free_symbols(self):
        if False:
            while True:
                i = 10
        '\n        This method returns the symbols in the object, excluding those\n        that take on a specific value (i.e. the dummy symbols).\n\n        Examples\n        ========\n\n        >>> from sympy import Sum\n        >>> from sympy.abc import x, y\n        >>> Sum(x, (x, y, 1)).free_symbols\n        {y}\n        '
        (function, limits) = (self.function, self.limits)
        reps = {i[0]: i[0] if i[0].free_symbols == {i[0]} else Dummy() for i in self.limits}
        function = function.xreplace(reps)
        isyms = function.free_symbols
        for xab in limits:
            v = reps[xab[0]]
            if len(xab) == 1:
                isyms.add(v)
                continue
            if v in isyms:
                isyms.remove(v)
            for i in xab[1:]:
                isyms.update(i.free_symbols)
        reps = {v: k for (k, v) in reps.items()}
        return {reps.get(_, _) for _ in isyms}

    @property
    def is_number(self):
        if False:
            i = 10
            return i + 15
        'Return True if the Sum has no free symbols, else False.'
        return not self.free_symbols

    def _eval_interval(self, x, a, b):
        if False:
            i = 10
            return i + 15
        limits = [i if i[0] != x else (x, a, b) for i in self.limits]
        integrand = self.function
        return self.func(integrand, *limits)

    def _eval_subs(self, old, new):
        if False:
            i = 10
            return i + 15
        '\n        Perform substitutions over non-dummy variables\n        of an expression with limits.  Also, can be used\n        to specify point-evaluation of an abstract antiderivative.\n\n        Examples\n        ========\n\n        >>> from sympy import Sum, oo\n        >>> from sympy.abc import s, n\n        >>> Sum(1/n**s, (n, 1, oo)).subs(s, 2)\n        Sum(n**(-2), (n, 1, oo))\n\n        >>> from sympy import Integral\n        >>> from sympy.abc import x, a\n        >>> Integral(a*x**2, x).subs(x, 4)\n        Integral(a*x**2, (x, 4))\n\n        See Also\n        ========\n\n        variables : Lists the integration variables\n        transform : Perform mapping on the dummy variable for integrals\n        change_index : Perform mapping on the sum and product dummy variables\n\n        '
        (func, limits) = (self.function, list(self.limits))
        limits.reverse()
        if not isinstance(old, Symbol) or old.free_symbols.intersection(self.free_symbols):
            sub_into_func = True
            for (i, xab) in enumerate(limits):
                if 1 == len(xab) and old == xab[0]:
                    if new._diff_wrt:
                        xab = (new,)
                    else:
                        xab = (old, old)
                limits[i] = Tuple(xab[0], *[l._subs(old, new) for l in xab[1:]])
                if len(xab[0].free_symbols.intersection(old.free_symbols)) != 0:
                    sub_into_func = False
                    break
            if isinstance(old, (AppliedUndef, UndefinedFunction)):
                sy2 = set(self.variables).intersection(set(new.atoms(Symbol)))
                sy1 = set(self.variables).intersection(set(old.args))
                if not sy2.issubset(sy1):
                    raise ValueError('substitution cannot create dummy dependencies')
                sub_into_func = True
            if sub_into_func:
                func = func.subs(old, new)
        else:
            for (i, xab) in enumerate(limits):
                if len(xab) == 3:
                    limits[i] = Tuple(xab[0], *[l._subs(old, new) for l in xab[1:]])
                    if old == xab[0]:
                        break
        for (i, xab) in enumerate(limits):
            if len(xab) == 2 and (xab[0] - xab[1]).is_zero:
                limits[i] = Tuple(xab[0])
        limits.reverse()
        return self.func(func, *limits)

    @property
    def has_finite_limits(self):
        if False:
            print('Hello World!')
        "\n        Returns True if the limits are known to be finite, either by the\n        explicit bounds, assumptions on the bounds, or assumptions on the\n        variables.  False if known to be infinite, based on the bounds.\n        None if not enough information is available to determine.\n\n        Examples\n        ========\n\n        >>> from sympy import Sum, Integral, Product, oo, Symbol\n        >>> x = Symbol('x')\n        >>> Sum(x, (x, 1, 8)).has_finite_limits\n        True\n\n        >>> Integral(x, (x, 1, oo)).has_finite_limits\n        False\n\n        >>> M = Symbol('M')\n        >>> Sum(x, (x, 1, M)).has_finite_limits\n\n        >>> N = Symbol('N', integer=True)\n        >>> Product(x, (x, 1, N)).has_finite_limits\n        True\n\n        See Also\n        ========\n\n        has_reversed_limits\n\n        "
        ret_None = False
        for lim in self.limits:
            if len(lim) == 3:
                if any((l.is_infinite for l in lim[1:])):
                    return False
                elif any((l.is_infinite is None for l in lim[1:])):
                    if lim[0].is_infinite is None:
                        ret_None = True
            elif lim[0].is_infinite is None:
                ret_None = True
        if ret_None:
            return None
        return True

    @property
    def has_reversed_limits(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns True if the limits are known to be in reversed order, either\n        by the explicit bounds, assumptions on the bounds, or assumptions on the\n        variables.  False if known to be in normal order, based on the bounds.\n        None if not enough information is available to determine.\n\n        Examples\n        ========\n\n        >>> from sympy import Sum, Integral, Product, oo, Symbol\n        >>> x = Symbol('x')\n        >>> Sum(x, (x, 8, 1)).has_reversed_limits\n        True\n\n        >>> Sum(x, (x, 1, oo)).has_reversed_limits\n        False\n\n        >>> M = Symbol('M')\n        >>> Integral(x, (x, 1, M)).has_reversed_limits\n\n        >>> N = Symbol('N', integer=True, positive=True)\n        >>> Sum(x, (x, 1, N)).has_reversed_limits\n        False\n\n        >>> Product(x, (x, 2, N)).has_reversed_limits\n\n        >>> Product(x, (x, 2, N)).subs(N, N + 2).has_reversed_limits\n        False\n\n        See Also\n        ========\n\n        sympy.concrete.expr_with_intlimits.ExprWithIntLimits.has_empty_sequence\n\n        "
        ret_None = False
        for lim in self.limits:
            if len(lim) == 3:
                (var, a, b) = lim
                dif = b - a
                if dif.is_extended_negative:
                    return True
                elif dif.is_extended_nonnegative:
                    continue
                else:
                    ret_None = True
            else:
                return None
        if ret_None:
            return None
        return False

class AddWithLimits(ExprWithLimits):
    """Represents unevaluated oriented additions.
        Parent class for Integral and Sum.
    """
    __slots__ = ()

    def __new__(cls, function, *symbols, **assumptions):
        if False:
            print('Hello World!')
        from sympy.concrete.summations import Sum
        pre = _common_new(cls, function, *symbols, discrete=issubclass(cls, Sum), **assumptions)
        if isinstance(pre, tuple):
            (function, limits, orientation) = pre
        else:
            return pre
        obj = Expr.__new__(cls, **assumptions)
        arglist = [orientation * function]
        arglist.extend(limits)
        obj._args = tuple(arglist)
        obj.is_commutative = function.is_commutative
        return obj

    def _eval_adjoint(self):
        if False:
            while True:
                i = 10
        if all((x.is_real for x in flatten(self.limits))):
            return self.func(self.function.adjoint(), *self.limits)
        return None

    def _eval_conjugate(self):
        if False:
            for i in range(10):
                print('nop')
        if all((x.is_real for x in flatten(self.limits))):
            return self.func(self.function.conjugate(), *self.limits)
        return None

    def _eval_transpose(self):
        if False:
            while True:
                i = 10
        if all((x.is_real for x in flatten(self.limits))):
            return self.func(self.function.transpose(), *self.limits)
        return None

    def _eval_factor(self, **hints):
        if False:
            print('Hello World!')
        if 1 == len(self.limits):
            summand = self.function.factor(**hints)
            if summand.is_Mul:
                out = sift(summand.args, lambda w: w.is_commutative and (not set(self.variables) & w.free_symbols))
                return Mul(*out[True]) * self.func(Mul(*out[False]), *self.limits)
        else:
            summand = self.func(self.function, *self.limits[0:-1]).factor()
            if not summand.has(self.variables[-1]):
                return self.func(1, [self.limits[-1]]).doit() * summand
            elif isinstance(summand, Mul):
                return self.func(summand, self.limits[-1]).factor()
        return self

    def _eval_expand_basic(self, **hints):
        if False:
            while True:
                i = 10
        summand = self.function.expand(**hints)
        force = hints.get('force', False)
        if summand.is_Add and (force or (summand.is_commutative and self.has_finite_limits is not False)):
            return Add(*[self.func(i, *self.limits) for i in summand.args])
        elif isinstance(summand, MatrixBase):
            return summand.applyfunc(lambda x: self.func(x, *self.limits))
        elif summand != self.function:
            return self.func(summand, *self.limits)
        return self