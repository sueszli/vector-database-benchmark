from typing import Tuple as tTuple
from sympy.concrete.expr_with_limits import AddWithLimits
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.function import diff
from sympy.core.logic import fuzzy_bool
from sympy.core.mul import Mul
from sympy.core.numbers import oo, pi
from sympy.core.relational import Ne
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol, Wild
from sympy.core.sympify import sympify
from sympy.functions import Piecewise, sqrt, piecewise_fold, tan, cot, atan
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.complexes import Abs, sign
from sympy.functions.elementary.miscellaneous import Min, Max
from sympy.functions.special.singularity_functions import Heaviside
from .rationaltools import ratint
from sympy.matrices import MatrixBase
from sympy.polys import Poly, PolynomialError
from sympy.series.formal import FormalPowerSeries
from sympy.series.limits import limit
from sympy.series.order import Order
from sympy.tensor.functions import shape
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import filldedent

class Integral(AddWithLimits):
    """Represents unevaluated integral."""
    __slots__ = ()
    args: tTuple[Expr, Tuple]

    def __new__(cls, function, *symbols, **assumptions):
        if False:
            for i in range(10):
                print('nop')
        'Create an unevaluated integral.\n\n        Explanation\n        ===========\n\n        Arguments are an integrand followed by one or more limits.\n\n        If no limits are given and there is only one free symbol in the\n        expression, that symbol will be used, otherwise an error will be\n        raised.\n\n        >>> from sympy import Integral\n        >>> from sympy.abc import x, y\n        >>> Integral(x)\n        Integral(x, x)\n        >>> Integral(y)\n        Integral(y, y)\n\n        When limits are provided, they are interpreted as follows (using\n        ``x`` as though it were the variable of integration):\n\n            (x,) or x - indefinite integral\n            (x, a) - "evaluate at" integral is an abstract antiderivative\n            (x, a, b) - definite integral\n\n        The ``as_dummy`` method can be used to see which symbols cannot be\n        targeted by subs: those with a prepended underscore cannot be\n        changed with ``subs``. (Also, the integration variables themselves --\n        the first element of a limit -- can never be changed by subs.)\n\n        >>> i = Integral(x, x)\n        >>> at = Integral(x, (x, x))\n        >>> i.as_dummy()\n        Integral(x, x)\n        >>> at.as_dummy()\n        Integral(_0, (_0, x))\n\n        '
        if hasattr(function, '_eval_Integral'):
            return function._eval_Integral(*symbols, **assumptions)
        if isinstance(function, Poly):
            sympy_deprecation_warning('\n                integrate(Poly) and Integral(Poly) are deprecated. Instead,\n                use the Poly.integrate() method, or convert the Poly to an\n                Expr first with the Poly.as_expr() method.\n                ', deprecated_since_version='1.6', active_deprecations_target='deprecated-integrate-poly')
        obj = AddWithLimits.__new__(cls, function, *symbols, **assumptions)
        return obj

    def __getnewargs__(self):
        if False:
            print('Hello World!')
        return (self.function,) + tuple([tuple(xab) for xab in self.limits])

    @property
    def free_symbols(self):
        if False:
            print('Hello World!')
        '\n        This method returns the symbols that will exist when the\n        integral is evaluated. This is useful if one is trying to\n        determine whether an integral depends on a certain\n        symbol or not.\n\n        Examples\n        ========\n\n        >>> from sympy import Integral\n        >>> from sympy.abc import x, y\n        >>> Integral(x, (x, y, 1)).free_symbols\n        {y}\n\n        See Also\n        ========\n\n        sympy.concrete.expr_with_limits.ExprWithLimits.function\n        sympy.concrete.expr_with_limits.ExprWithLimits.limits\n        sympy.concrete.expr_with_limits.ExprWithLimits.variables\n        '
        return super().free_symbols

    def _eval_is_zero(self):
        if False:
            while True:
                i = 10
        if self.function.is_zero:
            return True
        got_none = False
        for l in self.limits:
            if len(l) == 3:
                z = l[1] == l[2] or (l[1] - l[2]).is_zero
                if z:
                    return True
                elif z is None:
                    got_none = True
        free = self.function.free_symbols
        for xab in self.limits:
            if len(xab) == 1:
                free.add(xab[0])
                continue
            if len(xab) == 2 and xab[0] not in free:
                if xab[1].is_zero:
                    return True
                elif xab[1].is_zero is None:
                    got_none = True
            free.discard(xab[0])
            for i in xab[1:]:
                free.update(i.free_symbols)
        if self.function.is_zero is False and got_none is False:
            return False

    def transform(self, x, u):
        if False:
            print('Hello World!')
        '\n        Performs a change of variables from `x` to `u` using the relationship\n        given by `x` and `u` which will define the transformations `f` and `F`\n        (which are inverses of each other) as follows:\n\n        1) If `x` is a Symbol (which is a variable of integration) then `u`\n           will be interpreted as some function, f(u), with inverse F(u).\n           This, in effect, just makes the substitution of x with f(x).\n\n        2) If `u` is a Symbol then `x` will be interpreted as some function,\n           F(x), with inverse f(u). This is commonly referred to as\n           u-substitution.\n\n        Once f and F have been identified, the transformation is made as\n        follows:\n\n        .. math:: \\int_a^b x \\mathrm{d}x \\rightarrow \\int_{F(a)}^{F(b)} f(x)\n                  \\frac{\\mathrm{d}}{\\mathrm{d}x}\n\n        where `F(x)` is the inverse of `f(x)` and the limits and integrand have\n        been corrected so as to retain the same value after integration.\n\n        Notes\n        =====\n\n        The mappings, F(x) or f(u), must lead to a unique integral. Linear\n        or rational linear expression, ``2*x``, ``1/x`` and ``sqrt(x)``, will\n        always work; quadratic expressions like ``x**2 - 1`` are acceptable\n        as long as the resulting integrand does not depend on the sign of\n        the solutions (see examples).\n\n        The integral will be returned unchanged if ``x`` is not a variable of\n        integration.\n\n        ``x`` must be (or contain) only one of of the integration variables. If\n        ``u`` has more than one free symbol then it should be sent as a tuple\n        (``u``, ``uvar``) where ``uvar`` identifies which variable is replacing\n        the integration variable.\n        XXX can it contain another integration variable?\n\n        Examples\n        ========\n\n        >>> from sympy.abc import a, x, u\n        >>> from sympy import Integral, cos, sqrt\n\n        >>> i = Integral(x*cos(x**2 - 1), (x, 0, 1))\n\n        transform can change the variable of integration\n\n        >>> i.transform(x, u)\n        Integral(u*cos(u**2 - 1), (u, 0, 1))\n\n        transform can perform u-substitution as long as a unique\n        integrand is obtained:\n\n        >>> i.transform(x**2 - 1, u)\n        Integral(cos(u)/2, (u, -1, 0))\n\n        This attempt fails because x = +/-sqrt(u + 1) and the\n        sign does not cancel out of the integrand:\n\n        >>> Integral(cos(x**2 - 1), (x, 0, 1)).transform(x**2 - 1, u)\n        Traceback (most recent call last):\n        ...\n        ValueError:\n        The mapping between F(x) and f(u) did not give a unique integrand.\n\n        transform can do a substitution. Here, the previous\n        result is transformed back into the original expression\n        using "u-substitution":\n\n        >>> ui = _\n        >>> _.transform(sqrt(u + 1), x) == i\n        True\n\n        We can accomplish the same with a regular substitution:\n\n        >>> ui.transform(u, x**2 - 1) == i\n        True\n\n        If the `x` does not contain a symbol of integration then\n        the integral will be returned unchanged. Integral `i` does\n        not have an integration variable `a` so no change is made:\n\n        >>> i.transform(a, x) == i\n        True\n\n        When `u` has more than one free symbol the symbol that is\n        replacing `x` must be identified by passing `u` as a tuple:\n\n        >>> Integral(x, (x, 0, 1)).transform(x, (u + a, u))\n        Integral(a + u, (u, -a, 1 - a))\n        >>> Integral(x, (x, 0, 1)).transform(x, (u + a, a))\n        Integral(a + u, (a, -u, 1 - u))\n\n        See Also\n        ========\n\n        sympy.concrete.expr_with_limits.ExprWithLimits.variables : Lists the integration variables\n        as_dummy : Replace integration variables with dummy ones\n        '
        d = Dummy('d')
        xfree = x.free_symbols.intersection(self.variables)
        if len(xfree) > 1:
            raise ValueError('F(x) can only contain one of: %s' % self.variables)
        xvar = xfree.pop() if xfree else d
        if xvar not in self.variables:
            return self
        u = sympify(u)
        if isinstance(u, Expr):
            ufree = u.free_symbols
            if len(ufree) == 0:
                raise ValueError(filldedent('\n                f(u) cannot be a constant'))
            if len(ufree) > 1:
                raise ValueError(filldedent('\n                When f(u) has more than one free symbol, the one replacing x\n                must be identified: pass f(u) as (f(u), u)'))
            uvar = ufree.pop()
        else:
            (u, uvar) = u
            if uvar not in u.free_symbols:
                raise ValueError(filldedent("\n                Expecting a tuple (expr, symbol) where symbol identified\n                a free symbol in expr, but symbol is not in expr's free\n                symbols."))
            if not isinstance(uvar, Symbol):
                raise ValueError(filldedent("\n                Expecting a tuple (expr, symbol) but didn't get\n                a symbol; got %s" % uvar))
        if x.is_Symbol and u.is_Symbol:
            return self.xreplace({x: u})
        if not x.is_Symbol and (not u.is_Symbol):
            raise ValueError('either x or u must be a symbol')
        if uvar == xvar:
            return self.transform(x, (u.subs(uvar, d), d)).xreplace({d: uvar})
        if uvar in self.limits:
            raise ValueError(filldedent('\n            u must contain the same variable as in x\n            or a variable that is not already an integration variable'))
        from sympy.solvers.solvers import solve
        if not x.is_Symbol:
            F = [x.subs(xvar, d)]
            soln = solve(u - x, xvar, check=False)
            if not soln:
                raise ValueError('no solution for solve(F(x) - f(u), x)')
            f = [fi.subs(uvar, d) for fi in soln]
        else:
            f = [u.subs(uvar, d)]
            from sympy.simplify.simplify import posify
            (pdiff, reps) = posify(u - x)
            puvar = uvar.subs([(v, k) for (k, v) in reps.items()])
            soln = [s.subs(reps) for s in solve(pdiff, puvar)]
            if not soln:
                raise ValueError('no solution for solve(F(x) - f(u), u)')
            F = [fi.subs(xvar, d) for fi in soln]
        newfuncs = {(self.function.subs(xvar, fi) * fi.diff(d)).subs(d, uvar) for fi in f}
        if len(newfuncs) > 1:
            raise ValueError(filldedent('\n            The mapping between F(x) and f(u) did not give\n            a unique integrand.'))
        newfunc = newfuncs.pop()

        def _calc_limit_1(F, a, b):
            if False:
                while True:
                    i = 10
            '\n            replace d with a, using subs if possible, otherwise limit\n            where sign of b is considered\n            '
            wok = F.subs(d, a)
            if wok is S.NaN or (wok.is_finite is False and a.is_finite):
                return limit(sign(b) * F, d, a)
            return wok

        def _calc_limit(a, b):
            if False:
                i = 10
                return i + 15
            '\n            replace d with a, using subs if possible, otherwise limit\n            where sign of b is considered\n            '
            avals = list({_calc_limit_1(Fi, a, b) for Fi in F})
            if len(avals) > 1:
                raise ValueError(filldedent('\n                The mapping between F(x) and f(u) did not\n                give a unique limit.'))
            return avals[0]
        newlimits = []
        for xab in self.limits:
            sym = xab[0]
            if sym == xvar:
                if len(xab) == 3:
                    (a, b) = xab[1:]
                    (a, b) = (_calc_limit(a, b), _calc_limit(b, a))
                    if fuzzy_bool(a - b > 0):
                        (a, b) = (b, a)
                        newfunc = -newfunc
                    newlimits.append((uvar, a, b))
                elif len(xab) == 2:
                    a = _calc_limit(xab[1], 1)
                    newlimits.append((uvar, a))
                else:
                    newlimits.append(uvar)
            else:
                newlimits.append(xab)
        return self.func(newfunc, *newlimits)

    def doit(self, **hints):
        if False:
            return 10
        '\n        Perform the integration using any hints given.\n\n        Examples\n        ========\n\n        >>> from sympy import Piecewise, S\n        >>> from sympy.abc import x, t\n        >>> p = x**2 + Piecewise((0, x/t < 0), (1, True))\n        >>> p.integrate((t, S(4)/5, 1), (x, -1, 1))\n        1/3\n\n        See Also\n        ========\n\n        sympy.integrals.trigonometry.trigintegrate\n        sympy.integrals.heurisch.heurisch\n        sympy.integrals.rationaltools.ratint\n        as_sum : Approximate the integral using a sum\n        '
        if not hints.get('integrals', True):
            return self
        deep = hints.get('deep', True)
        meijerg = hints.get('meijerg', None)
        conds = hints.get('conds', 'piecewise')
        risch = hints.get('risch', None)
        heurisch = hints.get('heurisch', None)
        manual = hints.get('manual', None)
        if len(list(filter(None, (manual, meijerg, risch, heurisch)))) > 1:
            raise ValueError('At most one of manual, meijerg, risch, heurisch can be True')
        elif manual:
            meijerg = risch = heurisch = False
        elif meijerg:
            manual = risch = heurisch = False
        elif risch:
            manual = meijerg = heurisch = False
        elif heurisch:
            manual = meijerg = risch = False
        eval_kwargs = {'meijerg': meijerg, 'risch': risch, 'manual': manual, 'heurisch': heurisch, 'conds': conds}
        if conds not in ('separate', 'piecewise', 'none'):
            raise ValueError('conds must be one of "separate", "piecewise", "none", got: %s' % conds)
        if risch and any((len(xab) > 1 for xab in self.limits)):
            raise ValueError('risch=True is only allowed for indefinite integrals.')
        if self.is_zero:
            return S.Zero
        from sympy.concrete.summations import Sum
        if isinstance(self.function, Sum):
            if any((v in self.function.limits[0] for v in self.variables)):
                raise ValueError('Limit of the sum cannot be an integration variable.')
            if any((l.is_infinite for l in self.function.limits[0][1:])):
                return self
            _i = self
            _sum = self.function
            return _sum.func(_i.func(_sum.function, *_i.limits).doit(), *_sum.limits).doit()
        function = self.function
        function = function.replace(lambda x: isinstance(x, Heaviside) and x.args[1] * 2 != 1, lambda x: Heaviside(x.args[0]))
        if deep:
            function = function.doit(**hints)
        if function.is_zero:
            return S.Zero
        if isinstance(function, MatrixBase):
            return function.applyfunc(lambda f: self.func(f, *self.limits).doit(**hints))
        if isinstance(function, FormalPowerSeries):
            if len(self.limits) > 1:
                raise NotImplementedError
            xab = self.limits[0]
            if len(xab) > 1:
                return function.integrate(xab, **eval_kwargs)
            else:
                return function.integrate(xab[0], **eval_kwargs)
        reps = {}
        for xab in self.limits:
            if len(xab) != 3:
                continue
            (x, a, b) = xab
            l = (a, b)
            if all((i.is_nonnegative for i in l)) and (not x.is_nonnegative):
                d = Dummy(positive=True)
            elif all((i.is_nonpositive for i in l)) and (not x.is_nonpositive):
                d = Dummy(negative=True)
            elif all((i.is_real for i in l)) and (not x.is_real):
                d = Dummy(real=True)
            else:
                d = None
            if d:
                reps[x] = d
        if reps:
            undo = {v: k for (k, v) in reps.items()}
            did = self.xreplace(reps).doit(**hints)
            if isinstance(did, tuple):
                did = tuple([i.xreplace(undo) for i in did])
            else:
                did = did.xreplace(undo)
            return did
        undone_limits = []
        ulj = set()
        for xab in self.limits:
            if len(xab) == 1:
                uli = set(xab[:1])
            elif len(xab) == 2:
                uli = xab[1].free_symbols
            elif len(xab) == 3:
                uli = xab[1].free_symbols.union(xab[2].free_symbols)
            if xab[0] in ulj or any((v[0] in uli for v in undone_limits)):
                undone_limits.append(xab)
                ulj.update(uli)
                function = self.func(*[function] + [xab])
                factored_function = function.factor()
                if not isinstance(factored_function, Integral):
                    function = factored_function
                continue
            if function.has(Abs, sign) and (len(xab) < 3 and all((x.is_extended_real for x in xab)) or (len(xab) == 3 and all((x.is_extended_real and (not x.is_infinite) for x in xab[1:])))):
                xr = Dummy('xr', real=True)
                function = function.xreplace({xab[0]: xr}).rewrite(Piecewise).xreplace({xr: xab[0]})
            elif function.has(Min, Max):
                function = function.rewrite(Piecewise)
            if function.has(Piecewise) and (not isinstance(function, Piecewise)):
                function = piecewise_fold(function)
            if isinstance(function, Piecewise):
                if len(xab) == 1:
                    antideriv = function._eval_integral(xab[0], **eval_kwargs)
                else:
                    antideriv = self._eval_integral(function, xab[0], **eval_kwargs)
            else:

                def try_meijerg(function, xab):
                    if False:
                        for i in range(10):
                            print('nop')
                    ret = None
                    if len(xab) == 3 and meijerg is not False:
                        (x, a, b) = xab
                        try:
                            res = meijerint_definite(function, x, a, b)
                        except NotImplementedError:
                            _debug('NotImplementedError from meijerint_definite')
                            res = None
                        if res is not None:
                            (f, cond) = res
                            if conds == 'piecewise':
                                u = self.func(function, (x, a, b))
                                return Piecewise((f, cond), (u, True), evaluate=False)
                            elif conds == 'separate':
                                if len(self.limits) != 1:
                                    raise ValueError(filldedent('\n                                        conds=separate not supported in\n                                        multiple integrals'))
                                ret = (f, cond)
                            else:
                                ret = f
                    return ret
                meijerg1 = meijerg
                if meijerg is not False and len(xab) == 3 and xab[1].is_extended_real and xab[2].is_extended_real and (not function.is_Poly) and (xab[1].has(oo, -oo) or xab[2].has(oo, -oo)):
                    ret = try_meijerg(function, xab)
                    if ret is not None:
                        function = ret
                        continue
                    meijerg1 = False
                if meijerg1 is False and meijerg is True:
                    antideriv = None
                else:
                    antideriv = self._eval_integral(function, xab[0], **eval_kwargs)
                    if antideriv is None and meijerg is True:
                        ret = try_meijerg(function, xab)
                        if ret is not None:
                            function = ret
                            continue
            final = hints.get('final', True)
            if final and (not isinstance(antideriv, Integral)) and (antideriv is not None):
                for atan_term in antideriv.atoms(atan):
                    atan_arg = atan_term.args[0]
                    for tan_part in atan_arg.atoms(tan):
                        x1 = Dummy('x1')
                        tan_exp1 = atan_arg.subs(tan_part, x1)
                        coeff = tan_exp1.diff(x1)
                        if x1 not in coeff.free_symbols:
                            a = tan_part.args[0]
                            antideriv = antideriv.subs(atan_term, Add(atan_term, sign(coeff) * pi * floor((a - pi / 2) / pi)))
                    for cot_part in atan_arg.atoms(cot):
                        x1 = Dummy('x1')
                        cot_exp1 = atan_arg.subs(cot_part, x1)
                        coeff = cot_exp1.diff(x1)
                        if x1 not in coeff.free_symbols:
                            a = cot_part.args[0]
                            antideriv = antideriv.subs(atan_term, Add(atan_term, sign(coeff) * pi * floor(a / pi)))
            if antideriv is None:
                undone_limits.append(xab)
                function = self.func(*[function] + [xab]).factor()
                factored_function = function.factor()
                if not isinstance(factored_function, Integral):
                    function = factored_function
                continue
            elif len(xab) == 1:
                function = antideriv
            else:
                if len(xab) == 3:
                    (x, a, b) = xab
                elif len(xab) == 2:
                    (x, b) = xab
                    a = None
                else:
                    raise NotImplementedError
                if deep:
                    if isinstance(a, Basic):
                        a = a.doit(**hints)
                    if isinstance(b, Basic):
                        b = b.doit(**hints)
                if antideriv.is_Poly:
                    gens = list(antideriv.gens)
                    gens.remove(x)
                    antideriv = antideriv.as_expr()
                    function = antideriv._eval_interval(x, a, b)
                    function = Poly(function, *gens)
                else:

                    def is_indef_int(g, x):
                        if False:
                            while True:
                                i = 10
                        return isinstance(g, Integral) and any((i == (x,) for i in g.limits))

                    def eval_factored(f, x, a, b):
                        if False:
                            return 10
                        args = []
                        for g in Mul.make_args(f):
                            if is_indef_int(g, x):
                                args.append(g._eval_interval(x, a, b))
                            else:
                                args.append(g)
                        return Mul(*args)
                    (integrals, others, piecewises) = ([], [], [])
                    for f in Add.make_args(antideriv):
                        if any((is_indef_int(g, x) for g in Mul.make_args(f))):
                            integrals.append(f)
                        elif any((isinstance(g, Piecewise) for g in Mul.make_args(f))):
                            piecewises.append(piecewise_fold(f))
                        else:
                            others.append(f)
                    uneval = Add(*[eval_factored(f, x, a, b) for f in integrals])
                    try:
                        evalued = Add(*others)._eval_interval(x, a, b)
                        evalued_pw = piecewise_fold(Add(*piecewises))._eval_interval(x, a, b)
                        function = uneval + evalued + evalued_pw
                    except NotImplementedError:
                        undone_limits.append(xab)
                        function = self.func(*[function] + [xab])
                        factored_function = function.factor()
                        if not isinstance(factored_function, Integral):
                            function = factored_function
        return function

    def _eval_derivative(self, sym):
        if False:
            print('Hello World!')
        'Evaluate the derivative of the current Integral object by\n        differentiating under the integral sign [1], using the Fundamental\n        Theorem of Calculus [2] when possible.\n\n        Explanation\n        ===========\n\n        Whenever an Integral is encountered that is equivalent to zero or\n        has an integrand that is independent of the variable of integration\n        those integrals are performed. All others are returned as Integral\n        instances which can be resolved with doit() (provided they are integrable).\n\n        References\n        ==========\n\n        .. [1] https://en.wikipedia.org/wiki/Differentiation_under_the_integral_sign\n        .. [2] https://en.wikipedia.org/wiki/Fundamental_theorem_of_calculus\n\n        Examples\n        ========\n\n        >>> from sympy import Integral\n        >>> from sympy.abc import x, y\n        >>> i = Integral(x + y, y, (y, 1, x))\n        >>> i.diff(x)\n        Integral(x + y, (y, x)) + Integral(1, y, (y, 1, x))\n        >>> i.doit().diff(x) == i.diff(x).doit()\n        True\n        >>> i.diff(y)\n        0\n\n        The previous must be true since there is no y in the evaluated integral:\n\n        >>> i.free_symbols\n        {x}\n        >>> i.doit()\n        2*x**3/3 - x/2 - 1/6\n\n        '
        (f, limits) = (self.function, list(self.limits))
        limit = limits.pop(-1)
        if len(limit) == 3:
            (x, a, b) = limit
        elif len(limit) == 2:
            (x, b) = limit
            a = None
        else:
            a = b = None
            x = limit[0]
        if limits:
            f = self.func(f, *tuple(limits))

        def _do(f, ab):
            if False:
                print('Hello World!')
            dab_dsym = diff(ab, sym)
            if not dab_dsym:
                return S.Zero
            if isinstance(f, Integral):
                limits = [(x, x) if len(l) == 1 and l[0] == x else l for l in f.limits]
                f = self.func(f.function, *limits)
            return f.subs(x, ab) * dab_dsym
        rv = S.Zero
        if b is not None:
            rv += _do(f, b)
        if a is not None:
            rv -= _do(f, a)
        if len(limit) == 1 and sym == x:
            arg = f
            rv += arg
        else:
            u = Dummy('u')
            arg = f.subs(x, u).diff(sym).subs(u, x)
            if arg:
                rv += self.func(arg, (x, a, b))
        return rv

    def _eval_integral(self, f, x, meijerg=None, risch=None, manual=None, heurisch=None, conds='piecewise', final=None):
        if False:
            print('Hello World!')
        '\n        Calculate the anti-derivative to the function f(x).\n\n        Explanation\n        ===========\n\n        The following algorithms are applied (roughly in this order):\n\n        1. Simple heuristics (based on pattern matching and integral table):\n\n           - most frequently used functions (e.g. polynomials, products of\n             trig functions)\n\n        2. Integration of rational functions:\n\n           - A complete algorithm for integrating rational functions is\n             implemented (the Lazard-Rioboo-Trager algorithm).  The algorithm\n             also uses the partial fraction decomposition algorithm\n             implemented in apart() as a preprocessor to make this process\n             faster.  Note that the integral of a rational function is always\n             elementary, but in general, it may include a RootSum.\n\n        3. Full Risch algorithm:\n\n           - The Risch algorithm is a complete decision\n             procedure for integrating elementary functions, which means that\n             given any elementary function, it will either compute an\n             elementary antiderivative, or else prove that none exists.\n             Currently, part of transcendental case is implemented, meaning\n             elementary integrals containing exponentials, logarithms, and\n             (soon!) trigonometric functions can be computed.  The algebraic\n             case, e.g., functions containing roots, is much more difficult\n             and is not implemented yet.\n\n           - If the routine fails (because the integrand is not elementary, or\n             because a case is not implemented yet), it continues on to the\n             next algorithms below.  If the routine proves that the integrals\n             is nonelementary, it still moves on to the algorithms below,\n             because we might be able to find a closed-form solution in terms\n             of special functions.  If risch=True, however, it will stop here.\n\n        4. The Meijer G-Function algorithm:\n\n           - This algorithm works by first rewriting the integrand in terms of\n             very general Meijer G-Function (meijerg in SymPy), integrating\n             it, and then rewriting the result back, if possible.  This\n             algorithm is particularly powerful for definite integrals (which\n             is actually part of a different method of Integral), since it can\n             compute closed-form solutions of definite integrals even when no\n             closed-form indefinite integral exists.  But it also is capable\n             of computing many indefinite integrals as well.\n\n           - Another advantage of this method is that it can use some results\n             about the Meijer G-Function to give a result in terms of a\n             Piecewise expression, which allows to express conditionally\n             convergent integrals.\n\n           - Setting meijerg=True will cause integrate() to use only this\n             method.\n\n        5. The "manual integration" algorithm:\n\n           - This algorithm tries to mimic how a person would find an\n             antiderivative by hand, for example by looking for a\n             substitution or applying integration by parts. This algorithm\n             does not handle as many integrands but can return results in a\n             more familiar form.\n\n           - Sometimes this algorithm can evaluate parts of an integral; in\n             this case integrate() will try to evaluate the rest of the\n             integrand using the other methods here.\n\n           - Setting manual=True will cause integrate() to use only this\n             method.\n\n        6. The Heuristic Risch algorithm:\n\n           - This is a heuristic version of the Risch algorithm, meaning that\n             it is not deterministic.  This is tried as a last resort because\n             it can be very slow.  It is still used because not enough of the\n             full Risch algorithm is implemented, so that there are still some\n             integrals that can only be computed using this method.  The goal\n             is to implement enough of the Risch and Meijer G-function methods\n             so that this can be deleted.\n\n             Setting heurisch=True will cause integrate() to use only this\n             method. Set heurisch=False to not use it.\n\n        '
        from sympy.integrals.risch import risch_integrate, NonElementaryIntegral
        from sympy.integrals.manualintegrate import manualintegrate
        if risch:
            try:
                return risch_integrate(f, x, conds=conds)
            except NotImplementedError:
                return None
        if manual:
            try:
                result = manualintegrate(f, x)
                if result is not None and result.func != Integral:
                    return result
            except (ValueError, PolynomialError):
                pass
        eval_kwargs = {'meijerg': meijerg, 'risch': risch, 'manual': manual, 'heurisch': heurisch, 'conds': conds}
        if isinstance(f, Poly) and (not (manual or meijerg or risch)):
            return f.integrate(x)
        if isinstance(f, Piecewise):
            return f.piecewise_integrate(x, **eval_kwargs)
        if not f.has(x):
            return f * x
        poly = f.as_poly(x)
        if poly is not None and (not (manual or meijerg or risch)):
            return poly.integrate().as_expr()
        if risch is not False:
            try:
                (result, i) = risch_integrate(f, x, separate_integral=True, conds=conds)
            except NotImplementedError:
                pass
            else:
                if i:
                    if result == 0:
                        return NonElementaryIntegral(f, x).doit(risch=False)
                    else:
                        return result + i.doit(risch=False)
                else:
                    return result
        from sympy.simplify.fu import sincos_to_sum
        parts = []
        args = Add.make_args(f)
        for g in args:
            (coeff, g) = g.as_independent(x)
            if g is S.One and (not meijerg):
                parts.append(coeff * x)
                continue
            order_term = g.getO()
            if order_term is not None:
                h = self._eval_integral(g.removeO(), x, **eval_kwargs)
                if h is not None:
                    h_order_expr = self._eval_integral(order_term.expr, x, **eval_kwargs)
                    if h_order_expr is not None:
                        h_order_term = order_term.func(h_order_expr, *order_term.variables)
                        parts.append(coeff * (h + h_order_term))
                        continue
                return None
            if g.is_Pow and (not g.exp.has(x)) and (not meijerg):
                a = Wild('a', exclude=[x])
                b = Wild('b', exclude=[x])
                M = g.base.match(a * x + b)
                if M is not None:
                    if g.exp == -1:
                        h = log(g.base)
                    elif conds != 'piecewise':
                        h = g.base ** (g.exp + 1) / (g.exp + 1)
                    else:
                        h1 = log(g.base)
                        h2 = g.base ** (g.exp + 1) / (g.exp + 1)
                        h = Piecewise((h2, Ne(g.exp, -1)), (h1, True))
                    parts.append(coeff * h / M[a])
                    continue
            if g.is_rational_function(x) and (not (manual or meijerg or risch)):
                parts.append(coeff * ratint(g, x))
                continue
            if not (manual or meijerg or risch):
                h = trigintegrate(g, x, conds=conds)
                if h is not None:
                    parts.append(coeff * h)
                    continue
                h = deltaintegrate(g, x)
                if h is not None:
                    parts.append(coeff * h)
                    continue
                from .singularityfunctions import singularityintegrate
                h = singularityintegrate(g, x)
                if h is not None:
                    parts.append(coeff * h)
                    continue
                if risch is not False:
                    try:
                        (h, i) = risch_integrate(g, x, separate_integral=True, conds=conds)
                    except NotImplementedError:
                        h = None
                    else:
                        if i:
                            h = h + i.doit(risch=False)
                        parts.append(coeff * h)
                        continue
                if heurisch is not False:
                    from sympy.integrals.heurisch import heurisch as heurisch_, heurisch_wrapper
                    try:
                        if conds == 'piecewise':
                            h = heurisch_wrapper(g, x, hints=[])
                        else:
                            h = heurisch_(g, x, hints=[])
                    except PolynomialError:
                        h = None
            else:
                h = None
            if meijerg is not False and h is None:
                try:
                    h = meijerint_indefinite(g, x)
                except NotImplementedError:
                    _debug('NotImplementedError from meijerint_definite')
                if h is not None:
                    parts.append(coeff * h)
                    continue
            if h is None and manual is not False:
                try:
                    result = manualintegrate(g, x)
                    if result is not None and (not isinstance(result, Integral)):
                        if result.has(Integral) and (not manual):
                            new_eval_kwargs = eval_kwargs
                            new_eval_kwargs['manual'] = False
                            new_eval_kwargs['final'] = False
                            result = result.func(*[arg.doit(**new_eval_kwargs) if arg.has(Integral) else arg for arg in result.args]).expand(multinomial=False, log=False, power_exp=False, power_base=False)
                        if not result.has(Integral):
                            parts.append(coeff * result)
                            continue
                except (ValueError, PolynomialError):
                    pass
            if not h and len(args) == 1:
                f = sincos_to_sum(f).expand(mul=True, deep=False)
                if f.is_Add:
                    return self._eval_integral(f, x, **eval_kwargs)
            if h is not None:
                parts.append(coeff * h)
            else:
                return None
        return Add(*parts)

    def _eval_lseries(self, x, logx=None, cdir=0):
        if False:
            i = 10
            return i + 15
        expr = self.as_dummy()
        symb = x
        for l in expr.limits:
            if x in l[1:]:
                symb = l[0]
                break
        for term in expr.function.lseries(symb, logx):
            yield integrate(term, *expr.limits)

    def _eval_nseries(self, x, n, logx=None, cdir=0):
        if False:
            while True:
                i = 10
        symb = x
        for l in self.limits:
            if x in l[1:]:
                symb = l[0]
                break
        (terms, order) = self.function.nseries(x=symb, n=n, logx=logx).as_coeff_add(Order)
        order = [o.subs(symb, x) for o in order]
        return integrate(terms, *self.limits) + Add(*order) * x

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        if False:
            i = 10
            return i + 15
        series_gen = self.args[0].lseries(x)
        for leading_term in series_gen:
            if leading_term != 0:
                break
        return integrate(leading_term, *self.args[1:])

    def _eval_simplify(self, **kwargs):
        if False:
            while True:
                i = 10
        expr = factor_terms(self)
        if isinstance(expr, Integral):
            from sympy.simplify.simplify import simplify
            return expr.func(*[simplify(i, **kwargs) for i in expr.args])
        return expr.simplify(**kwargs)

    def as_sum(self, n=None, method='midpoint', evaluate=True):
        if False:
            i = 10
            return i + 15
        "\n        Approximates a definite integral by a sum.\n\n        Parameters\n        ==========\n\n        n :\n            The number of subintervals to use, optional.\n        method :\n            One of: 'left', 'right', 'midpoint', 'trapezoid'.\n        evaluate : bool\n            If False, returns an unevaluated Sum expression. The default\n            is True, evaluate the sum.\n\n        Notes\n        =====\n\n        These methods of approximate integration are described in [1].\n\n        Examples\n        ========\n\n        >>> from sympy import Integral, sin, sqrt\n        >>> from sympy.abc import x, n\n        >>> e = Integral(sin(x), (x, 3, 7))\n        >>> e\n        Integral(sin(x), (x, 3, 7))\n\n        For demonstration purposes, this interval will only be split into 2\n        regions, bounded by [3, 5] and [5, 7].\n\n        The left-hand rule uses function evaluations at the left of each\n        interval:\n\n        >>> e.as_sum(2, 'left')\n        2*sin(5) + 2*sin(3)\n\n        The midpoint rule uses evaluations at the center of each interval:\n\n        >>> e.as_sum(2, 'midpoint')\n        2*sin(4) + 2*sin(6)\n\n        The right-hand rule uses function evaluations at the right of each\n        interval:\n\n        >>> e.as_sum(2, 'right')\n        2*sin(5) + 2*sin(7)\n\n        The trapezoid rule uses function evaluations on both sides of the\n        intervals. This is equivalent to taking the average of the left and\n        right hand rule results:\n\n        >>> e.as_sum(2, 'trapezoid')\n        2*sin(5) + sin(3) + sin(7)\n        >>> (e.as_sum(2, 'left') + e.as_sum(2, 'right'))/2 == _\n        True\n\n        Here, the discontinuity at x = 0 can be avoided by using the\n        midpoint or right-hand method:\n\n        >>> e = Integral(1/sqrt(x), (x, 0, 1))\n        >>> e.as_sum(5).n(4)\n        1.730\n        >>> e.as_sum(10).n(4)\n        1.809\n        >>> e.doit().n(4)  # the actual value is 2\n        2.000\n\n        The left- or trapezoid method will encounter the discontinuity and\n        return infinity:\n\n        >>> e.as_sum(5, 'left')\n        zoo\n\n        The number of intervals can be symbolic. If omitted, a dummy symbol\n        will be used for it.\n\n        >>> e = Integral(x**2, (x, 0, 2))\n        >>> e.as_sum(n, 'right').expand()\n        8/3 + 4/n + 4/(3*n**2)\n\n        This shows that the midpoint rule is more accurate, as its error\n        term decays as the square of n:\n\n        >>> e.as_sum(method='midpoint').expand()\n        8/3 - 2/(3*_n**2)\n\n        A symbolic sum is returned with evaluate=False:\n\n        >>> e.as_sum(n, 'midpoint', evaluate=False)\n        2*Sum((2*_k/n - 1/n)**2, (_k, 1, n))/n\n\n        See Also\n        ========\n\n        Integral.doit : Perform the integration using any hints\n\n        References\n        ==========\n\n        .. [1] https://en.wikipedia.org/wiki/Riemann_sum#Riemann_summation_methods\n        "
        from sympy.concrete.summations import Sum
        limits = self.limits
        if len(limits) > 1:
            raise NotImplementedError('Multidimensional midpoint rule not implemented yet')
        else:
            limit = limits[0]
            if len(limit) != 3 or limit[1].is_finite is False or limit[2].is_finite is False:
                raise ValueError('Expecting a definite integral over a finite interval.')
        if n is None:
            n = Dummy('n', integer=True, positive=True)
        else:
            n = sympify(n)
        if n.is_positive is False or n.is_integer is False or n.is_finite is False:
            raise ValueError('n must be a positive integer, got %s' % n)
        (x, a, b) = limit
        dx = (b - a) / n
        k = Dummy('k', integer=True, positive=True)
        f = self.function
        if method == 'left':
            result = dx * Sum(f.subs(x, a + (k - 1) * dx), (k, 1, n))
        elif method == 'right':
            result = dx * Sum(f.subs(x, a + k * dx), (k, 1, n))
        elif method == 'midpoint':
            result = dx * Sum(f.subs(x, a + k * dx - dx / 2), (k, 1, n))
        elif method == 'trapezoid':
            result = dx * ((f.subs(x, a) + f.subs(x, b)) / 2 + Sum(f.subs(x, a + k * dx), (k, 1, n - 1)))
        else:
            raise ValueError('Unknown method %s' % method)
        return result.doit() if evaluate else result

    def principal_value(self, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Compute the Cauchy Principal Value of the definite integral of a real function in the given interval\n        on the real axis.\n\n        Explanation\n        ===========\n\n        In mathematics, the Cauchy principal value, is a method for assigning values to certain improper\n        integrals which would otherwise be undefined.\n\n        Examples\n        ========\n\n        >>> from sympy import Integral, oo\n        >>> from sympy.abc import x\n        >>> Integral(x+1, (x, -oo, oo)).principal_value()\n        oo\n        >>> f = 1 / (x**3)\n        >>> Integral(f, (x, -oo, oo)).principal_value()\n        0\n        >>> Integral(f, (x, -10, 10)).principal_value()\n        0\n        >>> Integral(f, (x, -10, oo)).principal_value() + Integral(f, (x, -oo, 10)).principal_value()\n        0\n\n        References\n        ==========\n\n        .. [1] https://en.wikipedia.org/wiki/Cauchy_principal_value\n        .. [2] https://mathworld.wolfram.com/CauchyPrincipalValue.html\n        '
        if len(self.limits) != 1 or len(list(self.limits[0])) != 3:
            raise ValueError("You need to insert a variable, lower_limit, and upper_limit correctly to calculate cauchy's principal value")
        (x, a, b) = self.limits[0]
        if not (a.is_comparable and b.is_comparable and (a <= b)):
            raise ValueError("The lower_limit must be smaller than or equal to the upper_limit to calculate cauchy's principal value. Also, a and b need to be comparable.")
        if a == b:
            return S.Zero
        from sympy.calculus.singularities import singularities
        r = Dummy('r')
        f = self.function
        singularities_list = [s for s in singularities(f, x) if s.is_comparable and a <= s <= b]
        for i in singularities_list:
            if i in (a, b):
                raise ValueError('The principal value is not defined in the given interval due to singularity at %d.' % i)
        F = integrate(f, x, **kwargs)
        if F.has(Integral):
            return self
        if a is -oo and b is oo:
            I = limit(F - F.subs(x, -x), x, oo)
        else:
            I = limit(F, x, b, '-') - limit(F, x, a, '+')
        for s in singularities_list:
            I += limit(F.subs(x, s - r) - F.subs(x, s + r), r, 0, '+')
        return I

def integrate(*args, meijerg=None, conds='piecewise', risch=None, heurisch=None, manual=None, **kwargs):
    if False:
        while True:
            i = 10
    "integrate(f, var, ...)\n\n    .. deprecated:: 1.6\n\n       Using ``integrate()`` with :class:`~.Poly` is deprecated. Use\n       :meth:`.Poly.integrate` instead. See :ref:`deprecated-integrate-poly`.\n\n    Explanation\n    ===========\n\n    Compute definite or indefinite integral of one or more variables\n    using Risch-Norman algorithm and table lookup. This procedure is\n    able to handle elementary algebraic and transcendental functions\n    and also a huge class of special functions, including Airy,\n    Bessel, Whittaker and Lambert.\n\n    var can be:\n\n    - a symbol                   -- indefinite integration\n    - a tuple (symbol, a)        -- indefinite integration with result\n                                    given with ``a`` replacing ``symbol``\n    - a tuple (symbol, a, b)     -- definite integration\n\n    Several variables can be specified, in which case the result is\n    multiple integration. (If var is omitted and the integrand is\n    univariate, the indefinite integral in that variable will be performed.)\n\n    Indefinite integrals are returned without terms that are independent\n    of the integration variables. (see examples)\n\n    Definite improper integrals often entail delicate convergence\n    conditions. Pass conds='piecewise', 'separate' or 'none' to have\n    these returned, respectively, as a Piecewise function, as a separate\n    result (i.e. result will be a tuple), or not at all (default is\n    'piecewise').\n\n    **Strategy**\n\n    SymPy uses various approaches to definite integration. One method is to\n    find an antiderivative for the integrand, and then use the fundamental\n    theorem of calculus. Various functions are implemented to integrate\n    polynomial, rational and trigonometric functions, and integrands\n    containing DiracDelta terms.\n\n    SymPy also implements the part of the Risch algorithm, which is a decision\n    procedure for integrating elementary functions, i.e., the algorithm can\n    either find an elementary antiderivative, or prove that one does not\n    exist.  There is also a (very successful, albeit somewhat slow) general\n    implementation of the heuristic Risch algorithm.  This algorithm will\n    eventually be phased out as more of the full Risch algorithm is\n    implemented. See the docstring of Integral._eval_integral() for more\n    details on computing the antiderivative using algebraic methods.\n\n    The option risch=True can be used to use only the (full) Risch algorithm.\n    This is useful if you want to know if an elementary function has an\n    elementary antiderivative.  If the indefinite Integral returned by this\n    function is an instance of NonElementaryIntegral, that means that the\n    Risch algorithm has proven that integral to be non-elementary.  Note that\n    by default, additional methods (such as the Meijer G method outlined\n    below) are tried on these integrals, as they may be expressible in terms\n    of special functions, so if you only care about elementary answers, use\n    risch=True.  Also note that an unevaluated Integral returned by this\n    function is not necessarily a NonElementaryIntegral, even with risch=True,\n    as it may just be an indication that the particular part of the Risch\n    algorithm needed to integrate that function is not yet implemented.\n\n    Another family of strategies comes from re-writing the integrand in\n    terms of so-called Meijer G-functions. Indefinite integrals of a\n    single G-function can always be computed, and the definite integral\n    of a product of two G-functions can be computed from zero to\n    infinity. Various strategies are implemented to rewrite integrands\n    as G-functions, and use this information to compute integrals (see\n    the ``meijerint`` module).\n\n    The option manual=True can be used to use only an algorithm that tries\n    to mimic integration by hand. This algorithm does not handle as many\n    integrands as the other algorithms implemented but may return results in\n    a more familiar form. The ``manualintegrate`` module has functions that\n    return the steps used (see the module docstring for more information).\n\n    In general, the algebraic methods work best for computing\n    antiderivatives of (possibly complicated) combinations of elementary\n    functions. The G-function methods work best for computing definite\n    integrals from zero to infinity of moderately complicated\n    combinations of special functions, or indefinite integrals of very\n    simple combinations of special functions.\n\n    The strategy employed by the integration code is as follows:\n\n    - If computing a definite integral, and both limits are real,\n      and at least one limit is +- oo, try the G-function method of\n      definite integration first.\n\n    - Try to find an antiderivative, using all available methods, ordered\n      by performance (that is try fastest method first, slowest last; in\n      particular polynomial integration is tried first, Meijer\n      G-functions second to last, and heuristic Risch last).\n\n    - If still not successful, try G-functions irrespective of the\n      limits.\n\n    The option meijerg=True, False, None can be used to, respectively:\n    always use G-function methods and no others, never use G-function\n    methods, or use all available methods (in order as described above).\n    It defaults to None.\n\n    Examples\n    ========\n\n    >>> from sympy import integrate, log, exp, oo\n    >>> from sympy.abc import a, x, y\n\n    >>> integrate(x*y, x)\n    x**2*y/2\n\n    >>> integrate(log(x), x)\n    x*log(x) - x\n\n    >>> integrate(log(x), (x, 1, a))\n    a*log(a) - a + 1\n\n    >>> integrate(x)\n    x**2/2\n\n    Terms that are independent of x are dropped by indefinite integration:\n\n    >>> from sympy import sqrt\n    >>> integrate(sqrt(1 + x), (x, 0, x))\n    2*(x + 1)**(3/2)/3 - 2/3\n    >>> integrate(sqrt(1 + x), x)\n    2*(x + 1)**(3/2)/3\n\n    >>> integrate(x*y)\n    Traceback (most recent call last):\n    ...\n    ValueError: specify integration variables to integrate x*y\n\n    Note that ``integrate(x)`` syntax is meant only for convenience\n    in interactive sessions and should be avoided in library code.\n\n    >>> integrate(x**a*exp(-x), (x, 0, oo)) # same as conds='piecewise'\n    Piecewise((gamma(a + 1), re(a) > -1),\n        (Integral(x**a*exp(-x), (x, 0, oo)), True))\n\n    >>> integrate(x**a*exp(-x), (x, 0, oo), conds='none')\n    gamma(a + 1)\n\n    >>> integrate(x**a*exp(-x), (x, 0, oo), conds='separate')\n    (gamma(a + 1), re(a) > -1)\n\n    See Also\n    ========\n\n    Integral, Integral.doit\n\n    "
    doit_flags = {'deep': False, 'meijerg': meijerg, 'conds': conds, 'risch': risch, 'heurisch': heurisch, 'manual': manual}
    integral = Integral(*args, **kwargs)
    if isinstance(integral, Integral):
        return integral.doit(**doit_flags)
    else:
        new_args = [a.doit(**doit_flags) if isinstance(a, Integral) else a for a in integral.args]
        return integral.func(*new_args)

def line_integrate(field, curve, vars):
    if False:
        while True:
            i = 10
    'line_integrate(field, Curve, variables)\n\n    Compute the line integral.\n\n    Examples\n    ========\n\n    >>> from sympy import Curve, line_integrate, E, ln\n    >>> from sympy.abc import x, y, t\n    >>> C = Curve([E**t + 1, E**t - 1], (t, 0, ln(2)))\n    >>> line_integrate(x + y, C, [x, y])\n    3*sqrt(2)\n\n    See Also\n    ========\n\n    sympy.integrals.integrals.integrate, Integral\n    '
    from sympy.geometry import Curve
    F = sympify(field)
    if not F:
        raise ValueError('Expecting function specifying field as first argument.')
    if not isinstance(curve, Curve):
        raise ValueError('Expecting Curve entity as second argument.')
    if not is_sequence(vars):
        raise ValueError('Expecting ordered iterable for variables.')
    if len(curve.functions) != len(vars):
        raise ValueError('Field variable size does not match curve dimension.')
    if curve.parameter in vars:
        raise ValueError('Curve parameter clashes with field parameters.')
    Ft = F
    dldt = 0
    for (i, var) in enumerate(vars):
        _f = curve.functions[i]
        _dn = diff(_f, curve.parameter)
        dldt = dldt + _dn * _dn
        Ft = Ft.subs(var, _f)
    Ft = Ft * sqrt(dldt)
    integral = Integral(Ft, curve.limits).doit(deep=False)
    return integral

@shape.register(Integral)
def _(expr):
    if False:
        i = 10
        return i + 15
    return shape(expr.function)
from .deltafunctions import deltaintegrate
from .meijerint import meijerint_definite, meijerint_indefinite, _debug
from .trigonometry import trigintegrate