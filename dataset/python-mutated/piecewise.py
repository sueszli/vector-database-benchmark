from sympy.core import S, Function, diff, Tuple, Dummy, Mul
from sympy.core.basic import Basic, as_Basic
from sympy.core.numbers import Rational, NumberSymbol, _illegal
from sympy.core.parameters import global_parameters
from sympy.core.relational import Lt, Gt, Eq, Ne, Relational, _canonical, _canonical_coeff
from sympy.core.sorting import ordered
from sympy.functions.elementary.miscellaneous import Max, Min
from sympy.logic.boolalg import And, Boolean, distribute_and_over_or, Not, true, false, Or, ITE, simplify_logic, to_cnf, distribute_or_over_and
from sympy.utilities.iterables import uniq, sift, common_prefix
from sympy.utilities.misc import filldedent, func_name
from itertools import product
Undefined = S.NaN

class ExprCondPair(Tuple):
    """Represents an expression, condition pair."""

    def __new__(cls, expr, cond):
        if False:
            for i in range(10):
                print('nop')
        expr = as_Basic(expr)
        if cond == True:
            return Tuple.__new__(cls, expr, true)
        elif cond == False:
            return Tuple.__new__(cls, expr, false)
        elif isinstance(cond, Basic) and cond.has(Piecewise):
            cond = piecewise_fold(cond)
            if isinstance(cond, Piecewise):
                cond = cond.rewrite(ITE)
        if not isinstance(cond, Boolean):
            raise TypeError(filldedent('\n                Second argument must be a Boolean,\n                not `%s`' % func_name(cond)))
        return Tuple.__new__(cls, expr, cond)

    @property
    def expr(self):
        if False:
            return 10
        '\n        Returns the expression of this pair.\n        '
        return self.args[0]

    @property
    def cond(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the condition of this pair.\n        '
        return self.args[1]

    @property
    def is_commutative(self):
        if False:
            return 10
        return self.expr.is_commutative

    def __iter__(self):
        if False:
            print('Hello World!')
        yield self.expr
        yield self.cond

    def _eval_simplify(self, **kwargs):
        if False:
            i = 10
            return i + 15
        return self.func(*[a.simplify(**kwargs) for a in self.args])

class Piecewise(Function):
    """
    Represents a piecewise function.

    Usage:

      Piecewise( (expr,cond), (expr,cond), ... )
        - Each argument is a 2-tuple defining an expression and condition
        - The conds are evaluated in turn returning the first that is True.
          If any of the evaluated conds are not explicitly False,
          e.g. ``x < 1``, the function is returned in symbolic form.
        - If the function is evaluated at a place where all conditions are False,
          nan will be returned.
        - Pairs where the cond is explicitly False, will be removed and no pair
          appearing after a True condition will ever be retained. If a single
          pair with a True condition remains, it will be returned, even when
          evaluation is False.

    Examples
    ========

    >>> from sympy import Piecewise, log, piecewise_fold
    >>> from sympy.abc import x, y
    >>> f = x**2
    >>> g = log(x)
    >>> p = Piecewise((0, x < -1), (f, x <= 1), (g, True))
    >>> p.subs(x,1)
    1
    >>> p.subs(x,5)
    log(5)

    Booleans can contain Piecewise elements:

    >>> cond = (x < y).subs(x, Piecewise((2, x < 0), (3, True))); cond
    Piecewise((2, x < 0), (3, True)) < y

    The folded version of this results in a Piecewise whose
    expressions are Booleans:

    >>> folded_cond = piecewise_fold(cond); folded_cond
    Piecewise((2 < y, x < 0), (3 < y, True))

    When a Boolean containing Piecewise (like cond) or a Piecewise
    with Boolean expressions (like folded_cond) is used as a condition,
    it is converted to an equivalent :class:`~.ITE` object:

    >>> Piecewise((1, folded_cond))
    Piecewise((1, ITE(x < 0, y > 2, y > 3)))

    When a condition is an ``ITE``, it will be converted to a simplified
    Boolean expression:

    >>> piecewise_fold(_)
    Piecewise((1, ((x >= 0) | (y > 2)) & ((y > 3) | (x < 0))))

    See Also
    ========

    piecewise_fold
    piecewise_exclusive
    ITE
    """
    nargs = None
    is_Piecewise = True

    def __new__(cls, *args, **options):
        if False:
            for i in range(10):
                print('nop')
        if len(args) == 0:
            raise TypeError('At least one (expr, cond) pair expected.')
        newargs = []
        for ec in args:
            pair = ExprCondPair(*getattr(ec, 'args', ec))
            cond = pair.cond
            if cond is false:
                continue
            newargs.append(pair)
            if cond is true:
                break
        eval = options.pop('evaluate', global_parameters.evaluate)
        if eval:
            r = cls.eval(*newargs)
            if r is not None:
                return r
        elif len(newargs) == 1 and newargs[0].cond == True:
            return newargs[0].expr
        return Basic.__new__(cls, *newargs, **options)

    @classmethod
    def eval(cls, *_args):
        if False:
            return 10
        'Either return a modified version of the args or, if no\n        modifications were made, return None.\n\n        Modifications that are made here:\n\n        1. relationals are made canonical\n        2. any False conditions are dropped\n        3. any repeat of a previous condition is ignored\n        4. any args past one with a true condition are dropped\n\n        If there are no args left, nan will be returned.\n        If there is a single arg with a True condition, its\n        corresponding expression will be returned.\n\n        EXAMPLES\n        ========\n\n        >>> from sympy import Piecewise\n        >>> from sympy.abc import x\n        >>> cond = -x < -1\n        >>> args = [(1, cond), (4, cond), (3, False), (2, True), (5, x < 1)]\n        >>> Piecewise(*args, evaluate=False)\n        Piecewise((1, -x < -1), (4, -x < -1), (2, True))\n        >>> Piecewise(*args)\n        Piecewise((1, x > 1), (2, True))\n        '
        if not _args:
            return Undefined
        if len(_args) == 1 and _args[0][-1] == True:
            return _args[0][0]
        newargs = _piecewise_collapse_arguments(_args)
        missing = len(newargs) != len(_args)
        same = all((a == b for (a, b) in zip(newargs, _args)))
        if not newargs:
            raise ValueError(filldedent('\n                There are no conditions (or none that\n                are not trivially false) to define an\n                expression.'))
        if missing or not same:
            return cls(*newargs)

    def doit(self, **hints):
        if False:
            while True:
                i = 10
        '\n        Evaluate this piecewise function.\n        '
        newargs = []
        for (e, c) in self.args:
            if hints.get('deep', True):
                if isinstance(e, Basic):
                    newe = e.doit(**hints)
                    if newe != self:
                        e = newe
                if isinstance(c, Basic):
                    c = c.doit(**hints)
            newargs.append((e, c))
        return self.func(*newargs)

    def _eval_simplify(self, **kwargs):
        if False:
            return 10
        return piecewise_simplify(self, **kwargs)

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        if False:
            for i in range(10):
                print('nop')
        for (e, c) in self.args:
            if c == True or c.subs(x, 0) == True:
                return e.as_leading_term(x)

    def _eval_adjoint(self):
        if False:
            return 10
        return self.func(*[(e.adjoint(), c) for (e, c) in self.args])

    def _eval_conjugate(self):
        if False:
            for i in range(10):
                print('nop')
        return self.func(*[(e.conjugate(), c) for (e, c) in self.args])

    def _eval_derivative(self, x):
        if False:
            while True:
                i = 10
        return self.func(*[(diff(e, x), c) for (e, c) in self.args])

    def _eval_evalf(self, prec):
        if False:
            print('Hello World!')
        return self.func(*[(e._evalf(prec), c) for (e, c) in self.args])

    def _eval_is_meromorphic(self, x, a):
        if False:
            for i in range(10):
                print('nop')
        if not a.is_real:
            return None
        for (e, c) in self.args:
            cond = c.subs(x, a)
            if cond.is_Relational:
                return None
            if a in c.as_set().boundary:
                return None
            if cond:
                return e._eval_is_meromorphic(x, a)

    def piecewise_integrate(self, x, **kwargs):
        if False:
            return 10
        'Return the Piecewise with each expression being\n        replaced with its antiderivative. To obtain a continuous\n        antiderivative, use the :func:`~.integrate` function or method.\n\n        Examples\n        ========\n\n        >>> from sympy import Piecewise\n        >>> from sympy.abc import x\n        >>> p = Piecewise((0, x < 0), (1, x < 1), (2, True))\n        >>> p.piecewise_integrate(x)\n        Piecewise((0, x < 0), (x, x < 1), (2*x, True))\n\n        Note that this does not give a continuous function, e.g.\n        at x = 1 the 3rd condition applies and the antiderivative\n        there is 2*x so the value of the antiderivative is 2:\n\n        >>> anti = _\n        >>> anti.subs(x, 1)\n        2\n\n        The continuous derivative accounts for the integral *up to*\n        the point of interest, however:\n\n        >>> p.integrate(x)\n        Piecewise((0, x < 0), (x, x < 1), (2*x - 1, True))\n        >>> _.subs(x, 1)\n        1\n\n        See Also\n        ========\n        Piecewise._eval_integral\n        '
        from sympy.integrals import integrate
        return self.func(*[(integrate(e, x, **kwargs), c) for (e, c) in self.args])

    def _handle_irel(self, x, handler):
        if False:
            i = 10
            return i + 15
        'Return either None (if the conditions of self depend only on x) else\n        a Piecewise expression whose expressions (handled by the handler that\n        was passed) are paired with the governing x-independent relationals,\n        e.g. Piecewise((A, a(x) & b(y)), (B, c(x) | c(y)) ->\n        Piecewise(\n            (handler(Piecewise((A, a(x) & True), (B, c(x) | True)), b(y) & c(y)),\n            (handler(Piecewise((A, a(x) & True), (B, c(x) | False)), b(y)),\n            (handler(Piecewise((A, a(x) & False), (B, c(x) | True)), c(y)),\n            (handler(Piecewise((A, a(x) & False), (B, c(x) | False)), True))\n        '
        rel = self.atoms(Relational)
        irel = list(ordered([r for r in rel if x not in r.free_symbols and r not in (S.true, S.false)]))
        if irel:
            args = {}
            exprinorder = []
            for truth in product((1, 0), repeat=len(irel)):
                reps = dict(zip(irel, truth))
                if 1 not in truth:
                    cond = None
                else:
                    andargs = Tuple(*[i for i in reps if reps[i]])
                    free = list(andargs.free_symbols)
                    if len(free) == 1:
                        from sympy.solvers.inequalities import reduce_inequalities, _solve_inequality
                        try:
                            t = reduce_inequalities(andargs, free[0])
                        except (ValueError, NotImplementedError):
                            t = And(*[_solve_inequality(a, free[0], linear=True) for a in andargs])
                    else:
                        t = And(*andargs)
                    if t is S.false:
                        continue
                    cond = t
                expr = handler(self.xreplace(reps))
                if isinstance(expr, self.func) and len(expr.args) == 1:
                    (expr, econd) = expr.args[0]
                    cond = And(econd, True if cond is None else cond)
                if cond is not None:
                    args.setdefault(expr, []).append(cond)
                    exprinorder.append(expr)
            for k in args:
                args[k] = Or(*args[k])
            args = [(e, args[e]) for e in uniq(exprinorder)]
            args.append((expr, True))
            return Piecewise(*args)

    def _eval_integral(self, x, _first=True, **kwargs):
        if False:
            print('Hello World!')
        'Return the indefinite integral of the\n        Piecewise such that subsequent substitution of x with a\n        value will give the value of the integral (not including\n        the constant of integration) up to that point. To only\n        integrate the individual parts of Piecewise, use the\n        ``piecewise_integrate`` method.\n\n        Examples\n        ========\n\n        >>> from sympy import Piecewise\n        >>> from sympy.abc import x\n        >>> p = Piecewise((0, x < 0), (1, x < 1), (2, True))\n        >>> p.integrate(x)\n        Piecewise((0, x < 0), (x, x < 1), (2*x - 1, True))\n        >>> p.piecewise_integrate(x)\n        Piecewise((0, x < 0), (x, x < 1), (2*x, True))\n\n        See Also\n        ========\n        Piecewise.piecewise_integrate\n        '
        from sympy.integrals.integrals import integrate
        if _first:

            def handler(ipw):
                if False:
                    for i in range(10):
                        print('nop')
                if isinstance(ipw, self.func):
                    return ipw._eval_integral(x, _first=False, **kwargs)
                else:
                    return ipw.integrate(x, **kwargs)
            irv = self._handle_irel(x, handler)
            if irv is not None:
                return irv
        (ok, abei) = self._intervals(x)
        if not ok:
            from sympy.integrals.integrals import Integral
            return Integral(self, x)
        pieces = [(a, b) for (a, b, _, _) in abei]
        oo = S.Infinity
        done = [(-oo, oo, -1)]
        for (k, p) in enumerate(pieces):
            if p == (-oo, oo):
                for (j, (a, b, i)) in enumerate(done):
                    if i == -1:
                        done[j] = (a, b, k)
                break
            N = len(done) - 1
            for (j, (a, b, i)) in enumerate(reversed(done)):
                if i == -1:
                    j = N - j
                    done[j:j + 1] = _clip(p, (a, b), k)
        done = [(a, b, i) for (a, b, i) in done if a != b]
        if any((i == -1 for (a, b, i) in done)):
            abei.append((-oo, oo, Undefined, -1))
        args = []
        sum = None
        for (a, b, i) in done:
            anti = integrate(abei[i][-2], x, **kwargs)
            if sum is None:
                sum = anti
            else:
                sum = sum.subs(x, a)
                e = anti._eval_interval(x, a, x)
                if sum.has(*_illegal) or e.has(*_illegal):
                    sum = anti
                else:
                    sum += e
            if b is S.Infinity:
                cond = True
            elif self.args[abei[i][-1]].cond.subs(x, b) == False:
                cond = x < b
            else:
                cond = x <= b
            args.append((sum, cond))
        return Piecewise(*args)

    def _eval_interval(self, sym, a, b, _first=True):
        if False:
            for i in range(10):
                print('nop')
        'Evaluates the function along the sym in a given interval [a, b]'
        if a is None or b is None:
            return super()._eval_interval(sym, a, b)
        else:
            (x, lo, hi) = map(as_Basic, (sym, a, b))
        if _first:

            def handler(ipw):
                if False:
                    return 10
                if isinstance(ipw, self.func):
                    return ipw._eval_interval(x, lo, hi, _first=None)
                else:
                    return ipw._eval_interval(x, lo, hi)
            irv = self._handle_irel(x, handler)
            if irv is not None:
                return irv
            if (lo < hi) is S.false or (lo is S.Infinity or hi is S.NegativeInfinity):
                rv = self._eval_interval(x, hi, lo, _first=False)
                if isinstance(rv, Piecewise):
                    rv = Piecewise(*[(-e, c) for (e, c) in rv.args])
                else:
                    rv = -rv
                return rv
            if (lo < hi) is S.true or (hi is S.Infinity or lo is S.NegativeInfinity):
                pass
            else:
                _a = Dummy('lo')
                _b = Dummy('hi')
                a = lo if lo.is_comparable else _a
                b = hi if hi.is_comparable else _b
                pos = self._eval_interval(x, a, b, _first=False)
                if a == _a and b == _b:
                    (neg, pos) = (-pos.xreplace({_a: hi, _b: lo}), pos.xreplace({_a: lo, _b: hi}))
                else:
                    (neg, pos) = (-self._eval_interval(x, hi, lo, _first=False), pos.xreplace({_a: lo, _b: hi}))
                p = Dummy('', positive=True)
                if lo.is_Symbol:
                    pos = pos.xreplace({lo: hi - p}).xreplace({p: hi - lo})
                    neg = neg.xreplace({lo: hi + p}).xreplace({p: lo - hi})
                elif hi.is_Symbol:
                    pos = pos.xreplace({hi: lo + p}).xreplace({p: hi - lo})
                    neg = neg.xreplace({hi: lo - p}).xreplace({p: lo - hi})
                touch = lambda _: _.replace(lambda x: isinstance(x, (Min, Max)), lambda x: x.func(*x.args))
                neg = touch(neg)
                pos = touch(pos)
                if a == _a:
                    rv = Piecewise((pos, lo < hi), (neg, True))
                else:
                    rv = Piecewise((neg, hi < lo), (pos, True))
                if rv == Undefined:
                    raise ValueError("Can't integrate across undefined region.")
                if any((isinstance(i, Piecewise) for i in (pos, neg))):
                    rv = piecewise_fold(rv)
                return rv
        (ok, abei) = self._intervals(x)
        if not ok:
            from sympy.integrals.integrals import Integral
            return Integral(self.diff(x), (x, lo, hi))
        pieces = [(a, b) for (a, b, _, _) in abei]
        done = [(lo, hi, -1)]
        oo = S.Infinity
        for (k, p) in enumerate(pieces):
            if p[:2] == (-oo, oo):
                for (j, (a, b, i)) in enumerate(done):
                    if i == -1:
                        done[j] = (a, b, k)
                break
            N = len(done) - 1
            for (j, (a, b, i)) in enumerate(reversed(done)):
                if i == -1:
                    j = N - j
                    done[j:j + 1] = _clip(p, (a, b), k)
        done = [(a, b, i) for (a, b, i) in done if a != b]
        sum = S.Zero
        upto = None
        for (a, b, i) in done:
            if i == -1:
                if upto is None:
                    return Undefined
                return Piecewise((sum, hi <= upto), (Undefined, True))
            sum += abei[i][-2]._eval_interval(x, a, b)
            upto = b
        return sum

    def _intervals(self, sym, err_on_Eq=False):
        if False:
            return 10
        'Return a bool and a message (when bool is False), else a\n        list of unique tuples, (a, b, e, i), where a and b\n        are the lower and upper bounds in which the expression e of\n        argument i in self is defined and $a < b$ (when involving\n        numbers) or $a \\le b$ when involving symbols.\n\n        If there are any relationals not involving sym, or any\n        relational cannot be solved for sym, the bool will be False\n        a message be given as the second return value. The calling\n        routine should have removed such relationals before calling\n        this routine.\n\n        The evaluated conditions will be returned as ranges.\n        Discontinuous ranges will be returned separately with\n        identical expressions. The first condition that evaluates to\n        True will be returned as the last tuple with a, b = -oo, oo.\n        '
        from sympy.solvers.inequalities import _solve_inequality
        assert isinstance(self, Piecewise)

        def nonsymfail(cond):
            if False:
                for i in range(10):
                    print('nop')
            return (False, filldedent('\n                A condition not involving\n                %s appeared: %s' % (sym, cond)))

        def _solve_relational(r):
            if False:
                while True:
                    i = 10
            if sym not in r.free_symbols:
                return nonsymfail(r)
            try:
                rv = _solve_inequality(r, sym)
            except NotImplementedError:
                return (False, 'Unable to solve relational %s for %s.' % (r, sym))
            if isinstance(rv, Relational):
                free = rv.args[1].free_symbols
                if rv.args[0] != sym or sym in free:
                    return (False, 'Unable to solve relational %s for %s.' % (r, sym))
                if rv.rel_op == '==':
                    rv = S.false
                elif rv.rel_op == '!=':
                    try:
                        rv = Or(sym < rv.rhs, sym > rv.rhs)
                    except TypeError:
                        rv = S.true
            elif rv == (S.NegativeInfinity < sym) & (sym < S.Infinity):
                rv = S.true
            return (True, rv)
        args = list(self.args)
        keys = self.atoms(Relational)
        reps = {}
        for r in keys:
            (ok, s) = _solve_relational(r)
            if ok != True:
                return (False, ok)
            reps[r] = s
        args = [i.xreplace(reps) for i in self.args]
        expr_cond = []
        default = idefault = None
        for (i, (expr, cond)) in enumerate(args):
            if cond is S.false:
                continue
            if cond is S.true:
                default = expr
                idefault = i
                break
            if isinstance(cond, Eq):
                if err_on_Eq:
                    return (False, 'encountered Eq condition: %s' % cond)
                continue
            cond = to_cnf(cond)
            if isinstance(cond, And):
                cond = distribute_or_over_and(cond)
            if isinstance(cond, Or):
                expr_cond.extend([(i, expr, o) for o in cond.args if not isinstance(o, Eq)])
            elif cond is not S.false:
                expr_cond.append((i, expr, cond))
            elif cond is S.true:
                default = expr
                idefault = i
                break
        int_expr = []
        for (iarg, expr, cond) in expr_cond:
            if isinstance(cond, And):
                lower = S.NegativeInfinity
                upper = S.Infinity
                exclude = []
                for cond2 in cond.args:
                    if not isinstance(cond2, Relational):
                        return (False, 'expecting only Relationals')
                    if isinstance(cond2, Eq):
                        lower = upper
                        if err_on_Eq:
                            return (False, 'encountered secondary Eq condition')
                        break
                    elif isinstance(cond2, Ne):
                        (l, r) = cond2.args
                        if l == sym:
                            exclude.append(r)
                        elif r == sym:
                            exclude.append(l)
                        else:
                            return nonsymfail(cond2)
                        continue
                    elif cond2.lts == sym:
                        upper = Min(cond2.gts, upper)
                    elif cond2.gts == sym:
                        lower = Max(cond2.lts, lower)
                    else:
                        return nonsymfail(cond2)
                if exclude:
                    exclude = list(ordered(exclude))
                    newcond = []
                    for (i, e) in enumerate(exclude):
                        if e < lower == True or e > upper == True:
                            continue
                        if not newcond:
                            newcond.append((None, lower))
                        newcond.append((newcond[-1][1], e))
                    newcond.append((newcond[-1][1], upper))
                    newcond.pop(0)
                    expr_cond.extend([(iarg, expr, And(i[0] < sym, sym < i[1])) for i in newcond])
                    continue
            elif isinstance(cond, Relational) and cond.rel_op != '!=':
                (lower, upper) = (cond.lts, cond.gts)
                if cond.lts == sym:
                    lower = S.NegativeInfinity
                elif cond.gts == sym:
                    upper = S.Infinity
                else:
                    return nonsymfail(cond)
            else:
                return (False, 'unrecognized condition: %s' % cond)
            (lower, upper) = (lower, Max(lower, upper))
            if err_on_Eq and lower == upper:
                return (False, 'encountered Eq condition')
            if (lower >= upper) is not S.true:
                int_expr.append((lower, upper, expr, iarg))
        if default is not None:
            int_expr.append((S.NegativeInfinity, S.Infinity, default, idefault))
        return (True, list(uniq(int_expr)))

    def _eval_nseries(self, x, n, logx, cdir=0):
        if False:
            print('Hello World!')
        args = [(ec.expr._eval_nseries(x, n, logx), ec.cond) for ec in self.args]
        return self.func(*args)

    def _eval_power(self, s):
        if False:
            while True:
                i = 10
        return self.func(*[(e ** s, c) for (e, c) in self.args])

    def _eval_subs(self, old, new):
        if False:
            i = 10
            return i + 15
        args = list(self.args)
        args_exist = False
        for (i, (e, c)) in enumerate(args):
            c = c._subs(old, new)
            if c != False:
                args_exist = True
                e = e._subs(old, new)
            args[i] = (e, c)
            if c == True:
                break
        if not args_exist:
            args = ((Undefined, True),)
        return self.func(*args)

    def _eval_transpose(self):
        if False:
            return 10
        return self.func(*[(e.transpose(), c) for (e, c) in self.args])

    def _eval_template_is_attr(self, is_attr):
        if False:
            print('Hello World!')
        b = None
        for (expr, _) in self.args:
            a = getattr(expr, is_attr)
            if a is None:
                return
            if b is None:
                b = a
            elif b is not a:
                return
        return b
    _eval_is_finite = lambda self: self._eval_template_is_attr('is_finite')
    _eval_is_complex = lambda self: self._eval_template_is_attr('is_complex')
    _eval_is_even = lambda self: self._eval_template_is_attr('is_even')
    _eval_is_imaginary = lambda self: self._eval_template_is_attr('is_imaginary')
    _eval_is_integer = lambda self: self._eval_template_is_attr('is_integer')
    _eval_is_irrational = lambda self: self._eval_template_is_attr('is_irrational')
    _eval_is_negative = lambda self: self._eval_template_is_attr('is_negative')
    _eval_is_nonnegative = lambda self: self._eval_template_is_attr('is_nonnegative')
    _eval_is_nonpositive = lambda self: self._eval_template_is_attr('is_nonpositive')
    _eval_is_nonzero = lambda self: self._eval_template_is_attr('is_nonzero')
    _eval_is_odd = lambda self: self._eval_template_is_attr('is_odd')
    _eval_is_polar = lambda self: self._eval_template_is_attr('is_polar')
    _eval_is_positive = lambda self: self._eval_template_is_attr('is_positive')
    _eval_is_extended_real = lambda self: self._eval_template_is_attr('is_extended_real')
    _eval_is_extended_positive = lambda self: self._eval_template_is_attr('is_extended_positive')
    _eval_is_extended_negative = lambda self: self._eval_template_is_attr('is_extended_negative')
    _eval_is_extended_nonzero = lambda self: self._eval_template_is_attr('is_extended_nonzero')
    _eval_is_extended_nonpositive = lambda self: self._eval_template_is_attr('is_extended_nonpositive')
    _eval_is_extended_nonnegative = lambda self: self._eval_template_is_attr('is_extended_nonnegative')
    _eval_is_real = lambda self: self._eval_template_is_attr('is_real')
    _eval_is_zero = lambda self: self._eval_template_is_attr('is_zero')

    @classmethod
    def __eval_cond(cls, cond):
        if False:
            print('Hello World!')
        'Return the truth value of the condition.'
        if cond == True:
            return True
        if isinstance(cond, Eq):
            try:
                diff = cond.lhs - cond.rhs
                if diff.is_commutative:
                    return diff.is_zero
            except TypeError:
                pass

    def as_expr_set_pairs(self, domain=None):
        if False:
            while True:
                i = 10
        'Return tuples for each argument of self that give\n        the expression and the interval in which it is valid\n        which is contained within the given domain.\n        If a condition cannot be converted to a set, an error\n        will be raised. The variable of the conditions is\n        assumed to be real; sets of real values are returned.\n\n        Examples\n        ========\n\n        >>> from sympy import Piecewise, Interval\n        >>> from sympy.abc import x\n        >>> p = Piecewise(\n        ...     (1, x < 2),\n        ...     (2,(x > 0) & (x < 4)),\n        ...     (3, True))\n        >>> p.as_expr_set_pairs()\n        [(1, Interval.open(-oo, 2)),\n         (2, Interval.Ropen(2, 4)),\n         (3, Interval(4, oo))]\n        >>> p.as_expr_set_pairs(Interval(0, 3))\n        [(1, Interval.Ropen(0, 2)),\n         (2, Interval(2, 3))]\n        '
        if domain is None:
            domain = S.Reals
        exp_sets = []
        U = domain
        complex = not domain.is_subset(S.Reals)
        cond_free = set()
        for (expr, cond) in self.args:
            cond_free |= cond.free_symbols
            if len(cond_free) > 1:
                raise NotImplementedError(filldedent('\n                    multivariate conditions are not handled.'))
            if complex:
                for i in cond.atoms(Relational):
                    if not isinstance(i, (Eq, Ne)):
                        raise ValueError(filldedent('\n                            Inequalities in the complex domain are\n                            not supported. Try the real domain by\n                            setting domain=S.Reals'))
            cond_int = U.intersect(cond.as_set())
            U = U - cond_int
            if cond_int != S.EmptySet:
                exp_sets.append((expr, cond_int))
        return exp_sets

    def _eval_rewrite_as_ITE(self, *args, **kwargs):
        if False:
            print('Hello World!')
        byfree = {}
        args = list(args)
        default = any((c == True for (b, c) in args))
        for (i, (b, c)) in enumerate(args):
            if not isinstance(b, Boolean) and b != True:
                raise TypeError(filldedent('\n                    Expecting Boolean or bool but got `%s`\n                    ' % func_name(b)))
            if c == True:
                break
            for c in c.args if isinstance(c, Or) else [c]:
                free = c.free_symbols
                x = free.pop()
                try:
                    byfree[x] = byfree.setdefault(x, S.EmptySet).union(c.as_set())
                except NotImplementedError:
                    if not default:
                        raise NotImplementedError(filldedent('\n                            A method to determine whether a multivariate\n                            conditional is consistent with a complete coverage\n                            of all variables has not been implemented so the\n                            rewrite is being stopped after encountering `%s`.\n                            This error would not occur if a default expression\n                            like `(foo, True)` were given.\n                            ' % c))
                if byfree[x] in (S.UniversalSet, S.Reals):
                    args[i] = list(args[i])
                    c = args[i][1] = True
                    break
            if c == True:
                break
        if c != True:
            raise ValueError(filldedent('\n                Conditions must cover all reals or a final default\n                condition `(foo, True)` must be given.\n                '))
        (last, _) = args[i]
        for (a, c) in reversed(args[:i]):
            last = ITE(c, a, last)
        return _canonical(last)

    def _eval_rewrite_as_KroneckerDelta(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        from sympy.functions.special.tensor_functions import KroneckerDelta
        rules = {And: [False, False], Or: [True, True], Not: [True, False], Eq: [None, None], Ne: [None, None]}

        class UnrecognizedCondition(Exception):
            pass

        def rewrite(cond):
            if False:
                i = 10
                return i + 15
            if isinstance(cond, Eq):
                return KroneckerDelta(*cond.args)
            if isinstance(cond, Ne):
                return 1 - KroneckerDelta(*cond.args)
            (cls, args) = (type(cond), cond.args)
            if cls not in rules:
                raise UnrecognizedCondition(cls)
            (b1, b2) = rules[cls]
            k = Mul(*[1 - rewrite(c) for c in args]) if b1 else Mul(*[rewrite(c) for c in args])
            if b2:
                return 1 - k
            return k
        conditions = []
        true_value = None
        for (value, cond) in args:
            if type(cond) in rules:
                conditions.append((value, cond))
            elif cond is S.true:
                if true_value is None:
                    true_value = value
            else:
                return
        if true_value is not None:
            result = true_value
            for (value, cond) in conditions[::-1]:
                try:
                    k = rewrite(cond)
                    result = k * value + (1 - k) * result
                except UnrecognizedCondition:
                    return
            return result

def piecewise_fold(expr, evaluate=True):
    if False:
        i = 10
        return i + 15
    '\n    Takes an expression containing a piecewise function and returns the\n    expression in piecewise form. In addition, any ITE conditions are\n    rewritten in negation normal form and simplified.\n\n    The final Piecewise is evaluated (default) but if the raw form\n    is desired, send ``evaluate=False``; if trivial evaluation is\n    desired, send ``evaluate=None`` and duplicate conditions and\n    processing of True and False will be handled.\n\n    Examples\n    ========\n\n    >>> from sympy import Piecewise, piecewise_fold, S\n    >>> from sympy.abc import x\n    >>> p = Piecewise((x, x < 1), (1, S(1) <= x))\n    >>> piecewise_fold(x*p)\n    Piecewise((x**2, x < 1), (x, True))\n\n    See Also\n    ========\n\n    Piecewise\n    piecewise_exclusive\n    '
    if not isinstance(expr, Basic) or not expr.has(Piecewise):
        return expr
    new_args = []
    if isinstance(expr, (ExprCondPair, Piecewise)):
        for (e, c) in expr.args:
            if not isinstance(e, Piecewise):
                e = piecewise_fold(e)
            assert not c.has(Piecewise)
            if isinstance(c, ITE):
                c = c.to_nnf()
                c = simplify_logic(c, form='cnf')
            if isinstance(e, Piecewise):
                new_args.extend([(piecewise_fold(ei), And(ci, c)) for (ei, ci) in e.args])
            else:
                new_args.append((e, c))
    else:
        if expr.is_Add or (expr.is_Mul and expr.is_commutative):
            (p, args) = sift(expr.args, lambda x: x.is_Piecewise, binary=True)
            pc = sift(p, lambda x: tuple([c for (e, c) in x.args]))
            for c in list(ordered(pc)):
                if len(pc[c]) > 1:
                    pargs = [list(i.args) for i in pc[c]]
                    com = common_prefix(*[[i.cond for i in j] for j in pargs])
                    n = len(com)
                    collected = []
                    for i in range(n):
                        collected.append((expr.func(*[ai[i].expr for ai in pargs]), com[i]))
                    remains = []
                    for a in pargs:
                        if n == len(a):
                            continue
                        if a[n].cond == True:
                            remains.append(a[n].expr)
                        else:
                            remains.append(Piecewise(*a[n:], evaluate=False))
                    if remains:
                        collected.append((expr.func(*remains), True))
                    args.append(Piecewise(*collected, evaluate=False))
                    continue
                args.extend(pc[c])
        else:
            args = expr.args
        folded = list(map(piecewise_fold, args))
        for ec in product(*[i.args if isinstance(i, Piecewise) else [(i, true)] for i in folded]):
            (e, c) = zip(*ec)
            new_args.append((expr.func(*e), And(*c)))
    if evaluate is None:
        new_args = list(reversed([(e, c) for (c, e) in {c: e for (e, c) in reversed(new_args)}.items()]))
    rv = Piecewise(*new_args, evaluate=evaluate)
    if evaluate is None and len(rv.args) == 1 and (rv.args[0].cond == True):
        return rv.args[0].expr
    if any((s.expr.has(Piecewise) for p in rv.atoms(Piecewise) for s in p.args)):
        return piecewise_fold(rv)
    return rv

def _clip(A, B, k):
    if False:
        i = 10
        return i + 15
    'Return interval B as intervals that are covered by A (keyed\n    to k) and all other intervals of B not covered by A keyed to -1.\n\n    The reference point of each interval is the rhs; if the lhs is\n    greater than the rhs then an interval of zero width interval will\n    result, e.g. (4, 1) is treated like (1, 1).\n\n    Examples\n    ========\n\n    >>> from sympy.functions.elementary.piecewise import _clip\n    >>> from sympy import Tuple\n    >>> A = Tuple(1, 3)\n    >>> B = Tuple(2, 4)\n    >>> _clip(A, B, 0)\n    [(2, 3, 0), (3, 4, -1)]\n\n    Interpretation: interval portion (2, 3) of interval (2, 4) is\n    covered by interval (1, 3) and is keyed to 0 as requested;\n    interval (3, 4) was not covered by (1, 3) and is keyed to -1.\n    '
    (a, b) = B
    (c, d) = A
    (c, d) = (Min(Max(c, a), b), Min(Max(d, a), b))
    (a, b) = (Min(a, b), b)
    p = []
    if a != c:
        p.append((a, c, -1))
    else:
        pass
    if c != d:
        p.append((c, d, k))
    else:
        pass
    if b != d:
        if d == c and p and (p[-1][-1] == -1):
            p[-1] = (p[-1][0], b, -1)
        else:
            p.append((d, b, -1))
    else:
        pass
    return p

def piecewise_simplify_arguments(expr, **kwargs):
    if False:
        while True:
            i = 10
    from sympy.simplify.simplify import simplify
    f1 = expr.args[0].cond.free_symbols
    args = None
    if len(f1) == 1 and (not expr.atoms(Eq)):
        x = f1.pop()
        (ok, abe_) = expr._intervals(x, err_on_Eq=True)

        def include(c, x, a):
            if False:
                print('Hello World!')
            'return True if c.subs(x, a) is True, else False'
            try:
                return c.subs(x, a) == True
            except TypeError:
                return False
        if ok:
            args = []
            covered = S.EmptySet
            from sympy.sets.sets import Interval
            for (a, b, e, i) in abe_:
                c = expr.args[i].cond
                incl_a = include(c, x, a)
                incl_b = include(c, x, b)
                iv = Interval(a, b, not incl_a, not incl_b)
                cset = iv - covered
                if not cset:
                    continue
                try:
                    a = cset.inf
                except NotImplementedError:
                    pass
                else:
                    incl_a = include(c, x, a)
                if incl_a and incl_b:
                    if a.is_infinite and b.is_infinite:
                        c = S.true
                    elif b.is_infinite:
                        c = x > a if a in covered else x >= a
                    elif a.is_infinite:
                        c = x <= b
                    elif a in covered:
                        c = And(a < x, x <= b)
                    else:
                        c = And(a <= x, x <= b)
                elif incl_a:
                    if a.is_infinite:
                        c = x < b
                    elif a in covered:
                        c = And(a < x, x < b)
                    else:
                        c = And(a <= x, x < b)
                elif incl_b:
                    if b.is_infinite:
                        c = x > a
                    else:
                        c = And(a < x, x <= b)
                elif a in covered:
                    c = x < b
                else:
                    c = And(a < x, x < b)
                covered |= iv
                if a is S.NegativeInfinity and incl_a:
                    covered |= {S.NegativeInfinity}
                if b is S.Infinity and incl_b:
                    covered |= {S.Infinity}
                args.append((e, c))
            if not S.Reals.is_subset(covered):
                args.append((Undefined, True))
    if args is None:
        args = list(expr.args)
        for i in range(len(args)):
            (e, c) = args[i]
            if isinstance(c, Basic):
                c = simplify(c, **kwargs)
            args[i] = (e, c)
    doit = kwargs.pop('doit', None)
    for i in range(len(args)):
        (e, c) = args[i]
        if isinstance(e, Basic):
            newe = simplify(e, doit=False, **kwargs)
            if newe != e:
                e = newe
        args[i] = (e, c)
    if doit is not None:
        kwargs['doit'] = doit
    return Piecewise(*args)

def _piecewise_collapse_arguments(_args):
    if False:
        for i in range(10):
            print('nop')
    newargs = []
    current_cond = set()
    for (expr, cond) in _args:
        cond = cond.replace(lambda _: _.is_Relational, _canonical_coeff)
        if isinstance(expr, Piecewise):
            unmatching = []
            for (i, (e, c)) in enumerate(expr.args):
                if c in current_cond:
                    continue
                if c == cond:
                    if c != True:
                        if unmatching:
                            expr = Piecewise(*unmatching + [(e, c)])
                        else:
                            expr = e
                    break
                else:
                    unmatching.append((e, c))
        got = False
        for i in [cond] + (list(cond.args) if isinstance(cond, And) else []):
            if i in current_cond:
                got = True
                break
        if got:
            continue
        if isinstance(cond, And):
            nonredundant = []
            for c in cond.args:
                if isinstance(c, Relational):
                    if c.negated.canonical in current_cond:
                        continue
                    if isinstance(c, (Lt, Gt)) and c.weak in current_cond:
                        cond = False
                        break
                nonredundant.append(c)
            else:
                cond = cond.func(*nonredundant)
        elif isinstance(cond, Relational):
            if cond.negated.canonical in current_cond:
                cond = S.true
        current_cond.add(cond)
        if newargs:
            if newargs[-1].expr == expr:
                orcond = Or(cond, newargs[-1].cond)
                if isinstance(orcond, (And, Or)):
                    orcond = distribute_and_over_or(orcond)
                newargs[-1] = ExprCondPair(expr, orcond)
                continue
            elif newargs[-1].cond == cond:
                continue
        newargs.append(ExprCondPair(expr, cond))
    return newargs
_blessed = lambda e: getattr(e.lhs, '_diff_wrt', False) and (getattr(e.rhs, '_diff_wrt', None) or isinstance(e.rhs, (Rational, NumberSymbol)))

def piecewise_simplify(expr, **kwargs):
    if False:
        i = 10
        return i + 15
    expr = piecewise_simplify_arguments(expr, **kwargs)
    if not isinstance(expr, Piecewise):
        return expr
    args = list(expr.args)
    args = _piecewise_simplify_eq_and(args)
    args = _piecewise_simplify_equal_to_next_segment(args)
    return Piecewise(*args)

def _piecewise_simplify_equal_to_next_segment(args):
    if False:
        print('Hello World!')
    '\n    See if expressions valid for an Equal expression happens to evaluate\n    to the same function as in the next piecewise segment, see:\n    https://github.com/sympy/sympy/issues/8458\n    '
    prevexpr = None
    for (i, (expr, cond)) in reversed(list(enumerate(args))):
        if prevexpr is not None:
            if isinstance(cond, And):
                (eqs, other) = sift(cond.args, lambda i: isinstance(i, Eq), binary=True)
            elif isinstance(cond, Eq):
                (eqs, other) = ([cond], [])
            else:
                eqs = other = []
            _prevexpr = prevexpr
            _expr = expr
            if eqs and (not other):
                eqs = list(ordered(eqs))
                for e in eqs:
                    if len(args) == 2 or _blessed(e):
                        _prevexpr = _prevexpr.subs(*e.args)
                        _expr = _expr.subs(*e.args)
            if _prevexpr == _expr:
                args[i] = args[i].func(args[i + 1][0], cond)
            else:
                prevexpr = expr
        else:
            prevexpr = expr
    return args

def _piecewise_simplify_eq_and(args):
    if False:
        print('Hello World!')
    '\n    Try to simplify conditions and the expression for\n    equalities that are part of the condition, e.g.\n    Piecewise((n, And(Eq(n,0), Eq(n + m, 0))), (1, True))\n    -> Piecewise((0, And(Eq(n, 0), Eq(m, 0))), (1, True))\n    '
    for (i, (expr, cond)) in enumerate(args):
        if isinstance(cond, And):
            (eqs, other) = sift(cond.args, lambda i: isinstance(i, Eq), binary=True)
        elif isinstance(cond, Eq):
            (eqs, other) = ([cond], [])
        else:
            eqs = other = []
        if eqs:
            eqs = list(ordered(eqs))
            for (j, e) in enumerate(eqs):
                if _blessed(e):
                    expr = expr.subs(*e.args)
                    eqs[j + 1:] = [ei.subs(*e.args) for ei in eqs[j + 1:]]
                    other = [ei.subs(*e.args) for ei in other]
            cond = And(*eqs + other)
            args[i] = args[i].func(expr, cond)
    return args

def piecewise_exclusive(expr, *, skip_nan=False, deep=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Rewrite :class:`Piecewise` with mutually exclusive conditions.\n\n    Explanation\n    ===========\n\n    SymPy represents the conditions of a :class:`Piecewise` in an\n    "if-elif"-fashion, allowing more than one condition to be simultaneously\n    True. The interpretation is that the first condition that is True is the\n    case that holds. While this is a useful representation computationally it\n    is not how a piecewise formula is typically shown in a mathematical text.\n    The :func:`piecewise_exclusive` function can be used to rewrite any\n    :class:`Piecewise` with more typical mutually exclusive conditions.\n\n    Note that further manipulation of the resulting :class:`Piecewise`, e.g.\n    simplifying it, will most likely make it non-exclusive. Hence, this is\n    primarily a function to be used in conjunction with printing the Piecewise\n    or if one would like to reorder the expression-condition pairs.\n\n    If it is not possible to determine that all possibilities are covered by\n    the different cases of the :class:`Piecewise` then a final\n    :class:`~sympy.core.numbers.NaN` case will be included explicitly. This\n    can be prevented by passing ``skip_nan=True``.\n\n    Examples\n    ========\n\n    >>> from sympy import piecewise_exclusive, Symbol, Piecewise, S\n    >>> x = Symbol(\'x\', real=True)\n    >>> p = Piecewise((0, x < 0), (S.Half, x <= 0), (1, True))\n    >>> piecewise_exclusive(p)\n    Piecewise((0, x < 0), (1/2, Eq(x, 0)), (1, x > 0))\n    >>> piecewise_exclusive(Piecewise((2, x > 1)))\n    Piecewise((2, x > 1), (nan, x <= 1))\n    >>> piecewise_exclusive(Piecewise((2, x > 1)), skip_nan=True)\n    Piecewise((2, x > 1))\n\n    Parameters\n    ==========\n\n    expr: a SymPy expression.\n        Any :class:`Piecewise` in the expression will be rewritten.\n    skip_nan: ``bool`` (default ``False``)\n        If ``skip_nan`` is set to ``True`` then a final\n        :class:`~sympy.core.numbers.NaN` case will not be included.\n    deep:  ``bool`` (default ``True``)\n        If ``deep`` is ``True`` then :func:`piecewise_exclusive` will rewrite\n        any :class:`Piecewise` subexpressions in ``expr`` rather than just\n        rewriting ``expr`` itself.\n\n    Returns\n    =======\n\n    An expression equivalent to ``expr`` but where all :class:`Piecewise` have\n    been rewritten with mutually exclusive conditions.\n\n    See Also\n    ========\n\n    Piecewise\n    piecewise_fold\n    '

    def make_exclusive(*pwargs):
        if False:
            print('Hello World!')
        cumcond = false
        newargs = []
        for (expr_i, cond_i) in pwargs[:-1]:
            cancond = And(cond_i, Not(cumcond)).simplify()
            cumcond = Or(cond_i, cumcond).simplify()
            newargs.append((expr_i, cancond))
        (expr_n, cond_n) = pwargs[-1]
        cancond_n = And(cond_n, Not(cumcond)).simplify()
        newargs.append((expr_n, cancond_n))
        if not skip_nan:
            cumcond = Or(cond_n, cumcond).simplify()
            if cumcond is not true:
                newargs.append((Undefined, Not(cumcond).simplify()))
        return Piecewise(*newargs, evaluate=False)
    if deep:
        return expr.replace(Piecewise, make_exclusive)
    elif isinstance(expr, Piecewise):
        return make_exclusive(*expr.args)
    else:
        return expr