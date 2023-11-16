from typing import Tuple as tTuple
from sympy.calculus.singularities import is_decreasing
from sympy.calculus.accumulationbounds import AccumulationBounds
from .expr_with_intlimits import ExprWithIntLimits
from .expr_with_limits import AddWithLimits
from .gosper import gosper_sum
from sympy.core.expr import Expr
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import Derivative, expand
from sympy.core.mul import Mul
from sympy.core.numbers import Float, _illegal
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, Wild, Symbol, symbols
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.combinatorial.numbers import bernoulli, harmonic
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import cot, csc
from sympy.functions.special.hyper import hyper
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.functions.special.zeta_functions import zeta
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import And
from sympy.polys.partfrac import apart
from sympy.polys.polyerrors import PolynomialError, PolificationFailed
from sympy.polys.polytools import parallel_poly_from_expr, Poly, factor
from sympy.polys.rationaltools import together
from sympy.series.limitseq import limit_seq
from sympy.series.order import O
from sympy.series.residues import residue
from sympy.sets.sets import FiniteSet, Interval
from sympy.utilities.iterables import sift
import itertools

class Sum(AddWithLimits, ExprWithIntLimits):
    """
    Represents unevaluated summation.

    Explanation
    ===========

    ``Sum`` represents a finite or infinite series, with the first argument
    being the general form of terms in the series, and the second argument
    being ``(dummy_variable, start, end)``, with ``dummy_variable`` taking
    all integer values from ``start`` through ``end``. In accordance with
    long-standing mathematical convention, the end term is included in the
    summation.

    Finite sums
    ===========

    For finite sums (and sums with symbolic limits assumed to be finite) we
    follow the summation convention described by Karr [1], especially
    definition 3 of section 1.4. The sum:

    .. math::

        \\sum_{m \\leq i < n} f(i)

    has *the obvious meaning* for `m < n`, namely:

    .. math::

        \\sum_{m \\leq i < n} f(i) = f(m) + f(m+1) + \\ldots + f(n-2) + f(n-1)

    with the upper limit value `f(n)` excluded. The sum over an empty set is
    zero if and only if `m = n`:

    .. math::

        \\sum_{m \\leq i < n} f(i) = 0  \\quad \\mathrm{for} \\quad  m = n

    Finally, for all other sums over empty sets we assume the following
    definition:

    .. math::

        \\sum_{m \\leq i < n} f(i) = - \\sum_{n \\leq i < m} f(i)  \\quad \\mathrm{for} \\quad  m > n

    It is important to note that Karr defines all sums with the upper
    limit being exclusive. This is in contrast to the usual mathematical notation,
    but does not affect the summation convention. Indeed we have:

    .. math::

        \\sum_{m \\leq i < n} f(i) = \\sum_{i = m}^{n - 1} f(i)

    where the difference in notation is intentional to emphasize the meaning,
    with limits typeset on the top being inclusive.

    Examples
    ========

    >>> from sympy.abc import i, k, m, n, x
    >>> from sympy import Sum, factorial, oo, IndexedBase, Function
    >>> Sum(k, (k, 1, m))
    Sum(k, (k, 1, m))
    >>> Sum(k, (k, 1, m)).doit()
    m**2/2 + m/2
    >>> Sum(k**2, (k, 1, m))
    Sum(k**2, (k, 1, m))
    >>> Sum(k**2, (k, 1, m)).doit()
    m**3/3 + m**2/2 + m/6
    >>> Sum(x**k, (k, 0, oo))
    Sum(x**k, (k, 0, oo))
    >>> Sum(x**k, (k, 0, oo)).doit()
    Piecewise((1/(1 - x), Abs(x) < 1), (Sum(x**k, (k, 0, oo)), True))
    >>> Sum(x**k/factorial(k), (k, 0, oo)).doit()
    exp(x)

    Here are examples to do summation with symbolic indices.  You
    can use either Function of IndexedBase classes:

    >>> f = Function('f')
    >>> Sum(f(n), (n, 0, 3)).doit()
    f(0) + f(1) + f(2) + f(3)
    >>> Sum(f(n), (n, 0, oo)).doit()
    Sum(f(n), (n, 0, oo))
    >>> f = IndexedBase('f')
    >>> Sum(f[n]**2, (n, 0, 3)).doit()
    f[0]**2 + f[1]**2 + f[2]**2 + f[3]**2

    An example showing that the symbolic result of a summation is still
    valid for seemingly nonsensical values of the limits. Then the Karr
    convention allows us to give a perfectly valid interpretation to
    those sums by interchanging the limits according to the above rules:

    >>> S = Sum(i, (i, 1, n)).doit()
    >>> S
    n**2/2 + n/2
    >>> S.subs(n, -4)
    6
    >>> Sum(i, (i, 1, -4)).doit()
    6
    >>> Sum(-i, (i, -3, 0)).doit()
    6

    An explicit example of the Karr summation convention:

    >>> S1 = Sum(i**2, (i, m, m+n-1)).doit()
    >>> S1
    m**2*n + m*n**2 - m*n + n**3/3 - n**2/2 + n/6
    >>> S2 = Sum(i**2, (i, m+n, m-1)).doit()
    >>> S2
    -m**2*n - m*n**2 + m*n - n**3/3 + n**2/2 - n/6
    >>> S1 + S2
    0
    >>> S3 = Sum(i, (i, m, m-1)).doit()
    >>> S3
    0

    See Also
    ========

    summation
    Product, sympy.concrete.products.product

    References
    ==========

    .. [1] Michael Karr, "Summation in Finite Terms", Journal of the ACM,
           Volume 28 Issue 2, April 1981, Pages 305-350
           https://dl.acm.org/doi/10.1145/322248.322255
    .. [2] https://en.wikipedia.org/wiki/Summation#Capital-sigma_notation
    .. [3] https://en.wikipedia.org/wiki/Empty_sum
    """
    __slots__ = ()
    limits: tTuple[tTuple[Symbol, Expr, Expr]]

    def __new__(cls, function, *symbols, **assumptions):
        if False:
            i = 10
            return i + 15
        obj = AddWithLimits.__new__(cls, function, *symbols, **assumptions)
        if not hasattr(obj, 'limits'):
            return obj
        if any((len(l) != 3 or None in l for l in obj.limits)):
            raise ValueError('Sum requires values for lower and upper bounds.')
        return obj

    def _eval_is_zero(self):
        if False:
            i = 10
            return i + 15
        if self.function.is_zero or self.has_empty_sequence:
            return True

    def _eval_is_extended_real(self):
        if False:
            while True:
                i = 10
        if self.has_empty_sequence:
            return True
        return self.function.is_extended_real

    def _eval_is_positive(self):
        if False:
            print('Hello World!')
        if self.has_finite_limits and self.has_reversed_limits is False:
            return self.function.is_positive

    def _eval_is_negative(self):
        if False:
            for i in range(10):
                print('nop')
        if self.has_finite_limits and self.has_reversed_limits is False:
            return self.function.is_negative

    def _eval_is_finite(self):
        if False:
            for i in range(10):
                print('nop')
        if self.has_finite_limits and self.function.is_finite:
            return True

    def doit(self, **hints):
        if False:
            print('Hello World!')
        if hints.get('deep', True):
            f = self.function.doit(**hints)
        else:
            f = self.function
        reps = {}
        for xab in self.limits:
            d = _dummy_with_inherited_properties_concrete(xab)
            if d:
                reps[xab[0]] = d
        if reps:
            undo = {v: k for (k, v) in reps.items()}
            did = self.xreplace(reps).doit(**hints)
            if isinstance(did, tuple):
                did = tuple([i.xreplace(undo) for i in did])
            elif did is not None:
                did = did.xreplace(undo)
            else:
                did = self
            return did
        if self.function.is_Matrix:
            expanded = self.expand()
            if self != expanded:
                return expanded.doit()
            return _eval_matrix_sum(self)
        for (n, limit) in enumerate(self.limits):
            (i, a, b) = limit
            dif = b - a
            if dif == -1:
                return S.Zero
            if dif.is_integer and dif.is_negative:
                (a, b) = (b + 1, a - 1)
                f = -f
            newf = eval_sum(f, (i, a, b))
            if newf is None:
                if f == self.function:
                    zeta_function = self.eval_zeta_function(f, (i, a, b))
                    if zeta_function is not None:
                        return zeta_function
                    return self
                else:
                    return self.func(f, *self.limits[n:])
            f = newf
        if hints.get('deep', True):
            if not isinstance(f, Piecewise):
                return f.doit(**hints)
        return f

    def eval_zeta_function(self, f, limits):
        if False:
            i = 10
            return i + 15
        '\n        Check whether the function matches with the zeta function.\n\n        If it matches, then return a `Piecewise` expression because\n        zeta function does not converge unless `s > 1` and `q > 0`\n        '
        (i, a, b) = limits
        (w, y, z) = (Wild('w', exclude=[i]), Wild('y', exclude=[i]), Wild('z', exclude=[i]))
        result = f.match((w * i + y) ** (-z))
        if result is not None and b is S.Infinity:
            coeff = 1 / result[w] ** result[z]
            s = result[z]
            q = result[y] / result[w] + a
            return Piecewise((coeff * zeta(s, q), And(q > 0, s > 1)), (self, True))

    def _eval_derivative(self, x):
        if False:
            print('Hello World!')
        '\n        Differentiate wrt x as long as x is not in the free symbols of any of\n        the upper or lower limits.\n\n        Explanation\n        ===========\n\n        Sum(a*b*x, (x, 1, a)) can be differentiated wrt x or b but not `a`\n        since the value of the sum is discontinuous in `a`. In a case\n        involving a limit variable, the unevaluated derivative is returned.\n        '
        if isinstance(x, Symbol) and x not in self.free_symbols:
            return S.Zero
        (f, limits) = (self.function, list(self.limits))
        limit = limits.pop(-1)
        if limits:
            f = self.func(f, *limits)
        (_, a, b) = limit
        if x in a.free_symbols or x in b.free_symbols:
            return None
        df = Derivative(f, x, evaluate=True)
        rv = self.func(df, limit)
        return rv

    def _eval_difference_delta(self, n, step):
        if False:
            for i in range(10):
                print('nop')
        (k, _, upper) = self.args[-1]
        new_upper = upper.subs(n, n + step)
        if len(self.args) == 2:
            f = self.args[0]
        else:
            f = self.func(*self.args[:-1])
        return Sum(f, (k, upper + 1, new_upper)).doit()

    def _eval_simplify(self, **kwargs):
        if False:
            while True:
                i = 10
        function = self.function
        if kwargs.get('deep', True):
            function = function.simplify(**kwargs)
        terms = Add.make_args(expand(function))
        s_t = []
        o_t = []
        for term in terms:
            if term.has(Sum):
                subterms = Mul.make_args(expand(term))
                out_terms = []
                for subterm in subterms:
                    if isinstance(subterm, Sum):
                        out_terms.append(subterm._eval_simplify(**kwargs))
                    else:
                        out_terms.append(subterm)
                s_t.append(Mul(*out_terms))
            else:
                o_t.append(term)
        from sympy.simplify.simplify import factor_sum, sum_combine
        result = Add(sum_combine(s_t), *o_t)
        return factor_sum(result, limits=self.limits)

    def is_convergent(self):
        if False:
            print('Hello World!')
        "\n        Checks for the convergence of a Sum.\n\n        Explanation\n        ===========\n\n        We divide the study of convergence of infinite sums and products in\n        two parts.\n\n        First Part:\n        One part is the question whether all the terms are well defined, i.e.,\n        they are finite in a sum and also non-zero in a product. Zero\n        is the analogy of (minus) infinity in products as\n        :math:`e^{-\\infty} = 0`.\n\n        Second Part:\n        The second part is the question of convergence after infinities,\n        and zeros in products, have been omitted assuming that their number\n        is finite. This means that we only consider the tail of the sum or\n        product, starting from some point after which all terms are well\n        defined.\n\n        For example, in a sum of the form:\n\n        .. math::\n\n            \\sum_{1 \\leq i < \\infty} \\frac{1}{n^2 + an + b}\n\n        where a and b are numbers. The routine will return true, even if there\n        are infinities in the term sequence (at most two). An analogous\n        product would be:\n\n        .. math::\n\n            \\prod_{1 \\leq i < \\infty} e^{\\frac{1}{n^2 + an + b}}\n\n        This is how convergence is interpreted. It is concerned with what\n        happens at the limit. Finding the bad terms is another independent\n        matter.\n\n        Note: It is responsibility of user to see that the sum or product\n        is well defined.\n\n        There are various tests employed to check the convergence like\n        divergence test, root test, integral test, alternating series test,\n        comparison tests, Dirichlet tests. It returns true if Sum is convergent\n        and false if divergent and NotImplementedError if it cannot be checked.\n\n        References\n        ==========\n\n        .. [1] https://en.wikipedia.org/wiki/Convergence_tests\n\n        Examples\n        ========\n\n        >>> from sympy import factorial, S, Sum, Symbol, oo\n        >>> n = Symbol('n', integer=True)\n        >>> Sum(n/(n - 1), (n, 4, 7)).is_convergent()\n        True\n        >>> Sum(n/(2*n + 1), (n, 1, oo)).is_convergent()\n        False\n        >>> Sum(factorial(n)/5**n, (n, 1, oo)).is_convergent()\n        False\n        >>> Sum(1/n**(S(6)/5), (n, 1, oo)).is_convergent()\n        True\n\n        See Also\n        ========\n\n        Sum.is_absolutely_convergent\n        sympy.concrete.products.Product.is_convergent\n        "
        (p, q, r) = symbols('p q r', cls=Wild)
        sym = self.limits[0][0]
        lower_limit = self.limits[0][1]
        upper_limit = self.limits[0][2]
        sequence_term = self.function.simplify()
        if len(sequence_term.free_symbols) > 1:
            raise NotImplementedError('convergence checking for more than one symbol containing series is not handled')
        if lower_limit.is_finite and upper_limit.is_finite:
            return S.true
        if lower_limit is S.NegativeInfinity:
            if upper_limit is S.Infinity:
                return Sum(sequence_term, (sym, 0, S.Infinity)).is_convergent() and Sum(sequence_term, (sym, S.NegativeInfinity, 0)).is_convergent()
            from sympy.simplify.simplify import simplify
            sequence_term = simplify(sequence_term.xreplace({sym: -sym}))
            lower_limit = -upper_limit
            upper_limit = S.Infinity
        sym_ = Dummy(sym.name, integer=True, positive=True)
        sequence_term = sequence_term.xreplace({sym: sym_})
        sym = sym_
        interval = Interval(lower_limit, upper_limit)
        if sequence_term.is_Piecewise:
            for (func, cond) in sequence_term.args:
                if cond == True or cond.as_set().sup is S.Infinity:
                    s = Sum(func, (sym, lower_limit, upper_limit))
                    return s.is_convergent()
            return S.true
        try:
            lim_val = limit_seq(sequence_term, sym)
            if lim_val is not None and lim_val.is_zero is False:
                return S.false
        except NotImplementedError:
            pass
        try:
            lim_val_abs = limit_seq(abs(sequence_term), sym)
            if lim_val_abs is not None and lim_val_abs.is_zero is False:
                return S.false
        except NotImplementedError:
            pass
        order = O(sequence_term, (sym, S.Infinity))
        p_series_test = order.expr.match(sym ** p)
        if p_series_test is not None:
            if p_series_test[p] < -1:
                return S.true
            if p_series_test[p] >= -1:
                return S.false
        n_log_test = order.expr.match(1 / (sym ** p * log(1 / sym) ** q * log(-log(1 / sym)) ** r)) or order.expr.match(1 / (sym ** p * (-log(1 / sym)) ** q * log(-log(1 / sym)) ** r))
        if n_log_test is not None:
            if n_log_test[p] > 1 or (n_log_test[p] == 1 and n_log_test[q] > 1) or (n_log_test[p] == n_log_test[q] == 1 and n_log_test[r] > 1):
                return S.true
            return S.false
        try:
            lim_comp = limit_seq(sym * sequence_term, sym)
            if lim_comp is not None and lim_comp.is_number and (lim_comp > 0):
                return S.false
        except NotImplementedError:
            pass
        next_sequence_term = sequence_term.xreplace({sym: sym + 1})
        from sympy.simplify.combsimp import combsimp
        from sympy.simplify.powsimp import powsimp
        ratio = combsimp(powsimp(next_sequence_term / sequence_term))
        try:
            lim_ratio = limit_seq(ratio, sym)
            if lim_ratio is not None and lim_ratio.is_number:
                if abs(lim_ratio) > 1:
                    return S.false
                if abs(lim_ratio) < 1:
                    return S.true
        except NotImplementedError:
            lim_ratio = None
        if lim_ratio == 1:
            test_val = sym * (sequence_term / sequence_term.subs(sym, sym + 1) - 1)
            test_val = test_val.gammasimp()
            try:
                lim_val = limit_seq(test_val, sym)
                if lim_val is not None and lim_val.is_number:
                    if lim_val > 1:
                        return S.true
                    if lim_val < 1:
                        return S.false
            except NotImplementedError:
                pass
        try:
            lim_evaluated = limit_seq(abs(sequence_term) ** (1 / sym), sym)
            if lim_evaluated is not None and lim_evaluated.is_number:
                if lim_evaluated < 1:
                    return S.true
                if lim_evaluated > 1:
                    return S.false
        except NotImplementedError:
            pass
        dict_val = sequence_term.match(S.NegativeOne ** (sym + p) * q)
        if not dict_val[p].has(sym) and is_decreasing(dict_val[q], interval):
            return S.true
        check_interval = None
        from sympy.solvers.solveset import solveset
        maxima = solveset(sequence_term.diff(sym), sym, interval)
        if not maxima:
            check_interval = interval
        elif isinstance(maxima, FiniteSet) and maxima.sup.is_number:
            check_interval = Interval(maxima.sup, interval.sup)
        if check_interval is not None and (is_decreasing(sequence_term, check_interval) or is_decreasing(-sequence_term, check_interval)):
            integral_val = Integral(sequence_term, (sym, lower_limit, upper_limit))
            try:
                integral_val_evaluated = integral_val.doit()
                if integral_val_evaluated.is_number:
                    return S(integral_val_evaluated.is_finite)
            except NotImplementedError:
                pass
        if order.expr.is_Mul:
            args = order.expr.args
            argset = set(args)
            m = Dummy('m', integer=True)

            def _dirichlet_test(g_n):
                if False:
                    while True:
                        i = 10
                try:
                    ing_val = limit_seq(Sum(g_n, (sym, interval.inf, m)).doit(), m)
                    if ing_val is not None and ing_val.is_finite:
                        return S.true
                except NotImplementedError:
                    pass

            def _bounded_convergent_test(g1_n, g2_n):
                if False:
                    return 10
                try:
                    lim_val = limit_seq(g1_n, sym)
                    if lim_val is not None and (lim_val.is_finite or (isinstance(lim_val, AccumulationBounds) and (lim_val.max - lim_val.min).is_finite)):
                        if Sum(g2_n, (sym, lower_limit, upper_limit)).is_absolutely_convergent():
                            return S.true
                except NotImplementedError:
                    pass
            for n in range(1, len(argset)):
                for a_tuple in itertools.combinations(args, n):
                    b_set = argset - set(a_tuple)
                    a_n = Mul(*a_tuple)
                    b_n = Mul(*b_set)
                    if is_decreasing(a_n, interval):
                        dirich = _dirichlet_test(b_n)
                        if dirich is not None:
                            return dirich
                    bc_test = _bounded_convergent_test(a_n, b_n)
                    if bc_test is not None:
                        return bc_test
        _sym = self.limits[0][0]
        sequence_term = sequence_term.xreplace({sym: _sym})
        raise NotImplementedError('The algorithm to find the Sum convergence of %s is not yet implemented' % sequence_term)

    def is_absolutely_convergent(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Checks for the absolute convergence of an infinite series.\n\n        Same as checking convergence of absolute value of sequence_term of\n        an infinite series.\n\n        References\n        ==========\n\n        .. [1] https://en.wikipedia.org/wiki/Absolute_convergence\n\n        Examples\n        ========\n\n        >>> from sympy import Sum, Symbol, oo\n        >>> n = Symbol('n', integer=True)\n        >>> Sum((-1)**n, (n, 1, oo)).is_absolutely_convergent()\n        False\n        >>> Sum((-1)**n/n**2, (n, 1, oo)).is_absolutely_convergent()\n        True\n\n        See Also\n        ========\n\n        Sum.is_convergent\n        "
        return Sum(abs(self.function), self.limits).is_convergent()

    def euler_maclaurin(self, m=0, n=0, eps=0, eval_integral=True):
        if False:
            while True:
                i = 10
        '\n        Return an Euler-Maclaurin approximation of self, where m is the\n        number of leading terms to sum directly and n is the number of\n        terms in the tail.\n\n        With m = n = 0, this is simply the corresponding integral\n        plus a first-order endpoint correction.\n\n        Returns (s, e) where s is the Euler-Maclaurin approximation\n        and e is the estimated error (taken to be the magnitude of\n        the first omitted term in the tail):\n\n            >>> from sympy.abc import k, a, b\n            >>> from sympy import Sum\n            >>> Sum(1/k, (k, 2, 5)).doit().evalf()\n            1.28333333333333\n            >>> s, e = Sum(1/k, (k, 2, 5)).euler_maclaurin()\n            >>> s\n            -log(2) + 7/20 + log(5)\n            >>> from sympy import sstr\n            >>> print(sstr((s.evalf(), e.evalf()), full_prec=True))\n            (1.26629073187415, 0.0175000000000000)\n\n        The endpoints may be symbolic:\n\n            >>> s, e = Sum(1/k, (k, a, b)).euler_maclaurin()\n            >>> s\n            -log(a) + log(b) + 1/(2*b) + 1/(2*a)\n            >>> e\n            Abs(1/(12*b**2) - 1/(12*a**2))\n\n        If the function is a polynomial of degree at most 2n+1, the\n        Euler-Maclaurin formula becomes exact (and e = 0 is returned):\n\n            >>> Sum(k, (k, 2, b)).euler_maclaurin()\n            (b**2/2 + b/2 - 1, 0)\n            >>> Sum(k, (k, 2, b)).doit()\n            b**2/2 + b/2 - 1\n\n        With a nonzero eps specified, the summation is ended\n        as soon as the remainder term is less than the epsilon.\n        '
        m = int(m)
        n = int(n)
        f = self.function
        if len(self.limits) != 1:
            raise ValueError('More than 1 limit')
        (i, a, b) = self.limits[0]
        if (a > b) == True:
            if a - b == 1:
                return (S.Zero, S.Zero)
            (a, b) = (b + 1, a - 1)
            f = -f
        s = S.Zero
        if m:
            if b.is_Integer and a.is_Integer:
                m = min(m, b - a + 1)
            if not eps or f.is_polynomial(i):
                s = Add(*[f.subs(i, a + k) for k in range(m)])
            else:
                term = f.subs(i, a)
                if term:
                    test = abs(term.evalf(3)) < eps
                    if test == True:
                        return (s, abs(term))
                    elif not test == False:
                        return (term, S.Zero)
                s = term
                for k in range(1, m):
                    term = f.subs(i, a + k)
                    if abs(term.evalf(3)) < eps and term != 0:
                        return (s, abs(term))
                    s += term
            if b - a + 1 == m:
                return (s, S.Zero)
            a += m
        x = Dummy('x')
        I = Integral(f.subs(i, x), (x, a, b))
        if eval_integral:
            I = I.doit()
        s += I

        def fpoint(expr):
            if False:
                i = 10
                return i + 15
            if b is S.Infinity:
                return (expr.subs(i, a), 0)
            return (expr.subs(i, a), expr.subs(i, b))
        (fa, fb) = fpoint(f)
        iterm = (fa + fb) / 2
        g = f.diff(i)
        for k in range(1, n + 2):
            (ga, gb) = fpoint(g)
            term = bernoulli(2 * k) / factorial(2 * k) * (gb - ga)
            if k > n:
                break
            if eps and term:
                term_evalf = term.evalf(3)
                if term_evalf is S.NaN:
                    return (S.NaN, S.NaN)
                if abs(term_evalf) < eps:
                    break
            s += term
            g = g.diff(i, 2, simplify=False)
        return (s + iterm, abs(term))

    def reverse_order(self, *indices):
        if False:
            for i in range(10):
                print('nop')
        '\n        Reverse the order of a limit in a Sum.\n\n        Explanation\n        ===========\n\n        ``reverse_order(self, *indices)`` reverses some limits in the expression\n        ``self`` which can be either a ``Sum`` or a ``Product``. The selectors in\n        the argument ``indices`` specify some indices whose limits get reversed.\n        These selectors are either variable names or numerical indices counted\n        starting from the inner-most limit tuple.\n\n        Examples\n        ========\n\n        >>> from sympy import Sum\n        >>> from sympy.abc import x, y, a, b, c, d\n\n        >>> Sum(x, (x, 0, 3)).reverse_order(x)\n        Sum(-x, (x, 4, -1))\n        >>> Sum(x*y, (x, 1, 5), (y, 0, 6)).reverse_order(x, y)\n        Sum(x*y, (x, 6, 0), (y, 7, -1))\n        >>> Sum(x, (x, a, b)).reverse_order(x)\n        Sum(-x, (x, b + 1, a - 1))\n        >>> Sum(x, (x, a, b)).reverse_order(0)\n        Sum(-x, (x, b + 1, a - 1))\n\n        While one should prefer variable names when specifying which limits\n        to reverse, the index counting notation comes in handy in case there\n        are several symbols with the same name.\n\n        >>> S = Sum(x**2, (x, a, b), (x, c, d))\n        >>> S\n        Sum(x**2, (x, a, b), (x, c, d))\n        >>> S0 = S.reverse_order(0)\n        >>> S0\n        Sum(-x**2, (x, b + 1, a - 1), (x, c, d))\n        >>> S1 = S0.reverse_order(1)\n        >>> S1\n        Sum(x**2, (x, b + 1, a - 1), (x, d + 1, c - 1))\n\n        Of course we can mix both notations:\n\n        >>> Sum(x*y, (x, a, b), (y, 2, 5)).reverse_order(x, 1)\n        Sum(x*y, (x, b + 1, a - 1), (y, 6, 1))\n        >>> Sum(x*y, (x, a, b), (y, 2, 5)).reverse_order(y, x)\n        Sum(x*y, (x, b + 1, a - 1), (y, 6, 1))\n\n        See Also\n        ========\n\n        sympy.concrete.expr_with_intlimits.ExprWithIntLimits.index, reorder_limit,\n        sympy.concrete.expr_with_intlimits.ExprWithIntLimits.reorder\n\n        References\n        ==========\n\n        .. [1] Michael Karr, "Summation in Finite Terms", Journal of the ACM,\n               Volume 28 Issue 2, April 1981, Pages 305-350\n               https://dl.acm.org/doi/10.1145/322248.322255\n        '
        l_indices = list(indices)
        for (i, indx) in enumerate(l_indices):
            if not isinstance(indx, int):
                l_indices[i] = self.index(indx)
        e = 1
        limits = []
        for (i, limit) in enumerate(self.limits):
            l = limit
            if i in l_indices:
                e = -e
                l = (limit[0], limit[2] + 1, limit[1] - 1)
            limits.append(l)
        return Sum(e * self.function, *limits)

    def _eval_rewrite_as_Product(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        from sympy.concrete.products import Product
        if self.function.is_extended_real:
            return log(Product(exp(self.function), *self.limits))

def summation(f, *symbols, **kwargs):
    if False:
        return 10
    "\n    Compute the summation of f with respect to symbols.\n\n    Explanation\n    ===========\n\n    The notation for symbols is similar to the notation used in Integral.\n    summation(f, (i, a, b)) computes the sum of f with respect to i from a to b,\n    i.e.,\n\n    ::\n\n                                    b\n                                  ____\n                                  \\   `\n        summation(f, (i, a, b)) =  )    f\n                                  /___,\n                                  i = a\n\n    If it cannot compute the sum, it returns an unevaluated Sum object.\n    Repeated sums can be computed by introducing additional symbols tuples::\n\n    Examples\n    ========\n\n    >>> from sympy import summation, oo, symbols, log\n    >>> i, n, m = symbols('i n m', integer=True)\n\n    >>> summation(2*i - 1, (i, 1, n))\n    n**2\n    >>> summation(1/2**i, (i, 0, oo))\n    2\n    >>> summation(1/log(n)**n, (n, 2, oo))\n    Sum(log(n)**(-n), (n, 2, oo))\n    >>> summation(i, (i, 0, n), (n, 0, m))\n    m**3/6 + m**2/2 + m/3\n\n    >>> from sympy.abc import x\n    >>> from sympy import factorial\n    >>> summation(x**n/factorial(n), (n, 0, oo))\n    exp(x)\n\n    See Also\n    ========\n\n    Sum\n    Product, sympy.concrete.products.product\n\n    "
    return Sum(f, *symbols, **kwargs).doit(deep=False)

def telescopic_direct(L, R, n, limits):
    if False:
        i = 10
        return i + 15
    '\n    Returns the direct summation of the terms of a telescopic sum\n\n    Explanation\n    ===========\n\n    L is the term with lower index\n    R is the term with higher index\n    n difference between the indexes of L and R\n\n    Examples\n    ========\n\n    >>> from sympy.concrete.summations import telescopic_direct\n    >>> from sympy.abc import k, a, b\n    >>> telescopic_direct(1/k, -1/(k+2), 2, (k, a, b))\n    -1/(b + 2) - 1/(b + 1) + 1/(a + 1) + 1/a\n\n    '
    (i, a, b) = limits
    return Add(*[L.subs(i, a + m) + R.subs(i, b - m) for m in range(n)])

def telescopic(L, R, limits):
    if False:
        print('Hello World!')
    '\n    Tries to perform the summation using the telescopic property.\n\n    Return None if not possible.\n    '
    (i, a, b) = limits
    if L.is_Add or R.is_Add:
        return None
    k = Wild('k')
    sol = (-R).match(L.subs(i, i + k))
    s = None
    if sol and k in sol:
        s = sol[k]
        if not (s.is_Integer and L.subs(i, i + s) + R == 0):
            s = None
    if s is None:
        m = Dummy('m')
        try:
            from sympy.solvers.solvers import solve
            sol = solve(L.subs(i, i + m) + R, m) or []
        except NotImplementedError:
            return None
        sol = [si for si in sol if si.is_Integer and (L.subs(i, i + si) + R).expand().is_zero]
        if len(sol) != 1:
            return None
        s = sol[0]
    if s < 0:
        return telescopic_direct(R, L, abs(s), (i, a, b))
    elif s > 0:
        return telescopic_direct(L, R, s, (i, a, b))

def eval_sum(f, limits):
    if False:
        i = 10
        return i + 15
    (i, a, b) = limits
    if f.is_zero:
        return S.Zero
    if i not in f.free_symbols:
        return f * (b - a + 1)
    if a == b:
        return f.subs(i, a)
    if isinstance(f, Piecewise):
        if not any((i in arg.args[1].free_symbols for arg in f.args)):
            newargs = []
            for arg in f.args:
                newexpr = eval_sum(arg.expr, limits)
                if newexpr is None:
                    return None
                newargs.append((newexpr, arg.cond))
            return f.func(*newargs)
    if f.has(KroneckerDelta):
        from .delta import deltasummation, _has_simple_delta
        f = f.replace(lambda x: isinstance(x, Sum), lambda x: x.factor())
        if _has_simple_delta(f, limits[0]):
            return deltasummation(f, limits)
    dif = b - a
    definite = dif.is_Integer
    if definite and dif < 100:
        return eval_sum_direct(f, (i, a, b))
    if isinstance(f, Piecewise):
        return None
    value = eval_sum_symbolic(f.expand(), (i, a, b))
    if value is not None:
        return value
    if definite:
        return eval_sum_direct(f, (i, a, b))

def eval_sum_direct(expr, limits):
    if False:
        for i in range(10):
            print('nop')
    '\n    Evaluate expression directly, but perform some simple checks first\n    to possibly result in a smaller expression and faster execution.\n    '
    (i, a, b) = limits
    dif = b - a
    if expr.is_Mul:
        (without_i, with_i) = expr.as_independent(i)
        if without_i != 1:
            s = eval_sum_direct(with_i, (i, a, b))
            if s:
                r = without_i * s
                if r is not S.NaN:
                    return r
        else:
            (L, R) = expr.as_two_terms()
            if not L.has(i):
                sR = eval_sum_direct(R, (i, a, b))
                if sR:
                    return L * sR
            if not R.has(i):
                sL = eval_sum_direct(L, (i, a, b))
                if sL:
                    return sL * R
    try:
        expr = apart(expr, i)
    except PolynomialError:
        pass
    if expr.is_Add:
        (without_i, with_i) = expr.as_independent(i)
        if without_i != 0:
            s = eval_sum_direct(with_i, (i, a, b))
            if s:
                r = without_i * (dif + 1) + s
                if r is not S.NaN:
                    return r
        else:
            (L, R) = expr.as_two_terms()
            lsum = eval_sum_direct(L, (i, a, b))
            rsum = eval_sum_direct(R, (i, a, b))
            if None not in (lsum, rsum):
                r = lsum + rsum
                if r is not S.NaN:
                    return r
    return Add(*[expr.subs(i, a + j) for j in range(dif + 1)])

def eval_sum_symbolic(f, limits):
    if False:
        while True:
            i = 10
    f_orig = f
    (i, a, b) = limits
    if not f.has(i):
        return f * (b - a + 1)
    if f.is_Mul:
        (without_i, with_i) = f.as_independent(i)
        if without_i != 1:
            s = eval_sum_symbolic(with_i, (i, a, b))
            if s:
                r = without_i * s
                if r is not S.NaN:
                    return r
        else:
            (L, R) = f.as_two_terms()
            if not L.has(i):
                sR = eval_sum_symbolic(R, (i, a, b))
                if sR:
                    return L * sR
            if not R.has(i):
                sL = eval_sum_symbolic(L, (i, a, b))
                if sL:
                    return sL * R
    try:
        f = apart(f, i)
    except PolynomialError:
        pass
    if f.is_Add:
        (L, R) = f.as_two_terms()
        lrsum = telescopic(L, R, (i, a, b))
        if lrsum:
            return lrsum
        (without_i, with_i) = f.as_independent(i)
        if without_i != 0:
            s = eval_sum_symbolic(with_i, (i, a, b))
            if s:
                r = without_i * (b - a + 1) + s
                if r is not S.NaN:
                    return r
        else:
            lsum = eval_sum_symbolic(L, (i, a, b))
            rsum = eval_sum_symbolic(R, (i, a, b))
            if None not in (lsum, rsum):
                r = lsum + rsum
                if r is not S.NaN:
                    return r
    n = Wild('n')
    result = f.match(i ** n)
    if result is not None:
        n = result[n]
        if n.is_Integer:
            if n >= 0:
                if b is S.Infinity and a is not S.NegativeInfinity or (a is S.NegativeInfinity and b is not S.Infinity):
                    return S.Infinity
                return ((bernoulli(n + 1, b + 1) - bernoulli(n + 1, a)) / (n + 1)).expand()
            elif a.is_Integer and a >= 1:
                if n == -1:
                    return harmonic(b) - harmonic(a - 1)
                else:
                    return harmonic(b, abs(n)) - harmonic(a - 1, abs(n))
    if not (a.has(S.Infinity, S.NegativeInfinity) or b.has(S.Infinity, S.NegativeInfinity)):
        c1 = Wild('c1', exclude=[i])
        c2 = Wild('c2', exclude=[i])
        c3 = Wild('c3', exclude=[i])
        wexp = Wild('wexp')
        e = f.powsimp().match(c1 ** wexp)
        if e is not None:
            e_exp = e.pop(wexp).expand().match(c2 * i + c3)
            if e_exp is not None:
                e.update(e_exp)
                p = (c1 ** c3).subs(e)
                q = (c1 ** c2).subs(e)
                r = p * (q ** a - q ** (b + 1)) / (1 - q)
                l = p * (b - a + 1)
                return Piecewise((l, Eq(q, S.One)), (r, True))
        r = gosper_sum(f, (i, a, b))
        if isinstance(r, (Mul, Add)):
            from sympy.simplify.radsimp import denom
            from sympy.solvers.solvers import solve
            non_limit = r.free_symbols - Tuple(*limits[1:]).free_symbols
            den = denom(together(r))
            den_sym = non_limit & den.free_symbols
            args = []
            for v in ordered(den_sym):
                try:
                    s = solve(den, v)
                    m = Eq(v, s[0]) if s else S.false
                    if m != False:
                        args.append((Sum(f_orig.subs(*m.args), limits).doit(), m))
                    break
                except NotImplementedError:
                    continue
            args.append((r, True))
            return Piecewise(*args)
        if r not in (None, S.NaN):
            return r
    h = eval_sum_hyper(f_orig, (i, a, b))
    if h is not None:
        return h
    r = eval_sum_residue(f_orig, (i, a, b))
    if r is not None:
        return r
    factored = f_orig.factor()
    if factored != f_orig:
        return eval_sum_symbolic(factored, (i, a, b))

def _eval_sum_hyper(f, i, a):
    if False:
        for i in range(10):
            print('nop')
    ' Returns (res, cond). Sums from a to oo. '
    if a != 0:
        return _eval_sum_hyper(f.subs(i, i + a), i, 0)
    if f.subs(i, 0) == 0:
        from sympy.simplify.simplify import simplify
        if simplify(f.subs(i, Dummy('i', integer=True, positive=True))) == 0:
            return (S.Zero, True)
        return _eval_sum_hyper(f.subs(i, i + 1), i, 0)
    from sympy.simplify.simplify import hypersimp
    hs = hypersimp(f, i)
    if hs is None:
        return None
    if isinstance(hs, Float):
        from sympy.simplify.simplify import nsimplify
        hs = nsimplify(hs)
    from sympy.simplify.combsimp import combsimp
    from sympy.simplify.hyperexpand import hyperexpand
    from sympy.simplify.radsimp import fraction
    (numer, denom) = fraction(factor(hs))
    (top, topl) = numer.as_coeff_mul(i)
    (bot, botl) = denom.as_coeff_mul(i)
    ab = [top, bot]
    factors = [topl, botl]
    params = [[], []]
    for k in range(2):
        for fac in factors[k]:
            mul = 1
            if fac.is_Pow:
                mul = fac.exp
                fac = fac.base
                if not mul.is_Integer:
                    return None
            p = Poly(fac, i)
            if p.degree() != 1:
                return None
            (m, n) = p.all_coeffs()
            ab[k] *= m ** mul
            params[k] += [n / m] * mul
    ap = params[0] + [1]
    bq = params[1]
    x = ab[0] / ab[1]
    h = hyper(ap, bq, x)
    f = combsimp(f)
    return (f.subs(i, 0) * hyperexpand(h), h.convergence_statement)

def eval_sum_hyper(f, i_a_b):
    if False:
        print('Hello World!')
    (i, a, b) = i_a_b
    if f.is_hypergeometric(i) is False:
        return
    if (b - a).is_Integer:
        return None
    old_sum = Sum(f, (i, a, b))
    if b != S.Infinity:
        if a is S.NegativeInfinity:
            res = _eval_sum_hyper(f.subs(i, -i), i, -b)
            if res is not None:
                return Piecewise(res, (old_sum, True))
        else:
            n_illegal = lambda x: sum((x.count(_) for _ in _illegal))
            had = n_illegal(f)
            res1 = _eval_sum_hyper(f, i, a)
            if res1 is None or n_illegal(res1) > had:
                return
            res2 = _eval_sum_hyper(f, i, b + 1)
            if res2 is None or n_illegal(res2) > had:
                return
            ((res1, cond1), (res2, cond2)) = (res1, res2)
            cond = And(cond1, cond2)
            if cond == False:
                return None
            return Piecewise((res1 - res2, cond), (old_sum, True))
    if a is S.NegativeInfinity:
        res1 = _eval_sum_hyper(f.subs(i, -i), i, 1)
        res2 = _eval_sum_hyper(f, i, 0)
        if res1 is None or res2 is None:
            return None
        (res1, cond1) = res1
        (res2, cond2) = res2
        cond = And(cond1, cond2)
        if cond == False or cond.as_set() == S.EmptySet:
            return None
        return Piecewise((res1 + res2, cond), (old_sum, True))
    res = _eval_sum_hyper(f, i, a)
    if res is not None:
        (r, c) = res
        if c == False:
            if r.is_number:
                f = f.subs(i, Dummy('i', integer=True, positive=True) + a)
                if f.is_positive or f.is_zero:
                    return S.Infinity
                elif f.is_negative:
                    return S.NegativeInfinity
            return None
        return Piecewise(res, (old_sum, True))

def eval_sum_residue(f, i_a_b):
    if False:
        i = 10
        return i + 15
    "Compute the infinite summation with residues\n\n    Notes\n    =====\n\n    If $f(n), g(n)$ are polynomials with $\\deg(g(n)) - \\deg(f(n)) \\ge 2$,\n    some infinite summations can be computed by the following residue\n    evaluations.\n\n    .. math::\n        \\sum_{n=-\\infty, g(n) \\ne 0}^{\\infty} \\frac{f(n)}{g(n)} =\n        -\\pi \\sum_{\\alpha|g(\\alpha)=0}\n        \\text{Res}(\\cot(\\pi x) \\frac{f(x)}{g(x)}, \\alpha)\n\n    .. math::\n        \\sum_{n=-\\infty, g(n) \\ne 0}^{\\infty} (-1)^n \\frac{f(n)}{g(n)} =\n        -\\pi \\sum_{\\alpha|g(\\alpha)=0}\n        \\text{Res}(\\csc(\\pi x) \\frac{f(x)}{g(x)}, \\alpha)\n\n    Examples\n    ========\n\n    >>> from sympy import Sum, oo, Symbol\n    >>> x = Symbol('x')\n\n    Doubly infinite series of rational functions.\n\n    >>> Sum(1 / (x**2 + 1), (x, -oo, oo)).doit()\n    pi/tanh(pi)\n\n    Doubly infinite alternating series of rational functions.\n\n    >>> Sum((-1)**x / (x**2 + 1), (x, -oo, oo)).doit()\n    pi/sinh(pi)\n\n    Infinite series of even rational functions.\n\n    >>> Sum(1 / (x**2 + 1), (x, 0, oo)).doit()\n    1/2 + pi/(2*tanh(pi))\n\n    Infinite series of alternating even rational functions.\n\n    >>> Sum((-1)**x / (x**2 + 1), (x, 0, oo)).doit()\n    pi/(2*sinh(pi)) + 1/2\n\n    This also have heuristics to transform arbitrarily shifted summand or\n    arbitrarily shifted summation range to the canonical problem the\n    formula can handle.\n\n    >>> Sum(1 / (x**2 + 2*x + 2), (x, -1, oo)).doit()\n    1/2 + pi/(2*tanh(pi))\n    >>> Sum(1 / (x**2 + 4*x + 5), (x, -2, oo)).doit()\n    1/2 + pi/(2*tanh(pi))\n    >>> Sum(1 / (x**2 + 1), (x, 1, oo)).doit()\n    -1/2 + pi/(2*tanh(pi))\n    >>> Sum(1 / (x**2 + 1), (x, 2, oo)).doit()\n    -1 + pi/(2*tanh(pi))\n\n    References\n    ==========\n\n    .. [#] http://www.supermath.info/InfiniteSeriesandtheResidueTheorem.pdf\n\n    .. [#] Asmar N.H., Grafakos L. (2018) Residue Theory.\n           In: Complex Analysis with Applications.\n           Undergraduate Texts in Mathematics. Springer, Cham.\n           https://doi.org/10.1007/978-3-319-94063-2_5\n    "
    (i, a, b) = i_a_b

    def is_even_function(numer, denom):
        if False:
            print('Hello World!')
        'Test if the rational function is an even function'
        numer_even = all((i % 2 == 0 for (i,) in numer.monoms()))
        denom_even = all((i % 2 == 0 for (i,) in denom.monoms()))
        numer_odd = all((i % 2 == 1 for (i,) in numer.monoms()))
        denom_odd = all((i % 2 == 1 for (i,) in denom.monoms()))
        return numer_even and denom_even or (numer_odd and denom_odd)

    def match_rational(f, i):
        if False:
            for i in range(10):
                print('nop')
        (numer, denom) = f.as_numer_denom()
        try:
            ((numer, denom), opt) = parallel_poly_from_expr((numer, denom), i)
        except (PolificationFailed, PolynomialError):
            return None
        return (numer, denom)

    def get_poles(denom):
        if False:
            while True:
                i = 10
        roots = denom.sqf_part().all_roots()
        roots = sift(roots, lambda x: x.is_integer)
        if None in roots:
            return None
        (int_roots, nonint_roots) = (roots[True], roots[False])
        return (int_roots, nonint_roots)

    def get_shift(denom):
        if False:
            return 10
        n = denom.degree(i)
        a = denom.coeff_monomial(i ** n)
        b = denom.coeff_monomial(i ** (n - 1))
        shift = -b / a / n
        return shift
    z = Dummy('z')

    def get_residue_factor(numer, denom, alternating):
        if False:
            i = 10
            return i + 15
        residue_factor = (numer.as_expr() / denom.as_expr()).subs(i, z)
        if not alternating:
            residue_factor *= cot(S.Pi * z)
        else:
            residue_factor *= csc(S.Pi * z)
        return residue_factor
    if f.free_symbols - {i}:
        return None
    if not (a.is_Integer or a in (S.Infinity, S.NegativeInfinity)):
        return None
    if not (b.is_Integer or b in (S.Infinity, S.NegativeInfinity)):
        return None
    if a != S.NegativeInfinity and b != S.Infinity:
        return None
    match = match_rational(f, i)
    if match:
        alternating = False
        (numer, denom) = match
    else:
        match = match_rational(f / S.NegativeOne ** i, i)
        if match:
            alternating = True
            (numer, denom) = match
        else:
            return None
    if denom.degree(i) - numer.degree(i) < 2:
        return None
    if (a, b) == (S.NegativeInfinity, S.Infinity):
        poles = get_poles(denom)
        if poles is None:
            return None
        (int_roots, nonint_roots) = poles
        if int_roots:
            return None
        residue_factor = get_residue_factor(numer, denom, alternating)
        residues = [residue(residue_factor, z, root) for root in nonint_roots]
        return -S.Pi * sum(residues)
    if not (a.is_finite and b is S.Infinity):
        return None
    if not is_even_function(numer, denom):
        shift = get_shift(denom)
        if not shift.is_Integer:
            return None
        if shift == 0:
            return None
        numer = numer.shift(shift)
        denom = denom.shift(shift)
        if not is_even_function(numer, denom):
            return None
        if alternating:
            f = S.NegativeOne ** i * (S.NegativeOne ** shift * numer.as_expr() / denom.as_expr())
        else:
            f = numer.as_expr() / denom.as_expr()
        return eval_sum_residue(f, (i, a - shift, b - shift))
    poles = get_poles(denom)
    if poles is None:
        return None
    (int_roots, nonint_roots) = poles
    if int_roots:
        int_roots = [int(root) for root in int_roots]
        int_roots_max = max(int_roots)
        int_roots_min = min(int_roots)
        if not len(int_roots) == int_roots_max - int_roots_min + 1:
            return None
        if a <= max(int_roots):
            return None
    residue_factor = get_residue_factor(numer, denom, alternating)
    residues = [residue(residue_factor, z, root) for root in int_roots + nonint_roots]
    full_sum = -S.Pi * sum(residues)
    if not int_roots:
        half_sum = (full_sum + f.xreplace({i: 0})) / 2
        extraneous_neg = [f.xreplace({i: i0}) for i0 in range(int(a), 0)]
        extraneous_pos = [f.xreplace({i: i0}) for i0 in range(0, int(a))]
        result = half_sum + sum(extraneous_neg) - sum(extraneous_pos)
        return result
    half_sum = full_sum / 2
    extraneous = [f.xreplace({i: i0}) for i0 in range(max(int_roots) + 1, int(a))]
    result = half_sum - sum(extraneous)
    return result

def _eval_matrix_sum(expression):
    if False:
        print('Hello World!')
    f = expression.function
    for (n, limit) in enumerate(expression.limits):
        (i, a, b) = limit
        dif = b - a
        if dif.is_Integer:
            if (dif < 0) == True:
                (a, b) = (b + 1, a - 1)
                f = -f
            newf = eval_sum_direct(f, (i, a, b))
            if newf is not None:
                return newf.doit()

def _dummy_with_inherited_properties_concrete(limits):
    if False:
        while True:
            i = 10
    '\n    Return a Dummy symbol that inherits as many assumptions as possible\n    from the provided symbol and limits.\n\n    If the symbol already has all True assumption shared by the limits\n    then return None.\n    '
    (x, a, b) = limits
    l = [a, b]
    assumptions_to_consider = ['extended_nonnegative', 'nonnegative', 'extended_nonpositive', 'nonpositive', 'extended_positive', 'positive', 'extended_negative', 'negative', 'integer', 'rational', 'finite', 'zero', 'real', 'extended_real']
    assumptions_to_keep = {}
    assumptions_to_add = {}
    for assum in assumptions_to_consider:
        assum_true = x._assumptions.get(assum, None)
        if assum_true:
            assumptions_to_keep[assum] = True
        elif all((getattr(i, 'is_' + assum) for i in l)):
            assumptions_to_add[assum] = True
    if assumptions_to_add:
        assumptions_to_keep.update(assumptions_to_add)
        return Dummy('d', **assumptions_to_keep)