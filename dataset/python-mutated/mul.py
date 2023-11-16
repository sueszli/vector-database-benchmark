from typing import Tuple as tTuple
from collections import defaultdict
from functools import reduce
from itertools import product
import operator
from .sympify import sympify
from .basic import Basic, _args_sortkey
from .singleton import S
from .operations import AssocOp, AssocOpDispatcher
from .cache import cacheit
from .intfunc import integer_nthroot, trailing
from .logic import fuzzy_not, _fuzzy_group
from .expr import Expr
from .parameters import global_parameters
from .kind import KindDispatcher
from .traversal import bottom_up
from sympy.utilities.iterables import sift

class NC_Marker:
    is_Order = False
    is_Mul = False
    is_Number = False
    is_Poly = False
    is_commutative = False

def _mulsort(args):
    if False:
        print('Hello World!')
    args.sort(key=_args_sortkey)

def _unevaluated_Mul(*args):
    if False:
        while True:
            i = 10
    'Return a well-formed unevaluated Mul: Numbers are collected and\n    put in slot 0, any arguments that are Muls will be flattened, and args\n    are sorted. Use this when args have changed but you still want to return\n    an unevaluated Mul.\n\n    Examples\n    ========\n\n    >>> from sympy.core.mul import _unevaluated_Mul as uMul\n    >>> from sympy import S, sqrt, Mul\n    >>> from sympy.abc import x\n    >>> a = uMul(*[S(3.0), x, S(2)])\n    >>> a.args[0]\n    6.00000000000000\n    >>> a.args[1]\n    x\n\n    Two unevaluated Muls with the same arguments will\n    always compare as equal during testing:\n\n    >>> m = uMul(sqrt(2), sqrt(3))\n    >>> m == uMul(sqrt(3), sqrt(2))\n    True\n    >>> u = Mul(sqrt(3), sqrt(2), evaluate=False)\n    >>> m == uMul(u)\n    True\n    >>> m == Mul(*m.args)\n    False\n\n    '
    args = list(args)
    newargs = []
    ncargs = []
    co = S.One
    while args:
        a = args.pop()
        if a.is_Mul:
            (c, nc) = a.args_cnc()
            args.extend(c)
            if nc:
                ncargs.append(Mul._from_args(nc))
        elif a.is_Number:
            co *= a
        else:
            newargs.append(a)
    _mulsort(newargs)
    if co is not S.One:
        newargs.insert(0, co)
    if ncargs:
        newargs.append(Mul._from_args(ncargs))
    return Mul._from_args(newargs)

class Mul(Expr, AssocOp):
    """
    Expression representing multiplication operation for algebraic field.

    .. deprecated:: 1.7

       Using arguments that aren't subclasses of :class:`~.Expr` in core
       operators (:class:`~.Mul`, :class:`~.Add`, and :class:`~.Pow`) is
       deprecated. See :ref:`non-expr-args-deprecated` for details.

    Every argument of ``Mul()`` must be ``Expr``. Infix operator ``*``
    on most scalar objects in SymPy calls this class.

    Another use of ``Mul()`` is to represent the structure of abstract
    multiplication so that its arguments can be substituted to return
    different class. Refer to examples section for this.

    ``Mul()`` evaluates the argument unless ``evaluate=False`` is passed.
    The evaluation logic includes:

    1. Flattening
        ``Mul(x, Mul(y, z))`` -> ``Mul(x, y, z)``

    2. Identity removing
        ``Mul(x, 1, y)`` -> ``Mul(x, y)``

    3. Exponent collecting by ``.as_base_exp()``
        ``Mul(x, x**2)`` -> ``Pow(x, 3)``

    4. Term sorting
        ``Mul(y, x, 2)`` -> ``Mul(2, x, y)``

    Since multiplication can be vector space operation, arguments may
    have the different :obj:`sympy.core.kind.Kind()`. Kind of the
    resulting object is automatically inferred.

    Examples
    ========

    >>> from sympy import Mul
    >>> from sympy.abc import x, y
    >>> Mul(x, 1)
    x
    >>> Mul(x, x)
    x**2

    If ``evaluate=False`` is passed, result is not evaluated.

    >>> Mul(1, 2, evaluate=False)
    1*2
    >>> Mul(x, x, evaluate=False)
    x*x

    ``Mul()`` also represents the general structure of multiplication
    operation.

    >>> from sympy import MatrixSymbol
    >>> A = MatrixSymbol('A', 2,2)
    >>> expr = Mul(x,y).subs({y:A})
    >>> expr
    x*A
    >>> type(expr)
    <class 'sympy.matrices.expressions.matmul.MatMul'>

    See Also
    ========

    MatMul

    """
    __slots__ = ()
    args: tTuple[Expr, ...]
    is_Mul = True
    _args_type = Expr
    _kind_dispatcher = KindDispatcher('Mul_kind_dispatcher', commutative=True)

    @property
    def kind(self):
        if False:
            while True:
                i = 10
        arg_kinds = (a.kind for a in self.args)
        return self._kind_dispatcher(*arg_kinds)

    def could_extract_minus_sign(self):
        if False:
            for i in range(10):
                print('nop')
        if self == -self:
            return False
        c = self.args[0]
        return c.is_Number and c.is_extended_negative

    def __neg__(self):
        if False:
            i = 10
            return i + 15
        (c, args) = self.as_coeff_mul()
        if args[0] is not S.ComplexInfinity:
            c = -c
        if c is not S.One:
            if args[0].is_Number:
                args = list(args)
                if c is S.NegativeOne:
                    args[0] = -args[0]
                else:
                    args[0] *= c
            else:
                args = (c,) + args
        return self._from_args(args, self.is_commutative)

    @classmethod
    def flatten(cls, seq):
        if False:
            while True:
                i = 10
        'Return commutative, noncommutative and order arguments by\n        combining related terms.\n\n        Notes\n        =====\n            * In an expression like ``a*b*c``, Python process this through SymPy\n              as ``Mul(Mul(a, b), c)``. This can have undesirable consequences.\n\n              -  Sometimes terms are not combined as one would like:\n                 {c.f. https://github.com/sympy/sympy/issues/4596}\n\n                >>> from sympy import Mul, sqrt\n                >>> from sympy.abc import x, y, z\n                >>> 2*(x + 1) # this is the 2-arg Mul behavior\n                2*x + 2\n                >>> y*(x + 1)*2\n                2*y*(x + 1)\n                >>> 2*(x + 1)*y # 2-arg result will be obtained first\n                y*(2*x + 2)\n                >>> Mul(2, x + 1, y) # all 3 args simultaneously processed\n                2*y*(x + 1)\n                >>> 2*((x + 1)*y) # parentheses can control this behavior\n                2*y*(x + 1)\n\n                Powers with compound bases may not find a single base to\n                combine with unless all arguments are processed at once.\n                Post-processing may be necessary in such cases.\n                {c.f. https://github.com/sympy/sympy/issues/5728}\n\n                >>> a = sqrt(x*sqrt(y))\n                >>> a**3\n                (x*sqrt(y))**(3/2)\n                >>> Mul(a,a,a)\n                (x*sqrt(y))**(3/2)\n                >>> a*a*a\n                x*sqrt(y)*sqrt(x*sqrt(y))\n                >>> _.subs(a.base, z).subs(z, a.base)\n                (x*sqrt(y))**(3/2)\n\n              -  If more than two terms are being multiplied then all the\n                 previous terms will be re-processed for each new argument.\n                 So if each of ``a``, ``b`` and ``c`` were :class:`Mul`\n                 expression, then ``a*b*c`` (or building up the product\n                 with ``*=``) will process all the arguments of ``a`` and\n                 ``b`` twice: once when ``a*b`` is computed and again when\n                 ``c`` is multiplied.\n\n                 Using ``Mul(a, b, c)`` will process all arguments once.\n\n            * The results of Mul are cached according to arguments, so flatten\n              will only be called once for ``Mul(a, b, c)``. If you can\n              structure a calculation so the arguments are most likely to be\n              repeats then this can save time in computing the answer. For\n              example, say you had a Mul, M, that you wished to divide by ``d[i]``\n              and multiply by ``n[i]`` and you suspect there are many repeats\n              in ``n``. It would be better to compute ``M*n[i]/d[i]`` rather\n              than ``M/d[i]*n[i]`` since every time n[i] is a repeat, the\n              product, ``M*n[i]`` will be returned without flattening -- the\n              cached value will be returned. If you divide by the ``d[i]``\n              first (and those are more unique than the ``n[i]``) then that will\n              create a new Mul, ``M/d[i]`` the args of which will be traversed\n              again when it is multiplied by ``n[i]``.\n\n              {c.f. https://github.com/sympy/sympy/issues/5706}\n\n              This consideration is moot if the cache is turned off.\n\n            NB\n            --\n              The validity of the above notes depends on the implementation\n              details of Mul and flatten which may change at any time. Therefore,\n              you should only consider them when your code is highly performance\n              sensitive.\n\n              Removal of 1 from the sequence is already handled by AssocOp.__new__.\n        '
        from sympy.calculus.accumulationbounds import AccumBounds
        from sympy.matrices.expressions import MatrixExpr
        rv = None
        if len(seq) == 2:
            (a, b) = seq
            if b.is_Rational:
                (a, b) = (b, a)
                seq = [a, b]
            assert a is not S.One
            if a.is_Rational and (not a.is_zero):
                (r, b) = b.as_coeff_Mul()
                if b.is_Add:
                    if r is not S.One:
                        ar = a * r
                        if ar is S.One:
                            arb = b
                        else:
                            arb = cls(a * r, b, evaluate=False)
                        rv = ([arb], [], None)
                    elif global_parameters.distribute and b.is_commutative:
                        newb = Add(*[_keep_coeff(a, bi) for bi in b.args])
                        rv = ([newb], [], None)
            if rv:
                return rv
        c_part = []
        nc_part = []
        nc_seq = []
        coeff = S.One
        c_powers = []
        num_exp = []
        neg1e = S.Zero
        pnum_rat = {}
        order_symbols = None
        for o in seq:
            if o.is_Order:
                (o, order_symbols) = o.as_expr_variables(order_symbols)
            if o.is_Mul:
                if o.is_commutative:
                    seq.extend(o.args)
                else:
                    for q in o.args:
                        if q.is_commutative:
                            seq.append(q)
                        else:
                            nc_seq.append(q)
                    seq.append(NC_Marker)
                continue
            elif o.is_Number:
                if o is S.NaN or (coeff is S.ComplexInfinity and o.is_zero):
                    return ([S.NaN], [], None)
                elif coeff.is_Number or isinstance(coeff, AccumBounds):
                    coeff *= o
                    if coeff is S.NaN:
                        return ([S.NaN], [], None)
                continue
            elif isinstance(o, AccumBounds):
                coeff = o.__mul__(coeff)
                continue
            elif o is S.ComplexInfinity:
                if not coeff:
                    return ([S.NaN], [], None)
                coeff = S.ComplexInfinity
                continue
            elif o is S.ImaginaryUnit:
                neg1e += S.Half
                continue
            elif o.is_commutative:
                (b, e) = o.as_base_exp()
                if o.is_Pow:
                    if b.is_Number:
                        if e.is_Rational:
                            if e.is_Integer:
                                coeff *= Pow(b, e)
                                continue
                            elif e.is_negative:
                                seq.append(Pow(b, e))
                                continue
                            elif b.is_negative:
                                neg1e += e
                                b = -b
                            if b is not S.One:
                                pnum_rat.setdefault(b, []).append(e)
                            continue
                        elif b.is_positive or e.is_integer:
                            num_exp.append((b, e))
                            continue
                c_powers.append((b, e))
            else:
                if o is not NC_Marker:
                    nc_seq.append(o)
                while nc_seq:
                    o = nc_seq.pop(0)
                    if not nc_part:
                        nc_part.append(o)
                        continue
                    o1 = nc_part.pop()
                    (b1, e1) = o1.as_base_exp()
                    (b2, e2) = o.as_base_exp()
                    new_exp = e1 + e2
                    if b1 == b2 and (not new_exp.is_Add):
                        o12 = b1 ** new_exp
                        if o12.is_commutative:
                            seq.append(o12)
                            continue
                        else:
                            nc_seq.insert(0, o12)
                    else:
                        nc_part.extend([o1, o])

        def _gather(c_powers):
            if False:
                print('Hello World!')
            common_b = {}
            for (b, e) in c_powers:
                co = e.as_coeff_Mul()
                common_b.setdefault(b, {}).setdefault(co[1], []).append(co[0])
            for (b, d) in common_b.items():
                for (di, li) in d.items():
                    d[di] = Add(*li)
            new_c_powers = []
            for (b, e) in common_b.items():
                new_c_powers.extend([(b, c * t) for (t, c) in e.items()])
            return new_c_powers
        c_powers = _gather(c_powers)
        num_exp = _gather(num_exp)
        for i in range(2):
            new_c_powers = []
            changed = False
            for (b, e) in c_powers:
                if e.is_zero:
                    if (b.is_Add or b.is_Mul) and any((infty in b.args for infty in (S.ComplexInfinity, S.Infinity, S.NegativeInfinity))):
                        return ([S.NaN], [], None)
                    continue
                if e is S.One:
                    if b.is_Number:
                        coeff *= b
                        continue
                    p = b
                if e is not S.One:
                    p = Pow(b, e)
                    if p.is_Pow and (not b.is_Pow):
                        bi = b
                        (b, e) = p.as_base_exp()
                        if b != bi:
                            changed = True
                c_part.append(p)
                new_c_powers.append((b, e))
            if changed and len({b for (b, e) in new_c_powers}) != len(new_c_powers):
                c_part = []
                c_powers = _gather(new_c_powers)
            else:
                break
        inv_exp_dict = {}
        for (b, e) in num_exp:
            inv_exp_dict.setdefault(e, []).append(b)
        for (e, b) in inv_exp_dict.items():
            inv_exp_dict[e] = cls(*b)
        c_part.extend([Pow(b, e) for (e, b) in inv_exp_dict.items() if e])
        comb_e = {}
        for (b, e) in pnum_rat.items():
            comb_e.setdefault(Add(*e), []).append(b)
        del pnum_rat
        num_rat = []
        for (e, b) in comb_e.items():
            b = cls(*b)
            if e.q == 1:
                coeff *= Pow(b, e)
                continue
            if e.p > e.q:
                (e_i, ep) = divmod(e.p, e.q)
                coeff *= Pow(b, e_i)
                e = Rational(ep, e.q)
            num_rat.append((b, e))
        del comb_e
        pnew = defaultdict(list)
        i = 0
        while i < len(num_rat):
            (bi, ei) = num_rat[i]
            if bi == 1:
                i += 1
                continue
            grow = []
            for j in range(i + 1, len(num_rat)):
                (bj, ej) = num_rat[j]
                g = bi.gcd(bj)
                if g is not S.One:
                    e = ei + ej
                    if e.q == 1:
                        coeff *= Pow(g, e)
                    else:
                        if e.p > e.q:
                            (e_i, ep) = divmod(e.p, e.q)
                            coeff *= Pow(g, e_i)
                            e = Rational(ep, e.q)
                        grow.append((g, e))
                    num_rat[j] = (bj / g, ej)
                    bi = bi / g
                    if bi is S.One:
                        break
            if bi is not S.One:
                obj = Pow(bi, ei)
                if obj.is_Number:
                    coeff *= obj
                else:
                    for obj in Mul.make_args(obj):
                        if obj.is_Number:
                            coeff *= obj
                        else:
                            assert obj.is_Pow
                            (bi, ei) = obj.args
                            pnew[ei].append(bi)
            num_rat.extend(grow)
            i += 1
        for (e, b) in pnew.items():
            pnew[e] = cls(*b)
        if neg1e:
            (p, q) = neg1e.as_numer_denom()
            (n, p) = divmod(p, q)
            if n % 2:
                coeff = -coeff
            if q == 2:
                c_part.append(S.ImaginaryUnit)
            elif p:
                neg1e = Rational(p, q)
                for (e, b) in pnew.items():
                    if e == neg1e and b.is_positive:
                        pnew[e] = -b
                        break
                else:
                    c_part.append(Pow(S.NegativeOne, neg1e, evaluate=False))
        c_part.extend([Pow(b, e) for (e, b) in pnew.items()])
        if coeff in (S.Infinity, S.NegativeInfinity):

            def _handle_for_oo(c_part, coeff_sign):
                if False:
                    i = 10
                    return i + 15
                new_c_part = []
                for t in c_part:
                    if t.is_extended_positive:
                        continue
                    if t.is_extended_negative:
                        coeff_sign *= -1
                        continue
                    new_c_part.append(t)
                return (new_c_part, coeff_sign)
            (c_part, coeff_sign) = _handle_for_oo(c_part, 1)
            (nc_part, coeff_sign) = _handle_for_oo(nc_part, coeff_sign)
            coeff *= coeff_sign
        if coeff is S.ComplexInfinity:
            c_part = [c for c in c_part if not (fuzzy_not(c.is_zero) and c.is_extended_real is not None)]
            nc_part = [c for c in nc_part if not (fuzzy_not(c.is_zero) and c.is_extended_real is not None)]
        elif coeff.is_zero:
            if any((isinstance(c, MatrixExpr) for c in nc_part)):
                return ([coeff], nc_part, order_symbols)
            if any((c.is_finite == False for c in c_part)):
                return ([S.NaN], [], order_symbols)
            return ([coeff], [], order_symbols)
        _new = []
        for i in c_part:
            if i.is_Number:
                coeff *= i
            else:
                _new.append(i)
        c_part = _new
        _mulsort(c_part)
        if coeff is not S.One:
            c_part.insert(0, coeff)
        if global_parameters.distribute and (not nc_part) and (len(c_part) == 2) and c_part[0].is_Number and c_part[0].is_finite and c_part[1].is_Add:
            coeff = c_part[0]
            c_part = [Add(*[coeff * f for f in c_part[1].args])]
        return (c_part, nc_part, order_symbols)

    def _eval_power(self, e):
        if False:
            for i in range(10):
                print('nop')
        (cargs, nc) = self.args_cnc(split_1=False)
        if e.is_Integer:
            return Mul(*[Pow(b, e, evaluate=False) for b in cargs]) * Pow(Mul._from_args(nc), e, evaluate=False)
        if e.is_Rational and e.q == 2:
            if self.is_imaginary:
                a = self.as_real_imag()[1]
                if a.is_Rational:
                    (n, d) = abs(a / 2).as_numer_denom()
                    (n, t) = integer_nthroot(n, 2)
                    if t:
                        (d, t) = integer_nthroot(d, 2)
                        if t:
                            from sympy.functions.elementary.complexes import sign
                            r = sympify(n) / d
                            return _unevaluated_Mul(r ** e.p, (1 + sign(a) * S.ImaginaryUnit) ** e.p)
        p = Pow(self, e, evaluate=False)
        if e.is_Rational or e.is_Float:
            return p._eval_expand_power_base()
        return p

    @classmethod
    def class_key(cls):
        if False:
            while True:
                i = 10
        return (3, 0, cls.__name__)

    def _eval_evalf(self, prec):
        if False:
            return 10
        (c, m) = self.as_coeff_Mul()
        if c is S.NegativeOne:
            if m.is_Mul:
                rv = -AssocOp._eval_evalf(m, prec)
            else:
                mnew = m._eval_evalf(prec)
                if mnew is not None:
                    m = mnew
                rv = -m
        else:
            rv = AssocOp._eval_evalf(self, prec)
        if rv.is_number:
            return rv.expand()
        return rv

    @property
    def _mpc_(self):
        if False:
            print('Hello World!')
        '\n        Convert self to an mpmath mpc if possible\n        '
        from .numbers import Float
        (im_part, imag_unit) = self.as_coeff_Mul()
        if imag_unit is not S.ImaginaryUnit:
            raise AttributeError('Cannot convert Mul to mpc. Must be of the form Number*I')
        return (Float(0)._mpf_, Float(im_part)._mpf_)

    @cacheit
    def as_two_terms(self):
        if False:
            return 10
        'Return head and tail of self.\n\n        This is the most efficient way to get the head and tail of an\n        expression.\n\n        - if you want only the head, use self.args[0];\n        - if you want to process the arguments of the tail then use\n          self.as_coef_mul() which gives the head and a tuple containing\n          the arguments of the tail when treated as a Mul.\n        - if you want the coefficient when self is treated as an Add\n          then use self.as_coeff_add()[0]\n\n        Examples\n        ========\n\n        >>> from sympy.abc import x, y\n        >>> (3*x*y).as_two_terms()\n        (3, x*y)\n        '
        args = self.args
        if len(args) == 1:
            return (S.One, self)
        elif len(args) == 2:
            return args
        else:
            return (args[0], self._new_rawargs(*args[1:]))

    @cacheit
    def as_coeff_mul(self, *deps, rational=True, **kwargs):
        if False:
            print('Hello World!')
        if deps:
            (l1, l2) = sift(self.args, lambda x: x.has(*deps), binary=True)
            return (self._new_rawargs(*l2), tuple(l1))
        args = self.args
        if args[0].is_Number:
            if not rational or args[0].is_Rational:
                return (args[0], args[1:])
            elif args[0].is_extended_negative:
                return (S.NegativeOne, (-args[0],) + args[1:])
        return (S.One, args)

    def as_coeff_Mul(self, rational=False):
        if False:
            return 10
        '\n        Efficiently extract the coefficient of a product.\n        '
        (coeff, args) = (self.args[0], self.args[1:])
        if coeff.is_Number:
            if not rational or coeff.is_Rational:
                if len(args) == 1:
                    return (coeff, args[0])
                else:
                    return (coeff, self._new_rawargs(*args))
            elif coeff.is_extended_negative:
                return (S.NegativeOne, self._new_rawargs(*(-coeff,) + args))
        return (S.One, self)

    def as_real_imag(self, deep=True, **hints):
        if False:
            for i in range(10):
                print('nop')
        from sympy.functions.elementary.complexes import Abs, im, re
        other = []
        coeffr = []
        coeffi = []
        addterms = S.One
        for a in self.args:
            (r, i) = a.as_real_imag()
            if i.is_zero:
                coeffr.append(r)
            elif r.is_zero:
                coeffi.append(i * S.ImaginaryUnit)
            elif a.is_commutative:
                aconj = a.conjugate() if other else None
                for (i, x) in enumerate(other):
                    if x == aconj:
                        coeffr.append(Abs(x) ** 2)
                        del other[i]
                        break
                else:
                    if a.is_Add:
                        addterms *= a
                    else:
                        other.append(a)
            else:
                other.append(a)
        m = self.func(*other)
        if hints.get('ignore') == m:
            return
        if len(coeffi) % 2:
            imco = im(coeffi.pop(0))
        else:
            imco = S.Zero
        reco = self.func(*coeffr + coeffi)
        (r, i) = (reco * re(m), reco * im(m))
        if addterms == 1:
            if m == 1:
                if imco.is_zero:
                    return (reco, S.Zero)
                else:
                    return (S.Zero, reco * imco)
            if imco is S.Zero:
                return (r, i)
            return (-imco * i, imco * r)
        from .function import expand_mul
        (addre, addim) = expand_mul(addterms, deep=False).as_real_imag()
        if imco is S.Zero:
            return (r * addre - i * addim, i * addre + r * addim)
        else:
            (r, i) = (-imco * i, imco * r)
            return (r * addre - i * addim, r * addim + i * addre)

    @staticmethod
    def _expandsums(sums):
        if False:
            return 10
        '\n        Helper function for _eval_expand_mul.\n\n        sums must be a list of instances of Basic.\n        '
        L = len(sums)
        if L == 1:
            return sums[0].args
        terms = []
        left = Mul._expandsums(sums[:L // 2])
        right = Mul._expandsums(sums[L // 2:])
        terms = [Mul(a, b) for a in left for b in right]
        added = Add(*terms)
        return Add.make_args(added)

    def _eval_expand_mul(self, **hints):
        if False:
            print('Hello World!')
        from sympy.simplify.radsimp import fraction
        expr = self
        (n, d) = fraction(expr)
        if d.is_Mul:
            (n, d) = [i._eval_expand_mul(**hints) if i.is_Mul else i for i in (n, d)]
        expr = n / d
        if not expr.is_Mul:
            return expr
        (plain, sums, rewrite) = ([], [], False)
        for factor in expr.args:
            if factor.is_Add:
                sums.append(factor)
                rewrite = True
            elif factor.is_commutative:
                plain.append(factor)
            else:
                sums.append(Basic(factor))
        if not rewrite:
            return expr
        else:
            plain = self.func(*plain)
            if sums:
                deep = hints.get('deep', False)
                terms = self.func._expandsums(sums)
                args = []
                for term in terms:
                    t = self.func(plain, term)
                    if t.is_Mul and any((a.is_Add for a in t.args)) and deep:
                        t = t._eval_expand_mul()
                    args.append(t)
                return Add(*args)
            else:
                return plain

    @cacheit
    def _eval_derivative(self, s):
        if False:
            i = 10
            return i + 15
        args = list(self.args)
        terms = []
        for i in range(len(args)):
            d = args[i].diff(s)
            if d:
                terms.append(reduce(lambda x, y: x * y, args[:i] + [d] + args[i + 1:], S.One))
        return Add.fromiter(terms)

    @cacheit
    def _eval_derivative_n_times(self, s, n):
        if False:
            while True:
                i = 10
        from .function import AppliedUndef
        from .symbol import Symbol, symbols, Dummy
        if not isinstance(s, (AppliedUndef, Symbol)):
            return super()._eval_derivative_n_times(s, n)
        from .numbers import Integer
        args = self.args
        m = len(args)
        if isinstance(n, (int, Integer)):
            terms = []
            from sympy.ntheory.multinomial import multinomial_coefficients_iterator
            for (kvals, c) in multinomial_coefficients_iterator(m, n):
                p = Mul(*[arg.diff((s, k)) for (k, arg) in zip(kvals, args)])
                terms.append(c * p)
            return Add(*terms)
        from sympy.concrete.summations import Sum
        from sympy.functions.combinatorial.factorials import factorial
        from sympy.functions.elementary.miscellaneous import Max
        kvals = symbols('k1:%i' % m, cls=Dummy)
        klast = n - sum(kvals)
        nfact = factorial(n)
        (e, l) = (nfact / prod(map(factorial, kvals)) / factorial(klast) * Mul(*[args[t].diff((s, kvals[t])) for t in range(m - 1)]) * args[-1].diff((s, Max(0, klast))), [(k, 0, n) for k in kvals])
        return Sum(e, *l)

    def _eval_difference_delta(self, n, step):
        if False:
            print('Hello World!')
        from sympy.series.limitseq import difference_delta as dd
        arg0 = self.args[0]
        rest = Mul(*self.args[1:])
        return arg0.subs(n, n + step) * dd(rest, n, step) + dd(arg0, n, step) * rest

    def _matches_simple(self, expr, repl_dict):
        if False:
            print('Hello World!')
        (coeff, terms) = self.as_coeff_Mul()
        terms = Mul.make_args(terms)
        if len(terms) == 1:
            newexpr = self.__class__._combine_inverse(expr, coeff)
            return terms[0].matches(newexpr, repl_dict)
        return

    def matches(self, expr, repl_dict=None, old=False):
        if False:
            print('Hello World!')
        expr = sympify(expr)
        if self.is_commutative and expr.is_commutative:
            return self._matches_commutative(expr, repl_dict, old)
        elif self.is_commutative is not expr.is_commutative:
            return None
        (c1, nc1) = self.args_cnc()
        (c2, nc2) = expr.args_cnc()
        (c1, c2) = [c or [1] for c in [c1, c2]]
        comm_mul_self = Mul(*c1)
        comm_mul_expr = Mul(*c2)
        repl_dict = comm_mul_self.matches(comm_mul_expr, repl_dict, old)
        if not repl_dict and c1 != c2:
            return None
        nc1 = Mul._matches_expand_pows(nc1)
        nc2 = Mul._matches_expand_pows(nc2)
        repl_dict = Mul._matches_noncomm(nc1, nc2, repl_dict)
        return repl_dict or None

    @staticmethod
    def _matches_expand_pows(arg_list):
        if False:
            i = 10
            return i + 15
        new_args = []
        for arg in arg_list:
            if arg.is_Pow and arg.exp > 0:
                new_args.extend([arg.base] * arg.exp)
            else:
                new_args.append(arg)
        return new_args

    @staticmethod
    def _matches_noncomm(nodes, targets, repl_dict=None):
        if False:
            for i in range(10):
                print('nop')
        'Non-commutative multiplication matcher.\n\n        `nodes` is a list of symbols within the matcher multiplication\n        expression, while `targets` is a list of arguments in the\n        multiplication expression being matched against.\n        '
        if repl_dict is None:
            repl_dict = {}
        else:
            repl_dict = repl_dict.copy()
        agenda = []
        state = (0, 0)
        (node_ind, target_ind) = state
        wildcard_dict = {}
        while target_ind < len(targets) and node_ind < len(nodes):
            node = nodes[node_ind]
            if node.is_Wild:
                Mul._matches_add_wildcard(wildcard_dict, state)
            states_matches = Mul._matches_new_states(wildcard_dict, state, nodes, targets)
            if states_matches:
                (new_states, new_matches) = states_matches
                agenda.extend(new_states)
                if new_matches:
                    for match in new_matches:
                        repl_dict[match] = new_matches[match]
            if not agenda:
                return None
            else:
                state = agenda.pop()
                (node_ind, target_ind) = state
        return repl_dict

    @staticmethod
    def _matches_add_wildcard(dictionary, state):
        if False:
            for i in range(10):
                print('nop')
        (node_ind, target_ind) = state
        if node_ind in dictionary:
            (begin, end) = dictionary[node_ind]
            dictionary[node_ind] = (begin, target_ind)
        else:
            dictionary[node_ind] = (target_ind, target_ind)

    @staticmethod
    def _matches_new_states(dictionary, state, nodes, targets):
        if False:
            print('Hello World!')
        (node_ind, target_ind) = state
        node = nodes[node_ind]
        target = targets[target_ind]
        if target_ind >= len(targets) - 1 and node_ind < len(nodes) - 1:
            return None
        if node.is_Wild:
            match_attempt = Mul._matches_match_wilds(dictionary, node_ind, nodes, targets)
            if match_attempt:
                other_node_inds = Mul._matches_get_other_nodes(dictionary, nodes, node_ind)
                for ind in other_node_inds:
                    (other_begin, other_end) = dictionary[ind]
                    (curr_begin, curr_end) = dictionary[node_ind]
                    other_targets = targets[other_begin:other_end + 1]
                    current_targets = targets[curr_begin:curr_end + 1]
                    for (curr, other) in zip(current_targets, other_targets):
                        if curr != other:
                            return None
                new_state = [(node_ind, target_ind + 1)]
                if node_ind < len(nodes) - 1:
                    new_state.append((node_ind + 1, target_ind + 1))
                return (new_state, match_attempt)
        else:
            if node_ind >= len(nodes) - 1 and target_ind < len(targets) - 1:
                return None
            match_attempt = node.matches(target)
            if match_attempt:
                return ([(node_ind + 1, target_ind + 1)], match_attempt)
            elif node == target:
                return ([(node_ind + 1, target_ind + 1)], None)
            else:
                return None

    @staticmethod
    def _matches_match_wilds(dictionary, wildcard_ind, nodes, targets):
        if False:
            i = 10
            return i + 15
        'Determine matches of a wildcard with sub-expression in `target`.'
        wildcard = nodes[wildcard_ind]
        (begin, end) = dictionary[wildcard_ind]
        terms = targets[begin:end + 1]
        mult = Mul(*terms) if len(terms) > 1 else terms[0]
        return wildcard.matches(mult)

    @staticmethod
    def _matches_get_other_nodes(dictionary, nodes, node_ind):
        if False:
            i = 10
            return i + 15
        'Find other wildcards that may have already been matched.'
        ind_node = nodes[node_ind]
        return [ind for ind in dictionary if nodes[ind] == ind_node]

    @staticmethod
    def _combine_inverse(lhs, rhs):
        if False:
            while True:
                i = 10
        '\n        Returns lhs/rhs, but treats arguments like symbols, so things\n        like oo/oo return 1 (instead of a nan) and ``I`` behaves like\n        a symbol instead of sqrt(-1).\n        '
        from sympy.simplify.simplify import signsimp
        from .symbol import Dummy
        if lhs == rhs:
            return S.One

        def check(l, r):
            if False:
                return 10
            if l.is_Float and r.is_comparable:
                return l.__add__(0) == r.evalf().__add__(0)
            return False
        if check(lhs, rhs) or check(rhs, lhs):
            return S.One
        if any((i.is_Pow or i.is_Mul for i in (lhs, rhs))):
            d = Dummy('I')
            _i = {S.ImaginaryUnit: d}
            i_ = {d: S.ImaginaryUnit}
            a = lhs.xreplace(_i).as_powers_dict()
            b = rhs.xreplace(_i).as_powers_dict()
            blen = len(b)
            for bi in tuple(b.keys()):
                if bi in a:
                    a[bi] -= b.pop(bi)
                    if not a[bi]:
                        a.pop(bi)
            if len(b) != blen:
                lhs = Mul(*[k ** v for (k, v) in a.items()]).xreplace(i_)
                rhs = Mul(*[k ** v for (k, v) in b.items()]).xreplace(i_)
        rv = lhs / rhs
        srv = signsimp(rv)
        return srv if srv.is_Number else rv

    def as_powers_dict(self):
        if False:
            for i in range(10):
                print('nop')
        d = defaultdict(int)
        for term in self.args:
            for (b, e) in term.as_powers_dict().items():
                d[b] += e
        return d

    def as_numer_denom(self):
        if False:
            for i in range(10):
                print('nop')
        (numers, denoms) = list(zip(*[f.as_numer_denom() for f in self.args]))
        return (self.func(*numers), self.func(*denoms))

    def as_base_exp(self):
        if False:
            for i in range(10):
                print('nop')
        e1 = None
        bases = []
        nc = 0
        for m in self.args:
            (b, e) = m.as_base_exp()
            if not b.is_commutative:
                nc += 1
            if e1 is None:
                e1 = e
            elif e != e1 or nc > 1:
                return (self, S.One)
            bases.append(b)
        return (self.func(*bases), e1)

    def _eval_is_polynomial(self, syms):
        if False:
            i = 10
            return i + 15
        return all((term._eval_is_polynomial(syms) for term in self.args))

    def _eval_is_rational_function(self, syms):
        if False:
            i = 10
            return i + 15
        return all((term._eval_is_rational_function(syms) for term in self.args))

    def _eval_is_meromorphic(self, x, a):
        if False:
            i = 10
            return i + 15
        return _fuzzy_group((arg.is_meromorphic(x, a) for arg in self.args), quick_exit=True)

    def _eval_is_algebraic_expr(self, syms):
        if False:
            i = 10
            return i + 15
        return all((term._eval_is_algebraic_expr(syms) for term in self.args))
    _eval_is_commutative = lambda self: _fuzzy_group((a.is_commutative for a in self.args))

    def _eval_is_complex(self):
        if False:
            i = 10
            return i + 15
        comp = _fuzzy_group((a.is_complex for a in self.args))
        if comp is False:
            if any((a.is_infinite for a in self.args)):
                if any((a.is_zero is not False for a in self.args)):
                    return None
                return False
        return comp

    def _eval_is_zero_infinite_helper(self):
        if False:
            print('Hello World!')
        seen_zero = seen_infinite = False
        for a in self.args:
            if a.is_zero:
                if seen_infinite is not False:
                    return (None, None)
                seen_zero = True
            elif a.is_infinite:
                if seen_zero is not False:
                    return (None, None)
                seen_infinite = True
            else:
                if seen_zero is False and a.is_zero is None:
                    if seen_infinite is not False:
                        return (None, None)
                    seen_zero = None
                if seen_infinite is False and a.is_infinite is None:
                    if seen_zero is not False:
                        return (None, None)
                    seen_infinite = None
        return (seen_zero, seen_infinite)

    def _eval_is_zero(self):
        if False:
            return 10
        (seen_zero, seen_infinite) = self._eval_is_zero_infinite_helper()
        if seen_zero is False:
            return False
        elif seen_zero is True and seen_infinite is False:
            return True
        else:
            return None

    def _eval_is_infinite(self):
        if False:
            print('Hello World!')
        (seen_zero, seen_infinite) = self._eval_is_zero_infinite_helper()
        if seen_infinite is True and seen_zero is False:
            return True
        elif seen_infinite is False:
            return False
        else:
            return None

    def _eval_is_rational(self):
        if False:
            while True:
                i = 10
        r = _fuzzy_group((a.is_rational for a in self.args), quick_exit=True)
        if r:
            return r
        elif r is False:
            if all((a.is_zero is False for a in self.args)):
                return False

    def _eval_is_algebraic(self):
        if False:
            for i in range(10):
                print('nop')
        r = _fuzzy_group((a.is_algebraic for a in self.args), quick_exit=True)
        if r:
            return r
        elif r is False:
            if all((a.is_zero is False for a in self.args)):
                return False

    def _eval_is_integer(self):
        if False:
            return 10
        is_rational = self._eval_is_rational()
        if is_rational is False:
            return False
        numerators = []
        denominators = []
        unknown = False
        for a in self.args:
            hit = False
            if a.is_integer:
                if abs(a) is not S.One:
                    numerators.append(a)
            elif a.is_Rational:
                (n, d) = a.as_numer_denom()
                if abs(n) is not S.One:
                    numerators.append(n)
                if d is not S.One:
                    denominators.append(d)
            elif a.is_Pow:
                (b, e) = a.as_base_exp()
                if not b.is_integer or not e.is_integer:
                    hit = unknown = True
                if e.is_negative:
                    denominators.append(2 if a is S.Half else Pow(a, S.NegativeOne))
                elif not hit:
                    assert not e.is_positive
                    assert not e.is_zero
                    return
                else:
                    return
            else:
                return
        if not denominators and (not unknown):
            return True
        allodd = lambda x: all((i.is_odd for i in x))
        alleven = lambda x: all((i.is_even for i in x))
        anyeven = lambda x: any((i.is_even for i in x))
        from .relational import is_gt
        if not numerators and denominators and all((is_gt(_, S.One) for _ in denominators)):
            return False
        elif unknown:
            return
        elif allodd(numerators) and anyeven(denominators):
            return False
        elif anyeven(numerators) and denominators == [2]:
            return True
        elif alleven(numerators) and allodd(denominators) and (Mul(*denominators, evaluate=False) - 1).is_positive:
            return False
        if len(denominators) == 1:
            d = denominators[0]
            if d.is_Integer and d.is_even:
                if (Add(*[i.as_base_exp()[1] for i in numerators if i.is_even]) - trailing(d.p)).is_nonnegative:
                    return True
        if len(numerators) == 1:
            n = numerators[0]
            if n.is_Integer and n.is_even:
                if (Add(*[i.as_base_exp()[1] for i in denominators if i.is_even]) - trailing(n.p)).is_positive:
                    return False

    def _eval_is_polar(self):
        if False:
            for i in range(10):
                print('nop')
        has_polar = any((arg.is_polar for arg in self.args))
        return has_polar and all((arg.is_polar or arg.is_positive for arg in self.args))

    def _eval_is_extended_real(self):
        if False:
            i = 10
            return i + 15
        return self._eval_real_imag(True)

    def _eval_real_imag(self, real):
        if False:
            i = 10
            return i + 15
        zero = False
        t_not_re_im = None
        for t in self.args:
            if (t.is_complex or t.is_infinite) is False and t.is_extended_real is False:
                return False
            elif t.is_imaginary:
                real = not real
            elif t.is_extended_real:
                if not zero:
                    z = t.is_zero
                    if not z and zero is False:
                        zero = z
                    elif z:
                        if all((a.is_finite for a in self.args)):
                            return True
                        return
            elif t.is_extended_real is False:
                if t_not_re_im:
                    return
                t_not_re_im = t
            elif t.is_imaginary is False:
                if t_not_re_im:
                    return
                t_not_re_im = t
            else:
                return
        if t_not_re_im:
            if t_not_re_im.is_extended_real is False:
                if real:
                    return zero
            if t_not_re_im.is_imaginary is False:
                if not real:
                    return zero
        elif zero is False:
            return real
        elif real:
            return real

    def _eval_is_imaginary(self):
        if False:
            for i in range(10):
                print('nop')
        if all((a.is_zero is False and a.is_finite for a in self.args)):
            return self._eval_real_imag(False)

    def _eval_is_hermitian(self):
        if False:
            return 10
        return self._eval_herm_antiherm(True)

    def _eval_is_antihermitian(self):
        if False:
            for i in range(10):
                print('nop')
        return self._eval_herm_antiherm(False)

    def _eval_herm_antiherm(self, herm):
        if False:
            i = 10
            return i + 15
        for t in self.args:
            if t.is_hermitian is None or t.is_antihermitian is None:
                return
            if t.is_hermitian:
                continue
            elif t.is_antihermitian:
                herm = not herm
            else:
                return
        if herm is not False:
            return herm
        is_zero = self._eval_is_zero()
        if is_zero:
            return True
        elif is_zero is False:
            return herm

    def _eval_is_irrational(self):
        if False:
            print('Hello World!')
        for t in self.args:
            a = t.is_irrational
            if a:
                others = list(self.args)
                others.remove(t)
                if all(((x.is_rational and fuzzy_not(x.is_zero)) is True for x in others)):
                    return True
                return
            if a is None:
                return
        if all((x.is_real for x in self.args)):
            return False

    def _eval_is_extended_positive(self):
        if False:
            i = 10
            return i + 15
        'Return True if self is positive, False if not, and None if it\n        cannot be determined.\n\n        Explanation\n        ===========\n\n        This algorithm is non-recursive and works by keeping track of the\n        sign which changes when a negative or nonpositive is encountered.\n        Whether a nonpositive or nonnegative is seen is also tracked since\n        the presence of these makes it impossible to return True, but\n        possible to return False if the end result is nonpositive. e.g.\n\n            pos * neg * nonpositive -> pos or zero -> None is returned\n            pos * neg * nonnegative -> neg or zero -> False is returned\n        '
        return self._eval_pos_neg(1)

    def _eval_pos_neg(self, sign):
        if False:
            return 10
        saw_NON = saw_NOT = False
        for t in self.args:
            if t.is_extended_positive:
                continue
            elif t.is_extended_negative:
                sign = -sign
            elif t.is_zero:
                if all((a.is_finite for a in self.args)):
                    return False
                return
            elif t.is_extended_nonpositive:
                sign = -sign
                saw_NON = True
            elif t.is_extended_nonnegative:
                saw_NON = True
            elif t.is_positive is False:
                sign = -sign
                if saw_NOT:
                    return
                saw_NOT = True
            elif t.is_negative is False:
                if saw_NOT:
                    return
                saw_NOT = True
            else:
                return
        if sign == 1 and saw_NON is False and (saw_NOT is False):
            return True
        if sign < 0:
            return False

    def _eval_is_extended_negative(self):
        if False:
            print('Hello World!')
        return self._eval_pos_neg(-1)

    def _eval_is_odd(self):
        if False:
            return 10
        is_integer = self._eval_is_integer()
        if is_integer is not True:
            return is_integer
        from sympy.simplify.radsimp import fraction
        (n, d) = fraction(self)
        if d.is_Integer and d.is_even:
            if (Add(*[i.as_base_exp()[1] for i in Mul.make_args(n) if i.is_even]) - trailing(d.p)).is_positive:
                return False
            return
        (r, acc) = (True, 1)
        for t in self.args:
            if abs(t) is S.One:
                continue
            if t.is_even:
                return False
            if r is False:
                pass
            elif acc != 1 and (acc + t).is_odd:
                r = False
            elif t.is_even is None:
                r = None
            acc = t
        return r

    def _eval_is_even(self):
        if False:
            i = 10
            return i + 15
        from sympy.simplify.radsimp import fraction
        (n, d) = fraction(self)
        if n.is_Integer and n.is_even:
            if (Add(*[i.as_base_exp()[1] for i in Mul.make_args(d) if i.is_even]) - trailing(n.p)).is_nonnegative:
                return False

    def _eval_is_composite(self):
        if False:
            while True:
                i = 10
        '\n        Here we count the number of arguments that have a minimum value\n        greater than two.\n        If there are more than one of such a symbol then the result is composite.\n        Else, the result cannot be determined.\n        '
        number_of_args = 0
        for arg in self.args:
            if not (arg.is_integer and arg.is_positive):
                return None
            if (arg - 1).is_positive:
                number_of_args += 1
        if number_of_args > 1:
            return True

    def _eval_subs(self, old, new):
        if False:
            return 10
        from sympy.functions.elementary.complexes import sign
        from sympy.ntheory.factor_ import multiplicity
        from sympy.simplify.powsimp import powdenest
        from sympy.simplify.radsimp import fraction
        if not old.is_Mul:
            return None
        if old.args[0].is_Number and old.args[0] < 0:
            if self.args[0].is_Number:
                if self.args[0] < 0:
                    return self._subs(-old, -new)
                return None

        def base_exp(a):
            if False:
                return 10
            from sympy.functions.elementary.exponential import exp
            if a.is_Pow or isinstance(a, exp):
                return a.as_base_exp()
            return (a, S.One)

        def breakup(eq):
            if False:
                for i in range(10):
                    print('nop')
            'break up powers of eq when treated as a Mul:\n                   b**(Rational*e) -> b**e, Rational\n                commutatives come back as a dictionary {b**e: Rational}\n                noncommutatives come back as a list [(b**e, Rational)]\n            '
            (c, nc) = (defaultdict(int), [])
            for a in Mul.make_args(eq):
                a = powdenest(a)
                (b, e) = base_exp(a)
                if e is not S.One:
                    (co, _) = e.as_coeff_mul()
                    b = Pow(b, e / co)
                    e = co
                if a.is_commutative:
                    c[b] += e
                else:
                    nc.append([b, e])
            return (c, nc)

        def rejoin(b, co):
            if False:
                return 10
            "\n            Put rational back with exponent; in general this is not ok, but\n            since we took it from the exponent for analysis, it's ok to put\n            it back.\n            "
            (b, e) = base_exp(b)
            return Pow(b, e * co)

        def ndiv(a, b):
            if False:
                return 10
            'if b divides a in an extractive way (like 1/4 divides 1/2\n            but not vice versa, and 2/5 does not divide 1/3) then return\n            the integer number of times it divides, else return 0.\n            '
            if not b.q % a.q or not a.q % b.q:
                return int(a / b)
            return 0
        rv = None
        (n, d) = fraction(self)
        self2 = self
        if d is not S.One:
            self2 = n._subs(old, new) / d._subs(old, new)
            if not self2.is_Mul:
                return self2._subs(old, new)
            if self2 != self:
                rv = self2
        co_self = self2.args[0]
        co_old = old.args[0]
        co_xmul = None
        if co_old.is_Rational and co_self.is_Rational:
            if co_old != co_self:
                co_xmul = co_self.extract_multiplicatively(co_old)
        elif co_old.is_Rational:
            return rv
        (c, nc) = breakup(self2)
        (old_c, old_nc) = breakup(old)
        if co_xmul and co_xmul.is_Rational and (abs(co_old) != 1):
            mult = S(multiplicity(abs(co_old), co_self))
            c.pop(co_self)
            if co_old in c:
                c[co_old] += mult
            else:
                c[co_old] = mult
            co_residual = co_self / co_old ** mult
        else:
            co_residual = 1
        ok = True
        if len(old_nc) > len(nc):
            ok = False
        elif len(old_c) > len(c):
            ok = False
        elif {i[0] for i in old_nc}.difference({i[0] for i in nc}):
            ok = False
        elif set(old_c).difference(set(c)):
            ok = False
        elif any((sign(c[b]) != sign(old_c[b]) for b in old_c)):
            ok = False
        if not ok:
            return rv
        if not old_c:
            cdid = None
        else:
            rat = []
            for (b, old_e) in old_c.items():
                c_e = c[b]
                rat.append(ndiv(c_e, old_e))
                if not rat[-1]:
                    return rv
            cdid = min(rat)
        if not old_nc:
            ncdid = None
            for i in range(len(nc)):
                nc[i] = rejoin(*nc[i])
        else:
            ncdid = 0
            take = len(old_nc)
            limit = cdid or S.Infinity
            failed = []
            i = 0
            while limit and i + take <= len(nc):
                hit = False
                rat = []
                for j in range(take):
                    if nc[i + j][0] != old_nc[j][0]:
                        break
                    elif j == 0:
                        rat.append(ndiv(nc[i + j][1], old_nc[j][1]))
                    elif j == take - 1:
                        rat.append(ndiv(nc[i + j][1], old_nc[j][1]))
                    elif nc[i + j][1] != old_nc[j][1]:
                        break
                    else:
                        rat.append(1)
                    j += 1
                else:
                    ndo = min(rat)
                    if ndo:
                        if take == 1:
                            if cdid:
                                ndo = min(cdid, ndo)
                            nc[i] = Pow(new, ndo) * rejoin(nc[i][0], nc[i][1] - ndo * old_nc[0][1])
                        else:
                            ndo = 1
                            l = rejoin(nc[i][0], nc[i][1] - ndo * old_nc[0][1])
                            mid = new
                            ir = i + take - 1
                            r = (nc[ir][0], nc[ir][1] - ndo * old_nc[-1][1])
                            if r[1]:
                                if i + take < len(nc):
                                    nc[i:i + take] = [l * mid, r]
                                else:
                                    r = rejoin(*r)
                                    nc[i:i + take] = [l * mid * r]
                            else:
                                nc[i:i + take] = [l * mid]
                        limit -= ndo
                        ncdid += ndo
                        hit = True
                if not hit:
                    failed.append(i)
                i += 1
            else:
                if not ncdid:
                    return rv
                failed.extend(range(i, len(nc)))
                for i in failed:
                    nc[i] = rejoin(*nc[i]).subs(old, new)
        if cdid is None:
            do = ncdid
        elif ncdid is None:
            do = cdid
        else:
            do = min(ncdid, cdid)
        margs = []
        for b in c:
            if b in old_c:
                e = c[b] - old_c[b] * do
                margs.append(rejoin(b, e))
            else:
                margs.append(rejoin(b.subs(old, new), c[b]))
        if cdid and (not ncdid):
            margs = [Pow(new, cdid)] + margs
        return co_residual * self2.func(*margs) * self2.func(*nc)

    def _eval_nseries(self, x, n, logx, cdir=0):
        if False:
            return 10
        from .function import PoleError
        from sympy.functions.elementary.integers import ceiling
        from sympy.series.order import Order

        def coeff_exp(term, x):
            if False:
                return 10
            lt = term.as_coeff_exponent(x)
            if lt[0].has(x):
                try:
                    lt = term.leadterm(x)
                except ValueError:
                    return (term, S.Zero)
            return lt
        ords = []
        try:
            for t in self.args:
                (coeff, exp) = t.leadterm(x)
                if not coeff.has(x):
                    ords.append((t, exp))
                else:
                    raise ValueError
            n0 = sum((t[1] for t in ords if t[1].is_number))
            facs = []
            for (t, m) in ords:
                n1 = ceiling(n - n0 + (m if m.is_number else 0))
                s = t.nseries(x, n=n1, logx=logx, cdir=cdir)
                ns = s.getn()
                if ns is not None:
                    if ns < n1:
                        n -= n1 - ns
                facs.append(s)
        except (ValueError, NotImplementedError, TypeError, PoleError):
            n0 = sympify(sum((t[1] for t in ords if t[1].is_number)))
            if n0.is_nonnegative:
                n0 = S.Zero
            facs = [t.nseries(x, n=ceiling(n - n0), logx=logx, cdir=cdir) for t in self.args]
            from sympy.simplify.powsimp import powsimp
            res = powsimp(self.func(*facs).expand(), combine='exp', deep=True)
            if res.has(Order):
                res += Order(x ** n, x)
            return res
        res = S.Zero
        ords2 = [Add.make_args(factor) for factor in facs]
        for fac in product(*ords2):
            ords3 = [coeff_exp(term, x) for term in fac]
            (coeffs, powers) = zip(*ords3)
            power = sum(powers)
            if (power - n).is_negative:
                res += Mul(*coeffs) * x ** power

        def max_degree(e, x):
            if False:
                i = 10
                return i + 15
            if e is x:
                return S.One
            if e.is_Atom:
                return S.Zero
            if e.is_Add:
                return max((max_degree(a, x) for a in e.args))
            if e.is_Mul:
                return Add(*[max_degree(a, x) for a in e.args])
            if e.is_Pow:
                return max_degree(e.base, x) * e.exp
            return S.Zero
        if self.is_polynomial(x):
            from sympy.polys.polyerrors import PolynomialError
            from sympy.polys.polytools import degree
            try:
                if max_degree(self, x) >= n or degree(self, x) != degree(res, x):
                    res += Order(x ** n, x)
            except PolynomialError:
                pass
            else:
                return res
        if res != self:
            if (self - res).subs(x, 0) == S.Zero and n > 0:
                lt = self._eval_as_leading_term(x, logx=logx, cdir=cdir)
                if lt == S.Zero:
                    return res
            res += Order(x ** n, x)
        return res

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        if False:
            for i in range(10):
                print('nop')
        return self.func(*[t.as_leading_term(x, logx=logx, cdir=cdir) for t in self.args])

    def _eval_conjugate(self):
        if False:
            i = 10
            return i + 15
        return self.func(*[t.conjugate() for t in self.args])

    def _eval_transpose(self):
        if False:
            return 10
        return self.func(*[t.transpose() for t in self.args[::-1]])

    def _eval_adjoint(self):
        if False:
            print('Hello World!')
        return self.func(*[t.adjoint() for t in self.args[::-1]])

    def as_content_primitive(self, radical=False, clear=True):
        if False:
            for i in range(10):
                print('nop')
        'Return the tuple (R, self/R) where R is the positive Rational\n        extracted from self.\n\n        Examples\n        ========\n\n        >>> from sympy import sqrt\n        >>> (-3*sqrt(2)*(2 - 2*sqrt(2))).as_content_primitive()\n        (6, -sqrt(2)*(1 - sqrt(2)))\n\n        See docstring of Expr.as_content_primitive for more examples.\n        '
        coef = S.One
        args = []
        for a in self.args:
            (c, p) = a.as_content_primitive(radical=radical, clear=clear)
            coef *= c
            if p is not S.One:
                args.append(p)
        return (coef, self.func(*args))

    def as_ordered_factors(self, order=None):
        if False:
            while True:
                i = 10
        'Transform an expression into an ordered list of factors.\n\n        Examples\n        ========\n\n        >>> from sympy import sin, cos\n        >>> from sympy.abc import x, y\n\n        >>> (2*x*y*sin(x)*cos(x)).as_ordered_factors()\n        [2, x, y, sin(x), cos(x)]\n\n        '
        (cpart, ncpart) = self.args_cnc()
        cpart.sort(key=lambda expr: expr.sort_key(order=order))
        return cpart + ncpart

    @property
    def _sorted_args(self):
        if False:
            i = 10
            return i + 15
        return tuple(self.as_ordered_factors())
mul = AssocOpDispatcher('mul')

def prod(a, start=1):
    if False:
        for i in range(10):
            print('nop')
    'Return product of elements of a. Start with int 1 so if only\n       ints are included then an int result is returned.\n\n    Examples\n    ========\n\n    >>> from sympy import prod, S\n    >>> prod(range(3))\n    0\n    >>> type(_) is int\n    True\n    >>> prod([S(2), 3])\n    6\n    >>> _.is_Integer\n    True\n\n    You can start the product at something other than 1:\n\n    >>> prod([1, 2], 3)\n    6\n\n    '
    return reduce(operator.mul, a, start)

def _keep_coeff(coeff, factors, clear=True, sign=False):
    if False:
        i = 10
        return i + 15
    'Return ``coeff*factors`` unevaluated if necessary.\n\n    If ``clear`` is False, do not keep the coefficient as a factor\n    if it can be distributed on a single factor such that one or\n    more terms will still have integer coefficients.\n\n    If ``sign`` is True, allow a coefficient of -1 to remain factored out.\n\n    Examples\n    ========\n\n    >>> from sympy.core.mul import _keep_coeff\n    >>> from sympy.abc import x, y\n    >>> from sympy import S\n\n    >>> _keep_coeff(S.Half, x + 2)\n    (x + 2)/2\n    >>> _keep_coeff(S.Half, x + 2, clear=False)\n    x/2 + 1\n    >>> _keep_coeff(S.Half, (x + 2)*y, clear=False)\n    y*(x + 2)/2\n    >>> _keep_coeff(S(-1), x + y)\n    -x - y\n    >>> _keep_coeff(S(-1), x + y, sign=True)\n    -(x + y)\n    '
    if not coeff.is_Number:
        if factors.is_Number:
            (factors, coeff) = (coeff, factors)
        else:
            return coeff * factors
    if factors is S.One:
        return coeff
    if coeff is S.One:
        return factors
    elif coeff is S.NegativeOne and (not sign):
        return -factors
    elif factors.is_Add:
        if not clear and coeff.is_Rational and (coeff.q != 1):
            args = [i.as_coeff_Mul() for i in factors.args]
            args = [(_keep_coeff(c, coeff), m) for (c, m) in args]
            if any((c.is_Integer for (c, _) in args)):
                return Add._from_args([Mul._from_args(i[1:] if i[0] == 1 else i) for i in args])
        return Mul(coeff, factors, evaluate=False)
    elif factors.is_Mul:
        margs = list(factors.args)
        if margs[0].is_Number:
            margs[0] *= coeff
            if margs[0] == 1:
                margs.pop(0)
        else:
            margs.insert(0, coeff)
        return Mul._from_args(margs)
    else:
        m = coeff * factors
        if m.is_Number and (not factors.is_Number):
            m = Mul._from_args((coeff, factors))
        return m

def expand_2arg(e):
    if False:
        print('Hello World!')

    def do(e):
        if False:
            while True:
                i = 10
        if e.is_Mul:
            (c, r) = e.as_coeff_Mul()
            if c.is_Number and r.is_Add:
                return _unevaluated_Add(*[c * ri for ri in r.args])
        return e
    return bottom_up(e, do)
from .numbers import Rational
from .power import Pow
from .add import Add, _unevaluated_Add