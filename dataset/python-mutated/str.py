"""
A Printer for generating readable representation of most SymPy classes.
"""
from __future__ import annotations
from typing import Any
from sympy.core import S, Rational, Pow, Basic, Mul, Number
from sympy.core.mul import _keep_coeff
from sympy.core.numbers import Integer
from sympy.core.relational import Relational
from sympy.core.sorting import default_sort_key
from sympy.utilities.iterables import sift
from .precedence import precedence, PRECEDENCE
from .printer import Printer, print_function
from mpmath.libmp import prec_to_dps, to_str as mlib_to_str

class StrPrinter(Printer):
    printmethod = '_sympystr'
    _default_settings: dict[str, Any] = {'order': None, 'full_prec': 'auto', 'sympy_integers': False, 'abbrev': False, 'perm_cyclic': True, 'min': None, 'max': None}
    _relationals: dict[str, str] = {}

    def parenthesize(self, item, level, strict=False):
        if False:
            i = 10
            return i + 15
        if precedence(item) < level or (not strict and precedence(item) <= level):
            return '(%s)' % self._print(item)
        else:
            return self._print(item)

    def stringify(self, args, sep, level=0):
        if False:
            i = 10
            return i + 15
        return sep.join([self.parenthesize(item, level) for item in args])

    def emptyPrinter(self, expr):
        if False:
            while True:
                i = 10
        if isinstance(expr, str):
            return expr
        elif isinstance(expr, Basic):
            return repr(expr)
        else:
            return str(expr)

    def _print_Add(self, expr, order=None):
        if False:
            i = 10
            return i + 15
        terms = self._as_ordered_terms(expr, order=order)
        prec = precedence(expr)
        l = []
        for term in terms:
            t = self._print(term)
            if t.startswith('-') and (not term.is_Add):
                sign = '-'
                t = t[1:]
            else:
                sign = '+'
            if precedence(term) < prec or term.is_Add:
                l.extend([sign, '(%s)' % t])
            else:
                l.extend([sign, t])
        sign = l.pop(0)
        if sign == '+':
            sign = ''
        return sign + ' '.join(l)

    def _print_BooleanTrue(self, expr):
        if False:
            i = 10
            return i + 15
        return 'True'

    def _print_BooleanFalse(self, expr):
        if False:
            while True:
                i = 10
        return 'False'

    def _print_Not(self, expr):
        if False:
            for i in range(10):
                print('nop')
        return '~%s' % self.parenthesize(expr.args[0], PRECEDENCE['Not'])

    def _print_And(self, expr):
        if False:
            while True:
                i = 10
        args = list(expr.args)
        for (j, i) in enumerate(args):
            if isinstance(i, Relational) and i.canonical.rhs is S.NegativeInfinity:
                args.insert(0, args.pop(j))
        return self.stringify(args, ' & ', PRECEDENCE['BitwiseAnd'])

    def _print_Or(self, expr):
        if False:
            while True:
                i = 10
        return self.stringify(expr.args, ' | ', PRECEDENCE['BitwiseOr'])

    def _print_Xor(self, expr):
        if False:
            return 10
        return self.stringify(expr.args, ' ^ ', PRECEDENCE['BitwiseXor'])

    def _print_AppliedPredicate(self, expr):
        if False:
            return 10
        return '%s(%s)' % (self._print(expr.function), self.stringify(expr.arguments, ', '))

    def _print_Basic(self, expr):
        if False:
            for i in range(10):
                print('nop')
        l = [self._print(o) for o in expr.args]
        return expr.__class__.__name__ + '(%s)' % ', '.join(l)

    def _print_BlockMatrix(self, B):
        if False:
            return 10
        if B.blocks.shape == (1, 1):
            self._print(B.blocks[0, 0])
        return self._print(B.blocks)

    def _print_Catalan(self, expr):
        if False:
            return 10
        return 'Catalan'

    def _print_ComplexInfinity(self, expr):
        if False:
            while True:
                i = 10
        return 'zoo'

    def _print_ConditionSet(self, s):
        if False:
            for i in range(10):
                print('nop')
        args = tuple([self._print(i) for i in (s.sym, s.condition)])
        if s.base_set is S.UniversalSet:
            return 'ConditionSet(%s, %s)' % args
        args += (self._print(s.base_set),)
        return 'ConditionSet(%s, %s, %s)' % args

    def _print_Derivative(self, expr):
        if False:
            while True:
                i = 10
        dexpr = expr.expr
        dvars = [i[0] if i[1] == 1 else i for i in expr.variable_count]
        return 'Derivative(%s)' % ', '.join((self._print(arg) for arg in [dexpr] + dvars))

    def _print_dict(self, d):
        if False:
            while True:
                i = 10
        keys = sorted(d.keys(), key=default_sort_key)
        items = []
        for key in keys:
            item = '%s: %s' % (self._print(key), self._print(d[key]))
            items.append(item)
        return '{%s}' % ', '.join(items)

    def _print_Dict(self, expr):
        if False:
            print('Hello World!')
        return self._print_dict(expr)

    def _print_RandomDomain(self, d):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(d, 'as_boolean'):
            return 'Domain: ' + self._print(d.as_boolean())
        elif hasattr(d, 'set'):
            return 'Domain: ' + self._print(d.symbols) + ' in ' + self._print(d.set)
        else:
            return 'Domain on ' + self._print(d.symbols)

    def _print_Dummy(self, expr):
        if False:
            while True:
                i = 10
        return '_' + expr.name

    def _print_EulerGamma(self, expr):
        if False:
            print('Hello World!')
        return 'EulerGamma'

    def _print_Exp1(self, expr):
        if False:
            print('Hello World!')
        return 'E'

    def _print_ExprCondPair(self, expr):
        if False:
            for i in range(10):
                print('nop')
        return '(%s, %s)' % (self._print(expr.expr), self._print(expr.cond))

    def _print_Function(self, expr):
        if False:
            print('Hello World!')
        return expr.func.__name__ + '(%s)' % self.stringify(expr.args, ', ')

    def _print_GoldenRatio(self, expr):
        if False:
            i = 10
            return i + 15
        return 'GoldenRatio'

    def _print_Heaviside(self, expr):
        if False:
            print('Hello World!')
        return expr.func.__name__ + '(%s)' % self.stringify(expr.pargs, ', ')

    def _print_TribonacciConstant(self, expr):
        if False:
            return 10
        return 'TribonacciConstant'

    def _print_ImaginaryUnit(self, expr):
        if False:
            while True:
                i = 10
        return 'I'

    def _print_Infinity(self, expr):
        if False:
            return 10
        return 'oo'

    def _print_Integral(self, expr):
        if False:
            print('Hello World!')

        def _xab_tostr(xab):
            if False:
                for i in range(10):
                    print('nop')
            if len(xab) == 1:
                return self._print(xab[0])
            else:
                return self._print((xab[0],) + tuple(xab[1:]))
        L = ', '.join([_xab_tostr(l) for l in expr.limits])
        return 'Integral(%s, %s)' % (self._print(expr.function), L)

    def _print_Interval(self, i):
        if False:
            for i in range(10):
                print('nop')
        fin = 'Interval{m}({a}, {b})'
        (a, b, l, r) = i.args
        if a.is_infinite and b.is_infinite:
            m = ''
        elif a.is_infinite and (not r):
            m = ''
        elif b.is_infinite and (not l):
            m = ''
        elif not l and (not r):
            m = ''
        elif l and r:
            m = '.open'
        elif l:
            m = '.Lopen'
        else:
            m = '.Ropen'
        return fin.format(**{'a': a, 'b': b, 'm': m})

    def _print_AccumulationBounds(self, i):
        if False:
            while True:
                i = 10
        return 'AccumBounds(%s, %s)' % (self._print(i.min), self._print(i.max))

    def _print_Inverse(self, I):
        if False:
            print('Hello World!')
        return '%s**(-1)' % self.parenthesize(I.arg, PRECEDENCE['Pow'])

    def _print_Lambda(self, obj):
        if False:
            i = 10
            return i + 15
        expr = obj.expr
        sig = obj.signature
        if len(sig) == 1 and sig[0].is_symbol:
            sig = sig[0]
        return 'Lambda(%s, %s)' % (self._print(sig), self._print(expr))

    def _print_LatticeOp(self, expr):
        if False:
            i = 10
            return i + 15
        args = sorted(expr.args, key=default_sort_key)
        return expr.func.__name__ + '(%s)' % ', '.join((self._print(arg) for arg in args))

    def _print_Limit(self, expr):
        if False:
            while True:
                i = 10
        (e, z, z0, dir) = expr.args
        return "Limit(%s, %s, %s, dir='%s')" % tuple(map(self._print, (e, z, z0, dir)))

    def _print_list(self, expr):
        if False:
            while True:
                i = 10
        return '[%s]' % self.stringify(expr, ', ')

    def _print_List(self, expr):
        if False:
            while True:
                i = 10
        return self._print_list(expr)

    def _print_MatrixBase(self, expr):
        if False:
            for i in range(10):
                print('nop')
        return expr._format_str(self)

    def _print_MatrixElement(self, expr):
        if False:
            while True:
                i = 10
        return self.parenthesize(expr.parent, PRECEDENCE['Atom'], strict=True) + '[%s, %s]' % (self._print(expr.i), self._print(expr.j))

    def _print_MatrixSlice(self, expr):
        if False:
            for i in range(10):
                print('nop')

        def strslice(x, dim):
            if False:
                i = 10
                return i + 15
            x = list(x)
            if x[2] == 1:
                del x[2]
            if x[0] == 0:
                x[0] = ''
            if x[1] == dim:
                x[1] = ''
            return ':'.join((self._print(arg) for arg in x))
        return self.parenthesize(expr.parent, PRECEDENCE['Atom'], strict=True) + '[' + strslice(expr.rowslice, expr.parent.rows) + ', ' + strslice(expr.colslice, expr.parent.cols) + ']'

    def _print_DeferredVector(self, expr):
        if False:
            for i in range(10):
                print('nop')
        return expr.name

    def _print_Mul(self, expr):
        if False:
            return 10
        prec = precedence(expr)
        args = expr.args
        if args[0] is S.One or any((isinstance(a, Number) or (a.is_Pow and all((ai.is_Integer for ai in a.args))) for a in args[1:])):
            (d, n) = sift(args, lambda x: isinstance(x, Pow) and bool(x.exp.as_coeff_Mul()[0] < 0), binary=True)
            for (i, di) in enumerate(d):
                if di.exp.is_Number:
                    e = -di.exp
                else:
                    dargs = list(di.exp.args)
                    dargs[0] = -dargs[0]
                    e = Mul._from_args(dargs)
                d[i] = Pow(di.base, e, evaluate=False) if e - 1 else di.base
            pre = []
            if n and (not n[0].is_Add) and n[0].could_extract_minus_sign():
                pre = [self._print(n.pop(0))]
            nfactors = pre + [self.parenthesize(a, prec, strict=False) for a in n]
            if not nfactors:
                nfactors = ['1']
            if len(d) > 1 and d[0].could_extract_minus_sign():
                pre = [self._print(d.pop(0))]
            else:
                pre = []
            dfactors = pre + [self.parenthesize(a, prec, strict=False) for a in d]
            n = '*'.join(nfactors)
            d = '*'.join(dfactors)
            if len(dfactors) > 1:
                return '%s/(%s)' % (n, d)
            elif dfactors:
                return '%s/%s' % (n, d)
            return n
        (c, e) = expr.as_coeff_Mul()
        if c < 0:
            expr = _keep_coeff(-c, e)
            sign = '-'
        else:
            sign = ''
        a = []
        b = []
        pow_paren = []
        if self.order not in ('old', 'none'):
            args = expr.as_ordered_factors()
        else:
            args = Mul.make_args(expr)

        def apow(i):
            if False:
                print('Hello World!')
            (b, e) = i.as_base_exp()
            eargs = list(Mul.make_args(e))
            if eargs[0] is S.NegativeOne:
                eargs = eargs[1:]
            else:
                eargs[0] = -eargs[0]
            e = Mul._from_args(eargs)
            if isinstance(i, Pow):
                return i.func(b, e, evaluate=False)
            return i.func(e, evaluate=False)
        for item in args:
            if item.is_commutative and isinstance(item, Pow) and bool(item.exp.as_coeff_Mul()[0] < 0):
                if item.exp is not S.NegativeOne:
                    b.append(apow(item))
                else:
                    if len(item.args[0].args) != 1 and isinstance(item.base, (Mul, Pow)):
                        pow_paren.append(item)
                    b.append(item.base)
            elif item.is_Rational and item is not S.Infinity:
                if item.p != 1:
                    a.append(Rational(item.p))
                if item.q != 1:
                    b.append(Rational(item.q))
            else:
                a.append(item)
        a = a or [S.One]
        a_str = [self.parenthesize(x, prec, strict=False) for x in a]
        b_str = [self.parenthesize(x, prec, strict=False) for x in b]
        for item in pow_paren:
            if item.base in b:
                b_str[b.index(item.base)] = '(%s)' % b_str[b.index(item.base)]
        if not b:
            return sign + '*'.join(a_str)
        elif len(b) == 1:
            return sign + '*'.join(a_str) + '/' + b_str[0]
        else:
            return sign + '*'.join(a_str) + '/(%s)' % '*'.join(b_str)

    def _print_MatMul(self, expr):
        if False:
            while True:
                i = 10
        (c, m) = expr.as_coeff_mmul()
        sign = ''
        if c.is_number:
            (re, im) = c.as_real_imag()
            if im.is_zero and re.is_negative:
                expr = _keep_coeff(-c, m)
                sign = '-'
            elif re.is_zero and im.is_negative:
                expr = _keep_coeff(-c, m)
                sign = '-'
        return sign + '*'.join([self.parenthesize(arg, precedence(expr)) for arg in expr.args])

    def _print_ElementwiseApplyFunction(self, expr):
        if False:
            for i in range(10):
                print('nop')
        return '{}.({})'.format(expr.function, self._print(expr.expr))

    def _print_NaN(self, expr):
        if False:
            i = 10
            return i + 15
        return 'nan'

    def _print_NegativeInfinity(self, expr):
        if False:
            i = 10
            return i + 15
        return '-oo'

    def _print_Order(self, expr):
        if False:
            for i in range(10):
                print('nop')
        if not expr.variables or all((p is S.Zero for p in expr.point)):
            if len(expr.variables) <= 1:
                return 'O(%s)' % self._print(expr.expr)
            else:
                return 'O(%s)' % self.stringify((expr.expr,) + expr.variables, ', ', 0)
        else:
            return 'O(%s)' % self.stringify(expr.args, ', ', 0)

    def _print_Ordinal(self, expr):
        if False:
            return 10
        return expr.__str__()

    def _print_Cycle(self, expr):
        if False:
            while True:
                i = 10
        return expr.__str__()

    def _print_Permutation(self, expr):
        if False:
            while True:
                i = 10
        from sympy.combinatorics.permutations import Permutation, Cycle
        from sympy.utilities.exceptions import sympy_deprecation_warning
        perm_cyclic = Permutation.print_cyclic
        if perm_cyclic is not None:
            sympy_deprecation_warning(f'\n                Setting Permutation.print_cyclic is deprecated. Instead use\n                init_printing(perm_cyclic={perm_cyclic}).\n                ', deprecated_since_version='1.6', active_deprecations_target='deprecated-permutation-print_cyclic', stacklevel=7)
        else:
            perm_cyclic = self._settings.get('perm_cyclic', True)
        if perm_cyclic:
            if not expr.size:
                return '()'
            s = Cycle(expr)(expr.size - 1).__repr__()[len('Cycle'):]
            last = s.rfind('(')
            if not last == 0 and ',' not in s[last:]:
                s = s[last:] + s[:last]
            s = s.replace(',', '')
            return s
        else:
            s = expr.support()
            if not s:
                if expr.size < 5:
                    return 'Permutation(%s)' % self._print(expr.array_form)
                return 'Permutation([], size=%s)' % self._print(expr.size)
            trim = self._print(expr.array_form[:s[-1] + 1]) + ', size=%s' % self._print(expr.size)
            use = full = self._print(expr.array_form)
            if len(trim) < len(full):
                use = trim
            return 'Permutation(%s)' % use

    def _print_Subs(self, obj):
        if False:
            return 10
        (expr, old, new) = obj.args
        if len(obj.point) == 1:
            old = old[0]
            new = new[0]
        return 'Subs(%s, %s, %s)' % (self._print(expr), self._print(old), self._print(new))

    def _print_TensorIndex(self, expr):
        if False:
            for i in range(10):
                print('nop')
        return expr._print()

    def _print_TensorHead(self, expr):
        if False:
            for i in range(10):
                print('nop')
        return expr._print()

    def _print_Tensor(self, expr):
        if False:
            print('Hello World!')
        return expr._print()

    def _print_TensMul(self, expr):
        if False:
            return 10
        (sign, args) = expr._get_args_for_traditional_printer()
        return sign + '*'.join([self.parenthesize(arg, precedence(expr)) for arg in args])

    def _print_TensAdd(self, expr):
        if False:
            while True:
                i = 10
        return expr._print()

    def _print_ArraySymbol(self, expr):
        if False:
            return 10
        return self._print(expr.name)

    def _print_ArrayElement(self, expr):
        if False:
            print('Hello World!')
        return '%s[%s]' % (self.parenthesize(expr.name, PRECEDENCE['Func'], True), ', '.join([self._print(i) for i in expr.indices]))

    def _print_PermutationGroup(self, expr):
        if False:
            while True:
                i = 10
        p = ['    %s' % self._print(a) for a in expr.args]
        return 'PermutationGroup([\n%s])' % ',\n'.join(p)

    def _print_Pi(self, expr):
        if False:
            print('Hello World!')
        return 'pi'

    def _print_PolyRing(self, ring):
        if False:
            for i in range(10):
                print('nop')
        return 'Polynomial ring in %s over %s with %s order' % (', '.join((self._print(rs) for rs in ring.symbols)), self._print(ring.domain), self._print(ring.order))

    def _print_FracField(self, field):
        if False:
            for i in range(10):
                print('nop')
        return 'Rational function field in %s over %s with %s order' % (', '.join((self._print(fs) for fs in field.symbols)), self._print(field.domain), self._print(field.order))

    def _print_FreeGroupElement(self, elm):
        if False:
            i = 10
            return i + 15
        return elm.__str__()

    def _print_GaussianElement(self, poly):
        if False:
            for i in range(10):
                print('nop')
        return '(%s + %s*I)' % (poly.x, poly.y)

    def _print_PolyElement(self, poly):
        if False:
            i = 10
            return i + 15
        return poly.str(self, PRECEDENCE, '%s**%s', '*')

    def _print_FracElement(self, frac):
        if False:
            for i in range(10):
                print('nop')
        if frac.denom == 1:
            return self._print(frac.numer)
        else:
            numer = self.parenthesize(frac.numer, PRECEDENCE['Mul'], strict=True)
            denom = self.parenthesize(frac.denom, PRECEDENCE['Atom'], strict=True)
            return numer + '/' + denom

    def _print_Poly(self, expr):
        if False:
            for i in range(10):
                print('nop')
        ATOM_PREC = PRECEDENCE['Atom'] - 1
        (terms, gens) = ([], [self.parenthesize(s, ATOM_PREC) for s in expr.gens])
        for (monom, coeff) in expr.terms():
            s_monom = []
            for (i, e) in enumerate(monom):
                if e > 0:
                    if e == 1:
                        s_monom.append(gens[i])
                    else:
                        s_monom.append(gens[i] + '**%d' % e)
            s_monom = '*'.join(s_monom)
            if coeff.is_Add:
                if s_monom:
                    s_coeff = '(' + self._print(coeff) + ')'
                else:
                    s_coeff = self._print(coeff)
            else:
                if s_monom:
                    if coeff is S.One:
                        terms.extend(['+', s_monom])
                        continue
                    if coeff is S.NegativeOne:
                        terms.extend(['-', s_monom])
                        continue
                s_coeff = self._print(coeff)
            if not s_monom:
                s_term = s_coeff
            else:
                s_term = s_coeff + '*' + s_monom
            if s_term.startswith('-'):
                terms.extend(['-', s_term[1:]])
            else:
                terms.extend(['+', s_term])
        if terms[0] in ('-', '+'):
            modifier = terms.pop(0)
            if modifier == '-':
                terms[0] = '-' + terms[0]
        format = expr.__class__.__name__ + '(%s, %s'
        from sympy.polys.polyerrors import PolynomialError
        try:
            format += ', modulus=%s' % expr.get_modulus()
        except PolynomialError:
            format += ", domain='%s'" % expr.get_domain()
        format += ')'
        for (index, item) in enumerate(gens):
            if len(item) > 2 and (item[:1] == '(' and item[len(item) - 1:] == ')'):
                gens[index] = item[1:len(item) - 1]
        return format % (' '.join(terms), ', '.join(gens))

    def _print_UniversalSet(self, p):
        if False:
            return 10
        return 'UniversalSet'

    def _print_AlgebraicNumber(self, expr):
        if False:
            i = 10
            return i + 15
        if expr.is_aliased:
            return self._print(expr.as_poly().as_expr())
        else:
            return self._print(expr.as_expr())

    def _print_Pow(self, expr, rational=False):
        if False:
            for i in range(10):
                print('nop')
        "Printing helper function for ``Pow``\n\n        Parameters\n        ==========\n\n        rational : bool, optional\n            If ``True``, it will not attempt printing ``sqrt(x)`` or\n            ``x**S.Half`` as ``sqrt``, and will use ``x**(1/2)``\n            instead.\n\n            See examples for additional details\n\n        Examples\n        ========\n\n        >>> from sympy import sqrt, StrPrinter\n        >>> from sympy.abc import x\n\n        How ``rational`` keyword works with ``sqrt``:\n\n        >>> printer = StrPrinter()\n        >>> printer._print_Pow(sqrt(x), rational=True)\n        'x**(1/2)'\n        >>> printer._print_Pow(sqrt(x), rational=False)\n        'sqrt(x)'\n        >>> printer._print_Pow(1/sqrt(x), rational=True)\n        'x**(-1/2)'\n        >>> printer._print_Pow(1/sqrt(x), rational=False)\n        '1/sqrt(x)'\n\n        Notes\n        =====\n\n        ``sqrt(x)`` is canonicalized as ``Pow(x, S.Half)`` in SymPy,\n        so there is no need of defining a separate printer for ``sqrt``.\n        Instead, it should be handled here as well.\n        "
        PREC = precedence(expr)
        if expr.exp is S.Half and (not rational):
            return 'sqrt(%s)' % self._print(expr.base)
        if expr.is_commutative:
            if -expr.exp is S.Half and (not rational):
                return '%s/sqrt(%s)' % tuple((self._print(arg) for arg in (S.One, expr.base)))
            if expr.exp is -S.One:
                return '%s/%s' % (self._print(S.One), self.parenthesize(expr.base, PREC, strict=False))
        e = self.parenthesize(expr.exp, PREC, strict=False)
        if self.printmethod == '_sympyrepr' and expr.exp.is_Rational and (expr.exp.q != 1):
            if e.startswith('(Rational'):
                return '%s**%s' % (self.parenthesize(expr.base, PREC, strict=False), e[1:-1])
        return '%s**%s' % (self.parenthesize(expr.base, PREC, strict=False), e)

    def _print_UnevaluatedExpr(self, expr):
        if False:
            i = 10
            return i + 15
        return self._print(expr.args[0])

    def _print_MatPow(self, expr):
        if False:
            return 10
        PREC = precedence(expr)
        return '%s**%s' % (self.parenthesize(expr.base, PREC, strict=False), self.parenthesize(expr.exp, PREC, strict=False))

    def _print_Integer(self, expr):
        if False:
            return 10
        if self._settings.get('sympy_integers', False):
            return 'S(%s)' % expr
        return str(expr.p)

    def _print_Integers(self, expr):
        if False:
            for i in range(10):
                print('nop')
        return 'Integers'

    def _print_Naturals(self, expr):
        if False:
            while True:
                i = 10
        return 'Naturals'

    def _print_Naturals0(self, expr):
        if False:
            i = 10
            return i + 15
        return 'Naturals0'

    def _print_Rationals(self, expr):
        if False:
            i = 10
            return i + 15
        return 'Rationals'

    def _print_Reals(self, expr):
        if False:
            return 10
        return 'Reals'

    def _print_Complexes(self, expr):
        if False:
            print('Hello World!')
        return 'Complexes'

    def _print_EmptySet(self, expr):
        if False:
            while True:
                i = 10
        return 'EmptySet'

    def _print_EmptySequence(self, expr):
        if False:
            i = 10
            return i + 15
        return 'EmptySequence'

    def _print_int(self, expr):
        if False:
            for i in range(10):
                print('nop')
        return str(expr)

    def _print_mpz(self, expr):
        if False:
            i = 10
            return i + 15
        return str(expr)

    def _print_Rational(self, expr):
        if False:
            return 10
        if expr.q == 1:
            return str(expr.p)
        else:
            if self._settings.get('sympy_integers', False):
                return 'S(%s)/%s' % (expr.p, expr.q)
            return '%s/%s' % (expr.p, expr.q)

    def _print_PythonRational(self, expr):
        if False:
            print('Hello World!')
        if expr.q == 1:
            return str(expr.p)
        else:
            return '%d/%d' % (expr.p, expr.q)

    def _print_Fraction(self, expr):
        if False:
            print('Hello World!')
        if expr.denominator == 1:
            return str(expr.numerator)
        else:
            return '%s/%s' % (expr.numerator, expr.denominator)

    def _print_mpq(self, expr):
        if False:
            i = 10
            return i + 15
        if expr.denominator == 1:
            return str(expr.numerator)
        else:
            return '%s/%s' % (expr.numerator, expr.denominator)

    def _print_Float(self, expr):
        if False:
            for i in range(10):
                print('nop')
        prec = expr._prec
        if prec < 5:
            dps = 0
        else:
            dps = prec_to_dps(expr._prec)
        if self._settings['full_prec'] is True:
            strip = False
        elif self._settings['full_prec'] is False:
            strip = True
        elif self._settings['full_prec'] == 'auto':
            strip = self._print_level > 1
        low = self._settings['min'] if 'min' in self._settings else None
        high = self._settings['max'] if 'max' in self._settings else None
        rv = mlib_to_str(expr._mpf_, dps, strip_zeros=strip, min_fixed=low, max_fixed=high)
        if rv.startswith('-.0'):
            rv = '-0.' + rv[3:]
        elif rv.startswith('.0'):
            rv = '0.' + rv[2:]
        if rv.startswith('+'):
            rv = rv[1:]
        return rv

    def _print_Relational(self, expr):
        if False:
            while True:
                i = 10
        charmap = {'==': 'Eq', '!=': 'Ne', ':=': 'Assignment', '+=': 'AddAugmentedAssignment', '-=': 'SubAugmentedAssignment', '*=': 'MulAugmentedAssignment', '/=': 'DivAugmentedAssignment', '%=': 'ModAugmentedAssignment'}
        if expr.rel_op in charmap:
            return '%s(%s, %s)' % (charmap[expr.rel_op], self._print(expr.lhs), self._print(expr.rhs))
        return '%s %s %s' % (self.parenthesize(expr.lhs, precedence(expr)), self._relationals.get(expr.rel_op) or expr.rel_op, self.parenthesize(expr.rhs, precedence(expr)))

    def _print_ComplexRootOf(self, expr):
        if False:
            return 10
        return 'CRootOf(%s, %d)' % (self._print_Add(expr.expr, order='lex'), expr.index)

    def _print_RootSum(self, expr):
        if False:
            print('Hello World!')
        args = [self._print_Add(expr.expr, order='lex')]
        if expr.fun is not S.IdentityFunction:
            args.append(self._print(expr.fun))
        return 'RootSum(%s)' % ', '.join(args)

    def _print_GroebnerBasis(self, basis):
        if False:
            i = 10
            return i + 15
        cls = basis.__class__.__name__
        exprs = [self._print_Add(arg, order=basis.order) for arg in basis.exprs]
        exprs = '[%s]' % ', '.join(exprs)
        gens = [self._print(gen) for gen in basis.gens]
        domain = "domain='%s'" % self._print(basis.domain)
        order = "order='%s'" % self._print(basis.order)
        args = [exprs] + gens + [domain, order]
        return '%s(%s)' % (cls, ', '.join(args))

    def _print_set(self, s):
        if False:
            while True:
                i = 10
        items = sorted(s, key=default_sort_key)
        args = ', '.join((self._print(item) for item in items))
        if not args:
            return 'set()'
        return '{%s}' % args

    def _print_FiniteSet(self, s):
        if False:
            print('Hello World!')
        from sympy.sets.sets import FiniteSet
        items = sorted(s, key=default_sort_key)
        args = ', '.join((self._print(item) for item in items))
        if any((item.has(FiniteSet) for item in items)):
            return 'FiniteSet({})'.format(args)
        return '{{{}}}'.format(args)

    def _print_Partition(self, s):
        if False:
            while True:
                i = 10
        items = sorted(s, key=default_sort_key)
        args = ', '.join((self._print(arg) for arg in items))
        return 'Partition({})'.format(args)

    def _print_frozenset(self, s):
        if False:
            print('Hello World!')
        if not s:
            return 'frozenset()'
        return 'frozenset(%s)' % self._print_set(s)

    def _print_Sum(self, expr):
        if False:
            while True:
                i = 10

        def _xab_tostr(xab):
            if False:
                return 10
            if len(xab) == 1:
                return self._print(xab[0])
            else:
                return self._print((xab[0],) + tuple(xab[1:]))
        L = ', '.join([_xab_tostr(l) for l in expr.limits])
        return 'Sum(%s, %s)' % (self._print(expr.function), L)

    def _print_Symbol(self, expr):
        if False:
            print('Hello World!')
        return expr.name
    _print_MatrixSymbol = _print_Symbol
    _print_RandomSymbol = _print_Symbol

    def _print_Identity(self, expr):
        if False:
            i = 10
            return i + 15
        return 'I'

    def _print_ZeroMatrix(self, expr):
        if False:
            i = 10
            return i + 15
        return '0'

    def _print_OneMatrix(self, expr):
        if False:
            print('Hello World!')
        return '1'

    def _print_Predicate(self, expr):
        if False:
            while True:
                i = 10
        return 'Q.%s' % expr.name

    def _print_str(self, expr):
        if False:
            while True:
                i = 10
        return str(expr)

    def _print_tuple(self, expr):
        if False:
            print('Hello World!')
        if len(expr) == 1:
            return '(%s,)' % self._print(expr[0])
        else:
            return '(%s)' % self.stringify(expr, ', ')

    def _print_Tuple(self, expr):
        if False:
            i = 10
            return i + 15
        return self._print_tuple(expr)

    def _print_Transpose(self, T):
        if False:
            return 10
        return '%s.T' % self.parenthesize(T.arg, PRECEDENCE['Pow'])

    def _print_Uniform(self, expr):
        if False:
            while True:
                i = 10
        return 'Uniform(%s, %s)' % (self._print(expr.a), self._print(expr.b))

    def _print_Quantity(self, expr):
        if False:
            i = 10
            return i + 15
        if self._settings.get('abbrev', False):
            return '%s' % expr.abbrev
        return '%s' % expr.name

    def _print_Quaternion(self, expr):
        if False:
            print('Hello World!')
        s = [self.parenthesize(i, PRECEDENCE['Mul'], strict=True) for i in expr.args]
        a = [s[0]] + [i + '*' + j for (i, j) in zip(s[1:], 'ijk')]
        return ' + '.join(a)

    def _print_Dimension(self, expr):
        if False:
            i = 10
            return i + 15
        return str(expr)

    def _print_Wild(self, expr):
        if False:
            while True:
                i = 10
        return expr.name + '_'

    def _print_WildFunction(self, expr):
        if False:
            return 10
        return expr.name + '_'

    def _print_WildDot(self, expr):
        if False:
            i = 10
            return i + 15
        return expr.name

    def _print_WildPlus(self, expr):
        if False:
            return 10
        return expr.name

    def _print_WildStar(self, expr):
        if False:
            for i in range(10):
                print('nop')
        return expr.name

    def _print_Zero(self, expr):
        if False:
            return 10
        if self._settings.get('sympy_integers', False):
            return 'S(0)'
        return self._print_Integer(Integer(0))

    def _print_DMP(self, p):
        if False:
            print('Hello World!')
        cls = p.__class__.__name__
        rep = self._print(p.to_list())
        dom = self._print(p.dom)
        return '%s(%s, %s)' % (cls, rep, dom)

    def _print_DMF(self, expr):
        if False:
            while True:
                i = 10
        cls = expr.__class__.__name__
        num = self._print(expr.num)
        den = self._print(expr.den)
        dom = self._print(expr.dom)
        return '%s(%s, %s, %s)' % (cls, num, den, dom)

    def _print_Object(self, obj):
        if False:
            for i in range(10):
                print('nop')
        return 'Object("%s")' % obj.name

    def _print_IdentityMorphism(self, morphism):
        if False:
            return 10
        return 'IdentityMorphism(%s)' % morphism.domain

    def _print_NamedMorphism(self, morphism):
        if False:
            i = 10
            return i + 15
        return 'NamedMorphism(%s, %s, "%s")' % (morphism.domain, morphism.codomain, morphism.name)

    def _print_Category(self, category):
        if False:
            for i in range(10):
                print('nop')
        return 'Category("%s")' % category.name

    def _print_Manifold(self, manifold):
        if False:
            print('Hello World!')
        return manifold.name.name

    def _print_Patch(self, patch):
        if False:
            return 10
        return patch.name.name

    def _print_CoordSystem(self, coords):
        if False:
            return 10
        return coords.name.name

    def _print_BaseScalarField(self, field):
        if False:
            print('Hello World!')
        return field._coord_sys.symbols[field._index].name

    def _print_BaseVectorField(self, field):
        if False:
            while True:
                i = 10
        return 'e_%s' % field._coord_sys.symbols[field._index].name

    def _print_Differential(self, diff):
        if False:
            return 10
        field = diff._form_field
        if hasattr(field, '_coord_sys'):
            return 'd%s' % field._coord_sys.symbols[field._index].name
        else:
            return 'd(%s)' % self._print(field)

    def _print_Tr(self, expr):
        if False:
            i = 10
            return i + 15
        return '%s(%s)' % ('Tr', self._print(expr.args[0]))

    def _print_Str(self, s):
        if False:
            return 10
        return self._print(s.name)

    def _print_AppliedBinaryRelation(self, expr):
        if False:
            print('Hello World!')
        rel = expr.function
        return '%s(%s, %s)' % (self._print(rel), self._print(expr.lhs), self._print(expr.rhs))

@print_function(StrPrinter)
def sstr(expr, **settings):
    if False:
        return 10
    "Returns the expression as a string.\n\n    For large expressions where speed is a concern, use the setting\n    order='none'. If abbrev=True setting is used then units are printed in\n    abbreviated form.\n\n    Examples\n    ========\n\n    >>> from sympy import symbols, Eq, sstr\n    >>> a, b = symbols('a b')\n    >>> sstr(Eq(a + b, 0))\n    'Eq(a + b, 0)'\n    "
    p = StrPrinter(settings)
    s = p.doprint(expr)
    return s

class StrReprPrinter(StrPrinter):
    """(internal) -- see sstrrepr"""

    def _print_str(self, s):
        if False:
            i = 10
            return i + 15
        return repr(s)

    def _print_Str(self, s):
        if False:
            i = 10
            return i + 15
        return '%s(%s)' % (s.__class__.__name__, self._print(s.name))

@print_function(StrReprPrinter)
def sstrrepr(expr, **settings):
    if False:
        return 10
    'return expr in mixed str/repr form\n\n       i.e. strings are returned in repr form with quotes, and everything else\n       is returned in str form.\n\n       This function could be useful for hooking into sys.displayhook\n    '
    p = StrReprPrinter(settings)
    s = p.doprint(expr)
    return s