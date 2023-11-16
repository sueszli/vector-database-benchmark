from collections import defaultdict
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.exprtools import Factors, gcd_terms, factor_terms
from sympy.core.function import expand_mul
from sympy.core.mul import Mul
from sympy.core.numbers import pi, I
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.core.traversal import bottom_up
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.hyperbolic import cosh, sinh, tanh, coth, sech, csch, HyperbolicFunction
from sympy.functions.elementary.trigonometric import cos, sin, tan, cot, sec, csc, sqrt, TrigonometricFunction
from sympy.ntheory.factor_ import perfect_power
from sympy.polys.polytools import factor
from sympy.strategies.tree import greedy
from sympy.strategies.core import identity, debug
from sympy import SYMPY_DEBUG

def TR0(rv):
    if False:
        return 10
    'Simplification of rational polynomials, trying to simplify\n    the expression, e.g. combine things like 3*x + 2*x, etc....\n    '
    return rv.normal().factor().expand()

def TR1(rv):
    if False:
        return 10
    'Replace sec, csc with 1/cos, 1/sin\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.fu import TR1, sec, csc\n    >>> from sympy.abc import x\n    >>> TR1(2*csc(x) + sec(x))\n    1/cos(x) + 2/sin(x)\n    '

    def f(rv):
        if False:
            return 10
        if isinstance(rv, sec):
            a = rv.args[0]
            return S.One / cos(a)
        elif isinstance(rv, csc):
            a = rv.args[0]
            return S.One / sin(a)
        return rv
    return bottom_up(rv, f)

def TR2(rv):
    if False:
        print('Hello World!')
    'Replace tan and cot with sin/cos and cos/sin\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.fu import TR2\n    >>> from sympy.abc import x\n    >>> from sympy import tan, cot, sin, cos\n    >>> TR2(tan(x))\n    sin(x)/cos(x)\n    >>> TR2(cot(x))\n    cos(x)/sin(x)\n    >>> TR2(tan(tan(x) - sin(x)/cos(x)))\n    0\n\n    '

    def f(rv):
        if False:
            while True:
                i = 10
        if isinstance(rv, tan):
            a = rv.args[0]
            return sin(a) / cos(a)
        elif isinstance(rv, cot):
            a = rv.args[0]
            return cos(a) / sin(a)
        return rv
    return bottom_up(rv, f)

def TR2i(rv, half=False):
    if False:
        return 10
    'Converts ratios involving sin and cos as follows::\n        sin(x)/cos(x) -> tan(x)\n        sin(x)/(cos(x) + 1) -> tan(x/2) if half=True\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.fu import TR2i\n    >>> from sympy.abc import x, a\n    >>> from sympy import sin, cos\n    >>> TR2i(sin(x)/cos(x))\n    tan(x)\n\n    Powers of the numerator and denominator are also recognized\n\n    >>> TR2i(sin(x)**2/(cos(x) + 1)**2, half=True)\n    tan(x/2)**2\n\n    The transformation does not take place unless assumptions allow\n    (i.e. the base must be positive or the exponent must be an integer\n    for both numerator and denominator)\n\n    >>> TR2i(sin(x)**a/(cos(x) + 1)**a)\n    sin(x)**a/(cos(x) + 1)**a\n\n    '

    def f(rv):
        if False:
            print('Hello World!')
        if not rv.is_Mul:
            return rv
        (n, d) = rv.as_numer_denom()
        if n.is_Atom or d.is_Atom:
            return rv

        def ok(k, e):
            if False:
                return 10
            return (e.is_integer or k.is_positive) and (k.func in (sin, cos) or (half and k.is_Add and (len(k.args) >= 2) and any((any((isinstance(ai, cos) or (ai.is_Pow and ai.base is cos) for ai in Mul.make_args(a))) for a in k.args))))
        n = n.as_powers_dict()
        ndone = [(k, n.pop(k)) for k in list(n.keys()) if not ok(k, n[k])]
        if not n:
            return rv
        d = d.as_powers_dict()
        ddone = [(k, d.pop(k)) for k in list(d.keys()) if not ok(k, d[k])]
        if not d:
            return rv

        def factorize(d, ddone):
            if False:
                i = 10
                return i + 15
            newk = []
            for k in d:
                if k.is_Add and len(k.args) > 1:
                    knew = factor(k) if half else factor_terms(k)
                    if knew != k:
                        newk.append((k, knew))
            if newk:
                for (i, (k, knew)) in enumerate(newk):
                    del d[k]
                    newk[i] = knew
                newk = Mul(*newk).as_powers_dict()
                for k in newk:
                    v = d[k] + newk[k]
                    if ok(k, v):
                        d[k] = v
                    else:
                        ddone.append((k, v))
                del newk
        factorize(n, ndone)
        factorize(d, ddone)
        t = []
        for k in n:
            if isinstance(k, sin):
                a = cos(k.args[0], evaluate=False)
                if a in d and d[a] == n[k]:
                    t.append(tan(k.args[0]) ** n[k])
                    n[k] = d[a] = None
                elif half:
                    a1 = 1 + a
                    if a1 in d and d[a1] == n[k]:
                        t.append(tan(k.args[0] / 2) ** n[k])
                        n[k] = d[a1] = None
            elif isinstance(k, cos):
                a = sin(k.args[0], evaluate=False)
                if a in d and d[a] == n[k]:
                    t.append(tan(k.args[0]) ** (-n[k]))
                    n[k] = d[a] = None
            elif half and k.is_Add and (k.args[0] is S.One) and isinstance(k.args[1], cos):
                a = sin(k.args[1].args[0], evaluate=False)
                if a in d and d[a] == n[k] and (d[a].is_integer or a.is_positive):
                    t.append(tan(a.args[0] / 2) ** (-n[k]))
                    n[k] = d[a] = None
        if t:
            rv = Mul(*t + [b ** e for (b, e) in n.items() if e]) / Mul(*[b ** e for (b, e) in d.items() if e])
            rv *= Mul(*[b ** e for (b, e) in ndone]) / Mul(*[b ** e for (b, e) in ddone])
        return rv
    return bottom_up(rv, f)

def TR3(rv):
    if False:
        return 10
    'Induced formula: example sin(-a) = -sin(a)\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.fu import TR3\n    >>> from sympy.abc import x, y\n    >>> from sympy import pi\n    >>> from sympy import cos\n    >>> TR3(cos(y - x*(y - x)))\n    cos(x*(x - y) + y)\n    >>> cos(pi/2 + x)\n    -sin(x)\n    >>> cos(30*pi/2 + x)\n    -cos(x)\n\n    '
    from sympy.simplify.simplify import signsimp

    def f(rv):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(rv, TrigonometricFunction):
            return rv
        rv = rv.func(signsimp(rv.args[0]))
        if not isinstance(rv, TrigonometricFunction):
            return rv
        if (rv.args[0] - S.Pi / 4).is_positive is (S.Pi / 2 - rv.args[0]).is_positive is True:
            fmap = {cos: sin, sin: cos, tan: cot, cot: tan, sec: csc, csc: sec}
            rv = fmap[type(rv)](S.Pi / 2 - rv.args[0])
        return rv
    return bottom_up(rv, f)

def TR4(rv):
    if False:
        for i in range(10):
            print('nop')
    "Identify values of special angles.\n\n        a=  0   pi/6        pi/4        pi/3        pi/2\n    ----------------------------------------------------\n    sin(a)  0   1/2         sqrt(2)/2   sqrt(3)/2   1\n    cos(a)  1   sqrt(3)/2   sqrt(2)/2   1/2         0\n    tan(a)  0   sqt(3)/3    1           sqrt(3)     --\n\n    Examples\n    ========\n\n    >>> from sympy import pi\n    >>> from sympy import cos, sin, tan, cot\n    >>> for s in (0, pi/6, pi/4, pi/3, pi/2):\n    ...    print('%s %s %s %s' % (cos(s), sin(s), tan(s), cot(s)))\n    ...\n    1 0 0 zoo\n    sqrt(3)/2 1/2 sqrt(3)/3 sqrt(3)\n    sqrt(2)/2 sqrt(2)/2 1 1\n    1/2 sqrt(3)/2 sqrt(3) sqrt(3)/3\n    0 1 zoo 0\n    "
    return rv

def _TR56(rv, f, g, h, max, pow):
    if False:
        for i in range(10):
            print('nop')
    'Helper for TR5 and TR6 to replace f**2 with h(g**2)\n\n    Options\n    =======\n\n    max :   controls size of exponent that can appear on f\n            e.g. if max=4 then f**4 will be changed to h(g**2)**2.\n    pow :   controls whether the exponent must be a perfect power of 2\n            e.g. if pow=True (and max >= 6) then f**6 will not be changed\n            but f**8 will be changed to h(g**2)**4\n\n    >>> from sympy.simplify.fu import _TR56 as T\n    >>> from sympy.abc import x\n    >>> from sympy import sin, cos\n    >>> h = lambda x: 1 - x\n    >>> T(sin(x)**3, sin, cos, h, 4, False)\n    (1 - cos(x)**2)*sin(x)\n    >>> T(sin(x)**6, sin, cos, h, 6, False)\n    (1 - cos(x)**2)**3\n    >>> T(sin(x)**6, sin, cos, h, 6, True)\n    sin(x)**6\n    >>> T(sin(x)**8, sin, cos, h, 10, True)\n    (1 - cos(x)**2)**4\n    '

    def _f(rv):
        if False:
            for i in range(10):
                print('nop')
        if not (rv.is_Pow and rv.base.func == f):
            return rv
        if not rv.exp.is_real:
            return rv
        if (rv.exp < 0) == True:
            return rv
        if (rv.exp > max) == True:
            return rv
        if rv.exp == 1:
            return rv
        if rv.exp == 2:
            return h(g(rv.base.args[0]) ** 2)
        else:
            if rv.exp % 2 == 1:
                e = rv.exp // 2
                return f(rv.base.args[0]) * h(g(rv.base.args[0]) ** 2) ** e
            elif rv.exp == 4:
                e = 2
            elif not pow:
                if rv.exp % 2:
                    return rv
                e = rv.exp // 2
            else:
                p = perfect_power(rv.exp)
                if not p:
                    return rv
                e = rv.exp // 2
            return h(g(rv.base.args[0]) ** 2) ** e
    return bottom_up(rv, _f)

def TR5(rv, max=4, pow=False):
    if False:
        while True:
            i = 10
    'Replacement of sin**2 with 1 - cos(x)**2.\n\n    See _TR56 docstring for advanced use of ``max`` and ``pow``.\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.fu import TR5\n    >>> from sympy.abc import x\n    >>> from sympy import sin\n    >>> TR5(sin(x)**2)\n    1 - cos(x)**2\n    >>> TR5(sin(x)**-2)  # unchanged\n    sin(x)**(-2)\n    >>> TR5(sin(x)**4)\n    (1 - cos(x)**2)**2\n    '
    return _TR56(rv, sin, cos, lambda x: 1 - x, max=max, pow=pow)

def TR6(rv, max=4, pow=False):
    if False:
        i = 10
        return i + 15
    'Replacement of cos**2 with 1 - sin(x)**2.\n\n    See _TR56 docstring for advanced use of ``max`` and ``pow``.\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.fu import TR6\n    >>> from sympy.abc import x\n    >>> from sympy import cos\n    >>> TR6(cos(x)**2)\n    1 - sin(x)**2\n    >>> TR6(cos(x)**-2)  #unchanged\n    cos(x)**(-2)\n    >>> TR6(cos(x)**4)\n    (1 - sin(x)**2)**2\n    '
    return _TR56(rv, cos, sin, lambda x: 1 - x, max=max, pow=pow)

def TR7(rv):
    if False:
        for i in range(10):
            print('nop')
    'Lowering the degree of cos(x)**2.\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.fu import TR7\n    >>> from sympy.abc import x\n    >>> from sympy import cos\n    >>> TR7(cos(x)**2)\n    cos(2*x)/2 + 1/2\n    >>> TR7(cos(x)**2 + 1)\n    cos(2*x)/2 + 3/2\n\n    '

    def f(rv):
        if False:
            for i in range(10):
                print('nop')
        if not (rv.is_Pow and rv.base.func == cos and (rv.exp == 2)):
            return rv
        return (1 + cos(2 * rv.base.args[0])) / 2
    return bottom_up(rv, f)

def TR8(rv, first=True):
    if False:
        return 10
    'Converting products of ``cos`` and/or ``sin`` to a sum or\n    difference of ``cos`` and or ``sin`` terms.\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.fu import TR8\n    >>> from sympy import cos, sin\n    >>> TR8(cos(2)*cos(3))\n    cos(5)/2 + cos(1)/2\n    >>> TR8(cos(2)*sin(3))\n    sin(5)/2 + sin(1)/2\n    >>> TR8(sin(2)*sin(3))\n    -cos(5)/2 + cos(1)/2\n    '

    def f(rv):
        if False:
            i = 10
            return i + 15
        if not (rv.is_Mul or (rv.is_Pow and rv.base.func in (cos, sin) and (rv.exp.is_integer or rv.base.is_positive))):
            return rv
        if first:
            (n, d) = [expand_mul(i) for i in rv.as_numer_denom()]
            newn = TR8(n, first=False)
            newd = TR8(d, first=False)
            if newn != n or newd != d:
                rv = gcd_terms(newn / newd)
                if rv.is_Mul and rv.args[0].is_Rational and (len(rv.args) == 2) and rv.args[1].is_Add:
                    rv = Mul(*rv.as_coeff_Mul())
            return rv
        args = {cos: [], sin: [], None: []}
        for a in Mul.make_args(rv):
            if a.func in (cos, sin):
                args[type(a)].append(a.args[0])
            elif a.is_Pow and a.exp.is_Integer and (a.exp > 0) and (a.base.func in (cos, sin)):
                args[type(a.base)].extend([a.base.args[0]] * a.exp)
            else:
                args[None].append(a)
        c = args[cos]
        s = args[sin]
        if not (c and s or len(c) > 1 or len(s) > 1):
            return rv
        args = args[None]
        n = min(len(c), len(s))
        for i in range(n):
            a1 = s.pop()
            a2 = c.pop()
            args.append((sin(a1 + a2) + sin(a1 - a2)) / 2)
        while len(c) > 1:
            a1 = c.pop()
            a2 = c.pop()
            args.append((cos(a1 + a2) + cos(a1 - a2)) / 2)
        if c:
            args.append(cos(c.pop()))
        while len(s) > 1:
            a1 = s.pop()
            a2 = s.pop()
            args.append((-cos(a1 + a2) + cos(a1 - a2)) / 2)
        if s:
            args.append(sin(s.pop()))
        return TR8(expand_mul(Mul(*args)))
    return bottom_up(rv, f)

def TR9(rv):
    if False:
        return 10
    'Sum of ``cos`` or ``sin`` terms as a product of ``cos`` or ``sin``.\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.fu import TR9\n    >>> from sympy import cos, sin\n    >>> TR9(cos(1) + cos(2))\n    2*cos(1/2)*cos(3/2)\n    >>> TR9(cos(1) + 2*sin(1) + 2*sin(2))\n    cos(1) + 4*sin(3/2)*cos(1/2)\n\n    If no change is made by TR9, no re-arrangement of the\n    expression will be made. For example, though factoring\n    of common term is attempted, if the factored expression\n    was not changed, the original expression will be returned:\n\n    >>> TR9(cos(3) + cos(3)*cos(2))\n    cos(3) + cos(2)*cos(3)\n\n    '

    def f(rv):
        if False:
            return 10
        if not rv.is_Add:
            return rv

        def do(rv, first=True):
            if False:
                for i in range(10):
                    print('nop')
            if not rv.is_Add:
                return rv
            args = list(ordered(rv.args))
            if len(args) != 2:
                hit = False
                for i in range(len(args)):
                    ai = args[i]
                    if ai is None:
                        continue
                    for j in range(i + 1, len(args)):
                        aj = args[j]
                        if aj is None:
                            continue
                        was = ai + aj
                        new = do(was)
                        if new != was:
                            args[i] = new
                            args[j] = None
                            hit = True
                            break
                if hit:
                    rv = Add(*[_f for _f in args if _f])
                    if rv.is_Add:
                        rv = do(rv)
                return rv
            split = trig_split(*args)
            if not split:
                return rv
            (gcd, n1, n2, a, b, iscos) = split
            if iscos:
                if n1 == n2:
                    return gcd * n1 * 2 * cos((a + b) / 2) * cos((a - b) / 2)
                if n1 < 0:
                    (a, b) = (b, a)
                return -2 * gcd * sin((a + b) / 2) * sin((a - b) / 2)
            else:
                if n1 == n2:
                    return gcd * n1 * 2 * sin((a + b) / 2) * cos((a - b) / 2)
                if n1 < 0:
                    (a, b) = (b, a)
                return 2 * gcd * cos((a + b) / 2) * sin((a - b) / 2)
        return process_common_addends(rv, do)
    return bottom_up(rv, f)

def TR10(rv, first=True):
    if False:
        return 10
    'Separate sums in ``cos`` and ``sin``.\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.fu import TR10\n    >>> from sympy.abc import a, b, c\n    >>> from sympy import cos, sin\n    >>> TR10(cos(a + b))\n    -sin(a)*sin(b) + cos(a)*cos(b)\n    >>> TR10(sin(a + b))\n    sin(a)*cos(b) + sin(b)*cos(a)\n    >>> TR10(sin(a + b + c))\n    (-sin(a)*sin(b) + cos(a)*cos(b))*sin(c) +     (sin(a)*cos(b) + sin(b)*cos(a))*cos(c)\n    '

    def f(rv):
        if False:
            for i in range(10):
                print('nop')
        if rv.func not in (cos, sin):
            return rv
        f = rv.func
        arg = rv.args[0]
        if arg.is_Add:
            if first:
                args = list(ordered(arg.args))
            else:
                args = list(arg.args)
            a = args.pop()
            b = Add._from_args(args)
            if b.is_Add:
                if f == sin:
                    return sin(a) * TR10(cos(b), first=False) + cos(a) * TR10(sin(b), first=False)
                else:
                    return cos(a) * TR10(cos(b), first=False) - sin(a) * TR10(sin(b), first=False)
            elif f == sin:
                return sin(a) * cos(b) + cos(a) * sin(b)
            else:
                return cos(a) * cos(b) - sin(a) * sin(b)
        return rv
    return bottom_up(rv, f)

def TR10i(rv):
    if False:
        while True:
            i = 10
    'Sum of products to function of sum.\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.fu import TR10i\n    >>> from sympy import cos, sin, sqrt\n    >>> from sympy.abc import x\n\n    >>> TR10i(cos(1)*cos(3) + sin(1)*sin(3))\n    cos(2)\n    >>> TR10i(cos(1)*sin(3) + sin(1)*cos(3) + cos(3))\n    cos(3) + sin(4)\n    >>> TR10i(sqrt(2)*cos(x)*x + sqrt(6)*sin(x)*x)\n    2*sqrt(2)*x*sin(x + pi/6)\n\n    '
    global _ROOT2, _ROOT3, _invROOT3
    if _ROOT2 is None:
        _roots()

    def f(rv):
        if False:
            print('Hello World!')
        if not rv.is_Add:
            return rv

        def do(rv, first=True):
            if False:
                print('Hello World!')
            if not rv.is_Add:
                return rv
            args = list(ordered(rv.args))
            if len(args) != 2:
                hit = False
                for i in range(len(args)):
                    ai = args[i]
                    if ai is None:
                        continue
                    for j in range(i + 1, len(args)):
                        aj = args[j]
                        if aj is None:
                            continue
                        was = ai + aj
                        new = do(was)
                        if new != was:
                            args[i] = new
                            args[j] = None
                            hit = True
                            break
                if hit:
                    rv = Add(*[_f for _f in args if _f])
                    if rv.is_Add:
                        rv = do(rv)
                return rv
            split = trig_split(*args, two=True)
            if not split:
                return rv
            (gcd, n1, n2, a, b, same) = split
            if same:
                gcd = n1 * gcd
                if n1 == n2:
                    return gcd * cos(a - b)
                return gcd * cos(a + b)
            else:
                gcd = n1 * gcd
                if n1 == n2:
                    return gcd * sin(a + b)
                return gcd * sin(b - a)
        rv = process_common_addends(rv, do, lambda x: tuple(ordered(x.free_symbols)))
        while rv.is_Add:
            byrad = defaultdict(list)
            for a in rv.args:
                hit = 0
                if a.is_Mul:
                    for ai in a.args:
                        if ai.is_Pow and ai.exp is S.Half and ai.base.is_Integer:
                            byrad[ai].append(a)
                            hit = 1
                            break
                if not hit:
                    byrad[S.One].append(a)
            args = []
            for a in byrad:
                for b in [_ROOT3 * a, _invROOT3]:
                    if b in byrad:
                        for i in range(len(byrad[a])):
                            if byrad[a][i] is None:
                                continue
                            for j in range(len(byrad[b])):
                                if byrad[b][j] is None:
                                    continue
                                was = Add(byrad[a][i] + byrad[b][j])
                                new = do(was)
                                if new != was:
                                    args.append(new)
                                    byrad[a][i] = None
                                    byrad[b][j] = None
                                    break
            if args:
                rv = Add(*args + [Add(*[_f for _f in v if _f]) for v in byrad.values()])
            else:
                rv = do(rv)
                break
        return rv
    return bottom_up(rv, f)

def TR11(rv, base=None):
    if False:
        i = 10
        return i + 15
    'Function of double angle to product. The ``base`` argument can be used\n    to indicate what is the un-doubled argument, e.g. if 3*pi/7 is the base\n    then cosine and sine functions with argument 6*pi/7 will be replaced.\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.fu import TR11\n    >>> from sympy import cos, sin, pi\n    >>> from sympy.abc import x\n    >>> TR11(sin(2*x))\n    2*sin(x)*cos(x)\n    >>> TR11(cos(2*x))\n    -sin(x)**2 + cos(x)**2\n    >>> TR11(sin(4*x))\n    4*(-sin(x)**2 + cos(x)**2)*sin(x)*cos(x)\n    >>> TR11(sin(4*x/3))\n    4*(-sin(x/3)**2 + cos(x/3)**2)*sin(x/3)*cos(x/3)\n\n    If the arguments are simply integers, no change is made\n    unless a base is provided:\n\n    >>> TR11(cos(2))\n    cos(2)\n    >>> TR11(cos(4), 2)\n    -sin(2)**2 + cos(2)**2\n\n    There is a subtle issue here in that autosimplification will convert\n    some higher angles to lower angles\n\n    >>> cos(6*pi/7) + cos(3*pi/7)\n    -cos(pi/7) + cos(3*pi/7)\n\n    The 6*pi/7 angle is now pi/7 but can be targeted with TR11 by supplying\n    the 3*pi/7 base:\n\n    >>> TR11(_, 3*pi/7)\n    -sin(3*pi/7)**2 + cos(3*pi/7)**2 + cos(3*pi/7)\n\n    '

    def f(rv):
        if False:
            for i in range(10):
                print('nop')
        if rv.func not in (cos, sin):
            return rv
        if base:
            f = rv.func
            t = f(base * 2)
            co = S.One
            if t.is_Mul:
                (co, t) = t.as_coeff_Mul()
            if t.func not in (cos, sin):
                return rv
            if rv.args[0] == t.args[0]:
                c = cos(base)
                s = sin(base)
                if f is cos:
                    return (c ** 2 - s ** 2) / co
                else:
                    return 2 * c * s / co
            return rv
        elif not rv.args[0].is_Number:
            (c, m) = rv.args[0].as_coeff_Mul(rational=True)
            if c.p % 2 == 0:
                arg = c.p // 2 * m / c.q
                c = TR11(cos(arg))
                s = TR11(sin(arg))
                if rv.func == sin:
                    rv = 2 * s * c
                else:
                    rv = c ** 2 - s ** 2
        return rv
    return bottom_up(rv, f)

def _TR11(rv):
    if False:
        i = 10
        return i + 15
    '\n    Helper for TR11 to find half-arguments for sin in factors of\n    num/den that appear in cos or sin factors in the den/num.\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.fu import TR11, _TR11\n    >>> from sympy import cos, sin\n    >>> from sympy.abc import x\n    >>> TR11(sin(x/3)/(cos(x/6)))\n    sin(x/3)/cos(x/6)\n    >>> _TR11(sin(x/3)/(cos(x/6)))\n    2*sin(x/6)\n    >>> TR11(sin(x/6)/(sin(x/3)))\n    sin(x/6)/sin(x/3)\n    >>> _TR11(sin(x/6)/(sin(x/3)))\n    1/(2*cos(x/6))\n\n    '

    def f(rv):
        if False:
            while True:
                i = 10
        if not isinstance(rv, Expr):
            return rv

        def sincos_args(flat):
            if False:
                while True:
                    i = 10
            args = defaultdict(set)
            for fi in Mul.make_args(flat):
                (b, e) = fi.as_base_exp()
                if e.is_Integer and e > 0:
                    if b.func in (cos, sin):
                        args[type(b)].add(b.args[0])
            return args
        (num_args, den_args) = map(sincos_args, rv.as_numer_denom())

        def handle_match(rv, num_args, den_args):
            if False:
                for i in range(10):
                    print('nop')
            for narg in num_args[sin]:
                half = narg / 2
                if half in den_args[cos]:
                    func = cos
                elif half in den_args[sin]:
                    func = sin
                else:
                    continue
                rv = TR11(rv, half)
                den_args[func].remove(half)
            return rv
        rv = handle_match(rv, num_args, den_args)
        rv = handle_match(rv, den_args, num_args)
        return rv
    return bottom_up(rv, f)

def TR12(rv, first=True):
    if False:
        while True:
            i = 10
    'Separate sums in ``tan``.\n\n    Examples\n    ========\n\n    >>> from sympy.abc import x, y\n    >>> from sympy import tan\n    >>> from sympy.simplify.fu import TR12\n    >>> TR12(tan(x + y))\n    (tan(x) + tan(y))/(-tan(x)*tan(y) + 1)\n    '

    def f(rv):
        if False:
            while True:
                i = 10
        if not rv.func == tan:
            return rv
        arg = rv.args[0]
        if arg.is_Add:
            if first:
                args = list(ordered(arg.args))
            else:
                args = list(arg.args)
            a = args.pop()
            b = Add._from_args(args)
            if b.is_Add:
                tb = TR12(tan(b), first=False)
            else:
                tb = tan(b)
            return (tan(a) + tb) / (1 - tan(a) * tb)
        return rv
    return bottom_up(rv, f)

def TR12i(rv):
    if False:
        i = 10
        return i + 15
    'Combine tan arguments as\n    (tan(y) + tan(x))/(tan(x)*tan(y) - 1) -> -tan(x + y).\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.fu import TR12i\n    >>> from sympy import tan\n    >>> from sympy.abc import a, b, c\n    >>> ta, tb, tc = [tan(i) for i in (a, b, c)]\n    >>> TR12i((ta + tb)/(-ta*tb + 1))\n    tan(a + b)\n    >>> TR12i((ta + tb)/(ta*tb - 1))\n    -tan(a + b)\n    >>> TR12i((-ta - tb)/(ta*tb - 1))\n    tan(a + b)\n    >>> eq = (ta + tb)/(-ta*tb + 1)**2*(-3*ta - 3*tc)/(2*(ta*tc - 1))\n    >>> TR12i(eq.expand())\n    -3*tan(a + b)*tan(a + c)/(2*(tan(a) + tan(b) - 1))\n    '

    def f(rv):
        if False:
            return 10
        if not (rv.is_Add or rv.is_Mul or rv.is_Pow):
            return rv
        (n, d) = rv.as_numer_denom()
        if not d.args or not n.args:
            return rv
        dok = {}

        def ok(di):
            if False:
                print('Hello World!')
            m = as_f_sign_1(di)
            if m:
                (g, f, s) = m
                if s is S.NegativeOne and f.is_Mul and (len(f.args) == 2) and all((isinstance(fi, tan) for fi in f.args)):
                    return (g, f)
        d_args = list(Mul.make_args(d))
        for (i, di) in enumerate(d_args):
            m = ok(di)
            if m:
                (g, t) = m
                s = Add(*[_.args[0] for _ in t.args])
                dok[s] = S.One
                d_args[i] = g
                continue
            if di.is_Add:
                di = factor(di)
                if di.is_Mul:
                    d_args.extend(di.args)
                    d_args[i] = S.One
            elif di.is_Pow and (di.exp.is_integer or di.base.is_positive):
                m = ok(di.base)
                if m:
                    (g, t) = m
                    s = Add(*[_.args[0] for _ in t.args])
                    dok[s] = di.exp
                    d_args[i] = g ** di.exp
                else:
                    di = factor(di)
                    if di.is_Mul:
                        d_args.extend(di.args)
                        d_args[i] = S.One
        if not dok:
            return rv

        def ok(ni):
            if False:
                print('Hello World!')
            if ni.is_Add and len(ni.args) == 2:
                (a, b) = ni.args
                if isinstance(a, tan) and isinstance(b, tan):
                    return (a, b)
        n_args = list(Mul.make_args(factor_terms(n)))
        hit = False
        for (i, ni) in enumerate(n_args):
            m = ok(ni)
            if not m:
                m = ok(-ni)
                if m:
                    n_args[i] = S.NegativeOne
                elif ni.is_Add:
                    ni = factor(ni)
                    if ni.is_Mul:
                        n_args.extend(ni.args)
                        n_args[i] = S.One
                    continue
                elif ni.is_Pow and (ni.exp.is_integer or ni.base.is_positive):
                    m = ok(ni.base)
                    if m:
                        n_args[i] = S.One
                    else:
                        ni = factor(ni)
                        if ni.is_Mul:
                            n_args.extend(ni.args)
                            n_args[i] = S.One
                        continue
                else:
                    continue
            else:
                n_args[i] = S.One
            hit = True
            s = Add(*[_.args[0] for _ in m])
            ed = dok[s]
            newed = ed.extract_additively(S.One)
            if newed is not None:
                if newed:
                    dok[s] = newed
                else:
                    dok.pop(s)
            n_args[i] *= -tan(s)
        if hit:
            rv = Mul(*n_args) / Mul(*d_args) / Mul(*[(Add(*[tan(a) for a in i.args]) - 1) ** e for (i, e) in dok.items()])
        return rv
    return bottom_up(rv, f)

def TR13(rv):
    if False:
        return 10
    'Change products of ``tan`` or ``cot``.\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.fu import TR13\n    >>> from sympy import tan, cot\n    >>> TR13(tan(3)*tan(2))\n    -tan(2)/tan(5) - tan(3)/tan(5) + 1\n    >>> TR13(cot(3)*cot(2))\n    cot(2)*cot(5) + 1 + cot(3)*cot(5)\n    '

    def f(rv):
        if False:
            for i in range(10):
                print('nop')
        if not rv.is_Mul:
            return rv
        args = {tan: [], cot: [], None: []}
        for a in Mul.make_args(rv):
            if a.func in (tan, cot):
                args[type(a)].append(a.args[0])
            else:
                args[None].append(a)
        t = args[tan]
        c = args[cot]
        if len(t) < 2 and len(c) < 2:
            return rv
        args = args[None]
        while len(t) > 1:
            t1 = t.pop()
            t2 = t.pop()
            args.append(1 - (tan(t1) / tan(t1 + t2) + tan(t2) / tan(t1 + t2)))
        if t:
            args.append(tan(t.pop()))
        while len(c) > 1:
            t1 = c.pop()
            t2 = c.pop()
            args.append(1 + cot(t1) * cot(t1 + t2) + cot(t2) * cot(t1 + t2))
        if c:
            args.append(cot(c.pop()))
        return Mul(*args)
    return bottom_up(rv, f)

def TRmorrie(rv):
    if False:
        while True:
            i = 10
    'Returns cos(x)*cos(2*x)*...*cos(2**(k-1)*x) -> sin(2**k*x)/(2**k*sin(x))\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.fu import TRmorrie, TR8, TR3\n    >>> from sympy.abc import x\n    >>> from sympy import Mul, cos, pi\n    >>> TRmorrie(cos(x)*cos(2*x))\n    sin(4*x)/(4*sin(x))\n    >>> TRmorrie(7*Mul(*[cos(x) for x in range(10)]))\n    7*sin(12)*sin(16)*cos(5)*cos(7)*cos(9)/(64*sin(1)*sin(3))\n\n    Sometimes autosimplification will cause a power to be\n    not recognized. e.g. in the following, cos(4*pi/7) automatically\n    simplifies to -cos(3*pi/7) so only 2 of the 3 terms are\n    recognized:\n\n    >>> TRmorrie(cos(pi/7)*cos(2*pi/7)*cos(4*pi/7))\n    -sin(3*pi/7)*cos(3*pi/7)/(4*sin(pi/7))\n\n    A touch by TR8 resolves the expression to a Rational\n\n    >>> TR8(_)\n    -1/8\n\n    In this case, if eq is unsimplified, the answer is obtained\n    directly:\n\n    >>> eq = cos(pi/9)*cos(2*pi/9)*cos(3*pi/9)*cos(4*pi/9)\n    >>> TRmorrie(eq)\n    1/16\n\n    But if angles are made canonical with TR3 then the answer\n    is not simplified without further work:\n\n    >>> TR3(eq)\n    sin(pi/18)*cos(pi/9)*cos(2*pi/9)/2\n    >>> TRmorrie(_)\n    sin(pi/18)*sin(4*pi/9)/(8*sin(pi/9))\n    >>> TR8(_)\n    cos(7*pi/18)/(16*sin(pi/9))\n    >>> TR3(_)\n    1/16\n\n    The original expression would have resolve to 1/16 directly with TR8,\n    however:\n\n    >>> TR8(eq)\n    1/16\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Morrie%27s_law\n\n    '

    def f(rv, first=True):
        if False:
            i = 10
            return i + 15
        if not rv.is_Mul:
            return rv
        if first:
            (n, d) = rv.as_numer_denom()
            return f(n, 0) / f(d, 0)
        args = defaultdict(list)
        coss = {}
        other = []
        for c in rv.args:
            (b, e) = c.as_base_exp()
            if e.is_Integer and isinstance(b, cos):
                (co, a) = b.args[0].as_coeff_Mul()
                args[a].append(co)
                coss[b] = e
            else:
                other.append(c)
        new = []
        for a in args:
            c = args[a]
            c.sort()
            while c:
                k = 0
                cc = ci = c[0]
                while cc in c:
                    k += 1
                    cc *= 2
                if k > 1:
                    newarg = sin(2 ** k * ci * a) / 2 ** k / sin(ci * a)
                    take = None
                    ccs = []
                    for i in range(k):
                        cc /= 2
                        key = cos(a * cc, evaluate=False)
                        ccs.append(cc)
                        take = min(coss[key], take or coss[key])
                    for i in range(k):
                        cc = ccs.pop()
                        key = cos(a * cc, evaluate=False)
                        coss[key] -= take
                        if not coss[key]:
                            c.remove(cc)
                    new.append(newarg ** take)
                else:
                    b = cos(c.pop(0) * a)
                    other.append(b ** coss[b])
        if new:
            rv = Mul(*new + other + [cos(k * a, evaluate=False) for a in args for k in args[a]])
        return rv
    return bottom_up(rv, f)

def TR14(rv, first=True):
    if False:
        while True:
            i = 10
    'Convert factored powers of sin and cos identities into simpler\n    expressions.\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.fu import TR14\n    >>> from sympy.abc import x, y\n    >>> from sympy import cos, sin\n    >>> TR14((cos(x) - 1)*(cos(x) + 1))\n    -sin(x)**2\n    >>> TR14((sin(x) - 1)*(sin(x) + 1))\n    -cos(x)**2\n    >>> p1 = (cos(x) + 1)*(cos(x) - 1)\n    >>> p2 = (cos(y) - 1)*2*(cos(y) + 1)\n    >>> p3 = (3*(cos(y) - 1))*(3*(cos(y) + 1))\n    >>> TR14(p1*p2*p3*(x - 1))\n    -18*(x - 1)*sin(x)**2*sin(y)**4\n\n    '

    def f(rv):
        if False:
            print('Hello World!')
        if not rv.is_Mul:
            return rv
        if first:
            (n, d) = rv.as_numer_denom()
            if d is not S.One:
                newn = TR14(n, first=False)
                newd = TR14(d, first=False)
                if newn != n or newd != d:
                    rv = newn / newd
                return rv
        other = []
        process = []
        for a in rv.args:
            if a.is_Pow:
                (b, e) = a.as_base_exp()
                if not (e.is_integer or b.is_positive):
                    other.append(a)
                    continue
                a = b
            else:
                e = S.One
            m = as_f_sign_1(a)
            if not m or m[1].func not in (cos, sin):
                if e is S.One:
                    other.append(a)
                else:
                    other.append(a ** e)
                continue
            (g, f, si) = m
            process.append((g, e.is_Number, e, f, si, a))
        process = list(ordered(process))
        nother = len(other)
        keys = (g, t, e, f, si, a) = list(range(6))
        while process:
            A = process.pop(0)
            if process:
                B = process[0]
                if A[e].is_Number and B[e].is_Number:
                    if A[f] == B[f]:
                        if A[si] != B[si]:
                            B = process.pop(0)
                            take = min(A[e], B[e])
                            if B[e] != take:
                                rem = [B[i] for i in keys]
                                rem[e] -= take
                                process.insert(0, rem)
                            elif A[e] != take:
                                rem = [A[i] for i in keys]
                                rem[e] -= take
                                process.insert(0, rem)
                            if isinstance(A[f], cos):
                                t = sin
                            else:
                                t = cos
                            other.append((-A[g] * B[g] * t(A[f].args[0]) ** 2) ** take)
                            continue
                elif A[e] == B[e]:
                    if A[f] == B[f]:
                        if A[si] != B[si]:
                            B = process.pop(0)
                            take = A[e]
                            if isinstance(A[f], cos):
                                t = sin
                            else:
                                t = cos
                            other.append((-A[g] * B[g] * t(A[f].args[0]) ** 2) ** take)
                            continue
            other.append(A[a] ** A[e])
        if len(other) != nother:
            rv = Mul(*other)
        return rv
    return bottom_up(rv, f)

def TR15(rv, max=4, pow=False):
    if False:
        i = 10
        return i + 15
    'Convert sin(x)**-2 to 1 + cot(x)**2.\n\n    See _TR56 docstring for advanced use of ``max`` and ``pow``.\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.fu import TR15\n    >>> from sympy.abc import x\n    >>> from sympy import sin\n    >>> TR15(1 - 1/sin(x)**2)\n    -cot(x)**2\n\n    '

    def f(rv):
        if False:
            print('Hello World!')
        if not (isinstance(rv, Pow) and isinstance(rv.base, sin)):
            return rv
        e = rv.exp
        if e % 2 == 1:
            return TR15(rv.base ** (e + 1)) / rv.base
        ia = 1 / rv
        a = _TR56(ia, sin, cot, lambda x: 1 + x, max=max, pow=pow)
        if a != ia:
            rv = a
        return rv
    return bottom_up(rv, f)

def TR16(rv, max=4, pow=False):
    if False:
        while True:
            i = 10
    'Convert cos(x)**-2 to 1 + tan(x)**2.\n\n    See _TR56 docstring for advanced use of ``max`` and ``pow``.\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.fu import TR16\n    >>> from sympy.abc import x\n    >>> from sympy import cos\n    >>> TR16(1 - 1/cos(x)**2)\n    -tan(x)**2\n\n    '

    def f(rv):
        if False:
            print('Hello World!')
        if not (isinstance(rv, Pow) and isinstance(rv.base, cos)):
            return rv
        e = rv.exp
        if e % 2 == 1:
            return TR15(rv.base ** (e + 1)) / rv.base
        ia = 1 / rv
        a = _TR56(ia, cos, tan, lambda x: 1 + x, max=max, pow=pow)
        if a != ia:
            rv = a
        return rv
    return bottom_up(rv, f)

def TR111(rv):
    if False:
        i = 10
        return i + 15
    'Convert f(x)**-i to g(x)**i where either ``i`` is an integer\n    or the base is positive and f, g are: tan, cot; sin, csc; or cos, sec.\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.fu import TR111\n    >>> from sympy.abc import x\n    >>> from sympy import tan\n    >>> TR111(1 - 1/tan(x)**2)\n    1 - cot(x)**2\n\n    '

    def f(rv):
        if False:
            while True:
                i = 10
        if not (isinstance(rv, Pow) and (rv.base.is_positive or (rv.exp.is_integer and rv.exp.is_negative))):
            return rv
        if isinstance(rv.base, tan):
            return cot(rv.base.args[0]) ** (-rv.exp)
        elif isinstance(rv.base, sin):
            return csc(rv.base.args[0]) ** (-rv.exp)
        elif isinstance(rv.base, cos):
            return sec(rv.base.args[0]) ** (-rv.exp)
        return rv
    return bottom_up(rv, f)

def TR22(rv, max=4, pow=False):
    if False:
        while True:
            i = 10
    'Convert tan(x)**2 to sec(x)**2 - 1 and cot(x)**2 to csc(x)**2 - 1.\n\n    See _TR56 docstring for advanced use of ``max`` and ``pow``.\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.fu import TR22\n    >>> from sympy.abc import x\n    >>> from sympy import tan, cot\n    >>> TR22(1 + tan(x)**2)\n    sec(x)**2\n    >>> TR22(1 + cot(x)**2)\n    csc(x)**2\n\n    '

    def f(rv):
        if False:
            while True:
                i = 10
        if not (isinstance(rv, Pow) and rv.base.func in (cot, tan)):
            return rv
        rv = _TR56(rv, tan, sec, lambda x: x - 1, max=max, pow=pow)
        rv = _TR56(rv, cot, csc, lambda x: x - 1, max=max, pow=pow)
        return rv
    return bottom_up(rv, f)

def TRpower(rv):
    if False:
        i = 10
        return i + 15
    'Convert sin(x)**n and cos(x)**n with positive n to sums.\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.fu import TRpower\n    >>> from sympy.abc import x\n    >>> from sympy import cos, sin\n    >>> TRpower(sin(x)**6)\n    -15*cos(2*x)/32 + 3*cos(4*x)/16 - cos(6*x)/32 + 5/16\n    >>> TRpower(sin(x)**3*cos(2*x)**4)\n    (3*sin(x)/4 - sin(3*x)/4)*(cos(4*x)/2 + cos(8*x)/8 + 3/8)\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/List_of_trigonometric_identities#Power-reduction_formulae\n\n    '

    def f(rv):
        if False:
            while True:
                i = 10
        if not (isinstance(rv, Pow) and isinstance(rv.base, (sin, cos))):
            return rv
        (b, n) = rv.as_base_exp()
        x = b.args[0]
        if n.is_Integer and n.is_positive:
            if n.is_odd and isinstance(b, cos):
                rv = 2 ** (1 - n) * Add(*[binomial(n, k) * cos((n - 2 * k) * x) for k in range((n + 1) / 2)])
            elif n.is_odd and isinstance(b, sin):
                rv = 2 ** (1 - n) * S.NegativeOne ** ((n - 1) / 2) * Add(*[binomial(n, k) * S.NegativeOne ** k * sin((n - 2 * k) * x) for k in range((n + 1) / 2)])
            elif n.is_even and isinstance(b, cos):
                rv = 2 ** (1 - n) * Add(*[binomial(n, k) * cos((n - 2 * k) * x) for k in range(n / 2)])
            elif n.is_even and isinstance(b, sin):
                rv = 2 ** (1 - n) * S.NegativeOne ** (n / 2) * Add(*[binomial(n, k) * S.NegativeOne ** k * cos((n - 2 * k) * x) for k in range(n / 2)])
            if n.is_even:
                rv += 2 ** (-n) * binomial(n, n / 2)
        return rv
    return bottom_up(rv, f)

def L(rv):
    if False:
        i = 10
        return i + 15
    'Return count of trigonometric functions in expression.\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.fu import L\n    >>> from sympy.abc import x\n    >>> from sympy import cos, sin\n    >>> L(cos(x)+sin(x))\n    2\n    '
    return S(rv.count(TrigonometricFunction))
if SYMPY_DEBUG:
    (TR0, TR1, TR2, TR3, TR4, TR5, TR6, TR7, TR8, TR9, TR10, TR11, TR12, TR13, TR2i, TRmorrie, TR14, TR15, TR16, TR12i, TR111, TR22) = list(map(debug, (TR0, TR1, TR2, TR3, TR4, TR5, TR6, TR7, TR8, TR9, TR10, TR11, TR12, TR13, TR2i, TRmorrie, TR14, TR15, TR16, TR12i, TR111, TR22)))
CTR1 = [(TR5, TR0), (TR6, TR0), identity]
CTR2 = (TR11, [(TR5, TR0), (TR6, TR0), TR0])
CTR3 = [(TRmorrie, TR8, TR0), (TRmorrie, TR8, TR10i, TR0), identity]
CTR4 = [(TR4, TR10i), identity]
RL1 = (TR4, TR3, TR4, TR12, TR4, TR13, TR4, TR0)
RL2 = [(TR4, TR3, TR10, TR4, TR3, TR11), (TR5, TR7, TR11, TR4), (CTR3, CTR1, TR9, CTR2, TR4, TR9, TR9, CTR4), identity]

def fu(rv, measure=lambda x: (L(x), x.count_ops())):
    if False:
        while True:
            i = 10
    'Attempt to simplify expression by using transformation rules given\n    in the algorithm by Fu et al.\n\n    :func:`fu` will try to minimize the objective function ``measure``.\n    By default this first minimizes the number of trig terms and then minimizes\n    the number of total operations.\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.fu import fu\n    >>> from sympy import cos, sin, tan, pi, S, sqrt\n    >>> from sympy.abc import x, y, a, b\n\n    >>> fu(sin(50)**2 + cos(50)**2 + sin(pi/6))\n    3/2\n    >>> fu(sqrt(6)*cos(x) + sqrt(2)*sin(x))\n    2*sqrt(2)*sin(x + pi/3)\n\n    CTR1 example\n\n    >>> eq = sin(x)**4 - cos(y)**2 + sin(y)**2 + 2*cos(x)**2\n    >>> fu(eq)\n    cos(x)**4 - 2*cos(y)**2 + 2\n\n    CTR2 example\n\n    >>> fu(S.Half - cos(2*x)/2)\n    sin(x)**2\n\n    CTR3 example\n\n    >>> fu(sin(a)*(cos(b) - sin(b)) + cos(a)*(sin(b) + cos(b)))\n    sqrt(2)*sin(a + b + pi/4)\n\n    CTR4 example\n\n    >>> fu(sqrt(3)*cos(x)/2 + sin(x)/2)\n    sin(x + pi/3)\n\n    Example 1\n\n    >>> fu(1-sin(2*x)**2/4-sin(y)**2-cos(x)**4)\n    -cos(x)**2 + cos(y)**2\n\n    Example 2\n\n    >>> fu(cos(4*pi/9))\n    sin(pi/18)\n    >>> fu(cos(pi/9)*cos(2*pi/9)*cos(3*pi/9)*cos(4*pi/9))\n    1/16\n\n    Example 3\n\n    >>> fu(tan(7*pi/18)+tan(5*pi/18)-sqrt(3)*tan(5*pi/18)*tan(7*pi/18))\n    -sqrt(3)\n\n    Objective function example\n\n    >>> fu(sin(x)/cos(x))  # default objective function\n    tan(x)\n    >>> fu(sin(x)/cos(x), measure=lambda x: -x.count_ops()) # maximize op count\n    sin(x)/cos(x)\n\n    References\n    ==========\n\n    .. [1] https://www.sciencedirect.com/science/article/pii/S0895717706001609\n    '
    fRL1 = greedy(RL1, measure)
    fRL2 = greedy(RL2, measure)
    was = rv
    rv = sympify(rv)
    if not isinstance(rv, Expr):
        return rv.func(*[fu(a, measure=measure) for a in rv.args])
    rv = TR1(rv)
    if rv.has(tan, cot):
        rv1 = fRL1(rv)
        if measure(rv1) < measure(rv):
            rv = rv1
        if rv.has(tan, cot):
            rv = TR2(rv)
    if rv.has(sin, cos):
        rv1 = fRL2(rv)
        rv2 = TR8(TRmorrie(rv1))
        rv = min([was, rv, rv1, rv2], key=measure)
    return min(TR2i(rv), rv, key=measure)

def process_common_addends(rv, do, key2=None, key1=True):
    if False:
        return 10
    'Apply ``do`` to addends of ``rv`` that (if ``key1=True``) share at least\n    a common absolute value of their coefficient and the value of ``key2`` when\n    applied to the argument. If ``key1`` is False ``key2`` must be supplied and\n    will be the only key applied.\n    '
    absc = defaultdict(list)
    if key1:
        for a in rv.args:
            (c, a) = a.as_coeff_Mul()
            if c < 0:
                c = -c
                a = -a
            absc[c, key2(a) if key2 else 1].append(a)
    elif key2:
        for a in rv.args:
            absc[S.One, key2(a)].append(a)
    else:
        raise ValueError('must have at least one key')
    args = []
    hit = False
    for k in absc:
        v = absc[k]
        (c, _) = k
        if len(v) > 1:
            e = Add(*v, evaluate=False)
            new = do(e)
            if new != e:
                e = new
                hit = True
            args.append(c * e)
        else:
            args.append(c * v[0])
    if hit:
        rv = Add(*args)
    return rv
fufuncs = '\n    TR0 TR1 TR2 TR3 TR4 TR5 TR6 TR7 TR8 TR9 TR10 TR10i TR11\n    TR12 TR13 L TR2i TRmorrie TR12i\n    TR14 TR15 TR16 TR111 TR22'.split()
FU = dict(list(zip(fufuncs, list(map(locals().get, fufuncs)))))

def _roots():
    if False:
        return 10
    global _ROOT2, _ROOT3, _invROOT3
    (_ROOT2, _ROOT3) = (sqrt(2), sqrt(3))
    _invROOT3 = 1 / _ROOT3
_ROOT2 = None

def trig_split(a, b, two=False):
    if False:
        return 10
    'Return the gcd, s1, s2, a1, a2, bool where\n\n    If two is False (default) then::\n        a + b = gcd*(s1*f(a1) + s2*f(a2)) where f = cos if bool else sin\n    else:\n        if bool, a + b was +/- cos(a1)*cos(a2) +/- sin(a1)*sin(a2) and equals\n            n1*gcd*cos(a - b) if n1 == n2 else\n            n1*gcd*cos(a + b)\n        else a + b was +/- cos(a1)*sin(a2) +/- sin(a1)*cos(a2) and equals\n            n1*gcd*sin(a + b) if n1 = n2 else\n            n1*gcd*sin(b - a)\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.fu import trig_split\n    >>> from sympy.abc import x, y, z\n    >>> from sympy import cos, sin, sqrt\n\n    >>> trig_split(cos(x), cos(y))\n    (1, 1, 1, x, y, True)\n    >>> trig_split(2*cos(x), -2*cos(y))\n    (2, 1, -1, x, y, True)\n    >>> trig_split(cos(x)*sin(y), cos(y)*sin(y))\n    (sin(y), 1, 1, x, y, True)\n\n    >>> trig_split(cos(x), -sqrt(3)*sin(x), two=True)\n    (2, 1, -1, x, pi/6, False)\n    >>> trig_split(cos(x), sin(x), two=True)\n    (sqrt(2), 1, 1, x, pi/4, False)\n    >>> trig_split(cos(x), -sin(x), two=True)\n    (sqrt(2), 1, -1, x, pi/4, False)\n    >>> trig_split(sqrt(2)*cos(x), -sqrt(6)*sin(x), two=True)\n    (2*sqrt(2), 1, -1, x, pi/6, False)\n    >>> trig_split(-sqrt(6)*cos(x), -sqrt(2)*sin(x), two=True)\n    (-2*sqrt(2), 1, 1, x, pi/3, False)\n    >>> trig_split(cos(x)/sqrt(6), sin(x)/sqrt(2), two=True)\n    (sqrt(6)/3, 1, 1, x, pi/6, False)\n    >>> trig_split(-sqrt(6)*cos(x)*sin(y), -sqrt(2)*sin(x)*sin(y), two=True)\n    (-2*sqrt(2)*sin(y), 1, 1, x, pi/3, False)\n\n    >>> trig_split(cos(x), sin(x))\n    >>> trig_split(cos(x), sin(z))\n    >>> trig_split(2*cos(x), -sin(x))\n    >>> trig_split(cos(x), -sqrt(3)*sin(x))\n    >>> trig_split(cos(x)*cos(y), sin(x)*sin(z))\n    >>> trig_split(cos(x)*cos(y), sin(x)*sin(y))\n    >>> trig_split(-sqrt(6)*cos(x), sqrt(2)*sin(x)*sin(y), two=True)\n    '
    global _ROOT2, _ROOT3, _invROOT3
    if _ROOT2 is None:
        _roots()
    (a, b) = [Factors(i) for i in (a, b)]
    (ua, ub) = a.normal(b)
    gcd = a.gcd(b).as_expr()
    n1 = n2 = 1
    if S.NegativeOne in ua.factors:
        ua = ua.quo(S.NegativeOne)
        n1 = -n1
    elif S.NegativeOne in ub.factors:
        ub = ub.quo(S.NegativeOne)
        n2 = -n2
    (a, b) = [i.as_expr() for i in (ua, ub)]

    def pow_cos_sin(a, two):
        if False:
            print('Hello World!')
        'Return ``a`` as a tuple (r, c, s) such that\n        ``a = (r or 1)*(c or 1)*(s or 1)``.\n\n        Three arguments are returned (radical, c-factor, s-factor) as\n        long as the conditions set by ``two`` are met; otherwise None is\n        returned. If ``two`` is True there will be one or two non-None\n        values in the tuple: c and s or c and r or s and r or s or c with c\n        being a cosine function (if possible) else a sine, and s being a sine\n        function (if possible) else oosine. If ``two`` is False then there\n        will only be a c or s term in the tuple.\n\n        ``two`` also require that either two cos and/or sin be present (with\n        the condition that if the functions are the same the arguments are\n        different or vice versa) or that a single cosine or a single sine\n        be present with an optional radical.\n\n        If the above conditions dictated by ``two`` are not met then None\n        is returned.\n        '
        c = s = None
        co = S.One
        if a.is_Mul:
            (co, a) = a.as_coeff_Mul()
            if len(a.args) > 2 or not two:
                return None
            if a.is_Mul:
                args = list(a.args)
            else:
                args = [a]
            a = args.pop(0)
            if isinstance(a, cos):
                c = a
            elif isinstance(a, sin):
                s = a
            elif a.is_Pow and a.exp is S.Half:
                co *= a
            else:
                return None
            if args:
                b = args[0]
                if isinstance(b, cos):
                    if c:
                        s = b
                    else:
                        c = b
                elif isinstance(b, sin):
                    if s:
                        c = b
                    else:
                        s = b
                elif b.is_Pow and b.exp is S.Half:
                    co *= b
                else:
                    return None
            return (co if co is not S.One else None, c, s)
        elif isinstance(a, cos):
            c = a
        elif isinstance(a, sin):
            s = a
        if c is None and s is None:
            return
        co = co if co is not S.One else None
        return (co, c, s)
    m = pow_cos_sin(a, two)
    if m is None:
        return
    (coa, ca, sa) = m
    m = pow_cos_sin(b, two)
    if m is None:
        return
    (cob, cb, sb) = m
    if not ca and cb or (ca and isinstance(ca, sin)):
        (coa, ca, sa, cob, cb, sb) = (cob, cb, sb, coa, ca, sa)
        (n1, n2) = (n2, n1)
    if not two:
        c = ca or sa
        s = cb or sb
        if not isinstance(c, s.func):
            return None
        return (gcd, n1, n2, c.args[0], s.args[0], isinstance(c, cos))
    else:
        if not coa and (not cob):
            if ca and cb and sa and sb:
                if isinstance(ca, sa.func) is not isinstance(cb, sb.func):
                    return
                args = {j.args for j in (ca, sa)}
                if not all((i.args in args for i in (cb, sb))):
                    return
                return (gcd, n1, n2, ca.args[0], sa.args[0], isinstance(ca, sa.func))
        if ca and sa or (cb and sb) or (two and (ca is None and sa is None or (cb is None and sb is None))):
            return
        c = ca or sa
        s = cb or sb
        if c.args != s.args:
            return
        if not coa:
            coa = S.One
        if not cob:
            cob = S.One
        if coa is cob:
            gcd *= _ROOT2
            return (gcd, n1, n2, c.args[0], pi / 4, False)
        elif coa / cob == _ROOT3:
            gcd *= 2 * cob
            return (gcd, n1, n2, c.args[0], pi / 3, False)
        elif coa / cob == _invROOT3:
            gcd *= 2 * coa
            return (gcd, n1, n2, c.args[0], pi / 6, False)

def as_f_sign_1(e):
    if False:
        i = 10
        return i + 15
    'If ``e`` is a sum that can be written as ``g*(a + s)`` where\n    ``s`` is ``+/-1``, return ``g``, ``a``, and ``s`` where ``a`` does\n    not have a leading negative coefficient.\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.fu import as_f_sign_1\n    >>> from sympy.abc import x\n    >>> as_f_sign_1(x + 1)\n    (1, x, 1)\n    >>> as_f_sign_1(x - 1)\n    (1, x, -1)\n    >>> as_f_sign_1(-x + 1)\n    (-1, x, -1)\n    >>> as_f_sign_1(-x - 1)\n    (-1, x, 1)\n    >>> as_f_sign_1(2*x + 2)\n    (2, x, 1)\n    '
    if not e.is_Add or len(e.args) != 2:
        return
    (a, b) = e.args
    if a in (S.NegativeOne, S.One):
        g = S.One
        if b.is_Mul and b.args[0].is_Number and (b.args[0] < 0):
            (a, b) = (-a, -b)
            g = -g
        return (g, b, a)
    (a, b) = [Factors(i) for i in e.args]
    (ua, ub) = a.normal(b)
    gcd = a.gcd(b).as_expr()
    if S.NegativeOne in ua.factors:
        ua = ua.quo(S.NegativeOne)
        n1 = -1
        n2 = 1
    elif S.NegativeOne in ub.factors:
        ub = ub.quo(S.NegativeOne)
        n1 = 1
        n2 = -1
    else:
        n1 = n2 = 1
    (a, b) = [i.as_expr() for i in (ua, ub)]
    if a is S.One:
        (a, b) = (b, a)
        (n1, n2) = (n2, n1)
    if n1 == -1:
        gcd = -gcd
        n2 = -n2
    if b is S.One:
        return (gcd, a, n2)

def _osborne(e, d):
    if False:
        print('Hello World!')
    'Replace all hyperbolic functions with trig functions using\n    the Osborne rule.\n\n    Notes\n    =====\n\n    ``d`` is a dummy variable to prevent automatic evaluation\n    of trigonometric/hyperbolic functions.\n\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Hyperbolic_function\n    '

    def f(rv):
        if False:
            while True:
                i = 10
        if not isinstance(rv, HyperbolicFunction):
            return rv
        a = rv.args[0]
        a = a * d if not a.is_Add else Add._from_args([i * d for i in a.args])
        if isinstance(rv, sinh):
            return I * sin(a)
        elif isinstance(rv, cosh):
            return cos(a)
        elif isinstance(rv, tanh):
            return I * tan(a)
        elif isinstance(rv, coth):
            return cot(a) / I
        elif isinstance(rv, sech):
            return sec(a)
        elif isinstance(rv, csch):
            return csc(a) / I
        else:
            raise NotImplementedError('unhandled %s' % rv.func)
    return bottom_up(e, f)

def _osbornei(e, d):
    if False:
        i = 10
        return i + 15
    'Replace all trig functions with hyperbolic functions using\n    the Osborne rule.\n\n    Notes\n    =====\n\n    ``d`` is a dummy variable to prevent automatic evaluation\n    of trigonometric/hyperbolic functions.\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Hyperbolic_function\n    '

    def f(rv):
        if False:
            print('Hello World!')
        if not isinstance(rv, TrigonometricFunction):
            return rv
        (const, x) = rv.args[0].as_independent(d, as_Add=True)
        a = x.xreplace({d: S.One}) + const * I
        if isinstance(rv, sin):
            return sinh(a) / I
        elif isinstance(rv, cos):
            return cosh(a)
        elif isinstance(rv, tan):
            return tanh(a) / I
        elif isinstance(rv, cot):
            return coth(a) * I
        elif isinstance(rv, sec):
            return sech(a)
        elif isinstance(rv, csc):
            return csch(a) * I
        else:
            raise NotImplementedError('unhandled %s' % rv.func)
    return bottom_up(e, f)

def hyper_as_trig(rv):
    if False:
        print('Hello World!')
    'Return an expression containing hyperbolic functions in terms\n    of trigonometric functions. Any trigonometric functions initially\n    present are replaced with Dummy symbols and the function to undo\n    the masking and the conversion back to hyperbolics is also returned. It\n    should always be true that::\n\n        t, f = hyper_as_trig(expr)\n        expr == f(t)\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.fu import hyper_as_trig, fu\n    >>> from sympy.abc import x\n    >>> from sympy import cosh, sinh\n    >>> eq = sinh(x)**2 + cosh(x)**2\n    >>> t, f = hyper_as_trig(eq)\n    >>> f(fu(t))\n    cosh(2*x)\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Hyperbolic_function\n    '
    from sympy.simplify.simplify import signsimp
    from sympy.simplify.radsimp import collect
    trigs = rv.atoms(TrigonometricFunction)
    reps = [(t, Dummy()) for t in trigs]
    masked = rv.xreplace(dict(reps))
    reps = [(v, k) for (k, v) in reps]
    d = Dummy()
    return (_osborne(masked, d), lambda x: collect(signsimp(_osbornei(x, d).xreplace(dict(reps))), S.ImaginaryUnit))

def sincos_to_sum(expr):
    if False:
        return 10
    'Convert products and powers of sin and cos to sums.\n\n    Explanation\n    ===========\n\n    Applied power reduction TRpower first, then expands products, and\n    converts products to sums with TR8.\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.fu import sincos_to_sum\n    >>> from sympy.abc import x\n    >>> from sympy import cos, sin\n    >>> sincos_to_sum(16*sin(x)**3*cos(2*x)**2)\n    7*sin(x) - 5*sin(3*x) + 3*sin(5*x) - sin(7*x)\n    '
    if not expr.has(cos, sin):
        return expr
    else:
        return TR8(expand_mul(TRpower(expr)))