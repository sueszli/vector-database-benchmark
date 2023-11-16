from collections import defaultdict
from functools import reduce
from math import prod
from sympy.core.function import expand_log, count_ops, _coeff_isneg
from sympy.core import sympify, Basic, Dummy, S, Add, Mul, Pow, expand_mul, factor_terms
from sympy.core.sorting import ordered, default_sort_key
from sympy.core.numbers import Integer, Rational
from sympy.core.mul import _keep_coeff
from sympy.core.rules import Transform
from sympy.functions import exp_polar, exp, log, root, polarify, unpolarify
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.polys import lcm, gcd
from sympy.ntheory.factor_ import multiplicity

def powsimp(expr, deep=False, combine='all', force=False, measure=count_ops):
    if False:
        for i in range(10):
            print('nop')
    "\n    Reduce expression by combining powers with similar bases and exponents.\n\n    Explanation\n    ===========\n\n    If ``deep`` is ``True`` then powsimp() will also simplify arguments of\n    functions. By default ``deep`` is set to ``False``.\n\n    If ``force`` is ``True`` then bases will be combined without checking for\n    assumptions, e.g. sqrt(x)*sqrt(y) -> sqrt(x*y) which is not true\n    if x and y are both negative.\n\n    You can make powsimp() only combine bases or only combine exponents by\n    changing combine='base' or combine='exp'.  By default, combine='all',\n    which does both.  combine='base' will only combine::\n\n         a   a          a                          2x      x\n        x * y  =>  (x*y)   as well as things like 2   =>  4\n\n    and combine='exp' will only combine\n    ::\n\n         a   b      (a + b)\n        x * x  =>  x\n\n    combine='exp' will strictly only combine exponents in the way that used\n    to be automatic.  Also use deep=True if you need the old behavior.\n\n    When combine='all', 'exp' is evaluated first.  Consider the first\n    example below for when there could be an ambiguity relating to this.\n    This is done so things like the second example can be completely\n    combined.  If you want 'base' combined first, do something like\n    powsimp(powsimp(expr, combine='base'), combine='exp').\n\n    Examples\n    ========\n\n    >>> from sympy import powsimp, exp, log, symbols\n    >>> from sympy.abc import x, y, z, n\n    >>> powsimp(x**y*x**z*y**z, combine='all')\n    x**(y + z)*y**z\n    >>> powsimp(x**y*x**z*y**z, combine='exp')\n    x**(y + z)*y**z\n    >>> powsimp(x**y*x**z*y**z, combine='base', force=True)\n    x**y*(x*y)**z\n\n    >>> powsimp(x**z*x**y*n**z*n**y, combine='all', force=True)\n    (n*x)**(y + z)\n    >>> powsimp(x**z*x**y*n**z*n**y, combine='exp')\n    n**(y + z)*x**(y + z)\n    >>> powsimp(x**z*x**y*n**z*n**y, combine='base', force=True)\n    (n*x)**y*(n*x)**z\n\n    >>> x, y = symbols('x y', positive=True)\n    >>> powsimp(log(exp(x)*exp(y)))\n    log(exp(x)*exp(y))\n    >>> powsimp(log(exp(x)*exp(y)), deep=True)\n    x + y\n\n    Radicals with Mul bases will be combined if combine='exp'\n\n    >>> from sympy import sqrt\n    >>> x, y = symbols('x y')\n\n    Two radicals are automatically joined through Mul:\n\n    >>> a=sqrt(x*sqrt(y))\n    >>> a*a**3 == a**4\n    True\n\n    But if an integer power of that radical has been\n    autoexpanded then Mul does not join the resulting factors:\n\n    >>> a**4 # auto expands to a Mul, no longer a Pow\n    x**2*y\n    >>> _*a # so Mul doesn't combine them\n    x**2*y*sqrt(x*sqrt(y))\n    >>> powsimp(_) # but powsimp will\n    (x*sqrt(y))**(5/2)\n    >>> powsimp(x*y*a) # but won't when doing so would violate assumptions\n    x*y*sqrt(x*sqrt(y))\n\n    "

    def recurse(arg, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        _deep = kwargs.get('deep', deep)
        _combine = kwargs.get('combine', combine)
        _force = kwargs.get('force', force)
        _measure = kwargs.get('measure', measure)
        return powsimp(arg, _deep, _combine, _force, _measure)
    expr = sympify(expr)
    if not isinstance(expr, Basic) or isinstance(expr, MatrixSymbol) or (expr.is_Atom or expr in (exp_polar(0), exp_polar(1))):
        return expr
    if deep or expr.is_Add or (expr.is_Mul and _y not in expr.args):
        expr = expr.func(*[recurse(w) for w in expr.args])
    if expr.is_Pow:
        return recurse(expr * _y, deep=False) / _y
    if not expr.is_Mul:
        return expr
    if combine in ('exp', 'all'):
        c_powers = defaultdict(list)
        nc_part = []
        newexpr = []
        coeff = S.One
        for term in expr.args:
            if term.is_Rational:
                coeff *= term
                continue
            if term.is_Pow:
                term = _denest_pow(term)
            if term.is_commutative:
                (b, e) = term.as_base_exp()
                if deep:
                    (b, e) = [recurse(i) for i in [b, e]]
                if b.is_Pow or isinstance(b, exp):
                    (b, e) = (b ** e, S.One)
                c_powers[b].append(e)
            else:
                if nc_part:
                    (b1, e1) = nc_part[-1].as_base_exp()
                    (b2, e2) = term.as_base_exp()
                    if b1 == b2 and e1.is_commutative and e2.is_commutative:
                        nc_part[-1] = Pow(b1, Add(e1, e2))
                        continue
                nc_part.append(term)
        for (b, e) in ordered(iter(c_powers.items())):
            if b and b.is_Rational and (not all((ei.is_Number for ei in e))) and (coeff is not S.One) and (b not in (S.One, S.NegativeOne)):
                m = multiplicity(abs(b), abs(coeff))
                if m:
                    e.append(m)
                    coeff /= b ** m
            c_powers[b] = Add(*e)
        if coeff is not S.One:
            if coeff in c_powers:
                c_powers[coeff] += S.One
            else:
                c_powers[coeff] = S.One
        c_powers = dict(c_powers)
        be = list(c_powers.items())
        skip = set()
        for (b, e) in be:
            if b in skip:
                continue
            bpos = b.is_positive or b.is_polar
            if bpos:
                binv = 1 / b
                if b != binv and binv in c_powers:
                    if b.as_numer_denom()[0] is S.One:
                        c_powers.pop(b)
                        c_powers[binv] -= e
                    else:
                        skip.add(binv)
                        e = c_powers.pop(binv)
                        c_powers[b] -= e
        be = list(c_powers.items())
        _n = S.NegativeOne
        for (b, e) in be:
            if (b.is_Symbol or b.is_Add) and -b in c_powers and (b in c_powers):
                if b.is_positive is not None or e.is_integer:
                    if e.is_integer or b.is_negative:
                        c_powers[-b] += c_powers.pop(b)
                    else:
                        e = c_powers.pop(-b)
                        c_powers[b] += e
                    if _n in c_powers:
                        c_powers[_n] += e
                    else:
                        c_powers[_n] = e
        c_powers = [(b, e) for (b, e) in c_powers.items() if e]

        def ratq(x):
            if False:
                return 10
            "Return Rational part of x's exponent as it appears in the bkey.\n            "
            return bkey(x)[0][1]

        def bkey(b, e=None):
            if False:
                return 10
            'Return (b**s, c.q), c.p where e -> c*s. If e is not given then\n            it will be taken by using as_base_exp() on the input b.\n            e.g.\n                x**3/2 -> (x, 2), 3\n                x**y -> (x**y, 1), 1\n                x**(2*y/3) -> (x**y, 3), 2\n                exp(x/2) -> (exp(a), 2), 1\n\n            '
            if e is not None:
                if e.is_Integer:
                    return ((b, S.One), e)
                elif e.is_Rational:
                    return ((b, Integer(e.q)), Integer(e.p))
                else:
                    (c, m) = e.as_coeff_Mul(rational=True)
                    if c is not S.One:
                        if m.is_integer:
                            return ((b, Integer(c.q)), m * Integer(c.p))
                        return ((b ** m, Integer(c.q)), Integer(c.p))
                    else:
                        return ((b ** e, S.One), S.One)
            else:
                return bkey(*b.as_base_exp())

        def update(b):
            if False:
                i = 10
                return i + 15
            'Decide what to do with base, b. If its exponent is now an\n            integer multiple of the Rational denominator, then remove it\n            and put the factors of its base in the common_b dictionary or\n            update the existing bases if necessary. If it has been zeroed\n            out, simply remove the base.\n            '
            (newe, r) = divmod(common_b[b], b[1])
            if not r:
                common_b.pop(b)
                if newe:
                    for m in Mul.make_args(b[0] ** newe):
                        (b, e) = bkey(m)
                        if b not in common_b:
                            common_b[b] = 0
                        common_b[b] += e
                        if b[1] != 1:
                            bases.append(b)
        common_b = {}
        done = []
        bases = []
        for (b, e) in c_powers:
            (b, e) = bkey(b, e)
            if b in common_b:
                common_b[b] = common_b[b] + e
            else:
                common_b[b] = e
            if b[1] != 1 and b[0].is_Mul:
                bases.append(b)
        bases.sort(key=default_sort_key)
        bases.sort(key=measure, reverse=True)
        for base in bases:
            if base not in common_b:
                continue
            (b, exponent) = base
            last = False
            qlcm = 1
            while True:
                bstart = b
                qstart = qlcm
                bb = []
                ee = []
                for bi in Mul.make_args(b):
                    (bib, bie) = bkey(bi)
                    if bib not in common_b or common_b[bib] < bie:
                        ee = bb = []
                        break
                    ee.append([bie, common_b[bib]])
                    bb.append(bib)
                if ee:
                    min1 = ee[0][1] // ee[0][0]
                    for i in range(1, len(ee)):
                        rat = ee[i][1] // ee[i][0]
                        if rat < 1:
                            break
                        min1 = min(min1, rat)
                    else:
                        for i in range(len(bb)):
                            common_b[bb[i]] -= min1 * ee[i][0]
                            update(bb[i])
                        common_b[base] += min1 * qstart * exponent
                if last or len(common_b) == 1 or all((k[1] == 1 for k in common_b)):
                    break
                qlcm = lcm([ratq(bi) for bi in Mul.make_args(bstart)])
                if qlcm == 1:
                    break
                b = bstart ** qlcm
                qlcm *= qstart
                if all((ratq(bi) == 1 for bi in Mul.make_args(b))):
                    last = True
            (b, q) = base
            done.append((b, common_b.pop(base) * Rational(1, q)))
        c_powers = done
        for ((b, q), e) in common_b.items():
            if (b.is_Pow or isinstance(b, exp)) and q is not S.One and (not b.exp.is_Rational):
                (b, be) = b.as_base_exp()
                b = b ** (be / q)
            else:
                b = root(b, q)
            c_powers.append((b, e))
        check = len(c_powers)
        c_powers = dict(c_powers)
        assert len(c_powers) == check
        newexpr = expr.func(*newexpr + [Pow(b, e) for (b, e) in c_powers.items()])
        if combine == 'exp':
            return expr.func(newexpr, expr.func(*nc_part))
        else:
            return recurse(expr.func(*nc_part), combine='base') * recurse(newexpr, combine='base')
    elif combine == 'base':
        c_powers = []
        nc_part = []
        for term in expr.args:
            if term.is_commutative:
                c_powers.append(list(term.as_base_exp()))
            else:
                nc_part.append(term)
        for i in range(len(c_powers)):
            (b, e) = c_powers[i]
            if not (all((x.is_nonnegative for x in b.as_numer_denom())) or e.is_integer or force or b.is_polar):
                continue
            (exp_c, exp_t) = e.as_coeff_Mul(rational=True)
            if exp_c is not S.One and exp_t is not S.One:
                c_powers[i] = [Pow(b, exp_c), exp_t]
        c_exp = defaultdict(list)
        for (b, e) in c_powers:
            if deep:
                e = recurse(e)
            if e.is_Add and (b.is_positive or e.is_integer):
                e = factor_terms(e)
                if _coeff_isneg(e):
                    e = -e
                    b = 1 / b
            c_exp[e].append(b)
        del c_powers
        c_powers = defaultdict(list)
        for e in c_exp:
            bases = c_exp[e]
            if len(bases) == 1:
                new_base = bases[0]
            elif e.is_integer or force:
                new_base = expr.func(*bases)
            else:
                unk = []
                nonneg = []
                neg = []
                for bi in bases:
                    if bi.is_negative:
                        neg.append(bi)
                    elif bi.is_nonnegative:
                        nonneg.append(bi)
                    elif bi.is_polar:
                        nonneg.append(bi)
                    else:
                        unk.append(bi)
                if len(unk) == 1 and (not neg) or (len(neg) == 1 and (not unk)):
                    nonneg.extend(unk + neg)
                    unk = neg = []
                elif neg:
                    israt = False
                    if e.is_Rational:
                        israt = True
                    else:
                        (p, d) = e.as_numer_denom()
                        if p.is_integer and d.is_integer:
                            israt = True
                    if israt:
                        neg = [-w for w in neg]
                        unk.extend([S.NegativeOne] * len(neg))
                    else:
                        unk.extend(neg)
                        neg = []
                    del israt
                for b in unk:
                    c_powers[b].append(e)
                new_base = expr.func(*nonneg + neg)

                def _terms(e):
                    if False:
                        for i in range(10):
                            print('nop')
                    if e.is_Add:
                        return sum([_terms(ai) for ai in e.args])
                    if e.is_Mul:
                        return prod([_terms(mi) for mi in e.args])
                    return 1
                xnew_base = expand_mul(new_base, deep=False)
                if len(Add.make_args(xnew_base)) < _terms(new_base):
                    new_base = factor_terms(xnew_base)
            c_powers[new_base].append(e)
        c_part = [Pow(b, ei) for (b, e) in c_powers.items() for ei in e]
        return expr.func(*c_part + nc_part)
    else:
        raise ValueError("combine must be one of ('all', 'exp', 'base').")

def powdenest(eq, force=False, polar=False):
    if False:
        while True:
            i = 10
    "\n    Collect exponents on powers as assumptions allow.\n\n    Explanation\n    ===========\n\n    Given ``(bb**be)**e``, this can be simplified as follows:\n        * if ``bb`` is positive, or\n        * ``e`` is an integer, or\n        * ``|be| < 1`` then this simplifies to ``bb**(be*e)``\n\n    Given a product of powers raised to a power, ``(bb1**be1 *\n    bb2**be2...)**e``, simplification can be done as follows:\n\n    - if e is positive, the gcd of all bei can be joined with e;\n    - all non-negative bb can be separated from those that are negative\n      and their gcd can be joined with e; autosimplification already\n      handles this separation.\n    - integer factors from powers that have integers in the denominator\n      of the exponent can be removed from any term and the gcd of such\n      integers can be joined with e\n\n    Setting ``force`` to ``True`` will make symbols that are not explicitly\n    negative behave as though they are positive, resulting in more\n    denesting.\n\n    Setting ``polar`` to ``True`` will do simplifications on the Riemann surface of\n    the logarithm, also resulting in more denestings.\n\n    When there are sums of logs in exp() then a product of powers may be\n    obtained e.g. ``exp(3*(log(a) + 2*log(b)))`` - > ``a**3*b**6``.\n\n    Examples\n    ========\n\n    >>> from sympy.abc import a, b, x, y, z\n    >>> from sympy import Symbol, exp, log, sqrt, symbols, powdenest\n\n    >>> powdenest((x**(2*a/3))**(3*x))\n    (x**(2*a/3))**(3*x)\n    >>> powdenest(exp(3*x*log(2)))\n    2**(3*x)\n\n    Assumptions may prevent expansion:\n\n    >>> powdenest(sqrt(x**2))\n    sqrt(x**2)\n\n    >>> p = symbols('p', positive=True)\n    >>> powdenest(sqrt(p**2))\n    p\n\n    No other expansion is done.\n\n    >>> i, j = symbols('i,j', integer=True)\n    >>> powdenest((x**x)**(i + j)) # -X-> (x**x)**i*(x**x)**j\n    x**(x*(i + j))\n\n    But exp() will be denested by moving all non-log terms outside of\n    the function; this may result in the collapsing of the exp to a power\n    with a different base:\n\n    >>> powdenest(exp(3*y*log(x)))\n    x**(3*y)\n    >>> powdenest(exp(y*(log(a) + log(b))))\n    (a*b)**y\n    >>> powdenest(exp(3*(log(a) + log(b))))\n    a**3*b**3\n\n    If assumptions allow, symbols can also be moved to the outermost exponent:\n\n    >>> i = Symbol('i', integer=True)\n    >>> powdenest(((x**(2*i))**(3*y))**x)\n    ((x**(2*i))**(3*y))**x\n    >>> powdenest(((x**(2*i))**(3*y))**x, force=True)\n    x**(6*i*x*y)\n\n    >>> powdenest(((x**(2*a/3))**(3*y/i))**x)\n    ((x**(2*a/3))**(3*y/i))**x\n    >>> powdenest((x**(2*i)*y**(4*i))**z, force=True)\n    (x*y**2)**(2*i*z)\n\n    >>> n = Symbol('n', negative=True)\n\n    >>> powdenest((x**i)**y, force=True)\n    x**(i*y)\n    >>> powdenest((n**i)**x, force=True)\n    (n**i)**x\n\n    "
    from sympy.simplify.simplify import posify
    if force:

        def _denest(b, e):
            if False:
                print('Hello World!')
            if not isinstance(b, (Pow, exp)):
                return (b.is_positive, Pow(b, e, evaluate=False))
            return _denest(b.base, b.exp * e)
        reps = []
        for p in eq.atoms(Pow, exp):
            if isinstance(p.base, (Pow, exp)):
                (ok, dp) = _denest(*p.args)
                if ok is not False:
                    reps.append((p, dp))
        if reps:
            eq = eq.subs(reps)
        (eq, reps) = posify(eq)
        return powdenest(eq, force=False, polar=polar).xreplace(reps)
    if polar:
        (eq, rep) = polarify(eq)
        return unpolarify(powdenest(unpolarify(eq, exponents_only=True)), rep)
    new = powsimp(eq)
    return new.xreplace(Transform(_denest_pow, filter=lambda m: m.is_Pow or isinstance(m, exp)))
_y = Dummy('y')

def _denest_pow(eq):
    if False:
        i = 10
        return i + 15
    '\n    Denest powers.\n\n    This is a helper function for powdenest that performs the actual\n    transformation.\n    '
    from sympy.simplify.simplify import logcombine
    (b, e) = eq.as_base_exp()
    if b.is_Pow or (isinstance(b, exp) and e != 1):
        new = b._eval_power(e)
        if new is not None:
            eq = new
            (b, e) = new.as_base_exp()
    if b is S.Exp1 and e.is_Mul:
        logs = []
        other = []
        for ei in e.args:
            if any((isinstance(ai, log) for ai in Add.make_args(ei))):
                logs.append(ei)
            else:
                other.append(ei)
        logs = logcombine(Mul(*logs))
        return Pow(exp(logs), Mul(*other))
    (_, be) = b.as_base_exp()
    if be is S.One and (not (b.is_Mul or (b.is_Rational and b.q != 1) or b.is_positive)):
        return eq
    (polars, nonpolars) = ([], [])
    for bb in Mul.make_args(b):
        if bb.is_polar:
            polars.append(bb.as_base_exp())
        else:
            nonpolars.append(bb)
    if len(polars) == 1 and (not polars[0][0].is_Mul):
        return Pow(polars[0][0], polars[0][1] * e) * powdenest(Mul(*nonpolars) ** e)
    elif polars:
        return Mul(*[powdenest(bb ** (ee * e)) for (bb, ee) in polars]) * powdenest(Mul(*nonpolars) ** e)
    if b.is_Integer:
        logb = expand_log(log(b))
        if logb.is_Mul:
            (c, logb) = logb.args
            e *= c
            base = logb.args[0]
            return Pow(base, e)
    if not b.is_Mul or any((s.is_Atom for s in Mul.make_args(b))):
        return eq

    def nc_gcd(aa, bb):
        if False:
            print('Hello World!')
        (a, b) = [i.as_coeff_Mul() for i in [aa, bb]]
        c = gcd(a[0], b[0]).as_numer_denom()[0]
        g = Mul(*a[1].args_cnc(cset=True)[0] & b[1].args_cnc(cset=True)[0])
        return _keep_coeff(c, g)
    glogb = expand_log(log(b))
    if glogb.is_Add:
        args = glogb.args
        g = reduce(nc_gcd, args)
        if g != 1:
            (cg, rg) = g.as_coeff_Mul()
            glogb = _keep_coeff(cg, rg * Add(*[a / g for a in args]))
    if isinstance(glogb, log) or not glogb.is_Mul:
        if glogb.args[0].is_Pow or isinstance(glogb.args[0], exp):
            glogb = _denest_pow(glogb.args[0])
            if (abs(glogb.exp) < 1) == True:
                return Pow(glogb.base, glogb.exp * e)
        return eq
    add = []
    other = []
    for a in glogb.args:
        if a.is_Add:
            add.append(a)
        else:
            other.append(a)
    return Pow(exp(logcombine(Mul(*add))), e * Mul(*other))