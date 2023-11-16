from collections import defaultdict
from sympy.core import sympify, S, Mul, Derivative, Pow
from sympy.core.add import _unevaluated_Add, Add
from sympy.core.assumptions import assumptions
from sympy.core.exprtools import Factors, gcd_terms
from sympy.core.function import _mexpand, expand_mul, expand_power_base
from sympy.core.mul import _keep_coeff, _unevaluated_Mul, _mulsort
from sympy.core.numbers import Rational, zoo, nan
from sympy.core.parameters import global_parameters
from sympy.core.sorting import ordered, default_sort_key
from sympy.core.symbol import Dummy, Wild, symbols
from sympy.functions import exp, sqrt, log
from sympy.functions.elementary.complexes import Abs
from sympy.polys import gcd
from sympy.simplify.sqrtdenest import sqrtdenest
from sympy.utilities.iterables import iterable, sift

def collect(expr, syms, func=None, evaluate=None, exact=False, distribute_order_term=True):
    if False:
        print('Hello World!')
    "\n    Collect additive terms of an expression.\n\n    Explanation\n    ===========\n\n    This function collects additive terms of an expression with respect\n    to a list of expression up to powers with rational exponents. By the\n    term symbol here are meant arbitrary expressions, which can contain\n    powers, products, sums etc. In other words symbol is a pattern which\n    will be searched for in the expression's terms.\n\n    The input expression is not expanded by :func:`collect`, so user is\n    expected to provide an expression in an appropriate form. This makes\n    :func:`collect` more predictable as there is no magic happening behind the\n    scenes. However, it is important to note, that powers of products are\n    converted to products of powers using the :func:`~.expand_power_base`\n    function.\n\n    There are two possible types of output. First, if ``evaluate`` flag is\n    set, this function will return an expression with collected terms or\n    else it will return a dictionary with expressions up to rational powers\n    as keys and collected coefficients as values.\n\n    Examples\n    ========\n\n    >>> from sympy import S, collect, expand, factor, Wild\n    >>> from sympy.abc import a, b, c, x, y\n\n    This function can collect symbolic coefficients in polynomials or\n    rational expressions. It will manage to find all integer or rational\n    powers of collection variable::\n\n        >>> collect(a*x**2 + b*x**2 + a*x - b*x + c, x)\n        c + x**2*(a + b) + x*(a - b)\n\n    The same result can be achieved in dictionary form::\n\n        >>> d = collect(a*x**2 + b*x**2 + a*x - b*x + c, x, evaluate=False)\n        >>> d[x**2]\n        a + b\n        >>> d[x]\n        a - b\n        >>> d[S.One]\n        c\n\n    You can also work with multivariate polynomials. However, remember that\n    this function is greedy so it will care only about a single symbol at time,\n    in specification order::\n\n        >>> collect(x**2 + y*x**2 + x*y + y + a*y, [x, y])\n        x**2*(y + 1) + x*y + y*(a + 1)\n\n    Also more complicated expressions can be used as patterns::\n\n        >>> from sympy import sin, log\n        >>> collect(a*sin(2*x) + b*sin(2*x), sin(2*x))\n        (a + b)*sin(2*x)\n\n        >>> collect(a*x*log(x) + b*(x*log(x)), x*log(x))\n        x*(a + b)*log(x)\n\n    You can use wildcards in the pattern::\n\n        >>> w = Wild('w1')\n        >>> collect(a*x**y - b*x**y, w**y)\n        x**y*(a - b)\n\n    It is also possible to work with symbolic powers, although it has more\n    complicated behavior, because in this case power's base and symbolic part\n    of the exponent are treated as a single symbol::\n\n        >>> collect(a*x**c + b*x**c, x)\n        a*x**c + b*x**c\n        >>> collect(a*x**c + b*x**c, x**c)\n        x**c*(a + b)\n\n    However if you incorporate rationals to the exponents, then you will get\n    well known behavior::\n\n        >>> collect(a*x**(2*c) + b*x**(2*c), x**c)\n        x**(2*c)*(a + b)\n\n    Note also that all previously stated facts about :func:`collect` function\n    apply to the exponential function, so you can get::\n\n        >>> from sympy import exp\n        >>> collect(a*exp(2*x) + b*exp(2*x), exp(x))\n        (a + b)*exp(2*x)\n\n    If you are interested only in collecting specific powers of some symbols\n    then set ``exact`` flag to True::\n\n        >>> collect(a*x**7 + b*x**7, x, exact=True)\n        a*x**7 + b*x**7\n        >>> collect(a*x**7 + b*x**7, x**7, exact=True)\n        x**7*(a + b)\n\n    If you want to collect on any object containing symbols, set\n    ``exact`` to None:\n\n        >>> collect(x*exp(x) + sin(x)*y + sin(x)*2 + 3*x, x, exact=None)\n        x*exp(x) + 3*x + (y + 2)*sin(x)\n        >>> collect(a*x*y + x*y + b*x + x, [x, y], exact=None)\n        x*y*(a + 1) + x*(b + 1)\n\n    You can also apply this function to differential equations, where\n    derivatives of arbitrary order can be collected. Note that if you\n    collect with respect to a function or a derivative of a function, all\n    derivatives of that function will also be collected. Use\n    ``exact=True`` to prevent this from happening::\n\n        >>> from sympy import Derivative as D, collect, Function\n        >>> f = Function('f') (x)\n\n        >>> collect(a*D(f,x) + b*D(f,x), D(f,x))\n        (a + b)*Derivative(f(x), x)\n\n        >>> collect(a*D(D(f,x),x) + b*D(D(f,x),x), f)\n        (a + b)*Derivative(f(x), (x, 2))\n\n        >>> collect(a*D(D(f,x),x) + b*D(D(f,x),x), D(f,x), exact=True)\n        a*Derivative(f(x), (x, 2)) + b*Derivative(f(x), (x, 2))\n\n        >>> collect(a*D(f,x) + b*D(f,x) + a*f + b*f, f)\n        (a + b)*f(x) + (a + b)*Derivative(f(x), x)\n\n    Or you can even match both derivative order and exponent at the same time::\n\n        >>> collect(a*D(D(f,x),x)**2 + b*D(D(f,x),x)**2, D(f,x))\n        (a + b)*Derivative(f(x), (x, 2))**2\n\n    Finally, you can apply a function to each of the collected coefficients.\n    For example you can factorize symbolic coefficients of polynomial::\n\n        >>> f = expand((x + a + 1)**3)\n\n        >>> collect(f, x, factor)\n        x**3 + 3*x**2*(a + 1) + 3*x*(a + 1)**2 + (a + 1)**3\n\n    .. note:: Arguments are expected to be in expanded form, so you might have\n              to call :func:`~.expand` prior to calling this function.\n\n    See Also\n    ========\n\n    collect_const, collect_sqrt, rcollect\n    "
    expr = sympify(expr)
    syms = [sympify(i) for i in (syms if iterable(syms) else [syms])]
    cond = lambda x: x.is_Symbol or (-x).is_Symbol or bool(x.atoms(Wild))
    (_, nonsyms) = sift(syms, cond, binary=True)
    if nonsyms:
        reps = dict(zip(nonsyms, [Dummy(**assumptions(i)) for i in nonsyms]))
        syms = [reps.get(s, s) for s in syms]
        rv = collect(expr.subs(reps), syms, func=func, evaluate=evaluate, exact=exact, distribute_order_term=distribute_order_term)
        urep = {v: k for (k, v) in reps.items()}
        if not isinstance(rv, dict):
            return rv.xreplace(urep)
        else:
            return {urep.get(k, k).xreplace(urep): v.xreplace(urep) for (k, v) in rv.items()}
    if exact is None:
        _syms = set()
        for i in Add.make_args(expr):
            if not i.has_free(*syms) or i in syms:
                continue
            if not i.is_Mul and i not in syms:
                _syms.add(i)
            else:
                g = i._new_rawargs(*i.as_coeff_mul(*syms)[1])
                if g not in syms:
                    _syms.add(g)
        simple = all((i.is_Pow and i.base in syms for i in _syms))
        syms = syms + list(ordered(_syms))
        if not simple:
            return collect(expr, syms, func=func, evaluate=evaluate, exact=False, distribute_order_term=distribute_order_term)
    if evaluate is None:
        evaluate = global_parameters.evaluate

    def make_expression(terms):
        if False:
            print('Hello World!')
        product = []
        for (term, rat, sym, deriv) in terms:
            if deriv is not None:
                (var, order) = deriv
                while order > 0:
                    (term, order) = (Derivative(term, var), order - 1)
            if sym is None:
                if rat is S.One:
                    product.append(term)
                else:
                    product.append(Pow(term, rat))
            else:
                product.append(Pow(term, rat * sym))
        return Mul(*product)

    def parse_derivative(deriv):
        if False:
            print('Hello World!')
        (expr, sym, order) = (deriv.expr, deriv.variables[0], 1)
        for s in deriv.variables[1:]:
            if s == sym:
                order += 1
            else:
                raise NotImplementedError('Improve MV Derivative support in collect')
        while isinstance(expr, Derivative):
            s0 = expr.variables[0]
            for s in expr.variables:
                if s != s0:
                    raise NotImplementedError('Improve MV Derivative support in collect')
            if s0 == sym:
                (expr, order) = (expr.expr, order + len(expr.variables))
            else:
                break
        return (expr, (sym, Rational(order)))

    def parse_term(expr):
        if False:
            print('Hello World!')
        'Parses expression expr and outputs tuple (sexpr, rat_expo,\n        sym_expo, deriv)\n        where:\n         - sexpr is the base expression\n         - rat_expo is the rational exponent that sexpr is raised to\n         - sym_expo is the symbolic exponent that sexpr is raised to\n         - deriv contains the derivatives of the expression\n\n         For example, the output of x would be (x, 1, None, None)\n         the output of 2**x would be (2, 1, x, None).\n        '
        (rat_expo, sym_expo) = (S.One, None)
        (sexpr, deriv) = (expr, None)
        if expr.is_Pow:
            if isinstance(expr.base, Derivative):
                (sexpr, deriv) = parse_derivative(expr.base)
            else:
                sexpr = expr.base
            if expr.base == S.Exp1:
                arg = expr.exp
                if arg.is_Rational:
                    (sexpr, rat_expo) = (S.Exp1, arg)
                elif arg.is_Mul:
                    (coeff, tail) = arg.as_coeff_Mul(rational=True)
                    (sexpr, rat_expo) = (exp(tail), coeff)
            elif expr.exp.is_Number:
                rat_expo = expr.exp
            else:
                (coeff, tail) = expr.exp.as_coeff_Mul()
                if coeff.is_Number:
                    (rat_expo, sym_expo) = (coeff, tail)
                else:
                    sym_expo = expr.exp
        elif isinstance(expr, exp):
            arg = expr.exp
            if arg.is_Rational:
                (sexpr, rat_expo) = (S.Exp1, arg)
            elif arg.is_Mul:
                (coeff, tail) = arg.as_coeff_Mul(rational=True)
                (sexpr, rat_expo) = (exp(tail), coeff)
        elif isinstance(expr, Derivative):
            (sexpr, deriv) = parse_derivative(expr)
        return (sexpr, rat_expo, sym_expo, deriv)

    def parse_expression(terms, pattern):
        if False:
            return 10
        'Parse terms searching for a pattern.\n        Terms is a list of tuples as returned by parse_terms;\n        Pattern is an expression treated as a product of factors.\n        '
        pattern = Mul.make_args(pattern)
        if len(terms) < len(pattern):
            return None
        else:
            pattern = [parse_term(elem) for elem in pattern]
            terms = terms[:]
            (elems, common_expo, has_deriv) = ([], None, False)
            for (elem, e_rat, e_sym, e_ord) in pattern:
                if elem.is_Number and e_rat == 1 and (e_sym is None):
                    continue
                for j in range(len(terms)):
                    if terms[j] is None:
                        continue
                    (term, t_rat, t_sym, t_ord) = terms[j]
                    if t_ord is not None:
                        has_deriv = True
                    if term.match(elem) is not None and (t_sym == e_sym or (t_sym is not None and e_sym is not None and (t_sym.match(e_sym) is not None))):
                        if exact is False:
                            expo = t_rat / e_rat
                            if common_expo is None:
                                common_expo = expo
                            elif common_expo != expo:
                                common_expo = 1
                        elif e_rat != t_rat or e_ord != t_ord:
                            continue
                        elems.append(terms[j])
                        terms[j] = None
                        break
                else:
                    return None
            return ([_f for _f in terms if _f], elems, common_expo, has_deriv)
    if evaluate:
        if expr.is_Add:
            o = expr.getO() or 0
            expr = expr.func(*[collect(a, syms, func, True, exact, distribute_order_term) for a in expr.args if a != o]) + o
        elif expr.is_Mul:
            return expr.func(*[collect(term, syms, func, True, exact, distribute_order_term) for term in expr.args])
        elif expr.is_Pow:
            b = collect(expr.base, syms, func, True, exact, distribute_order_term)
            return Pow(b, expr.exp)
    syms = [expand_power_base(i, deep=False) for i in syms]
    order_term = None
    if distribute_order_term:
        order_term = expr.getO()
        if order_term is not None:
            if order_term.has(*syms):
                order_term = None
            else:
                expr = expr.removeO()
    summa = [expand_power_base(i, deep=False) for i in Add.make_args(expr)]
    (collected, disliked) = (defaultdict(list), S.Zero)
    for product in summa:
        (c, nc) = product.args_cnc(split_1=False)
        args = list(ordered(c)) + nc
        terms = [parse_term(i) for i in args]
        small_first = True
        for symbol in syms:
            if isinstance(symbol, Derivative) and small_first:
                terms = list(reversed(terms))
                small_first = not small_first
            result = parse_expression(terms, symbol)
            if result is not None:
                if not symbol.is_commutative:
                    raise AttributeError('Can not collect noncommutative symbol')
                (terms, elems, common_expo, has_deriv) = result
                if not has_deriv:
                    margs = []
                    for elem in elems:
                        if elem[2] is None:
                            e = elem[1]
                        else:
                            e = elem[1] * elem[2]
                        margs.append(Pow(elem[0], e))
                    index = Mul(*margs)
                else:
                    index = make_expression(elems)
                terms = expand_power_base(make_expression(terms), deep=False)
                index = expand_power_base(index, deep=False)
                collected[index].append(terms)
                break
        else:
            disliked += product
    collected = {k: Add(*v) for (k, v) in collected.items()}
    if disliked is not S.Zero:
        collected[S.One] = disliked
    if order_term is not None:
        for (key, val) in collected.items():
            collected[key] = val + order_term
    if func is not None:
        collected = {key: func(val) for (key, val) in collected.items()}
    if evaluate:
        return Add(*[key * val for (key, val) in collected.items()])
    else:
        return collected

def rcollect(expr, *vars):
    if False:
        print('Hello World!')
    '\n    Recursively collect sums in an expression.\n\n    Examples\n    ========\n\n    >>> from sympy.simplify import rcollect\n    >>> from sympy.abc import x, y\n\n    >>> expr = (x**2*y + x*y + x + y)/(x + y)\n\n    >>> rcollect(expr, y)\n    (x + y*(x**2 + x + 1))/(x + y)\n\n    See Also\n    ========\n\n    collect, collect_const, collect_sqrt\n    '
    if expr.is_Atom or not expr.has(*vars):
        return expr
    else:
        expr = expr.__class__(*[rcollect(arg, *vars) for arg in expr.args])
        if expr.is_Add:
            return collect(expr, vars)
        else:
            return expr

def collect_sqrt(expr, evaluate=None):
    if False:
        return 10
    'Return expr with terms having common square roots collected together.\n    If ``evaluate`` is False a count indicating the number of sqrt-containing\n    terms will be returned and, if non-zero, the terms of the Add will be\n    returned, else the expression itself will be returned as a single term.\n    If ``evaluate`` is True, the expression with any collected terms will be\n    returned.\n\n    Note: since I = sqrt(-1), it is collected, too.\n\n    Examples\n    ========\n\n    >>> from sympy import sqrt\n    >>> from sympy.simplify.radsimp import collect_sqrt\n    >>> from sympy.abc import a, b\n\n    >>> r2, r3, r5 = [sqrt(i) for i in [2, 3, 5]]\n    >>> collect_sqrt(a*r2 + b*r2)\n    sqrt(2)*(a + b)\n    >>> collect_sqrt(a*r2 + b*r2 + a*r3 + b*r3)\n    sqrt(2)*(a + b) + sqrt(3)*(a + b)\n    >>> collect_sqrt(a*r2 + b*r2 + a*r3 + b*r5)\n    sqrt(3)*a + sqrt(5)*b + sqrt(2)*(a + b)\n\n    If evaluate is False then the arguments will be sorted and\n    returned as a list and a count of the number of sqrt-containing\n    terms will be returned:\n\n    >>> collect_sqrt(a*r2 + b*r2 + a*r3 + b*r5, evaluate=False)\n    ((sqrt(3)*a, sqrt(5)*b, sqrt(2)*(a + b)), 3)\n    >>> collect_sqrt(a*sqrt(2) + b, evaluate=False)\n    ((b, sqrt(2)*a), 1)\n    >>> collect_sqrt(a + b, evaluate=False)\n    ((a + b,), 0)\n\n    See Also\n    ========\n\n    collect, collect_const, rcollect\n    '
    if evaluate is None:
        evaluate = global_parameters.evaluate
    (coeff, expr) = expr.as_content_primitive()
    vars = set()
    for a in Add.make_args(expr):
        for m in a.args_cnc()[0]:
            if m.is_number and (m.is_Pow and m.exp.is_Rational and (m.exp.q == 2) or m is S.ImaginaryUnit):
                vars.add(m)
    d = collect_const(expr, *vars, Numbers=False)
    hit = expr != d
    if not evaluate:
        nrad = 0
        args = list(ordered(Add.make_args(d)))
        for (i, m) in enumerate(args):
            (c, nc) = m.args_cnc()
            for ci in c:
                if ci.is_Pow and ci.exp.is_Rational and (ci.exp.q == 2) or ci is S.ImaginaryUnit:
                    nrad += 1
                    break
            args[i] *= coeff
        if not (hit or nrad):
            args = [Add(*args)]
        return (tuple(args), nrad)
    return coeff * d

def collect_abs(expr):
    if False:
        return 10
    'Return ``expr`` with arguments of multiple Abs in a term collected\n    under a single instance.\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.radsimp import collect_abs\n    >>> from sympy.abc import x\n    >>> collect_abs(abs(x + 1)/abs(x**2 - 1))\n    Abs((x + 1)/(x**2 - 1))\n    >>> collect_abs(abs(1/x))\n    Abs(1/x)\n    '

    def _abs(mul):
        if False:
            for i in range(10):
                print('nop')
        (c, nc) = mul.args_cnc()
        a = []
        o = []
        for i in c:
            if isinstance(i, Abs):
                a.append(i.args[0])
            elif isinstance(i, Pow) and isinstance(i.base, Abs) and i.exp.is_real:
                a.append(i.base.args[0] ** i.exp)
            else:
                o.append(i)
        if len(a) < 2 and (not any((i.exp.is_negative for i in a if isinstance(i, Pow)))):
            return mul
        absarg = Mul(*a)
        A = Abs(absarg)
        args = [A]
        args.extend(o)
        if not A.has(Abs):
            args.extend(nc)
            return Mul(*args)
        if not isinstance(A, Abs):
            A = Abs(absarg, evaluate=False)
        args[0] = A
        _mulsort(args)
        args.extend(nc)
        return Mul._from_args(args, is_commutative=not nc)
    return expr.replace(lambda x: isinstance(x, Mul), lambda x: _abs(x)).replace(lambda x: isinstance(x, Pow), lambda x: _abs(x))

def collect_const(expr, *vars, Numbers=True):
    if False:
        for i in range(10):
            print('nop')
    'A non-greedy collection of terms with similar number coefficients in\n    an Add expr. If ``vars`` is given then only those constants will be\n    targeted. Although any Number can also be targeted, if this is not\n    desired set ``Numbers=False`` and no Float or Rational will be collected.\n\n    Parameters\n    ==========\n\n    expr : SymPy expression\n        This parameter defines the expression the expression from which\n        terms with similar coefficients are to be collected. A non-Add\n        expression is returned as it is.\n\n    vars : variable length collection of Numbers, optional\n        Specifies the constants to target for collection. Can be multiple in\n        number.\n\n    Numbers : bool\n        Specifies to target all instance of\n        :class:`sympy.core.numbers.Number` class. If ``Numbers=False``, then\n        no Float or Rational will be collected.\n\n    Returns\n    =======\n\n    expr : Expr\n        Returns an expression with similar coefficient terms collected.\n\n    Examples\n    ========\n\n    >>> from sympy import sqrt\n    >>> from sympy.abc import s, x, y, z\n    >>> from sympy.simplify.radsimp import collect_const\n    >>> collect_const(sqrt(3) + sqrt(3)*(1 + sqrt(2)))\n    sqrt(3)*(sqrt(2) + 2)\n    >>> collect_const(sqrt(3)*s + sqrt(7)*s + sqrt(3) + sqrt(7))\n    (sqrt(3) + sqrt(7))*(s + 1)\n    >>> s = sqrt(2) + 2\n    >>> collect_const(sqrt(3)*s + sqrt(3) + sqrt(7)*s + sqrt(7))\n    (sqrt(2) + 3)*(sqrt(3) + sqrt(7))\n    >>> collect_const(sqrt(3)*s + sqrt(3) + sqrt(7)*s + sqrt(7), sqrt(3))\n    sqrt(7) + sqrt(3)*(sqrt(2) + 3) + sqrt(7)*(sqrt(2) + 2)\n\n    The collection is sign-sensitive, giving higher precedence to the\n    unsigned values:\n\n    >>> collect_const(x - y - z)\n    x - (y + z)\n    >>> collect_const(-y - z)\n    -(y + z)\n    >>> collect_const(2*x - 2*y - 2*z, 2)\n    2*(x - y - z)\n    >>> collect_const(2*x - 2*y - 2*z, -2)\n    2*x - 2*(y + z)\n\n    See Also\n    ========\n\n    collect, collect_sqrt, rcollect\n    '
    if not expr.is_Add:
        return expr
    recurse = False
    if not vars:
        recurse = True
        vars = set()
        for a in expr.args:
            for m in Mul.make_args(a):
                if m.is_number:
                    vars.add(m)
    else:
        vars = sympify(vars)
    if not Numbers:
        vars = [v for v in vars if not v.is_Number]
    vars = list(ordered(vars))
    for v in vars:
        terms = defaultdict(list)
        Fv = Factors(v)
        for m in Add.make_args(expr):
            f = Factors(m)
            (q, r) = f.div(Fv)
            if r.is_one:
                fwas = f.factors.copy()
                fnow = q.factors
                if not any((k in fwas and fwas[k].is_Integer and (not fnow[k].is_Integer) for k in fnow)):
                    terms[v].append(q.as_expr())
                    continue
            terms[S.One].append(m)
        args = []
        hit = False
        uneval = False
        for k in ordered(terms):
            v = terms[k]
            if k is S.One:
                args.extend(v)
                continue
            if len(v) > 1:
                v = Add(*v)
                hit = True
                if recurse and v != expr:
                    vars.append(v)
            else:
                v = v[0]
            if Numbers and k.is_Number and v.is_Add:
                args.append(_keep_coeff(k, v, sign=True))
                uneval = True
            else:
                args.append(k * v)
        if hit:
            if uneval:
                expr = _unevaluated_Add(*args)
            else:
                expr = Add(*args)
            if not expr.is_Add:
                break
    return expr

def radsimp(expr, symbolic=True, max_terms=4):
    if False:
        return 10
    "\n    Rationalize the denominator by removing square roots.\n\n    Explanation\n    ===========\n\n    The expression returned from radsimp must be used with caution\n    since if the denominator contains symbols, it will be possible to make\n    substitutions that violate the assumptions of the simplification process:\n    that for a denominator matching a + b*sqrt(c), a != +/-b*sqrt(c). (If\n    there are no symbols, this assumptions is made valid by collecting terms\n    of sqrt(c) so the match variable ``a`` does not contain ``sqrt(c)``.) If\n    you do not want the simplification to occur for symbolic denominators, set\n    ``symbolic`` to False.\n\n    If there are more than ``max_terms`` radical terms then the expression is\n    returned unchanged.\n\n    Examples\n    ========\n\n    >>> from sympy import radsimp, sqrt, Symbol, pprint\n    >>> from sympy import factor_terms, fraction, signsimp\n    >>> from sympy.simplify.radsimp import collect_sqrt\n    >>> from sympy.abc import a, b, c\n\n    >>> radsimp(1/(2 + sqrt(2)))\n    (2 - sqrt(2))/2\n    >>> x,y = map(Symbol, 'xy')\n    >>> e = ((2 + 2*sqrt(2))*x + (2 + sqrt(8))*y)/(2 + sqrt(2))\n    >>> radsimp(e)\n    sqrt(2)*(x + y)\n\n    No simplification beyond removal of the gcd is done. One might\n    want to polish the result a little, however, by collecting\n    square root terms:\n\n    >>> r2 = sqrt(2)\n    >>> r5 = sqrt(5)\n    >>> ans = radsimp(1/(y*r2 + x*r2 + a*r5 + b*r5)); pprint(ans)\n        ___       ___       ___       ___\n      \\/ 5 *a + \\/ 5 *b - \\/ 2 *x - \\/ 2 *y\n    ------------------------------------------\n       2               2      2              2\n    5*a  + 10*a*b + 5*b  - 2*x  - 4*x*y - 2*y\n\n    >>> n, d = fraction(ans)\n    >>> pprint(factor_terms(signsimp(collect_sqrt(n))/d, radical=True))\n            ___             ___\n          \\/ 5 *(a + b) - \\/ 2 *(x + y)\n    ------------------------------------------\n       2               2      2              2\n    5*a  + 10*a*b + 5*b  - 2*x  - 4*x*y - 2*y\n\n    If radicals in the denominator cannot be removed or there is no denominator,\n    the original expression will be returned.\n\n    >>> radsimp(sqrt(2)*x + sqrt(2))\n    sqrt(2)*x + sqrt(2)\n\n    Results with symbols will not always be valid for all substitutions:\n\n    >>> eq = 1/(a + b*sqrt(c))\n    >>> eq.subs(a, b*sqrt(c))\n    1/(2*b*sqrt(c))\n    >>> radsimp(eq).subs(a, b*sqrt(c))\n    nan\n\n    If ``symbolic=False``, symbolic denominators will not be transformed (but\n    numeric denominators will still be processed):\n\n    >>> radsimp(eq, symbolic=False)\n    1/(a + b*sqrt(c))\n\n    "
    from sympy.simplify.simplify import signsimp
    syms = symbols('a:d A:D')

    def _num(rterms):
        if False:
            return 10
        (a, b, c, d, A, B, C, D) = syms
        if len(rterms) == 2:
            reps = dict(list(zip([A, a, B, b], [j for i in rterms for j in i])))
            return (sqrt(A) * a - sqrt(B) * b).xreplace(reps)
        if len(rterms) == 3:
            reps = dict(list(zip([A, a, B, b, C, c], [j for i in rterms for j in i])))
            return ((sqrt(A) * a + sqrt(B) * b - sqrt(C) * c) * (2 * sqrt(A) * sqrt(B) * a * b - A * a ** 2 - B * b ** 2 + C * c ** 2)).xreplace(reps)
        elif len(rterms) == 4:
            reps = dict(list(zip([A, a, B, b, C, c, D, d], [j for i in rterms for j in i])))
            return ((sqrt(A) * a + sqrt(B) * b - sqrt(C) * c - sqrt(D) * d) * (2 * sqrt(A) * sqrt(B) * a * b - A * a ** 2 - B * b ** 2 - 2 * sqrt(C) * sqrt(D) * c * d + C * c ** 2 + D * d ** 2) * (-8 * sqrt(A) * sqrt(B) * sqrt(C) * sqrt(D) * a * b * c * d + A ** 2 * a ** 4 - 2 * A * B * a ** 2 * b ** 2 - 2 * A * C * a ** 2 * c ** 2 - 2 * A * D * a ** 2 * d ** 2 + B ** 2 * b ** 4 - 2 * B * C * b ** 2 * c ** 2 - 2 * B * D * b ** 2 * d ** 2 + C ** 2 * c ** 4 - 2 * C * D * c ** 2 * d ** 2 + D ** 2 * d ** 4)).xreplace(reps)
        elif len(rterms) == 1:
            return sqrt(rterms[0][0])
        else:
            raise NotImplementedError

    def ispow2(d, log2=False):
        if False:
            return 10
        if not d.is_Pow:
            return False
        e = d.exp
        if e.is_Rational and e.q == 2 or (symbolic and denom(e) == 2):
            return True
        if log2:
            q = 1
            if e.is_Rational:
                q = e.q
            elif symbolic:
                d = denom(e)
                if d.is_Integer:
                    q = d
            if q != 1 and log(q, 2).is_Integer:
                return True
        return False

    def handle(expr):
        if False:
            i = 10
            return i + 15
        from sympy.simplify.simplify import nsimplify
        (n, d) = fraction(expr)
        if expr.is_Atom or (d.is_Atom and n.is_Atom):
            return expr
        elif not n.is_Atom:
            n = n.func(*[handle(a) for a in n.args])
            return _unevaluated_Mul(n, handle(1 / d))
        elif n is not S.One:
            return _unevaluated_Mul(n, handle(1 / d))
        elif d.is_Mul:
            return _unevaluated_Mul(*[handle(1 / d) for d in d.args])
        if not symbolic and d.free_symbols:
            return expr
        if ispow2(d):
            d2 = sqrtdenest(sqrt(d.base)) ** numer(d.exp)
            if d2 != d:
                return handle(1 / d2)
        elif d.is_Pow and (d.exp.is_integer or d.base.is_positive):
            return handle(1 / d.base) ** d.exp
        if not (d.is_Add or ispow2(d)):
            return 1 / d.func(*[handle(a) for a in d.args])
        keep = True
        d = _mexpand(d)
        if d.is_Atom:
            return 1 / d
        if d.is_number:
            _d = nsimplify(d)
            if _d.is_Number and _d.equals(d):
                return 1 / _d
        while True:
            collected = defaultdict(list)
            for m in Add.make_args(d):
                p2 = []
                other = []
                for i in Mul.make_args(m):
                    if ispow2(i, log2=True):
                        p2.append(i.base if i.exp is S.Half else i.base ** (2 * i.exp))
                    elif i is S.ImaginaryUnit:
                        p2.append(S.NegativeOne)
                    else:
                        other.append(i)
                collected[tuple(ordered(p2))].append(Mul(*other))
            rterms = list(ordered(list(collected.items())))
            rterms = [(Mul(*i), Add(*j)) for (i, j) in rterms]
            nrad = len(rterms) - (1 if rterms[0][0] is S.One else 0)
            if nrad < 1:
                break
            elif nrad > max_terms:
                keep = False
                break
            if len(rterms) > 4:
                if all((x.is_Integer and (y ** 2).is_Rational for (x, y) in rterms)):
                    (nd, d) = rad_rationalize(S.One, Add._from_args([sqrt(x) * y for (x, y) in rterms]))
                    n *= nd
                else:
                    keep = False
                break
            from sympy.simplify.powsimp import powsimp, powdenest
            num = powsimp(_num(rterms))
            n *= num
            d *= num
            d = powdenest(_mexpand(d), force=symbolic)
            if d.has(S.Zero, nan, zoo):
                return expr
            if d.is_Atom:
                break
        if not keep:
            return expr
        return _unevaluated_Mul(n, 1 / d)
    (coeff, expr) = expr.as_coeff_Add()
    expr = expr.normal()
    old = fraction(expr)
    (n, d) = fraction(handle(expr))
    if old != (n, d):
        if not d.is_Atom:
            was = (n, d)
            n = signsimp(n, evaluate=False)
            d = signsimp(d, evaluate=False)
            u = Factors(_unevaluated_Mul(n, 1 / d))
            u = _unevaluated_Mul(*[k ** v for (k, v) in u.factors.items()])
            (n, d) = fraction(u)
            if old == (n, d):
                (n, d) = was
        n = expand_mul(n)
        if d.is_Number or d.is_Add:
            (n2, d2) = fraction(gcd_terms(_unevaluated_Mul(n, 1 / d)))
            if d2.is_Number or d2.count_ops() <= d.count_ops():
                (n, d) = [signsimp(i) for i in (n2, d2)]
                if n.is_Mul and n.args[0].is_Number:
                    n = n.func(*n.args)
    return coeff + _unevaluated_Mul(n, 1 / d)

def rad_rationalize(num, den):
    if False:
        for i in range(10):
            print('nop')
    '\n    Rationalize ``num/den`` by removing square roots in the denominator;\n    num and den are sum of terms whose squares are positive rationals.\n\n    Examples\n    ========\n\n    >>> from sympy import sqrt\n    >>> from sympy.simplify.radsimp import rad_rationalize\n    >>> rad_rationalize(sqrt(3), 1 + sqrt(2)/3)\n    (-sqrt(3) + sqrt(6)/3, -7/9)\n    '
    if not den.is_Add:
        return (num, den)
    (g, a, b) = split_surds(den)
    a = a * sqrt(g)
    num = _mexpand((a - b) * num)
    den = _mexpand(a ** 2 - b ** 2)
    return rad_rationalize(num, den)

def fraction(expr, exact=False):
    if False:
        return 10
    "Returns a pair with expression's numerator and denominator.\n       If the given expression is not a fraction then this function\n       will return the tuple (expr, 1).\n\n       This function will not make any attempt to simplify nested\n       fractions or to do any term rewriting at all.\n\n       If only one of the numerator/denominator pair is needed then\n       use numer(expr) or denom(expr) functions respectively.\n\n       >>> from sympy import fraction, Rational, Symbol\n       >>> from sympy.abc import x, y\n\n       >>> fraction(x/y)\n       (x, y)\n       >>> fraction(x)\n       (x, 1)\n\n       >>> fraction(1/y**2)\n       (1, y**2)\n\n       >>> fraction(x*y/2)\n       (x*y, 2)\n       >>> fraction(Rational(1, 2))\n       (1, 2)\n\n       This function will also work fine with assumptions:\n\n       >>> k = Symbol('k', negative=True)\n       >>> fraction(x * y**k)\n       (x, y**(-k))\n\n       If we know nothing about sign of some exponent and ``exact``\n       flag is unset, then structure this exponent's structure will\n       be analyzed and pretty fraction will be returned:\n\n       >>> from sympy import exp, Mul\n       >>> fraction(2*x**(-y))\n       (2, x**y)\n\n       >>> fraction(exp(-x))\n       (1, exp(x))\n\n       >>> fraction(exp(-x), exact=True)\n       (exp(-x), 1)\n\n       The ``exact`` flag will also keep any unevaluated Muls from\n       being evaluated:\n\n       >>> u = Mul(2, x + 1, evaluate=False)\n       >>> fraction(u)\n       (2*x + 2, 1)\n       >>> fraction(u, exact=True)\n       (2*(x  + 1), 1)\n    "
    expr = sympify(expr)
    (numer, denom) = ([], [])
    for term in Mul.make_args(expr):
        if term.is_commutative and (term.is_Pow or isinstance(term, exp)):
            (b, ex) = term.as_base_exp()
            if ex.is_negative:
                if ex is S.NegativeOne:
                    denom.append(b)
                elif exact:
                    if ex.is_constant():
                        denom.append(Pow(b, -ex))
                    else:
                        numer.append(term)
                else:
                    denom.append(Pow(b, -ex))
            elif ex.is_positive:
                numer.append(term)
            elif not exact and ex.is_Mul:
                (n, d) = term.as_numer_denom()
                if n != 1:
                    numer.append(n)
                denom.append(d)
            else:
                numer.append(term)
        elif term.is_Rational and (not term.is_Integer):
            if term.p != 1:
                numer.append(term.p)
            denom.append(term.q)
        else:
            numer.append(term)
    return (Mul(*numer, evaluate=not exact), Mul(*denom, evaluate=not exact))

def numer(expr):
    if False:
        for i in range(10):
            print('nop')
    return fraction(expr)[0]

def denom(expr):
    if False:
        print('Hello World!')
    return fraction(expr)[1]

def fraction_expand(expr, **hints):
    if False:
        while True:
            i = 10
    return expr.expand(frac=True, **hints)

def numer_expand(expr, **hints):
    if False:
        for i in range(10):
            print('nop')
    (a, b) = fraction(expr)
    return a.expand(numer=True, **hints) / b

def denom_expand(expr, **hints):
    if False:
        i = 10
        return i + 15
    (a, b) = fraction(expr)
    return a / b.expand(denom=True, **hints)
expand_numer = numer_expand
expand_denom = denom_expand
expand_fraction = fraction_expand

def split_surds(expr):
    if False:
        i = 10
        return i + 15
    '\n    Split an expression with terms whose squares are positive rationals\n    into a sum of terms whose surds squared have gcd equal to g\n    and a sum of terms with surds squared prime with g.\n\n    Examples\n    ========\n\n    >>> from sympy import sqrt\n    >>> from sympy.simplify.radsimp import split_surds\n    >>> split_surds(3*sqrt(3) + sqrt(5)/7 + sqrt(6) + sqrt(10) + sqrt(15))\n    (3, sqrt(2) + sqrt(5) + 3, sqrt(5)/7 + sqrt(10))\n    '
    args = sorted(expr.args, key=default_sort_key)
    coeff_muls = [x.as_coeff_Mul() for x in args]
    surds = [x[1] ** 2 for x in coeff_muls if x[1].is_Pow]
    surds.sort(key=default_sort_key)
    (g, b1, b2) = _split_gcd(*surds)
    g2 = g
    if not b2 and len(b1) >= 2:
        b1n = [x / g for x in b1]
        b1n = [x for x in b1n if x != 1]
        (g1, b1n, b2) = _split_gcd(*b1n)
        g2 = g * g1
    (a1v, a2v) = ([], [])
    for (c, s) in coeff_muls:
        if s.is_Pow and s.exp == S.Half:
            s1 = s.base
            if s1 in b1:
                a1v.append(c * sqrt(s1 / g2))
            else:
                a2v.append(c * s)
        else:
            a2v.append(c * s)
    a = Add(*a1v)
    b = Add(*a2v)
    return (g2, a, b)

def _split_gcd(*a):
    if False:
        return 10
    '\n    Split the list of integers ``a`` into a list of integers, ``a1`` having\n    ``g = gcd(a1)``, and a list ``a2`` whose elements are not divisible by\n    ``g``.  Returns ``g, a1, a2``.\n\n    Examples\n    ========\n\n    >>> from sympy.simplify.radsimp import _split_gcd\n    >>> _split_gcd(55, 35, 22, 14, 77, 10)\n    (5, [55, 35, 10], [22, 14, 77])\n    '
    g = a[0]
    b1 = [g]
    b2 = []
    for x in a[1:]:
        g1 = gcd(g, x)
        if g1 == 1:
            b2.append(x)
        else:
            g = g1
            b1.append(x)
    return (g, b1, b2)