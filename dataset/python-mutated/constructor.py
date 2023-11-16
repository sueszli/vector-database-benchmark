"""Tools for constructing domains for expressions. """
from math import prod
from sympy.core import sympify
from sympy.core.evalf import pure_complex
from sympy.core.sorting import ordered
from sympy.polys.domains import ZZ, QQ, ZZ_I, QQ_I, EX
from sympy.polys.domains.complexfield import ComplexField
from sympy.polys.domains.realfield import RealField
from sympy.polys.polyoptions import build_options
from sympy.polys.polyutils import parallel_dict_from_basic
from sympy.utilities import public

def _construct_simple(coeffs, opt):
    if False:
        return 10
    'Handle simple domains, e.g.: ZZ, QQ, RR and algebraic domains. '
    rationals = floats = complexes = algebraics = False
    float_numbers = []
    if opt.extension is True:
        is_algebraic = lambda coeff: coeff.is_number and coeff.is_algebraic
    else:
        is_algebraic = lambda coeff: False
    for coeff in coeffs:
        if coeff.is_Rational:
            if not coeff.is_Integer:
                rationals = True
        elif coeff.is_Float:
            if algebraics:
                return False
            else:
                floats = True
                float_numbers.append(coeff)
        else:
            is_complex = pure_complex(coeff)
            if is_complex:
                complexes = True
                (x, y) = is_complex
                if x.is_Rational and y.is_Rational:
                    if not (x.is_Integer and y.is_Integer):
                        rationals = True
                    continue
                else:
                    floats = True
                    if x.is_Float:
                        float_numbers.append(x)
                    if y.is_Float:
                        float_numbers.append(y)
            elif is_algebraic(coeff):
                if floats:
                    return False
                algebraics = True
            else:
                return None
    max_prec = max((c._prec for c in float_numbers)) if float_numbers else 53
    if algebraics:
        (domain, result) = _construct_algebraic(coeffs, opt)
    else:
        if floats and complexes:
            domain = ComplexField(prec=max_prec)
        elif floats:
            domain = RealField(prec=max_prec)
        elif rationals or opt.field:
            domain = QQ_I if complexes else QQ
        else:
            domain = ZZ_I if complexes else ZZ
        result = [domain.from_sympy(coeff) for coeff in coeffs]
    return (domain, result)

def _construct_algebraic(coeffs, opt):
    if False:
        print('Hello World!')
    'We know that coefficients are algebraic so construct the extension. '
    from sympy.polys.numberfields import primitive_element
    exts = set()

    def build_trees(args):
        if False:
            while True:
                i = 10
        trees = []
        for a in args:
            if a.is_Rational:
                tree = ('Q', QQ.from_sympy(a))
            elif a.is_Add:
                tree = ('+', build_trees(a.args))
            elif a.is_Mul:
                tree = ('*', build_trees(a.args))
            else:
                tree = ('e', a)
                exts.add(a)
            trees.append(tree)
        return trees
    trees = build_trees(coeffs)
    exts = list(ordered(exts))
    (g, span, H) = primitive_element(exts, ex=True, polys=True)
    root = sum([s * ext for (s, ext) in zip(span, exts)])
    (domain, g) = (QQ.algebraic_field((g, root)), g.rep.to_list())
    exts_dom = [domain.dtype.from_list(h, g, QQ) for h in H]
    exts_map = dict(zip(exts, exts_dom))

    def convert_tree(tree):
        if False:
            for i in range(10):
                print('nop')
        (op, args) = tree
        if op == 'Q':
            return domain.dtype.from_list([args], g, QQ)
        elif op == '+':
            return sum((convert_tree(a) for a in args), domain.zero)
        elif op == '*':
            return prod((convert_tree(a) for a in args))
        elif op == 'e':
            return exts_map[args]
        else:
            raise RuntimeError
    result = [convert_tree(tree) for tree in trees]
    return (domain, result)

def _construct_composite(coeffs, opt):
    if False:
        i = 10
        return i + 15
    'Handle composite domains, e.g.: ZZ[X], QQ[X], ZZ(X), QQ(X). '
    (numers, denoms) = ([], [])
    for coeff in coeffs:
        (numer, denom) = coeff.as_numer_denom()
        numers.append(numer)
        denoms.append(denom)
    (polys, gens) = parallel_dict_from_basic(numers + denoms)
    if not gens:
        return None
    if opt.composite is None:
        if any((gen.is_number and gen.is_algebraic for gen in gens)):
            return None
        all_symbols = set()
        for gen in gens:
            symbols = gen.free_symbols
            if all_symbols & symbols:
                return None
            else:
                all_symbols |= symbols
    n = len(gens)
    k = len(polys) // 2
    numers = polys[:k]
    denoms = polys[k:]
    if opt.field:
        fractions = True
    else:
        (fractions, zeros) = (False, (0,) * n)
        for denom in denoms:
            if len(denom) > 1 or zeros not in denom:
                fractions = True
                break
    coeffs = set()
    if not fractions:
        for (numer, denom) in zip(numers, denoms):
            denom = denom[zeros]
            for (monom, coeff) in numer.items():
                coeff /= denom
                coeffs.add(coeff)
                numer[monom] = coeff
    else:
        for (numer, denom) in zip(numers, denoms):
            coeffs.update(list(numer.values()))
            coeffs.update(list(denom.values()))
    rationals = floats = complexes = False
    float_numbers = []
    for coeff in coeffs:
        if coeff.is_Rational:
            if not coeff.is_Integer:
                rationals = True
        elif coeff.is_Float:
            floats = True
            float_numbers.append(coeff)
        else:
            is_complex = pure_complex(coeff)
            if is_complex is not None:
                complexes = True
                (x, y) = is_complex
                if x.is_Rational and y.is_Rational:
                    if not (x.is_Integer and y.is_Integer):
                        rationals = True
                else:
                    floats = True
                    if x.is_Float:
                        float_numbers.append(x)
                    if y.is_Float:
                        float_numbers.append(y)
    max_prec = max((c._prec for c in float_numbers)) if float_numbers else 53
    if floats and complexes:
        ground = ComplexField(prec=max_prec)
    elif floats:
        ground = RealField(prec=max_prec)
    elif complexes:
        if rationals:
            ground = QQ_I
        else:
            ground = ZZ_I
    elif rationals:
        ground = QQ
    else:
        ground = ZZ
    result = []
    if not fractions:
        domain = ground.poly_ring(*gens)
        for numer in numers:
            for (monom, coeff) in numer.items():
                numer[monom] = ground.from_sympy(coeff)
            result.append(domain(numer))
    else:
        domain = ground.frac_field(*gens)
        for (numer, denom) in zip(numers, denoms):
            for (monom, coeff) in numer.items():
                numer[monom] = ground.from_sympy(coeff)
            for (monom, coeff) in denom.items():
                denom[monom] = ground.from_sympy(coeff)
            result.append(domain((numer, denom)))
    return (domain, result)

def _construct_expression(coeffs, opt):
    if False:
        i = 10
        return i + 15
    'The last resort case, i.e. use the expression domain. '
    (domain, result) = (EX, [])
    for coeff in coeffs:
        result.append(domain.from_sympy(coeff))
    return (domain, result)

@public
def construct_domain(obj, **args):
    if False:
        return 10
    "Construct a minimal domain for a list of expressions.\n\n    Explanation\n    ===========\n\n    Given a list of normal SymPy expressions (of type :py:class:`~.Expr`)\n    ``construct_domain`` will find a minimal :py:class:`~.Domain` that can\n    represent those expressions. The expressions will be converted to elements\n    of the domain and both the domain and the domain elements are returned.\n\n    Parameters\n    ==========\n\n    obj: list or dict\n        The expressions to build a domain for.\n\n    **args: keyword arguments\n        Options that affect the choice of domain.\n\n    Returns\n    =======\n\n    (K, elements): Domain and list of domain elements\n        The domain K that can represent the expressions and the list or dict\n        of domain elements representing the same expressions as elements of K.\n\n    Examples\n    ========\n\n    Given a list of :py:class:`~.Integer` ``construct_domain`` will return the\n    domain :ref:`ZZ` and a list of integers as elements of :ref:`ZZ`.\n\n    >>> from sympy import construct_domain, S\n    >>> expressions = [S(2), S(3), S(4)]\n    >>> K, elements = construct_domain(expressions)\n    >>> K\n    ZZ\n    >>> elements\n    [2, 3, 4]\n    >>> type(elements[0])  # doctest: +SKIP\n    <class 'int'>\n    >>> type(expressions[0])\n    <class 'sympy.core.numbers.Integer'>\n\n    If there are any :py:class:`~.Rational` then :ref:`QQ` is returned\n    instead.\n\n    >>> construct_domain([S(1)/2, S(3)/4])\n    (QQ, [1/2, 3/4])\n\n    If there are symbols then a polynomial ring :ref:`K[x]` is returned.\n\n    >>> from sympy import symbols\n    >>> x, y = symbols('x, y')\n    >>> construct_domain([2*x + 1, S(3)/4])\n    (QQ[x], [2*x + 1, 3/4])\n    >>> construct_domain([2*x + 1, y])\n    (ZZ[x,y], [2*x + 1, y])\n\n    If any symbols appear with negative powers then a rational function field\n    :ref:`K(x)` will be returned.\n\n    >>> construct_domain([y/x, x/(1 - y)])\n    (ZZ(x,y), [y/x, -x/(y - 1)])\n\n    Irrational algebraic numbers will result in the :ref:`EX` domain by\n    default. The keyword argument ``extension=True`` leads to the construction\n    of an algebraic number field :ref:`QQ(a)`.\n\n    >>> from sympy import sqrt\n    >>> construct_domain([sqrt(2)])\n    (EX, [EX(sqrt(2))])\n    >>> construct_domain([sqrt(2)], extension=True)  # doctest: +SKIP\n    (QQ<sqrt(2)>, [ANP([1, 0], [1, 0, -2], QQ)])\n\n    See also\n    ========\n\n    Domain\n    Expr\n    "
    opt = build_options(args)
    if hasattr(obj, '__iter__'):
        if isinstance(obj, dict):
            if not obj:
                (monoms, coeffs) = ([], [])
            else:
                (monoms, coeffs) = list(zip(*list(obj.items())))
        else:
            coeffs = obj
    else:
        coeffs = [obj]
    coeffs = list(map(sympify, coeffs))
    result = _construct_simple(coeffs, opt)
    if result is not None:
        if result is not False:
            (domain, coeffs) = result
        else:
            (domain, coeffs) = _construct_expression(coeffs, opt)
    else:
        if opt.composite is False:
            result = None
        else:
            result = _construct_composite(coeffs, opt)
        if result is not None:
            (domain, coeffs) = result
        else:
            (domain, coeffs) = _construct_expression(coeffs, opt)
    if hasattr(obj, '__iter__'):
        if isinstance(obj, dict):
            return (domain, dict(list(zip(monoms, coeffs))))
        else:
            return (domain, coeffs)
    else:
        return (domain, coeffs[0])