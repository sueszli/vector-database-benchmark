"""Implementation of :class:`PolynomialRing` class. """
from sympy.polys.agca.modules import FreeModulePolyRing
from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.compositedomain import CompositeDomain
from sympy.polys.domains.old_fractionfield import FractionField
from sympy.polys.domains.ring import Ring
from sympy.polys.orderings import monomial_key, build_product_order
from sympy.polys.polyclasses import DMP, DMF
from sympy.polys.polyerrors import GeneratorsNeeded, PolynomialError, CoercionFailed, ExactQuotientFailed, NotReversible
from sympy.polys.polyutils import dict_from_basic, basic_from_dict, _dict_reorder
from sympy.utilities import public
from sympy.utilities.iterables import iterable

@public
class PolynomialRingBase(Ring, CharacteristicZero, CompositeDomain):
    """
    Base class for generalized polynomial rings.

    This base class should be used for uniform access to generalized polynomial
    rings. Subclasses only supply information about the element storage etc.

    Do not instantiate.
    """
    has_assoc_Ring = True
    has_assoc_Field = True
    default_order = 'grevlex'

    def __init__(self, dom, *gens, **opts):
        if False:
            print('Hello World!')
        if not gens:
            raise GeneratorsNeeded('generators not specified')
        lev = len(gens) - 1
        self.ngens = len(gens)
        self.zero = self.dtype.zero(lev, dom)
        self.one = self.dtype.one(lev, dom)
        self.domain = self.dom = dom
        self.symbols = self.gens = gens
        self.order = opts.get('order', monomial_key(self.default_order))

    def new(self, element):
        if False:
            i = 10
            return i + 15
        return self.dtype(element, self.dom, len(self.gens) - 1)

    def _ground_new(self, element):
        if False:
            while True:
                i = 10
        return self.one.ground_new(element)

    def _from_dict(self, element):
        if False:
            while True:
                i = 10
        return DMP.from_dict(element, len(self.gens) - 1, self.dom)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        s_order = str(self.order)
        orderstr = ' order=' + s_order if s_order != self.default_order else ''
        return str(self.dom) + '[' + ','.join(map(str, self.gens)) + orderstr + ']'

    def __hash__(self):
        if False:
            return 10
        return hash((self.__class__.__name__, self.dtype, self.dom, self.gens, self.order))

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        'Returns ``True`` if two domains are equivalent. '
        return isinstance(other, PolynomialRingBase) and self.dtype == other.dtype and (self.dom == other.dom) and (self.gens == other.gens) and (self.order == other.order)

    @property
    def has_CharacteristicZero(self):
        if False:
            while True:
                i = 10
        return self.dom.has_CharacteristicZero

    def characteristic(self):
        if False:
            return 10
        return self.dom.characteristic()

    def from_ZZ(K1, a, K0):
        if False:
            print('Hello World!')
        'Convert a Python ``int`` object to ``dtype``. '
        return K1._ground_new(K1.dom.convert(a, K0))

    def from_ZZ_python(K1, a, K0):
        if False:
            i = 10
            return i + 15
        'Convert a Python ``int`` object to ``dtype``. '
        return K1._ground_new(K1.dom.convert(a, K0))

    def from_QQ(K1, a, K0):
        if False:
            return 10
        'Convert a Python ``Fraction`` object to ``dtype``. '
        return K1._ground_new(K1.dom.convert(a, K0))

    def from_QQ_python(K1, a, K0):
        if False:
            for i in range(10):
                print('nop')
        'Convert a Python ``Fraction`` object to ``dtype``. '
        return K1._ground_new(K1.dom.convert(a, K0))

    def from_ZZ_gmpy(K1, a, K0):
        if False:
            return 10
        'Convert a GMPY ``mpz`` object to ``dtype``. '
        return K1._ground_new(K1.dom.convert(a, K0))

    def from_QQ_gmpy(K1, a, K0):
        if False:
            while True:
                i = 10
        'Convert a GMPY ``mpq`` object to ``dtype``. '
        return K1._ground_new(K1.dom.convert(a, K0))

    def from_RealField(K1, a, K0):
        if False:
            i = 10
            return i + 15
        'Convert a mpmath ``mpf`` object to ``dtype``. '
        return K1._ground_new(K1.dom.convert(a, K0))

    def from_AlgebraicField(K1, a, K0):
        if False:
            i = 10
            return i + 15
        'Convert a ``ANP`` object to ``dtype``. '
        if K1.dom == K0:
            return K1._ground_new(a)

    def from_PolynomialRing(K1, a, K0):
        if False:
            for i in range(10):
                print('nop')
        'Convert a ``PolyElement`` object to ``dtype``. '
        if K1.gens == K0.symbols:
            if K1.dom == K0.dom:
                return K1(dict(a))
            else:
                convert_dom = lambda c: K1.dom.convert_from(c, K0.dom)
                return K1._from_dict({m: convert_dom(c) for (m, c) in a.items()})
        else:
            (monoms, coeffs) = _dict_reorder(a.to_dict(), K0.symbols, K1.gens)
            if K1.dom != K0.dom:
                coeffs = [K1.dom.convert(c, K0.dom) for c in coeffs]
            return K1._from_dict(dict(zip(monoms, coeffs)))

    def from_GlobalPolynomialRing(K1, a, K0):
        if False:
            return 10
        'Convert a ``DMP`` object to ``dtype``. '
        if K1.gens == K0.gens:
            if K1.dom != K0.dom:
                a = a.convert(K1.dom)
            return K1(a.to_list())
        else:
            (monoms, coeffs) = _dict_reorder(a.to_dict(), K0.gens, K1.gens)
            if K1.dom != K0.dom:
                coeffs = [K1.dom.convert(c, K0.dom) for c in coeffs]
            return K1(dict(zip(monoms, coeffs)))

    def get_field(self):
        if False:
            while True:
                i = 10
        'Returns a field associated with ``self``. '
        return FractionField(self.dom, *self.gens)

    def poly_ring(self, *gens):
        if False:
            for i in range(10):
                print('nop')
        'Returns a polynomial ring, i.e. ``K[X]``. '
        raise NotImplementedError('nested domains not allowed')

    def frac_field(self, *gens):
        if False:
            for i in range(10):
                print('nop')
        'Returns a fraction field, i.e. ``K(X)``. '
        raise NotImplementedError('nested domains not allowed')

    def revert(self, a):
        if False:
            print('Hello World!')
        try:
            return self.exquo(self.one, a)
        except (ExactQuotientFailed, ZeroDivisionError):
            raise NotReversible('%s is not a unit' % a)

    def gcdex(self, a, b):
        if False:
            while True:
                i = 10
        'Extended GCD of ``a`` and ``b``. '
        return a.gcdex(b)

    def gcd(self, a, b):
        if False:
            while True:
                i = 10
        'Returns GCD of ``a`` and ``b``. '
        return a.gcd(b)

    def lcm(self, a, b):
        if False:
            while True:
                i = 10
        'Returns LCM of ``a`` and ``b``. '
        return a.lcm(b)

    def factorial(self, a):
        if False:
            i = 10
            return i + 15
        'Returns factorial of ``a``. '
        return self.dtype(self.dom.factorial(a))

    def _vector_to_sdm(self, v, order):
        if False:
            while True:
                i = 10
        '\n        For internal use by the modules class.\n\n        Convert an iterable of elements of this ring into a sparse distributed\n        module element.\n        '
        raise NotImplementedError

    def _sdm_to_dics(self, s, n):
        if False:
            return 10
        'Helper for _sdm_to_vector.'
        from sympy.polys.distributedmodules import sdm_to_dict
        dic = sdm_to_dict(s)
        res = [{} for _ in range(n)]
        for (k, v) in dic.items():
            res[k[0]][k[1:]] = v
        return res

    def _sdm_to_vector(self, s, n):
        if False:
            print('Hello World!')
        '\n        For internal use by the modules class.\n\n        Convert a sparse distributed module into a list of length ``n``.\n\n        Examples\n        ========\n\n        >>> from sympy import QQ, ilex\n        >>> from sympy.abc import x, y\n        >>> R = QQ.old_poly_ring(x, y, order=ilex)\n        >>> L = [((1, 1, 1), QQ(1)), ((0, 1, 0), QQ(1)), ((0, 0, 1), QQ(2))]\n        >>> R._sdm_to_vector(L, 2)\n        [DMF([[1], [2, 0]], [[1]], QQ), DMF([[1, 0], []], [[1]], QQ)]\n        '
        dics = self._sdm_to_dics(s, n)
        return [self(x) for x in dics]

    def free_module(self, rank):
        if False:
            i = 10
            return i + 15
        '\n        Generate a free module of rank ``rank`` over ``self``.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import x\n        >>> from sympy import QQ\n        >>> QQ.old_poly_ring(x).free_module(2)\n        QQ[x]**2\n        '
        return FreeModulePolyRing(self, rank)

def _vector_to_sdm_helper(v, order):
    if False:
        print('Hello World!')
    'Helper method for common code in Global and Local poly rings.'
    from sympy.polys.distributedmodules import sdm_from_dict
    d = {}
    for (i, e) in enumerate(v):
        for (key, value) in e.to_dict().items():
            d[(i,) + key] = value
    return sdm_from_dict(d, order)

@public
class GlobalPolynomialRing(PolynomialRingBase):
    """A true polynomial ring, with objects DMP. """
    is_PolynomialRing = is_Poly = True
    dtype = DMP

    def new(self, element):
        if False:
            return 10
        if isinstance(element, dict):
            return DMP.from_dict(element, len(self.gens) - 1, self.dom)
        elif element in self.dom:
            return self._ground_new(self.dom.convert(element))
        else:
            return self.dtype(element, self.dom, len(self.gens) - 1)

    def from_FractionField(K1, a, K0):
        if False:
            return 10
        "\n        Convert a ``DMF`` object to ``DMP``.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.polyclasses import DMP, DMF\n        >>> from sympy.polys.domains import ZZ\n        >>> from sympy.abc import x\n\n        >>> f = DMF(([ZZ(1), ZZ(1)], [ZZ(1)]), ZZ)\n        >>> K = ZZ.old_frac_field(x)\n\n        >>> F = ZZ.old_poly_ring(x).from_FractionField(f, K)\n\n        >>> F == DMP([ZZ(1), ZZ(1)], ZZ)\n        True\n        >>> type(F)  # doctest: +SKIP\n        <class 'sympy.polys.polyclasses.DMP_Python'>\n\n        "
        if a.denom().is_one:
            return K1.from_GlobalPolynomialRing(a.numer(), K0)

    def to_sympy(self, a):
        if False:
            print('Hello World!')
        'Convert ``a`` to a SymPy object. '
        return basic_from_dict(a.to_sympy_dict(), *self.gens)

    def from_sympy(self, a):
        if False:
            for i in range(10):
                print('nop')
        "Convert SymPy's expression to ``dtype``. "
        try:
            (rep, _) = dict_from_basic(a, gens=self.gens)
        except PolynomialError:
            raise CoercionFailed('Cannot convert %s to type %s' % (a, self))
        for (k, v) in rep.items():
            rep[k] = self.dom.from_sympy(v)
        return DMP.from_dict(rep, self.ngens - 1, self.dom)

    def is_positive(self, a):
        if False:
            while True:
                i = 10
        'Returns True if ``LC(a)`` is positive. '
        return self.dom.is_positive(a.LC())

    def is_negative(self, a):
        if False:
            for i in range(10):
                print('nop')
        'Returns True if ``LC(a)`` is negative. '
        return self.dom.is_negative(a.LC())

    def is_nonpositive(self, a):
        if False:
            while True:
                i = 10
        'Returns True if ``LC(a)`` is non-positive. '
        return self.dom.is_nonpositive(a.LC())

    def is_nonnegative(self, a):
        if False:
            print('Hello World!')
        'Returns True if ``LC(a)`` is non-negative. '
        return self.dom.is_nonnegative(a.LC())

    def _vector_to_sdm(self, v, order):
        if False:
            for i in range(10):
                print('nop')
        '\n        Examples\n        ========\n\n        >>> from sympy import lex, QQ\n        >>> from sympy.abc import x, y\n        >>> R = QQ.old_poly_ring(x, y)\n        >>> f = R.convert(x + 2*y)\n        >>> g = R.convert(x * y)\n        >>> R._vector_to_sdm([f, g], lex)\n        [((1, 1, 1), 1), ((0, 1, 0), 1), ((0, 0, 1), 2)]\n        '
        return _vector_to_sdm_helper(v, order)

class GeneralizedPolynomialRing(PolynomialRingBase):
    """A generalized polynomial ring, with objects DMF. """
    dtype = DMF

    def new(self, a):
        if False:
            return 10
        'Construct an element of ``self`` domain from ``a``. '
        res = self.dtype(a, self.dom, len(self.gens) - 1)
        if res.denom().terms(order=self.order)[0][0] != (0,) * len(self.gens):
            from sympy.printing.str import sstr
            raise CoercionFailed('denominator %s not allowed in %s' % (sstr(res), self))
        return res

    def __contains__(self, a):
        if False:
            print('Hello World!')
        try:
            a = self.convert(a)
        except CoercionFailed:
            return False
        return a.denom().terms(order=self.order)[0][0] == (0,) * len(self.gens)

    def to_sympy(self, a):
        if False:
            print('Hello World!')
        'Convert ``a`` to a SymPy object. '
        return basic_from_dict(a.numer().to_sympy_dict(), *self.gens) / basic_from_dict(a.denom().to_sympy_dict(), *self.gens)

    def from_sympy(self, a):
        if False:
            return 10
        "Convert SymPy's expression to ``dtype``. "
        (p, q) = a.as_numer_denom()
        (num, _) = dict_from_basic(p, gens=self.gens)
        (den, _) = dict_from_basic(q, gens=self.gens)
        for (k, v) in num.items():
            num[k] = self.dom.from_sympy(v)
        for (k, v) in den.items():
            den[k] = self.dom.from_sympy(v)
        return self((num, den)).cancel()

    def exquo(self, a, b):
        if False:
            return 10
        'Exact quotient of ``a`` and ``b``. '
        r = a / b
        try:
            r = self.new((r.num, r.den))
        except CoercionFailed:
            raise ExactQuotientFailed(a, b, self)
        return r

    def from_FractionField(K1, a, K0):
        if False:
            while True:
                i = 10
        dmf = K1.get_field().from_FractionField(a, K0)
        return K1((dmf.num, dmf.den))

    def _vector_to_sdm(self, v, order):
        if False:
            for i in range(10):
                print('nop')
        '\n        Turn an iterable into a sparse distributed module.\n\n        Note that the vector is multiplied by a unit first to make all entries\n        polynomials.\n\n        Examples\n        ========\n\n        >>> from sympy import ilex, QQ\n        >>> from sympy.abc import x, y\n        >>> R = QQ.old_poly_ring(x, y, order=ilex)\n        >>> f = R.convert((x + 2*y) / (1 + x))\n        >>> g = R.convert(x * y)\n        >>> R._vector_to_sdm([f, g], ilex)\n        [((0, 0, 1), 2), ((0, 1, 0), 1), ((1, 1, 1), 1), ((1,\n          2, 1), 1)]\n        '
        u = self.one.numer()
        for x in v:
            u *= x.denom()
        return _vector_to_sdm_helper([x.numer() * u / x.denom() for x in v], order)

@public
def PolynomialRing(dom, *gens, **opts):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a generalized multivariate polynomial ring.\n\n    A generalized polynomial ring is defined by a ground field `K`, a set\n    of generators (typically `x_1, \\ldots, x_n`) and a monomial order `<`.\n    The monomial order can be global, local or mixed. In any case it induces\n    a total ordering on the monomials, and there exists for every (non-zero)\n    polynomial `f \\in K[x_1, \\ldots, x_n]` a well-defined "leading monomial"\n    `LM(f) = LM(f, >)`. One can then define a multiplicative subset\n    `S = S_> = \\{f \\in K[x_1, \\ldots, x_n] | LM(f) = 1\\}`. The generalized\n    polynomial ring corresponding to the monomial order is\n    `R = S^{-1}K[x_1, \\ldots, x_n]`.\n\n    If `>` is a so-called global order, that is `1` is the smallest monomial,\n    then we just have `S = K` and `R = K[x_1, \\ldots, x_n]`.\n\n    Examples\n    ========\n\n    A few examples may make this clearer.\n\n    >>> from sympy.abc import x, y\n    >>> from sympy import QQ\n\n    Our first ring uses global lexicographic order.\n\n    >>> R1 = QQ.old_poly_ring(x, y, order=(("lex", x, y),))\n\n    The second ring uses local lexicographic order. Note that when using a\n    single (non-product) order, you can just specify the name and omit the\n    variables:\n\n    >>> R2 = QQ.old_poly_ring(x, y, order="ilex")\n\n    The third and fourth rings use a mixed orders:\n\n    >>> o1 = (("ilex", x), ("lex", y))\n    >>> o2 = (("lex", x), ("ilex", y))\n    >>> R3 = QQ.old_poly_ring(x, y, order=o1)\n    >>> R4 = QQ.old_poly_ring(x, y, order=o2)\n\n    We will investigate what elements of `K(x, y)` are contained in the various\n    rings.\n\n    >>> L = [x, 1/x, y/(1 + x), 1/(1 + y), 1/(1 + x*y)]\n    >>> test = lambda R: [f in R for f in L]\n\n    The first ring is just `K[x, y]`:\n\n    >>> test(R1)\n    [True, False, False, False, False]\n\n    The second ring is R1 localised at the maximal ideal (x, y):\n\n    >>> test(R2)\n    [True, False, True, True, True]\n\n    The third ring is R1 localised at the prime ideal (x):\n\n    >>> test(R3)\n    [True, False, True, False, True]\n\n    Finally the fourth ring is R1 localised at `S = K[x, y] \\setminus yK[y]`:\n\n    >>> test(R4)\n    [True, False, False, True, False]\n    '
    order = opts.get('order', GeneralizedPolynomialRing.default_order)
    if iterable(order):
        order = build_product_order(order, gens)
    order = monomial_key(order)
    opts['order'] = order
    if order.is_global:
        return GlobalPolynomialRing(dom, *gens, **opts)
    else:
        return GeneralizedPolynomialRing(dom, *gens, **opts)