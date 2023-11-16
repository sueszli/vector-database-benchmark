"""Sparse polynomial rings. """
from __future__ import annotations
from typing import Any
from operator import add, mul, lt, le, gt, ge
from functools import reduce
from types import GeneratorType
from sympy.core.expr import Expr
from sympy.core.intfunc import igcd
from sympy.core.symbol import Symbol, symbols as _symbols
from sympy.core.sympify import CantSympify, sympify
from sympy.ntheory.multinomial import multinomial_coefficients
from sympy.polys.compatibility import IPolys
from sympy.polys.constructor import construct_domain
from sympy.polys.densebasic import ninf, dmp_to_dict, dmp_from_dict
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.domains.polynomialring import PolynomialRing
from sympy.polys.heuristicgcd import heugcd
from sympy.polys.monomials import MonomialOps
from sympy.polys.orderings import lex
from sympy.polys.polyerrors import CoercionFailed, GeneratorsError, ExactQuotientFailed, MultivariatePolynomialError
from sympy.polys.polyoptions import Domain as DomainOpt, Order as OrderOpt, build_options
from sympy.polys.polyutils import expr_from_dict, _dict_reorder, _parallel_dict_from_expr
from sympy.printing.defaults import DefaultPrinting
from sympy.utilities import public, subsets
from sympy.utilities.iterables import is_sequence
from sympy.utilities.magic import pollute

@public
def ring(symbols, domain, order=lex):
    if False:
        for i in range(10):
            print('nop')
    'Construct a polynomial ring returning ``(ring, x_1, ..., x_n)``.\n\n    Parameters\n    ==========\n\n    symbols : str\n        Symbol/Expr or sequence of str, Symbol/Expr (non-empty)\n    domain : :class:`~.Domain` or coercible\n    order : :class:`~.MonomialOrder` or coercible, optional, defaults to ``lex``\n\n    Examples\n    ========\n\n    >>> from sympy.polys.rings import ring\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.orderings import lex\n\n    >>> R, x, y, z = ring("x,y,z", ZZ, lex)\n    >>> R\n    Polynomial ring in x, y, z over ZZ with lex order\n    >>> x + y + z\n    x + y + z\n    >>> type(_)\n    <class \'sympy.polys.rings.PolyElement\'>\n\n    '
    _ring = PolyRing(symbols, domain, order)
    return (_ring,) + _ring.gens

@public
def xring(symbols, domain, order=lex):
    if False:
        while True:
            i = 10
    'Construct a polynomial ring returning ``(ring, (x_1, ..., x_n))``.\n\n    Parameters\n    ==========\n\n    symbols : str\n        Symbol/Expr or sequence of str, Symbol/Expr (non-empty)\n    domain : :class:`~.Domain` or coercible\n    order : :class:`~.MonomialOrder` or coercible, optional, defaults to ``lex``\n\n    Examples\n    ========\n\n    >>> from sympy.polys.rings import xring\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.orderings import lex\n\n    >>> R, (x, y, z) = xring("x,y,z", ZZ, lex)\n    >>> R\n    Polynomial ring in x, y, z over ZZ with lex order\n    >>> x + y + z\n    x + y + z\n    >>> type(_)\n    <class \'sympy.polys.rings.PolyElement\'>\n\n    '
    _ring = PolyRing(symbols, domain, order)
    return (_ring, _ring.gens)

@public
def vring(symbols, domain, order=lex):
    if False:
        print('Hello World!')
    'Construct a polynomial ring and inject ``x_1, ..., x_n`` into the global namespace.\n\n    Parameters\n    ==========\n\n    symbols : str\n        Symbol/Expr or sequence of str, Symbol/Expr (non-empty)\n    domain : :class:`~.Domain` or coercible\n    order : :class:`~.MonomialOrder` or coercible, optional, defaults to ``lex``\n\n    Examples\n    ========\n\n    >>> from sympy.polys.rings import vring\n    >>> from sympy.polys.domains import ZZ\n    >>> from sympy.polys.orderings import lex\n\n    >>> vring("x,y,z", ZZ, lex)\n    Polynomial ring in x, y, z over ZZ with lex order\n    >>> x + y + z # noqa:\n    x + y + z\n    >>> type(_)\n    <class \'sympy.polys.rings.PolyElement\'>\n\n    '
    _ring = PolyRing(symbols, domain, order)
    pollute([sym.name for sym in _ring.symbols], _ring.gens)
    return _ring

@public
def sring(exprs, *symbols, **options):
    if False:
        i = 10
        return i + 15
    'Construct a ring deriving generators and domain from options and input expressions.\n\n    Parameters\n    ==========\n\n    exprs : :class:`~.Expr` or sequence of :class:`~.Expr` (sympifiable)\n    symbols : sequence of :class:`~.Symbol`/:class:`~.Expr`\n    options : keyword arguments understood by :class:`~.Options`\n\n    Examples\n    ========\n\n    >>> from sympy import sring, symbols\n\n    >>> x, y, z = symbols("x,y,z")\n    >>> R, f = sring(x + 2*y + 3*z)\n    >>> R\n    Polynomial ring in x, y, z over ZZ with lex order\n    >>> f\n    x + 2*y + 3*z\n    >>> type(_)\n    <class \'sympy.polys.rings.PolyElement\'>\n\n    '
    single = False
    if not is_sequence(exprs):
        (exprs, single) = ([exprs], True)
    exprs = list(map(sympify, exprs))
    opt = build_options(symbols, options)
    (reps, opt) = _parallel_dict_from_expr(exprs, opt)
    if opt.domain is None:
        coeffs = sum([list(rep.values()) for rep in reps], [])
        (opt.domain, coeffs_dom) = construct_domain(coeffs, opt=opt)
        coeff_map = dict(zip(coeffs, coeffs_dom))
        reps = [{m: coeff_map[c] for (m, c) in rep.items()} for rep in reps]
    _ring = PolyRing(opt.gens, opt.domain, opt.order)
    polys = list(map(_ring.from_dict, reps))
    if single:
        return (_ring, polys[0])
    else:
        return (_ring, polys)

def _parse_symbols(symbols):
    if False:
        while True:
            i = 10
    if isinstance(symbols, str):
        return _symbols(symbols, seq=True) if symbols else ()
    elif isinstance(symbols, Expr):
        return (symbols,)
    elif is_sequence(symbols):
        if all((isinstance(s, str) for s in symbols)):
            return _symbols(symbols)
        elif all((isinstance(s, Expr) for s in symbols)):
            return symbols
    raise GeneratorsError('expected a string, Symbol or expression or a non-empty sequence of strings, Symbols or expressions')
_ring_cache: dict[Any, Any] = {}

class PolyRing(DefaultPrinting, IPolys):
    """Multivariate distributed polynomial ring. """

    def __new__(cls, symbols, domain, order=lex):
        if False:
            for i in range(10):
                print('nop')
        symbols = tuple(_parse_symbols(symbols))
        ngens = len(symbols)
        domain = DomainOpt.preprocess(domain)
        order = OrderOpt.preprocess(order)
        _hash_tuple = (cls.__name__, symbols, ngens, domain, order)
        obj = _ring_cache.get(_hash_tuple)
        if obj is None:
            if domain.is_Composite and set(symbols) & set(domain.symbols):
                raise GeneratorsError("polynomial ring and it's ground domain share generators")
            obj = object.__new__(cls)
            obj._hash_tuple = _hash_tuple
            obj._hash = hash(_hash_tuple)
            obj.dtype = type('PolyElement', (PolyElement,), {'ring': obj})
            obj.symbols = symbols
            obj.ngens = ngens
            obj.domain = domain
            obj.order = order
            obj.zero_monom = (0,) * ngens
            obj.gens = obj._gens()
            obj._gens_set = set(obj.gens)
            obj._one = [(obj.zero_monom, domain.one)]
            if ngens:
                codegen = MonomialOps(ngens)
                obj.monomial_mul = codegen.mul()
                obj.monomial_pow = codegen.pow()
                obj.monomial_mulpow = codegen.mulpow()
                obj.monomial_ldiv = codegen.ldiv()
                obj.monomial_div = codegen.div()
                obj.monomial_lcm = codegen.lcm()
                obj.monomial_gcd = codegen.gcd()
            else:
                monunit = lambda a, b: ()
                obj.monomial_mul = monunit
                obj.monomial_pow = monunit
                obj.monomial_mulpow = lambda a, b, c: ()
                obj.monomial_ldiv = monunit
                obj.monomial_div = monunit
                obj.monomial_lcm = monunit
                obj.monomial_gcd = monunit
            if order is lex:
                obj.leading_expv = max
            else:
                obj.leading_expv = lambda f: max(f, key=order)
            for (symbol, generator) in zip(obj.symbols, obj.gens):
                if isinstance(symbol, Symbol):
                    name = symbol.name
                    if not hasattr(obj, name):
                        setattr(obj, name, generator)
            _ring_cache[_hash_tuple] = obj
        return obj

    def _gens(self):
        if False:
            return 10
        'Return a list of polynomial generators. '
        one = self.domain.one
        _gens = []
        for i in range(self.ngens):
            expv = self.monomial_basis(i)
            poly = self.zero
            poly[expv] = one
            _gens.append(poly)
        return tuple(_gens)

    def __getnewargs__(self):
        if False:
            while True:
                i = 10
        return (self.symbols, self.domain, self.order)

    def __getstate__(self):
        if False:
            for i in range(10):
                print('nop')
        state = self.__dict__.copy()
        del state['leading_expv']
        for (key, value) in state.items():
            if key.startswith('monomial_'):
                del state[key]
        return state

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return self._hash

    def __eq__(self, other):
        if False:
            return 10
        return isinstance(other, PolyRing) and (self.symbols, self.domain, self.ngens, self.order) == (other.symbols, other.domain, other.ngens, other.order)

    def __ne__(self, other):
        if False:
            print('Hello World!')
        return not self == other

    def clone(self, symbols=None, domain=None, order=None):
        if False:
            while True:
                i = 10
        return self.__class__(symbols or self.symbols, domain or self.domain, order or self.order)

    def monomial_basis(self, i):
        if False:
            while True:
                i = 10
        'Return the ith-basis element. '
        basis = [0] * self.ngens
        basis[i] = 1
        return tuple(basis)

    @property
    def zero(self):
        if False:
            i = 10
            return i + 15
        return self.dtype()

    @property
    def one(self):
        if False:
            for i in range(10):
                print('nop')
        return self.dtype(self._one)

    def domain_new(self, element, orig_domain=None):
        if False:
            i = 10
            return i + 15
        return self.domain.convert(element, orig_domain)

    def ground_new(self, coeff):
        if False:
            i = 10
            return i + 15
        return self.term_new(self.zero_monom, coeff)

    def term_new(self, monom, coeff):
        if False:
            print('Hello World!')
        coeff = self.domain_new(coeff)
        poly = self.zero
        if coeff:
            poly[monom] = coeff
        return poly

    def ring_new(self, element):
        if False:
            return 10
        if isinstance(element, PolyElement):
            if self == element.ring:
                return element
            elif isinstance(self.domain, PolynomialRing) and self.domain.ring == element.ring:
                return self.ground_new(element)
            else:
                raise NotImplementedError('conversion')
        elif isinstance(element, str):
            raise NotImplementedError('parsing')
        elif isinstance(element, dict):
            return self.from_dict(element)
        elif isinstance(element, list):
            try:
                return self.from_terms(element)
            except ValueError:
                return self.from_list(element)
        elif isinstance(element, Expr):
            return self.from_expr(element)
        else:
            return self.ground_new(element)
    __call__ = ring_new

    def from_dict(self, element, orig_domain=None):
        if False:
            i = 10
            return i + 15
        domain_new = self.domain_new
        poly = self.zero
        for (monom, coeff) in element.items():
            coeff = domain_new(coeff, orig_domain)
            if coeff:
                poly[monom] = coeff
        return poly

    def from_terms(self, element, orig_domain=None):
        if False:
            i = 10
            return i + 15
        return self.from_dict(dict(element), orig_domain)

    def from_list(self, element):
        if False:
            i = 10
            return i + 15
        return self.from_dict(dmp_to_dict(element, self.ngens - 1, self.domain))

    def _rebuild_expr(self, expr, mapping):
        if False:
            while True:
                i = 10
        domain = self.domain

        def _rebuild(expr):
            if False:
                for i in range(10):
                    print('nop')
            generator = mapping.get(expr)
            if generator is not None:
                return generator
            elif expr.is_Add:
                return reduce(add, list(map(_rebuild, expr.args)))
            elif expr.is_Mul:
                return reduce(mul, list(map(_rebuild, expr.args)))
            else:
                (base, exp) = expr.as_base_exp()
                if exp.is_Integer and exp > 1:
                    return _rebuild(base) ** int(exp)
                else:
                    return self.ground_new(domain.convert(expr))
        return _rebuild(sympify(expr))

    def from_expr(self, expr):
        if False:
            for i in range(10):
                print('nop')
        mapping = dict(list(zip(self.symbols, self.gens)))
        try:
            poly = self._rebuild_expr(expr, mapping)
        except CoercionFailed:
            raise ValueError('expected an expression convertible to a polynomial in %s, got %s' % (self, expr))
        else:
            return self.ring_new(poly)

    def index(self, gen):
        if False:
            print('Hello World!')
        'Compute index of ``gen`` in ``self.gens``. '
        if gen is None:
            if self.ngens:
                i = 0
            else:
                i = -1
        elif isinstance(gen, int):
            i = gen
            if 0 <= i and i < self.ngens:
                pass
            elif -self.ngens <= i and i <= -1:
                i = -i - 1
            else:
                raise ValueError('invalid generator index: %s' % gen)
        elif isinstance(gen, self.dtype):
            try:
                i = self.gens.index(gen)
            except ValueError:
                raise ValueError('invalid generator: %s' % gen)
        elif isinstance(gen, str):
            try:
                i = self.symbols.index(gen)
            except ValueError:
                raise ValueError('invalid generator: %s' % gen)
        else:
            raise ValueError('expected a polynomial generator, an integer, a string or None, got %s' % gen)
        return i

    def drop(self, *gens):
        if False:
            return 10
        'Remove specified generators from this ring. '
        indices = set(map(self.index, gens))
        symbols = [s for (i, s) in enumerate(self.symbols) if i not in indices]
        if not symbols:
            return self.domain
        else:
            return self.clone(symbols=symbols)

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        symbols = self.symbols[key]
        if not symbols:
            return self.domain
        else:
            return self.clone(symbols=symbols)

    def to_ground(self):
        if False:
            i = 10
            return i + 15
        if self.domain.is_Composite or hasattr(self.domain, 'domain'):
            return self.clone(domain=self.domain.domain)
        else:
            raise ValueError('%s is not a composite domain' % self.domain)

    def to_domain(self):
        if False:
            i = 10
            return i + 15
        return PolynomialRing(self)

    def to_field(self):
        if False:
            return 10
        from sympy.polys.fields import FracField
        return FracField(self.symbols, self.domain, self.order)

    @property
    def is_univariate(self):
        if False:
            while True:
                i = 10
        return len(self.gens) == 1

    @property
    def is_multivariate(self):
        if False:
            while True:
                i = 10
        return len(self.gens) > 1

    def add(self, *objs):
        if False:
            print('Hello World!')
        '\n        Add a sequence of polynomials or containers of polynomials.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.rings import ring\n        >>> from sympy.polys.domains import ZZ\n\n        >>> R, x = ring("x", ZZ)\n        >>> R.add([ x**2 + 2*i + 3 for i in range(4) ])\n        4*x**2 + 24\n        >>> _.factor_list()\n        (4, [(x**2 + 6, 1)])\n\n        '
        p = self.zero
        for obj in objs:
            if is_sequence(obj, include=GeneratorType):
                p += self.add(*obj)
            else:
                p += obj
        return p

    def mul(self, *objs):
        if False:
            print('Hello World!')
        '\n        Multiply a sequence of polynomials or containers of polynomials.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.rings import ring\n        >>> from sympy.polys.domains import ZZ\n\n        >>> R, x = ring("x", ZZ)\n        >>> R.mul([ x**2 + 2*i + 3 for i in range(4) ])\n        x**8 + 24*x**6 + 206*x**4 + 744*x**2 + 945\n        >>> _.factor_list()\n        (1, [(x**2 + 3, 1), (x**2 + 5, 1), (x**2 + 7, 1), (x**2 + 9, 1)])\n\n        '
        p = self.one
        for obj in objs:
            if is_sequence(obj, include=GeneratorType):
                p *= self.mul(*obj)
            else:
                p *= obj
        return p

    def drop_to_ground(self, *gens):
        if False:
            i = 10
            return i + 15
        '\n        Remove specified generators from the ring and inject them into\n        its domain.\n        '
        indices = set(map(self.index, gens))
        symbols = [s for (i, s) in enumerate(self.symbols) if i not in indices]
        gens = [gen for (i, gen) in enumerate(self.gens) if i not in indices]
        if not symbols:
            return self
        else:
            return self.clone(symbols=symbols, domain=self.drop(*gens))

    def compose(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Add the generators of ``other`` to ``self``'
        if self != other:
            syms = set(self.symbols).union(set(other.symbols))
            return self.clone(symbols=list(syms))
        else:
            return self

    def add_gens(self, symbols):
        if False:
            print('Hello World!')
        'Add the elements of ``symbols`` as generators to ``self``'
        syms = set(self.symbols).union(set(symbols))
        return self.clone(symbols=list(syms))

    def symmetric_poly(self, n):
        if False:
            i = 10
            return i + 15
        "\n        Return the elementary symmetric polynomial of degree *n* over\n        this ring's generators.\n        "
        if n < 0 or n > self.ngens:
            raise ValueError('Cannot generate symmetric polynomial of order %s for %s' % (n, self.gens))
        elif not n:
            return self.one
        else:
            poly = self.zero
            for s in subsets(range(self.ngens), int(n)):
                monom = tuple((int(i in s) for i in range(self.ngens)))
                poly += self.term_new(monom, self.domain.one)
            return poly

class PolyElement(DomainElement, DefaultPrinting, CantSympify, dict):
    """Element of multivariate distributed polynomial ring. """

    def new(self, init):
        if False:
            for i in range(10):
                print('nop')
        return self.__class__(init)

    def parent(self):
        if False:
            for i in range(10):
                print('nop')
        return self.ring.to_domain()

    def __getnewargs__(self):
        if False:
            while True:
                i = 10
        return (self.ring, list(self.iterterms()))
    _hash = None

    def __hash__(self):
        if False:
            while True:
                i = 10
        _hash = self._hash
        if _hash is None:
            self._hash = _hash = hash((self.ring, frozenset(self.items())))
        return _hash

    def copy(self):
        if False:
            i = 10
            return i + 15
        "Return a copy of polynomial self.\n\n        Polynomials are mutable; if one is interested in preserving\n        a polynomial, and one plans to use inplace operations, one\n        can copy the polynomial. This method makes a shallow copy.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.domains import ZZ\n        >>> from sympy.polys.rings import ring\n\n        >>> R, x, y = ring('x, y', ZZ)\n        >>> p = (x + y)**2\n        >>> p1 = p.copy()\n        >>> p2 = p\n        >>> p[R.zero_monom] = 3\n        >>> p\n        x**2 + 2*x*y + y**2 + 3\n        >>> p1\n        x**2 + 2*x*y + y**2\n        >>> p2\n        x**2 + 2*x*y + y**2 + 3\n\n        "
        return self.new(self)

    def set_ring(self, new_ring):
        if False:
            while True:
                i = 10
        if self.ring == new_ring:
            return self
        elif self.ring.symbols != new_ring.symbols:
            terms = list(zip(*_dict_reorder(self, self.ring.symbols, new_ring.symbols)))
            return new_ring.from_terms(terms, self.ring.domain)
        else:
            return new_ring.from_dict(self, self.ring.domain)

    def as_expr(self, *symbols):
        if False:
            while True:
                i = 10
        if not symbols:
            symbols = self.ring.symbols
        elif len(symbols) != self.ring.ngens:
            raise ValueError('Wrong number of symbols, expected %s got %s' % (self.ring.ngens, len(symbols)))
        return expr_from_dict(self.as_expr_dict(), *symbols)

    def as_expr_dict(self):
        if False:
            print('Hello World!')
        to_sympy = self.ring.domain.to_sympy
        return {monom: to_sympy(coeff) for (monom, coeff) in self.iterterms()}

    def clear_denoms(self):
        if False:
            while True:
                i = 10
        domain = self.ring.domain
        if not domain.is_Field or not domain.has_assoc_Ring:
            return (domain.one, self)
        ground_ring = domain.get_ring()
        common = ground_ring.one
        lcm = ground_ring.lcm
        denom = domain.denom
        for coeff in self.values():
            common = lcm(common, denom(coeff))
        poly = self.new([(k, v * common) for (k, v) in self.items()])
        return (common, poly)

    def strip_zero(self):
        if False:
            i = 10
            return i + 15
        'Eliminate monomials with zero coefficient. '
        for (k, v) in list(self.items()):
            if not v:
                del self[k]

    def __eq__(p1, p2):
        if False:
            while True:
                i = 10
        "Equality test for polynomials.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.domains import ZZ\n        >>> from sympy.polys.rings import ring\n\n        >>> _, x, y = ring('x, y', ZZ)\n        >>> p1 = (x + y)**2 + (x - y)**2\n        >>> p1 == 4*x*y\n        False\n        >>> p1 == 2*(x**2 + y**2)\n        True\n\n        "
        if not p2:
            return not p1
        elif isinstance(p2, PolyElement) and p2.ring == p1.ring:
            return dict.__eq__(p1, p2)
        elif len(p1) > 1:
            return False
        else:
            return p1.get(p1.ring.zero_monom) == p2

    def __ne__(p1, p2):
        if False:
            for i in range(10):
                print('nop')
        return not p1 == p2

    def almosteq(p1, p2, tolerance=None):
        if False:
            for i in range(10):
                print('nop')
        'Approximate equality test for polynomials. '
        ring = p1.ring
        if isinstance(p2, ring.dtype):
            if set(p1.keys()) != set(p2.keys()):
                return False
            almosteq = ring.domain.almosteq
            for k in p1.keys():
                if not almosteq(p1[k], p2[k], tolerance):
                    return False
            return True
        elif len(p1) > 1:
            return False
        else:
            try:
                p2 = ring.domain.convert(p2)
            except CoercionFailed:
                return False
            else:
                return ring.domain.almosteq(p1.const(), p2, tolerance)

    def sort_key(self):
        if False:
            while True:
                i = 10
        return (len(self), self.terms())

    def _cmp(p1, p2, op):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(p2, p1.ring.dtype):
            return op(p1.sort_key(), p2.sort_key())
        else:
            return NotImplemented

    def __lt__(p1, p2):
        if False:
            i = 10
            return i + 15
        return p1._cmp(p2, lt)

    def __le__(p1, p2):
        if False:
            while True:
                i = 10
        return p1._cmp(p2, le)

    def __gt__(p1, p2):
        if False:
            return 10
        return p1._cmp(p2, gt)

    def __ge__(p1, p2):
        if False:
            for i in range(10):
                print('nop')
        return p1._cmp(p2, ge)

    def _drop(self, gen):
        if False:
            while True:
                i = 10
        ring = self.ring
        i = ring.index(gen)
        if ring.ngens == 1:
            return (i, ring.domain)
        else:
            symbols = list(ring.symbols)
            del symbols[i]
            return (i, ring.clone(symbols=symbols))

    def drop(self, gen):
        if False:
            i = 10
            return i + 15
        (i, ring) = self._drop(gen)
        if self.ring.ngens == 1:
            if self.is_ground:
                return self.coeff(1)
            else:
                raise ValueError('Cannot drop %s' % gen)
        else:
            poly = ring.zero
            for (k, v) in self.items():
                if k[i] == 0:
                    K = list(k)
                    del K[i]
                    poly[tuple(K)] = v
                else:
                    raise ValueError('Cannot drop %s' % gen)
            return poly

    def _drop_to_ground(self, gen):
        if False:
            for i in range(10):
                print('nop')
        ring = self.ring
        i = ring.index(gen)
        symbols = list(ring.symbols)
        del symbols[i]
        return (i, ring.clone(symbols=symbols, domain=ring[i]))

    def drop_to_ground(self, gen):
        if False:
            i = 10
            return i + 15
        if self.ring.ngens == 1:
            raise ValueError('Cannot drop only generator to ground')
        (i, ring) = self._drop_to_ground(gen)
        poly = ring.zero
        gen = ring.domain.gens[0]
        for (monom, coeff) in self.iterterms():
            mon = monom[:i] + monom[i + 1:]
            if mon not in poly:
                poly[mon] = (gen ** monom[i]).mul_ground(coeff)
            else:
                poly[mon] += (gen ** monom[i]).mul_ground(coeff)
        return poly

    def to_dense(self):
        if False:
            i = 10
            return i + 15
        return dmp_from_dict(self, self.ring.ngens - 1, self.ring.domain)

    def to_dict(self):
        if False:
            while True:
                i = 10
        return dict(self)

    def str(self, printer, precedence, exp_pattern, mul_symbol):
        if False:
            for i in range(10):
                print('nop')
        if not self:
            return printer._print(self.ring.domain.zero)
        prec_mul = precedence['Mul']
        prec_atom = precedence['Atom']
        ring = self.ring
        symbols = ring.symbols
        ngens = ring.ngens
        zm = ring.zero_monom
        sexpvs = []
        for (expv, coeff) in self.terms():
            negative = ring.domain.is_negative(coeff)
            sign = ' - ' if negative else ' + '
            sexpvs.append(sign)
            if expv == zm:
                scoeff = printer._print(coeff)
                if negative and scoeff.startswith('-'):
                    scoeff = scoeff[1:]
            else:
                if negative:
                    coeff = -coeff
                if coeff != self.ring.domain.one:
                    scoeff = printer.parenthesize(coeff, prec_mul, strict=True)
                else:
                    scoeff = ''
            sexpv = []
            for i in range(ngens):
                exp = expv[i]
                if not exp:
                    continue
                symbol = printer.parenthesize(symbols[i], prec_atom, strict=True)
                if exp != 1:
                    if exp != int(exp) or exp < 0:
                        sexp = printer.parenthesize(exp, prec_atom, strict=False)
                    else:
                        sexp = exp
                    sexpv.append(exp_pattern % (symbol, sexp))
                else:
                    sexpv.append('%s' % symbol)
            if scoeff:
                sexpv = [scoeff] + sexpv
            sexpvs.append(mul_symbol.join(sexpv))
        if sexpvs[0] in [' + ', ' - ']:
            head = sexpvs.pop(0)
            if head == ' - ':
                sexpvs.insert(0, '-')
        return ''.join(sexpvs)

    @property
    def is_generator(self):
        if False:
            for i in range(10):
                print('nop')
        return self in self.ring._gens_set

    @property
    def is_ground(self):
        if False:
            while True:
                i = 10
        return not self or (len(self) == 1 and self.ring.zero_monom in self)

    @property
    def is_monomial(self):
        if False:
            for i in range(10):
                print('nop')
        return not self or (len(self) == 1 and self.LC == 1)

    @property
    def is_term(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self) <= 1

    @property
    def is_negative(self):
        if False:
            for i in range(10):
                print('nop')
        return self.ring.domain.is_negative(self.LC)

    @property
    def is_positive(self):
        if False:
            return 10
        return self.ring.domain.is_positive(self.LC)

    @property
    def is_nonnegative(self):
        if False:
            for i in range(10):
                print('nop')
        return self.ring.domain.is_nonnegative(self.LC)

    @property
    def is_nonpositive(self):
        if False:
            print('Hello World!')
        return self.ring.domain.is_nonpositive(self.LC)

    @property
    def is_zero(f):
        if False:
            i = 10
            return i + 15
        return not f

    @property
    def is_one(f):
        if False:
            for i in range(10):
                print('nop')
        return f == f.ring.one

    @property
    def is_monic(f):
        if False:
            return 10
        return f.ring.domain.is_one(f.LC)

    @property
    def is_primitive(f):
        if False:
            while True:
                i = 10
        return f.ring.domain.is_one(f.content())

    @property
    def is_linear(f):
        if False:
            print('Hello World!')
        return all((sum(monom) <= 1 for monom in f.itermonoms()))

    @property
    def is_quadratic(f):
        if False:
            for i in range(10):
                print('nop')
        return all((sum(monom) <= 2 for monom in f.itermonoms()))

    @property
    def is_squarefree(f):
        if False:
            i = 10
            return i + 15
        if not f.ring.ngens:
            return True
        return f.ring.dmp_sqf_p(f)

    @property
    def is_irreducible(f):
        if False:
            print('Hello World!')
        if not f.ring.ngens:
            return True
        return f.ring.dmp_irreducible_p(f)

    @property
    def is_cyclotomic(f):
        if False:
            return 10
        if f.ring.is_univariate:
            return f.ring.dup_cyclotomic_p(f)
        else:
            raise MultivariatePolynomialError('cyclotomic polynomial')

    def __neg__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.new([(monom, -coeff) for (monom, coeff) in self.iterterms()])

    def __pos__(self):
        if False:
            return 10
        return self

    def __add__(p1, p2):
        if False:
            print('Hello World!')
        "Add two polynomials.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.domains import ZZ\n        >>> from sympy.polys.rings import ring\n\n        >>> _, x, y = ring('x, y', ZZ)\n        >>> (x + y)**2 + (x - y)**2\n        2*x**2 + 2*y**2\n\n        "
        if not p2:
            return p1.copy()
        ring = p1.ring
        if isinstance(p2, ring.dtype):
            p = p1.copy()
            get = p.get
            zero = ring.domain.zero
            for (k, v) in p2.items():
                v = get(k, zero) + v
                if v:
                    p[k] = v
                else:
                    del p[k]
            return p
        elif isinstance(p2, PolyElement):
            if isinstance(ring.domain, PolynomialRing) and ring.domain.ring == p2.ring:
                pass
            elif isinstance(p2.ring.domain, PolynomialRing) and p2.ring.domain.ring == ring:
                return p2.__radd__(p1)
            else:
                return NotImplemented
        try:
            cp2 = ring.domain_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            p = p1.copy()
            if not cp2:
                return p
            zm = ring.zero_monom
            if zm not in p1.keys():
                p[zm] = cp2
            elif p2 == -p[zm]:
                del p[zm]
            else:
                p[zm] += cp2
            return p

    def __radd__(p1, n):
        if False:
            return 10
        p = p1.copy()
        if not n:
            return p
        ring = p1.ring
        try:
            n = ring.domain_new(n)
        except CoercionFailed:
            return NotImplemented
        else:
            zm = ring.zero_monom
            if zm not in p1.keys():
                p[zm] = n
            elif n == -p[zm]:
                del p[zm]
            else:
                p[zm] += n
            return p

    def __sub__(p1, p2):
        if False:
            print('Hello World!')
        "Subtract polynomial p2 from p1.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.domains import ZZ\n        >>> from sympy.polys.rings import ring\n\n        >>> _, x, y = ring('x, y', ZZ)\n        >>> p1 = x + y**2\n        >>> p2 = x*y + y**2\n        >>> p1 - p2\n        -x*y + x\n\n        "
        if not p2:
            return p1.copy()
        ring = p1.ring
        if isinstance(p2, ring.dtype):
            p = p1.copy()
            get = p.get
            zero = ring.domain.zero
            for (k, v) in p2.items():
                v = get(k, zero) - v
                if v:
                    p[k] = v
                else:
                    del p[k]
            return p
        elif isinstance(p2, PolyElement):
            if isinstance(ring.domain, PolynomialRing) and ring.domain.ring == p2.ring:
                pass
            elif isinstance(p2.ring.domain, PolynomialRing) and p2.ring.domain.ring == ring:
                return p2.__rsub__(p1)
            else:
                return NotImplemented
        try:
            p2 = ring.domain_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            p = p1.copy()
            zm = ring.zero_monom
            if zm not in p1.keys():
                p[zm] = -p2
            elif p2 == p[zm]:
                del p[zm]
            else:
                p[zm] -= p2
            return p

    def __rsub__(p1, n):
        if False:
            i = 10
            return i + 15
        "n - p1 with n convertible to the coefficient domain.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.domains import ZZ\n        >>> from sympy.polys.rings import ring\n\n        >>> _, x, y = ring('x, y', ZZ)\n        >>> p = x + y\n        >>> 4 - p\n        -x - y + 4\n\n        "
        ring = p1.ring
        try:
            n = ring.domain_new(n)
        except CoercionFailed:
            return NotImplemented
        else:
            p = ring.zero
            for expv in p1:
                p[expv] = -p1[expv]
            p += n
            return p

    def __mul__(p1, p2):
        if False:
            return 10
        "Multiply two polynomials.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.domains import QQ\n        >>> from sympy.polys.rings import ring\n\n        >>> _, x, y = ring('x, y', QQ)\n        >>> p1 = x + y\n        >>> p2 = x - y\n        >>> p1*p2\n        x**2 - y**2\n\n        "
        ring = p1.ring
        p = ring.zero
        if not p1 or not p2:
            return p
        elif isinstance(p2, ring.dtype):
            get = p.get
            zero = ring.domain.zero
            monomial_mul = ring.monomial_mul
            p2it = list(p2.items())
            for (exp1, v1) in p1.items():
                for (exp2, v2) in p2it:
                    exp = monomial_mul(exp1, exp2)
                    p[exp] = get(exp, zero) + v1 * v2
            p.strip_zero()
            return p
        elif isinstance(p2, PolyElement):
            if isinstance(ring.domain, PolynomialRing) and ring.domain.ring == p2.ring:
                pass
            elif isinstance(p2.ring.domain, PolynomialRing) and p2.ring.domain.ring == ring:
                return p2.__rmul__(p1)
            else:
                return NotImplemented
        try:
            p2 = ring.domain_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            for (exp1, v1) in p1.items():
                v = v1 * p2
                if v:
                    p[exp1] = v
            return p

    def __rmul__(p1, p2):
        if False:
            print('Hello World!')
        "p2 * p1 with p2 in the coefficient domain of p1.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.domains import ZZ\n        >>> from sympy.polys.rings import ring\n\n        >>> _, x, y = ring('x, y', ZZ)\n        >>> p = x + y\n        >>> 4 * p\n        4*x + 4*y\n\n        "
        p = p1.ring.zero
        if not p2:
            return p
        try:
            p2 = p.ring.domain_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            for (exp1, v1) in p1.items():
                v = p2 * v1
                if v:
                    p[exp1] = v
            return p

    def __pow__(self, n):
        if False:
            print('Hello World!')
        "raise polynomial to power `n`\n\n        Examples\n        ========\n\n        >>> from sympy.polys.domains import ZZ\n        >>> from sympy.polys.rings import ring\n\n        >>> _, x, y = ring('x, y', ZZ)\n        >>> p = x + y**2\n        >>> p**3\n        x**3 + 3*x**2*y**2 + 3*x*y**4 + y**6\n\n        "
        ring = self.ring
        if not n:
            if self:
                return ring.one
            else:
                raise ValueError('0**0')
        elif len(self) == 1:
            (monom, coeff) = list(self.items())[0]
            p = ring.zero
            if coeff == ring.domain.one:
                p[ring.monomial_pow(monom, n)] = coeff
            else:
                p[ring.monomial_pow(monom, n)] = coeff ** n
            return p
        n = int(n)
        if n < 0:
            raise ValueError('Negative exponent')
        elif n == 1:
            return self.copy()
        elif n == 2:
            return self.square()
        elif n == 3:
            return self * self.square()
        elif len(self) <= 5:
            return self._pow_multinomial(n)
        else:
            return self._pow_generic(n)

    def _pow_generic(self, n):
        if False:
            return 10
        p = self.ring.one
        c = self
        while True:
            if n & 1:
                p = p * c
                n -= 1
                if not n:
                    break
            c = c.square()
            n = n // 2
        return p

    def _pow_multinomial(self, n):
        if False:
            return 10
        multinomials = multinomial_coefficients(len(self), n).items()
        monomial_mulpow = self.ring.monomial_mulpow
        zero_monom = self.ring.zero_monom
        terms = self.items()
        zero = self.ring.domain.zero
        poly = self.ring.zero
        for (multinomial, multinomial_coeff) in multinomials:
            product_monom = zero_monom
            product_coeff = multinomial_coeff
            for (exp, (monom, coeff)) in zip(multinomial, terms):
                if exp:
                    product_monom = monomial_mulpow(product_monom, monom, exp)
                    product_coeff *= coeff ** exp
            monom = tuple(product_monom)
            coeff = product_coeff
            coeff = poly.get(monom, zero) + coeff
            if coeff:
                poly[monom] = coeff
            elif monom in poly:
                del poly[monom]
        return poly

    def square(self):
        if False:
            for i in range(10):
                print('nop')
        "square of a polynomial\n\n        Examples\n        ========\n\n        >>> from sympy.polys.rings import ring\n        >>> from sympy.polys.domains import ZZ\n\n        >>> _, x, y = ring('x, y', ZZ)\n        >>> p = x + y**2\n        >>> p.square()\n        x**2 + 2*x*y**2 + y**4\n\n        "
        ring = self.ring
        p = ring.zero
        get = p.get
        keys = list(self.keys())
        zero = ring.domain.zero
        monomial_mul = ring.monomial_mul
        for i in range(len(keys)):
            k1 = keys[i]
            pk = self[k1]
            for j in range(i):
                k2 = keys[j]
                exp = monomial_mul(k1, k2)
                p[exp] = get(exp, zero) + pk * self[k2]
        p = p.imul_num(2)
        get = p.get
        for (k, v) in self.items():
            k2 = monomial_mul(k, k)
            p[k2] = get(k2, zero) + v ** 2
        p.strip_zero()
        return p

    def __divmod__(p1, p2):
        if False:
            print('Hello World!')
        ring = p1.ring
        if not p2:
            raise ZeroDivisionError('polynomial division')
        elif isinstance(p2, ring.dtype):
            return p1.div(p2)
        elif isinstance(p2, PolyElement):
            if isinstance(ring.domain, PolynomialRing) and ring.domain.ring == p2.ring:
                pass
            elif isinstance(p2.ring.domain, PolynomialRing) and p2.ring.domain.ring == ring:
                return p2.__rdivmod__(p1)
            else:
                return NotImplemented
        try:
            p2 = ring.domain_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            return (p1.quo_ground(p2), p1.rem_ground(p2))

    def __rdivmod__(p1, p2):
        if False:
            i = 10
            return i + 15
        return NotImplemented

    def __mod__(p1, p2):
        if False:
            for i in range(10):
                print('nop')
        ring = p1.ring
        if not p2:
            raise ZeroDivisionError('polynomial division')
        elif isinstance(p2, ring.dtype):
            return p1.rem(p2)
        elif isinstance(p2, PolyElement):
            if isinstance(ring.domain, PolynomialRing) and ring.domain.ring == p2.ring:
                pass
            elif isinstance(p2.ring.domain, PolynomialRing) and p2.ring.domain.ring == ring:
                return p2.__rmod__(p1)
            else:
                return NotImplemented
        try:
            p2 = ring.domain_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            return p1.rem_ground(p2)

    def __rmod__(p1, p2):
        if False:
            for i in range(10):
                print('nop')
        return NotImplemented

    def __truediv__(p1, p2):
        if False:
            print('Hello World!')
        ring = p1.ring
        if not p2:
            raise ZeroDivisionError('polynomial division')
        elif isinstance(p2, ring.dtype):
            if p2.is_monomial:
                return p1 * p2 ** (-1)
            else:
                return p1.quo(p2)
        elif isinstance(p2, PolyElement):
            if isinstance(ring.domain, PolynomialRing) and ring.domain.ring == p2.ring:
                pass
            elif isinstance(p2.ring.domain, PolynomialRing) and p2.ring.domain.ring == ring:
                return p2.__rtruediv__(p1)
            else:
                return NotImplemented
        try:
            p2 = ring.domain_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            return p1.quo_ground(p2)

    def __rtruediv__(p1, p2):
        if False:
            return 10
        return NotImplemented
    __floordiv__ = __truediv__
    __rfloordiv__ = __rtruediv__

    def _term_div(self):
        if False:
            return 10
        zm = self.ring.zero_monom
        domain = self.ring.domain
        domain_quo = domain.quo
        monomial_div = self.ring.monomial_div
        if domain.is_Field:

            def term_div(a_lm_a_lc, b_lm_b_lc):
                if False:
                    while True:
                        i = 10
                (a_lm, a_lc) = a_lm_a_lc
                (b_lm, b_lc) = b_lm_b_lc
                if b_lm == zm:
                    monom = a_lm
                else:
                    monom = monomial_div(a_lm, b_lm)
                if monom is not None:
                    return (monom, domain_quo(a_lc, b_lc))
                else:
                    return None
        else:

            def term_div(a_lm_a_lc, b_lm_b_lc):
                if False:
                    while True:
                        i = 10
                (a_lm, a_lc) = a_lm_a_lc
                (b_lm, b_lc) = b_lm_b_lc
                if b_lm == zm:
                    monom = a_lm
                else:
                    monom = monomial_div(a_lm, b_lm)
                if not (monom is None or a_lc % b_lc):
                    return (monom, domain_quo(a_lc, b_lc))
                else:
                    return None
        return term_div

    def div(self, fv):
        if False:
            for i in range(10):
                print('nop')
        "Division algorithm, see [CLO] p64.\n\n        fv array of polynomials\n           return qv, r such that\n           self = sum(fv[i]*qv[i]) + r\n\n        All polynomials are required not to be Laurent polynomials.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.rings import ring\n        >>> from sympy.polys.domains import ZZ\n\n        >>> _, x, y = ring('x, y', ZZ)\n        >>> f = x**3\n        >>> f0 = x - y**2\n        >>> f1 = x - y\n        >>> qv, r = f.div((f0, f1))\n        >>> qv[0]\n        x**2 + x*y**2 + y**4\n        >>> qv[1]\n        0\n        >>> r\n        y**6\n\n        "
        ring = self.ring
        ret_single = False
        if isinstance(fv, PolyElement):
            ret_single = True
            fv = [fv]
        if not all(fv):
            raise ZeroDivisionError('polynomial division')
        if not self:
            if ret_single:
                return (ring.zero, ring.zero)
            else:
                return ([], ring.zero)
        for f in fv:
            if f.ring != ring:
                raise ValueError('self and f must have the same ring')
        s = len(fv)
        qv = [ring.zero for i in range(s)]
        p = self.copy()
        r = ring.zero
        term_div = self._term_div()
        expvs = [fx.leading_expv() for fx in fv]
        while p:
            i = 0
            divoccurred = 0
            while i < s and divoccurred == 0:
                expv = p.leading_expv()
                term = term_div((expv, p[expv]), (expvs[i], fv[i][expvs[i]]))
                if term is not None:
                    (expv1, c) = term
                    qv[i] = qv[i]._iadd_monom((expv1, c))
                    p = p._iadd_poly_monom(fv[i], (expv1, -c))
                    divoccurred = 1
                else:
                    i += 1
            if not divoccurred:
                expv = p.leading_expv()
                r = r._iadd_monom((expv, p[expv]))
                del p[expv]
        if expv == ring.zero_monom:
            r += p
        if ret_single:
            if not qv:
                return (ring.zero, r)
            else:
                return (qv[0], r)
        else:
            return (qv, r)

    def rem(self, G):
        if False:
            while True:
                i = 10
        f = self
        if isinstance(G, PolyElement):
            G = [G]
        if not all(G):
            raise ZeroDivisionError('polynomial division')
        ring = f.ring
        domain = ring.domain
        zero = domain.zero
        monomial_mul = ring.monomial_mul
        r = ring.zero
        term_div = f._term_div()
        ltf = f.LT
        f = f.copy()
        get = f.get
        while f:
            for g in G:
                tq = term_div(ltf, g.LT)
                if tq is not None:
                    (m, c) = tq
                    for (mg, cg) in g.iterterms():
                        m1 = monomial_mul(mg, m)
                        c1 = get(m1, zero) - c * cg
                        if not c1:
                            del f[m1]
                        else:
                            f[m1] = c1
                    ltm = f.leading_expv()
                    if ltm is not None:
                        ltf = (ltm, f[ltm])
                    break
            else:
                (ltm, ltc) = ltf
                if ltm in r:
                    r[ltm] += ltc
                else:
                    r[ltm] = ltc
                del f[ltm]
                ltm = f.leading_expv()
                if ltm is not None:
                    ltf = (ltm, f[ltm])
        return r

    def quo(f, G):
        if False:
            for i in range(10):
                print('nop')
        return f.div(G)[0]

    def exquo(f, G):
        if False:
            for i in range(10):
                print('nop')
        (q, r) = f.div(G)
        if not r:
            return q
        else:
            raise ExactQuotientFailed(f, G)

    def _iadd_monom(self, mc):
        if False:
            print('Hello World!')
        "add to self the monomial coeff*x0**i0*x1**i1*...\n        unless self is a generator -- then just return the sum of the two.\n\n        mc is a tuple, (monom, coeff), where monomial is (i0, i1, ...)\n\n        Examples\n        ========\n\n        >>> from sympy.polys.rings import ring\n        >>> from sympy.polys.domains import ZZ\n\n        >>> _, x, y = ring('x, y', ZZ)\n        >>> p = x**4 + 2*y\n        >>> m = (1, 2)\n        >>> p1 = p._iadd_monom((m, 5))\n        >>> p1\n        x**4 + 5*x*y**2 + 2*y\n        >>> p1 is p\n        True\n        >>> p = x\n        >>> p1 = p._iadd_monom((m, 5))\n        >>> p1\n        5*x*y**2 + x\n        >>> p1 is p\n        False\n\n        "
        if self in self.ring._gens_set:
            cpself = self.copy()
        else:
            cpself = self
        (expv, coeff) = mc
        c = cpself.get(expv)
        if c is None:
            cpself[expv] = coeff
        else:
            c += coeff
            if c:
                cpself[expv] = c
            else:
                del cpself[expv]
        return cpself

    def _iadd_poly_monom(self, p2, mc):
        if False:
            while True:
                i = 10
        "add to self the product of (p)*(coeff*x0**i0*x1**i1*...)\n        unless self is a generator -- then just return the sum of the two.\n\n        mc is a tuple, (monom, coeff), where monomial is (i0, i1, ...)\n\n        Examples\n        ========\n\n        >>> from sympy.polys.rings import ring\n        >>> from sympy.polys.domains import ZZ\n\n        >>> _, x, y, z = ring('x, y, z', ZZ)\n        >>> p1 = x**4 + 2*y\n        >>> p2 = y + z\n        >>> m = (1, 2, 3)\n        >>> p1 = p1._iadd_poly_monom(p2, (m, 3))\n        >>> p1\n        x**4 + 3*x*y**3*z**3 + 3*x*y**2*z**4 + 2*y\n\n        "
        p1 = self
        if p1 in p1.ring._gens_set:
            p1 = p1.copy()
        (m, c) = mc
        get = p1.get
        zero = p1.ring.domain.zero
        monomial_mul = p1.ring.monomial_mul
        for (k, v) in p2.items():
            ka = monomial_mul(k, m)
            coeff = get(ka, zero) + v * c
            if coeff:
                p1[ka] = coeff
            else:
                del p1[ka]
        return p1

    def degree(f, x=None):
        if False:
            return 10
        "\n        The leading degree in ``x`` or the main variable.\n\n        Note that the degree of 0 is negative infinity (``float('-inf')``)\n\n        "
        i = f.ring.index(x)
        if not f:
            return ninf
        elif i < 0:
            return 0
        else:
            return max([monom[i] for monom in f.itermonoms()])

    def degrees(f):
        if False:
            print('Hello World!')
        "\n        A tuple containing leading degrees in all variables.\n\n        Note that the degree of 0 is negative infinity (``float('-inf')``)\n\n        "
        if not f:
            return (ninf,) * f.ring.ngens
        else:
            return tuple(map(max, list(zip(*f.itermonoms()))))

    def tail_degree(f, x=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        The tail degree in ``x`` or the main variable.\n\n        Note that the degree of 0 is negative infinity (``float('-inf')``)\n\n        "
        i = f.ring.index(x)
        if not f:
            return ninf
        elif i < 0:
            return 0
        else:
            return min([monom[i] for monom in f.itermonoms()])

    def tail_degrees(f):
        if False:
            print('Hello World!')
        "\n        A tuple containing tail degrees in all variables.\n\n        Note that the degree of 0 is negative infinity (``float('-inf')``)\n\n        "
        if not f:
            return (ninf,) * f.ring.ngens
        else:
            return tuple(map(min, list(zip(*f.itermonoms()))))

    def leading_expv(self):
        if False:
            i = 10
            return i + 15
        "Leading monomial tuple according to the monomial ordering.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.rings import ring\n        >>> from sympy.polys.domains import ZZ\n\n        >>> _, x, y, z = ring('x, y, z', ZZ)\n        >>> p = x**4 + x**3*y + x**2*z**2 + z**7\n        >>> p.leading_expv()\n        (4, 0, 0)\n\n        "
        if self:
            return self.ring.leading_expv(self)
        else:
            return None

    def _get_coeff(self, expv):
        if False:
            print('Hello World!')
        return self.get(expv, self.ring.domain.zero)

    def coeff(self, element):
        if False:
            print('Hello World!')
        '\n        Returns the coefficient that stands next to the given monomial.\n\n        Parameters\n        ==========\n\n        element : PolyElement (with ``is_monomial = True``) or 1\n\n        Examples\n        ========\n\n        >>> from sympy.polys.rings import ring\n        >>> from sympy.polys.domains import ZZ\n\n        >>> _, x, y, z = ring("x,y,z", ZZ)\n        >>> f = 3*x**2*y - x*y*z + 7*z**3 + 23\n\n        >>> f.coeff(x**2*y)\n        3\n        >>> f.coeff(x*y)\n        0\n        >>> f.coeff(1)\n        23\n\n        '
        if element == 1:
            return self._get_coeff(self.ring.zero_monom)
        elif isinstance(element, self.ring.dtype):
            terms = list(element.iterterms())
            if len(terms) == 1:
                (monom, coeff) = terms[0]
                if coeff == self.ring.domain.one:
                    return self._get_coeff(monom)
        raise ValueError('expected a monomial, got %s' % element)

    def const(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the constant coefficient. '
        return self._get_coeff(self.ring.zero_monom)

    @property
    def LC(self):
        if False:
            while True:
                i = 10
        return self._get_coeff(self.leading_expv())

    @property
    def LM(self):
        if False:
            i = 10
            return i + 15
        expv = self.leading_expv()
        if expv is None:
            return self.ring.zero_monom
        else:
            return expv

    def leading_monom(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Leading monomial as a polynomial element.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.rings import ring\n        >>> from sympy.polys.domains import ZZ\n\n        >>> _, x, y = ring('x, y', ZZ)\n        >>> (3*x*y + y**2).leading_monom()\n        x*y\n\n        "
        p = self.ring.zero
        expv = self.leading_expv()
        if expv:
            p[expv] = self.ring.domain.one
        return p

    @property
    def LT(self):
        if False:
            return 10
        expv = self.leading_expv()
        if expv is None:
            return (self.ring.zero_monom, self.ring.domain.zero)
        else:
            return (expv, self._get_coeff(expv))

    def leading_term(self):
        if False:
            while True:
                i = 10
        "Leading term as a polynomial element.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.rings import ring\n        >>> from sympy.polys.domains import ZZ\n\n        >>> _, x, y = ring('x, y', ZZ)\n        >>> (3*x*y + y**2).leading_term()\n        3*x*y\n\n        "
        p = self.ring.zero
        expv = self.leading_expv()
        if expv is not None:
            p[expv] = self[expv]
        return p

    def _sorted(self, seq, order):
        if False:
            while True:
                i = 10
        if order is None:
            order = self.ring.order
        else:
            order = OrderOpt.preprocess(order)
        if order is lex:
            return sorted(seq, key=lambda monom: monom[0], reverse=True)
        else:
            return sorted(seq, key=lambda monom: order(monom[0]), reverse=True)

    def coeffs(self, order=None):
        if False:
            i = 10
            return i + 15
        'Ordered list of polynomial coefficients.\n\n        Parameters\n        ==========\n\n        order : :class:`~.MonomialOrder` or coercible, optional\n\n        Examples\n        ========\n\n        >>> from sympy.polys.rings import ring\n        >>> from sympy.polys.domains import ZZ\n        >>> from sympy.polys.orderings import lex, grlex\n\n        >>> _, x, y = ring("x, y", ZZ, lex)\n        >>> f = x*y**7 + 2*x**2*y**3\n\n        >>> f.coeffs()\n        [2, 1]\n        >>> f.coeffs(grlex)\n        [1, 2]\n\n        '
        return [coeff for (_, coeff) in self.terms(order)]

    def monoms(self, order=None):
        if False:
            return 10
        'Ordered list of polynomial monomials.\n\n        Parameters\n        ==========\n\n        order : :class:`~.MonomialOrder` or coercible, optional\n\n        Examples\n        ========\n\n        >>> from sympy.polys.rings import ring\n        >>> from sympy.polys.domains import ZZ\n        >>> from sympy.polys.orderings import lex, grlex\n\n        >>> _, x, y = ring("x, y", ZZ, lex)\n        >>> f = x*y**7 + 2*x**2*y**3\n\n        >>> f.monoms()\n        [(2, 3), (1, 7)]\n        >>> f.monoms(grlex)\n        [(1, 7), (2, 3)]\n\n        '
        return [monom for (monom, _) in self.terms(order)]

    def terms(self, order=None):
        if False:
            for i in range(10):
                print('nop')
        'Ordered list of polynomial terms.\n\n        Parameters\n        ==========\n\n        order : :class:`~.MonomialOrder` or coercible, optional\n\n        Examples\n        ========\n\n        >>> from sympy.polys.rings import ring\n        >>> from sympy.polys.domains import ZZ\n        >>> from sympy.polys.orderings import lex, grlex\n\n        >>> _, x, y = ring("x, y", ZZ, lex)\n        >>> f = x*y**7 + 2*x**2*y**3\n\n        >>> f.terms()\n        [((2, 3), 2), ((1, 7), 1)]\n        >>> f.terms(grlex)\n        [((1, 7), 1), ((2, 3), 2)]\n\n        '
        return self._sorted(list(self.items()), order)

    def itercoeffs(self):
        if False:
            i = 10
            return i + 15
        'Iterator over coefficients of a polynomial. '
        return iter(self.values())

    def itermonoms(self):
        if False:
            return 10
        'Iterator over monomials of a polynomial. '
        return iter(self.keys())

    def iterterms(self):
        if False:
            i = 10
            return i + 15
        'Iterator over terms of a polynomial. '
        return iter(self.items())

    def listcoeffs(self):
        if False:
            while True:
                i = 10
        'Unordered list of polynomial coefficients. '
        return list(self.values())

    def listmonoms(self):
        if False:
            i = 10
            return i + 15
        'Unordered list of polynomial monomials. '
        return list(self.keys())

    def listterms(self):
        if False:
            for i in range(10):
                print('nop')
        'Unordered list of polynomial terms. '
        return list(self.items())

    def imul_num(p, c):
        if False:
            i = 10
            return i + 15
        "multiply inplace the polynomial p by an element in the\n        coefficient ring, provided p is not one of the generators;\n        else multiply not inplace\n\n        Examples\n        ========\n\n        >>> from sympy.polys.rings import ring\n        >>> from sympy.polys.domains import ZZ\n\n        >>> _, x, y = ring('x, y', ZZ)\n        >>> p = x + y**2\n        >>> p1 = p.imul_num(3)\n        >>> p1\n        3*x + 3*y**2\n        >>> p1 is p\n        True\n        >>> p = x\n        >>> p1 = p.imul_num(3)\n        >>> p1\n        3*x\n        >>> p1 is p\n        False\n\n        "
        if p in p.ring._gens_set:
            return p * c
        if not c:
            p.clear()
            return
        for exp in p:
            p[exp] *= c
        return p

    def content(f):
        if False:
            i = 10
            return i + 15
        "Returns GCD of polynomial's coefficients. "
        domain = f.ring.domain
        cont = domain.zero
        gcd = domain.gcd
        for coeff in f.itercoeffs():
            cont = gcd(cont, coeff)
        return cont

    def primitive(f):
        if False:
            while True:
                i = 10
        'Returns content and a primitive polynomial. '
        cont = f.content()
        return (cont, f.quo_ground(cont))

    def monic(f):
        if False:
            print('Hello World!')
        'Divides all coefficients by the leading coefficient. '
        if not f:
            return f
        else:
            return f.quo_ground(f.LC)

    def mul_ground(f, x):
        if False:
            while True:
                i = 10
        if not x:
            return f.ring.zero
        terms = [(monom, coeff * x) for (monom, coeff) in f.iterterms()]
        return f.new(terms)

    def mul_monom(f, monom):
        if False:
            for i in range(10):
                print('nop')
        monomial_mul = f.ring.monomial_mul
        terms = [(monomial_mul(f_monom, monom), f_coeff) for (f_monom, f_coeff) in f.items()]
        return f.new(terms)

    def mul_term(f, term):
        if False:
            return 10
        (monom, coeff) = term
        if not f or not coeff:
            return f.ring.zero
        elif monom == f.ring.zero_monom:
            return f.mul_ground(coeff)
        monomial_mul = f.ring.monomial_mul
        terms = [(monomial_mul(f_monom, monom), f_coeff * coeff) for (f_monom, f_coeff) in f.items()]
        return f.new(terms)

    def quo_ground(f, x):
        if False:
            print('Hello World!')
        domain = f.ring.domain
        if not x:
            raise ZeroDivisionError('polynomial division')
        if not f or x == domain.one:
            return f
        if domain.is_Field:
            quo = domain.quo
            terms = [(monom, quo(coeff, x)) for (monom, coeff) in f.iterterms()]
        else:
            terms = [(monom, coeff // x) for (monom, coeff) in f.iterterms() if not coeff % x]
        return f.new(terms)

    def quo_term(f, term):
        if False:
            for i in range(10):
                print('nop')
        (monom, coeff) = term
        if not coeff:
            raise ZeroDivisionError('polynomial division')
        elif not f:
            return f.ring.zero
        elif monom == f.ring.zero_monom:
            return f.quo_ground(coeff)
        term_div = f._term_div()
        terms = [term_div(t, term) for t in f.iterterms()]
        return f.new([t for t in terms if t is not None])

    def trunc_ground(f, p):
        if False:
            for i in range(10):
                print('nop')
        if f.ring.domain.is_ZZ:
            terms = []
            for (monom, coeff) in f.iterterms():
                coeff = coeff % p
                if coeff > p // 2:
                    coeff = coeff - p
                terms.append((monom, coeff))
        else:
            terms = [(monom, coeff % p) for (monom, coeff) in f.iterterms()]
        poly = f.new(terms)
        poly.strip_zero()
        return poly
    rem_ground = trunc_ground

    def extract_ground(self, g):
        if False:
            return 10
        f = self
        fc = f.content()
        gc = g.content()
        gcd = f.ring.domain.gcd(fc, gc)
        f = f.quo_ground(gcd)
        g = g.quo_ground(gcd)
        return (gcd, f, g)

    def _norm(f, norm_func):
        if False:
            while True:
                i = 10
        if not f:
            return f.ring.domain.zero
        else:
            ground_abs = f.ring.domain.abs
            return norm_func([ground_abs(coeff) for coeff in f.itercoeffs()])

    def max_norm(f):
        if False:
            i = 10
            return i + 15
        return f._norm(max)

    def l1_norm(f):
        if False:
            for i in range(10):
                print('nop')
        return f._norm(sum)

    def deflate(f, *G):
        if False:
            return 10
        ring = f.ring
        polys = [f] + list(G)
        J = [0] * ring.ngens
        for p in polys:
            for monom in p.itermonoms():
                for (i, m) in enumerate(monom):
                    J[i] = igcd(J[i], m)
        for (i, b) in enumerate(J):
            if not b:
                J[i] = 1
        J = tuple(J)
        if all((b == 1 for b in J)):
            return (J, polys)
        H = []
        for p in polys:
            h = ring.zero
            for (I, coeff) in p.iterterms():
                N = [i // j for (i, j) in zip(I, J)]
                h[tuple(N)] = coeff
            H.append(h)
        return (J, H)

    def inflate(f, J):
        if False:
            for i in range(10):
                print('nop')
        poly = f.ring.zero
        for (I, coeff) in f.iterterms():
            N = [i * j for (i, j) in zip(I, J)]
            poly[tuple(N)] = coeff
        return poly

    def lcm(self, g):
        if False:
            for i in range(10):
                print('nop')
        f = self
        domain = f.ring.domain
        if not domain.is_Field:
            (fc, f) = f.primitive()
            (gc, g) = g.primitive()
            c = domain.lcm(fc, gc)
        h = (f * g).quo(f.gcd(g))
        if not domain.is_Field:
            return h.mul_ground(c)
        else:
            return h.monic()

    def gcd(f, g):
        if False:
            i = 10
            return i + 15
        return f.cofactors(g)[0]

    def cofactors(f, g):
        if False:
            while True:
                i = 10
        if not f and (not g):
            zero = f.ring.zero
            return (zero, zero, zero)
        elif not f:
            (h, cff, cfg) = f._gcd_zero(g)
            return (h, cff, cfg)
        elif not g:
            (h, cfg, cff) = g._gcd_zero(f)
            return (h, cff, cfg)
        elif len(f) == 1:
            (h, cff, cfg) = f._gcd_monom(g)
            return (h, cff, cfg)
        elif len(g) == 1:
            (h, cfg, cff) = g._gcd_monom(f)
            return (h, cff, cfg)
        (J, (f, g)) = f.deflate(g)
        (h, cff, cfg) = f._gcd(g)
        return (h.inflate(J), cff.inflate(J), cfg.inflate(J))

    def _gcd_zero(f, g):
        if False:
            i = 10
            return i + 15
        (one, zero) = (f.ring.one, f.ring.zero)
        if g.is_nonnegative:
            return (g, zero, one)
        else:
            return (-g, zero, -one)

    def _gcd_monom(f, g):
        if False:
            while True:
                i = 10
        ring = f.ring
        ground_gcd = ring.domain.gcd
        ground_quo = ring.domain.quo
        monomial_gcd = ring.monomial_gcd
        monomial_ldiv = ring.monomial_ldiv
        (mf, cf) = list(f.iterterms())[0]
        (_mgcd, _cgcd) = (mf, cf)
        for (mg, cg) in g.iterterms():
            _mgcd = monomial_gcd(_mgcd, mg)
            _cgcd = ground_gcd(_cgcd, cg)
        h = f.new([(_mgcd, _cgcd)])
        cff = f.new([(monomial_ldiv(mf, _mgcd), ground_quo(cf, _cgcd))])
        cfg = f.new([(monomial_ldiv(mg, _mgcd), ground_quo(cg, _cgcd)) for (mg, cg) in g.iterterms()])
        return (h, cff, cfg)

    def _gcd(f, g):
        if False:
            return 10
        ring = f.ring
        if ring.domain.is_QQ:
            return f._gcd_QQ(g)
        elif ring.domain.is_ZZ:
            return f._gcd_ZZ(g)
        else:
            return ring.dmp_inner_gcd(f, g)

    def _gcd_ZZ(f, g):
        if False:
            return 10
        return heugcd(f, g)

    def _gcd_QQ(self, g):
        if False:
            return 10
        f = self
        ring = f.ring
        new_ring = ring.clone(domain=ring.domain.get_ring())
        (cf, f) = f.clear_denoms()
        (cg, g) = g.clear_denoms()
        f = f.set_ring(new_ring)
        g = g.set_ring(new_ring)
        (h, cff, cfg) = f._gcd_ZZ(g)
        h = h.set_ring(ring)
        (c, h) = (h.LC, h.monic())
        cff = cff.set_ring(ring).mul_ground(ring.domain.quo(c, cf))
        cfg = cfg.set_ring(ring).mul_ground(ring.domain.quo(c, cg))
        return (h, cff, cfg)

    def cancel(self, g):
        if False:
            print('Hello World!')
        '\n        Cancel common factors in a rational function ``f/g``.\n\n        Examples\n        ========\n\n        >>> from sympy.polys import ring, ZZ\n        >>> R, x,y = ring("x,y", ZZ)\n\n        >>> (2*x**2 - 2).cancel(x**2 - 2*x + 1)\n        (2*x + 2, x - 1)\n\n        '
        f = self
        ring = f.ring
        if not f:
            return (f, ring.one)
        domain = ring.domain
        if not (domain.is_Field and domain.has_assoc_Ring):
            (_, p, q) = f.cofactors(g)
        else:
            new_ring = ring.clone(domain=domain.get_ring())
            (cq, f) = f.clear_denoms()
            (cp, g) = g.clear_denoms()
            f = f.set_ring(new_ring)
            g = g.set_ring(new_ring)
            (_, p, q) = f.cofactors(g)
            (_, cp, cq) = new_ring.domain.cofactors(cp, cq)
            p = p.set_ring(ring)
            q = q.set_ring(ring)
            p = p.mul_ground(cp)
            q = q.mul_ground(cq)
        u = q.canonical_unit()
        if u == domain.one:
            (p, q) = (p, q)
        elif u == -domain.one:
            (p, q) = (-p, -q)
        else:
            p = p.mul_ground(u)
            q = q.mul_ground(u)
        return (p, q)

    def canonical_unit(f):
        if False:
            for i in range(10):
                print('nop')
        domain = f.ring.domain
        return domain.canonical_unit(f.LC)

    def diff(f, x):
        if False:
            i = 10
            return i + 15
        'Computes partial derivative in ``x``.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.rings import ring\n        >>> from sympy.polys.domains import ZZ\n\n        >>> _, x, y = ring("x,y", ZZ)\n        >>> p = x + x**2*y**3\n        >>> p.diff(x)\n        2*x*y**3 + 1\n\n        '
        ring = f.ring
        i = ring.index(x)
        m = ring.monomial_basis(i)
        g = ring.zero
        for (expv, coeff) in f.iterterms():
            if expv[i]:
                e = ring.monomial_ldiv(expv, m)
                g[e] = ring.domain_new(coeff * expv[i])
        return g

    def __call__(f, *values):
        if False:
            while True:
                i = 10
        if 0 < len(values) <= f.ring.ngens:
            return f.evaluate(list(zip(f.ring.gens, values)))
        else:
            raise ValueError('expected at least 1 and at most %s values, got %s' % (f.ring.ngens, len(values)))

    def evaluate(self, x, a=None):
        if False:
            print('Hello World!')
        f = self
        if isinstance(x, list) and a is None:
            ((X, a), x) = (x[0], x[1:])
            f = f.evaluate(X, a)
            if not x:
                return f
            else:
                x = [(Y.drop(X), a) for (Y, a) in x]
                return f.evaluate(x)
        ring = f.ring
        i = ring.index(x)
        a = ring.domain.convert(a)
        if ring.ngens == 1:
            result = ring.domain.zero
            for ((n,), coeff) in f.iterterms():
                result += coeff * a ** n
            return result
        else:
            poly = ring.drop(x).zero
            for (monom, coeff) in f.iterterms():
                (n, monom) = (monom[i], monom[:i] + monom[i + 1:])
                coeff = coeff * a ** n
                if monom in poly:
                    coeff = coeff + poly[monom]
                    if coeff:
                        poly[monom] = coeff
                    else:
                        del poly[monom]
                elif coeff:
                    poly[monom] = coeff
            return poly

    def subs(self, x, a=None):
        if False:
            i = 10
            return i + 15
        f = self
        if isinstance(x, list) and a is None:
            for (X, a) in x:
                f = f.subs(X, a)
            return f
        ring = f.ring
        i = ring.index(x)
        a = ring.domain.convert(a)
        if ring.ngens == 1:
            result = ring.domain.zero
            for ((n,), coeff) in f.iterterms():
                result += coeff * a ** n
            return ring.ground_new(result)
        else:
            poly = ring.zero
            for (monom, coeff) in f.iterterms():
                (n, monom) = (monom[i], monom[:i] + (0,) + monom[i + 1:])
                coeff = coeff * a ** n
                if monom in poly:
                    coeff = coeff + poly[monom]
                    if coeff:
                        poly[monom] = coeff
                    else:
                        del poly[monom]
                elif coeff:
                    poly[monom] = coeff
            return poly

    def symmetrize(self):
        if False:
            return 10
        '\n        Rewrite *self* in terms of elementary symmetric polynomials.\n\n        Explanation\n        ===========\n\n        If this :py:class:`~.PolyElement` belongs to a ring of $n$ variables,\n        we can try to write it as a function of the elementary symmetric\n        polynomials on $n$ variables. We compute a symmetric part, and a\n        remainder for any part we were not able to symmetrize.\n\n        Examples\n        ========\n\n        >>> from sympy.polys.rings import ring\n        >>> from sympy.polys.domains import ZZ\n        >>> R, x, y = ring("x,y", ZZ)\n\n        >>> f = x**2 + y**2\n        >>> f.symmetrize()\n        (x**2 - 2*y, 0, [(x, x + y), (y, x*y)])\n\n        >>> f = x**2 - y**2\n        >>> f.symmetrize()\n        (x**2 - 2*y, -2*y**2, [(x, x + y), (y, x*y)])\n\n        Returns\n        =======\n\n        Triple ``(p, r, m)``\n            ``p`` is a :py:class:`~.PolyElement` that represents our attempt\n            to express *self* as a function of elementary symmetric\n            polynomials. Each variable in ``p`` stands for one of the\n            elementary symmetric polynomials. The correspondence is given\n            by ``m``.\n\n            ``r`` is the remainder.\n\n            ``m`` is a list of pairs, giving the mapping from variables in\n            ``p`` to elementary symmetric polynomials.\n\n            The triple satisfies the equation ``p.compose(m) + r == self``.\n            If the remainder ``r`` is zero, *self* is symmetric. If it is\n            nonzero, we were not able to represent *self* as symmetric.\n\n        See Also\n        ========\n\n        sympy.polys.polyfuncs.symmetrize\n\n        References\n        ==========\n\n        .. [1] Lauer, E. Algorithms for symmetrical polynomials, Proc. 1976\n            ACM Symp. on Symbolic and Algebraic Computing, NY 242-247.\n            https://dl.acm.org/doi/pdf/10.1145/800205.806342\n\n        '
        f = self.copy()
        ring = f.ring
        n = ring.ngens
        if not n:
            return (f, ring.zero, [])
        polys = [ring.symmetric_poly(i + 1) for i in range(n)]
        poly_powers = {}

        def get_poly_power(i, n):
            if False:
                print('Hello World!')
            if (i, n) not in poly_powers:
                poly_powers[i, n] = polys[i] ** n
            return poly_powers[i, n]
        indices = list(range(n - 1))
        weights = list(range(n, 0, -1))
        symmetric = ring.zero
        while f:
            (_height, _monom, _coeff) = (-1, None, None)
            for (i, (monom, coeff)) in enumerate(f.terms()):
                if all((monom[i] >= monom[i + 1] for i in indices)):
                    height = max([n * m for (n, m) in zip(weights, monom)])
                    if height > _height:
                        (_height, _monom, _coeff) = (height, monom, coeff)
            if _height != -1:
                (monom, coeff) = (_monom, _coeff)
            else:
                break
            exponents = []
            for (m1, m2) in zip(monom, monom[1:] + (0,)):
                exponents.append(m1 - m2)
            symmetric += ring.term_new(tuple(exponents), coeff)
            product = coeff
            for (i, n) in enumerate(exponents):
                product *= get_poly_power(i, n)
            f -= product
        mapping = list(zip(ring.gens, polys))
        return (symmetric, f, mapping)

    def compose(f, x, a=None):
        if False:
            for i in range(10):
                print('nop')
        ring = f.ring
        poly = ring.zero
        gens_map = dict(zip(ring.gens, range(ring.ngens)))
        if a is not None:
            replacements = [(x, a)]
        elif isinstance(x, list):
            replacements = list(x)
        elif isinstance(x, dict):
            replacements = sorted(x.items(), key=lambda k: gens_map[k[0]])
        else:
            raise ValueError('expected a generator, value pair a sequence of such pairs')
        for (k, (x, g)) in enumerate(replacements):
            replacements[k] = (gens_map[x], ring.ring_new(g))
        for (monom, coeff) in f.iterterms():
            monom = list(monom)
            subpoly = ring.one
            for (i, g) in replacements:
                (n, monom[i]) = (monom[i], 0)
                if n:
                    subpoly *= g ** n
            subpoly = subpoly.mul_term((tuple(monom), coeff))
            poly += subpoly
        return poly

    def coeff_wrt(self, x, deg):
        if False:
            while True:
                i = 10
        '\n        Coefficient of ``self`` with respect to ``x**deg``.\n\n        Treating ``self`` as a univariate polynomial in ``x`` this finds the\n        coefficient of ``x**deg`` as a polynomial in the other generators.\n\n        Parameters\n        ==========\n\n        x : generator or generator index\n            The generator or generator index to compute the expression for.\n        deg : int\n            The degree of the monomial to compute the expression for.\n\n        Returns\n        =======\n\n        :py:class:`~.PolyElement`\n            The coefficient of ``x**deg`` as a polynomial in the same ring.\n\n        Examples\n        ========\n\n        >>> from sympy.polys import ring, ZZ\n        >>> R, x, y, z = ring("x, y, z", ZZ)\n\n        >>> p = 2*x**4 + 3*y**4 + 10*z**2 + 10*x*z**2\n        >>> deg = 2\n        >>> p.coeff_wrt(2, deg) # Using the generator index\n        10*x + 10\n        >>> p.coeff_wrt(z, deg) # Using the generator\n        10*x + 10\n        >>> p.coeff(z**2) # shows the difference between coeff and coeff_wrt\n        10\n\n        See Also\n        ========\n\n        coeff, coeffs\n\n        '
        p = self
        i = p.ring.index(x)
        terms = [(m, c) for (m, c) in p.iterterms() if m[i] == deg]
        if not terms:
            return p.ring.zero
        (monoms, coeffs) = zip(*terms)
        monoms = [m[:i] + (0,) + m[i + 1:] for m in monoms]
        return p.ring.from_dict(dict(zip(monoms, coeffs)))

    def prem(self, g, x=None):
        if False:
            print('Hello World!')
        '\n        Pseudo-remainder of the polynomial ``self`` with respect to ``g``.\n\n        The pseudo-quotient ``q`` and pseudo-remainder ``r`` with respect to\n        ``z`` when dividing ``f`` by ``g`` satisfy ``m*f = g*q + r``,\n        where ``deg(r,z) < deg(g,z)`` and\n        ``m = LC(g,z)**(deg(f,z) - deg(g,z)+1)``.\n\n        See :meth:`pdiv` for explanation of pseudo-division.\n\n\n        Parameters\n        ==========\n\n        g : :py:class:`~.PolyElement`\n            The polynomial to divide ``self`` by.\n        x : generator or generator index, optional\n            The main variable of the polynomials and default is first generator.\n\n        Returns\n        =======\n\n        :py:class:`~.PolyElement`\n            The pseudo-remainder polynomial.\n\n        Raises\n        ======\n\n        ZeroDivisionError : If ``g`` is the zero polynomial.\n\n        Examples\n        ========\n\n        >>> from sympy.polys import ring, ZZ\n        >>> R, x, y = ring("x, y", ZZ)\n\n        >>> f = x**2 + x*y\n        >>> g = 2*x + 2\n        >>> f.prem(g) # first generator is chosen by default if it is not given\n        -4*y + 4\n        >>> f.rem(g) # shows the differnce between prem and rem\n        x**2 + x*y\n        >>> f.prem(g, y) # generator is given\n        0\n        >>> f.prem(g, 1) # generator index is given\n        0\n\n        See Also\n        ========\n\n        pdiv, pquo, pexquo, sympy.polys.domains.ring.Ring.rem\n\n        '
        f = self
        x = f.ring.index(x)
        df = f.degree(x)
        dg = g.degree(x)
        if dg < 0:
            raise ZeroDivisionError('polynomial division')
        (r, dr) = (f, df)
        if df < dg:
            return r
        N = df - dg + 1
        lc_g = g.coeff_wrt(x, dg)
        xp = f.ring.gens[x]
        while True:
            lc_r = r.coeff_wrt(x, dr)
            (j, N) = (dr - dg, N - 1)
            R = r * lc_g
            G = g * lc_r * xp ** j
            r = R - G
            dr = r.degree(x)
            if dr < dg:
                break
        c = lc_g ** N
        return r * c

    def pdiv(self, g, x=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Computes the pseudo-division of the polynomial ``self`` with respect to ``g``.\n\n        The pseudo-division algorithm is used to find the pseudo-quotient ``q``\n        and pseudo-remainder ``r`` such that ``m*f = g*q + r``, where ``m``\n        represents the multiplier and ``f`` is the dividend polynomial.\n\n        The pseudo-quotient ``q`` and pseudo-remainder ``r`` are polynomials in\n        the variable ``x``, with the degree of ``r`` with respect to ``x``\n        being strictly less than the degree of ``g`` with respect to ``x``.\n\n        The multiplier ``m`` is defined as\n        ``LC(g, x) ^ (deg(f, x) - deg(g, x) + 1)``,\n        where ``LC(g, x)`` represents the leading coefficient of ``g``.\n\n        It is important to note that in the context of the ``prem`` method,\n        multivariate polynomials in a ring, such as ``R[x,y,z]``, are treated\n        as univariate polynomials with coefficients that are polynomials,\n        such as ``R[x,y][z]``. When dividing ``f`` by ``g`` with respect to the\n        variable ``z``, the pseudo-quotient ``q`` and pseudo-remainder ``r``\n        satisfy ``m*f = g*q + r``, where ``deg(r, z) < deg(g, z)``\n        and ``m = LC(g, z)^(deg(f, z) - deg(g, z) + 1)``.\n\n        In this function, the pseudo-remainder ``r`` can be obtained using the\n        ``prem`` method, the pseudo-quotient ``q`` can\n        be obtained using the ``pquo`` method, and\n        the function ``pdiv`` itself returns a tuple ``(q, r)``.\n\n\n        Parameters\n        ==========\n\n        g : :py:class:`~.PolyElement`\n            The polynomial to divide ``self`` by.\n        x : generator or generator index, optional\n            The main variable of the polynomials and default is first generator.\n\n        Returns\n        =======\n\n        :py:class:`~.PolyElement`\n            The pseudo-division polynomial (tuple of ``q`` and ``r``).\n\n        Raises\n        ======\n\n        ZeroDivisionError : If ``g`` is the zero polynomial.\n\n        Examples\n        ========\n\n        >>> from sympy.polys import ring, ZZ\n        >>> R, x, y = ring("x, y", ZZ)\n\n        >>> f = x**2 + x*y\n        >>> g = 2*x + 2\n        >>> f.pdiv(g) # first generator is chosen by default if it is not given\n        (2*x + 2*y - 2, -4*y + 4)\n        >>> f.div(g) # shows the difference between pdiv and div\n        (0, x**2 + x*y)\n        >>> f.pdiv(g, y) # generator is given\n        (2*x**3 + 2*x**2*y + 6*x**2 + 2*x*y + 8*x + 4, 0)\n        >>> f.pdiv(g, 1) # generator index is given\n        (2*x**3 + 2*x**2*y + 6*x**2 + 2*x*y + 8*x + 4, 0)\n\n        See Also\n        ========\n\n        prem\n            Computes only the pseudo-remainder more efficiently than\n            `f.pdiv(g)[1]`.\n        pquo\n            Returns only the pseudo-quotient.\n        pexquo\n            Returns only an exact pseudo-quotient having no remainder.\n        div\n            Returns quotient and remainder of f and g polynomials.\n\n        '
        f = self
        x = f.ring.index(x)
        df = f.degree(x)
        dg = g.degree(x)
        if dg < 0:
            raise ZeroDivisionError('polynomial division')
        (q, r, dr) = (x, f, df)
        if df < dg:
            return (q, r)
        N = df - dg + 1
        lc_g = g.coeff_wrt(x, dg)
        xp = f.ring.gens[x]
        while True:
            lc_r = r.coeff_wrt(x, dr)
            (j, N) = (dr - dg, N - 1)
            Q = q * lc_g
            q = Q + lc_r * xp ** j
            R = r * lc_g
            G = g * lc_r * xp ** j
            r = R - G
            dr = r.degree(x)
            if dr < dg:
                break
        c = lc_g ** N
        q = q * c
        r = r * c
        return (q, r)

    def pquo(self, g, x=None):
        if False:
            print('Hello World!')
        '\n        Polynomial pseudo-quotient in multivariate polynomial ring.\n\n        Examples\n        ========\n        >>> from sympy.polys import ring, ZZ\n        >>> R, x,y = ring("x,y", ZZ)\n\n        >>> f = x**2 + x*y\n        >>> g = 2*x + 2*y\n        >>> h = 2*x + 2\n        >>> f.pquo(g)\n        2*x\n        >>> f.quo(g) # shows the difference between pquo and quo\n        0\n        >>> f.pquo(h)\n        2*x + 2*y - 2\n        >>> f.quo(h) # shows the difference between pquo and quo\n        0\n\n        See Also\n        ========\n\n        prem, pdiv, pexquo, sympy.polys.domains.ring.Ring.quo\n\n        '
        f = self
        return f.pdiv(g, x)[0]

    def pexquo(self, g, x=None):
        if False:
            i = 10
            return i + 15
        '\n        Polynomial exact pseudo-quotient in multivariate polynomial ring.\n\n        Examples\n        ========\n        >>> from sympy.polys import ring, ZZ\n        >>> R, x,y = ring("x,y", ZZ)\n\n        >>> f = x**2 + x*y\n        >>> g = 2*x + 2*y\n        >>> h = 2*x + 2\n        >>> f.pexquo(g)\n        2*x\n        >>> f.exquo(g) # shows the differnce between pexquo and exquo\n        Traceback (most recent call last):\n        ...\n        ExactQuotientFailed: 2*x + 2*y does not divide x**2 + x*y\n        >>> f.pexquo(h)\n        Traceback (most recent call last):\n        ...\n        ExactQuotientFailed: 2*x + 2 does not divide x**2 + x*y\n\n        See Also\n        ========\n\n        prem, pdiv, pquo, sympy.polys.domains.ring.Ring.exquo\n\n        '
        f = self
        (q, r) = f.pdiv(g, x)
        if r.is_zero:
            return q
        else:
            raise ExactQuotientFailed(f, g)

    def subresultants(self, g, x=None):
        if False:
            print('Hello World!')
        '\n        Computes the subresultant PRS of two polynomials ``self`` and ``g``.\n\n        Parameters\n        ==========\n\n        g : :py:class:`~.PolyElement`\n            The second polynomial.\n        x : generator or generator index\n            The variable with respect to which the subresultant sequence is computed.\n\n        Returns\n        =======\n\n        R : list\n            Returns a list polynomials representing the subresultant PRS.\n\n        Examples\n        ========\n\n        >>> from sympy.polys import ring, ZZ\n        >>> R, x, y = ring("x, y", ZZ)\n\n        >>> f = x**2*y + x*y\n        >>> g = x + y\n        >>> f.subresultants(g) # first generator is chosen by default if not given\n        [x**2*y + x*y, x + y, y**3 - y**2]\n        >>> f.subresultants(g, 0) # generator index is given\n        [x**2*y + x*y, x + y, y**3 - y**2]\n        >>> f.subresultants(g, y) # generator is given\n        [x**2*y + x*y, x + y, x**3 + x**2]\n\n        '
        f = self
        x = f.ring.index(x)
        n = f.degree(x)
        m = g.degree(x)
        if n < m:
            (f, g) = (g, f)
            (n, m) = (m, n)
        if f == 0:
            return [0, 0]
        if g == 0:
            return [f, 1]
        R = [f, g]
        d = n - m
        b = (-1) ** (d + 1)
        h = f.prem(g, x)
        h = h * b
        lc = g.coeff_wrt(x, m)
        c = lc ** d
        S = [1, c]
        c = -c
        while h:
            k = h.degree(x)
            R.append(h)
            (f, g, m, d) = (g, h, k, m - k)
            b = -lc * c ** d
            h = f.prem(g, x)
            h = h.exquo(b)
            lc = g.coeff_wrt(x, k)
            if d > 1:
                p = (-lc) ** d
                q = c ** (d - 1)
                c = p.exquo(q)
            else:
                c = -lc
            S.append(-c)
        return R

    def half_gcdex(f, g):
        if False:
            i = 10
            return i + 15
        return f.ring.dmp_half_gcdex(f, g)

    def gcdex(f, g):
        if False:
            while True:
                i = 10
        return f.ring.dmp_gcdex(f, g)

    def resultant(f, g):
        if False:
            while True:
                i = 10
        return f.ring.dmp_resultant(f, g)

    def discriminant(f):
        if False:
            for i in range(10):
                print('nop')
        return f.ring.dmp_discriminant(f)

    def decompose(f):
        if False:
            return 10
        if f.ring.is_univariate:
            return f.ring.dup_decompose(f)
        else:
            raise MultivariatePolynomialError('polynomial decomposition')

    def shift(f, a):
        if False:
            i = 10
            return i + 15
        if f.ring.is_univariate:
            return f.ring.dup_shift(f, a)
        else:
            raise MultivariatePolynomialError('polynomial shift')

    def sturm(f):
        if False:
            return 10
        if f.ring.is_univariate:
            return f.ring.dup_sturm(f)
        else:
            raise MultivariatePolynomialError('sturm sequence')

    def gff_list(f):
        if False:
            print('Hello World!')
        return f.ring.dmp_gff_list(f)

    def sqf_norm(f):
        if False:
            i = 10
            return i + 15
        return f.ring.dmp_sqf_norm(f)

    def sqf_part(f):
        if False:
            for i in range(10):
                print('nop')
        return f.ring.dmp_sqf_part(f)

    def sqf_list(f, all=False):
        if False:
            print('Hello World!')
        return f.ring.dmp_sqf_list(f, all=all)

    def factor_list(f):
        if False:
            while True:
                i = 10
        return f.ring.dmp_factor_list(f)