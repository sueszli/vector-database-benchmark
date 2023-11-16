"""Implementation of RootOf class and related tools. """
from sympy.core.basic import Basic
from sympy.core import S, Expr, Integer, Float, I, oo, Add, Lambda, symbols, sympify, Rational, Dummy
from sympy.core.cache import cacheit
from sympy.core.relational import is_le
from sympy.core.sorting import ordered
from sympy.polys.domains import QQ
from sympy.polys.polyerrors import MultivariatePolynomialError, GeneratorsNeeded, PolynomialError, DomainError
from sympy.polys.polyfuncs import symmetrize, viete
from sympy.polys.polyroots import roots_linear, roots_quadratic, roots_binomial, preprocess_roots, roots
from sympy.polys.polytools import Poly, PurePoly, factor
from sympy.polys.rationaltools import together
from sympy.polys.rootisolation import dup_isolate_complex_roots_sqf, dup_isolate_real_roots_sqf
from sympy.utilities import lambdify, public, sift, numbered_symbols
from mpmath import mpf, mpc, findroot, workprec
from mpmath.libmp.libmpf import dps_to_prec, prec_to_dps
from sympy.multipledispatch import dispatch
from itertools import chain
__all__ = ['CRootOf']

class _pure_key_dict:
    """A minimal dictionary that makes sure that the key is a
    univariate PurePoly instance.

    Examples
    ========

    Only the following actions are guaranteed:

    >>> from sympy.polys.rootoftools import _pure_key_dict
    >>> from sympy import PurePoly
    >>> from sympy.abc import x, y

    1) creation

    >>> P = _pure_key_dict()

    2) assignment for a PurePoly or univariate polynomial

    >>> P[x] = 1
    >>> P[PurePoly(x - y, x)] = 2

    3) retrieval based on PurePoly key comparison (use this
       instead of the get method)

    >>> P[y]
    1

    4) KeyError when trying to retrieve a nonexisting key

    >>> P[y + 1]
    Traceback (most recent call last):
    ...
    KeyError: PurePoly(y + 1, y, domain='ZZ')

    5) ability to query with ``in``

    >>> x + 1 in P
    False

    NOTE: this is a *not* a dictionary. It is a very basic object
    for internal use that makes sure to always address its cache
    via PurePoly instances. It does not, for example, implement
    ``get`` or ``setdefault``.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._dict = {}

    def __getitem__(self, k):
        if False:
            while True:
                i = 10
        if not isinstance(k, PurePoly):
            if not (isinstance(k, Expr) and len(k.free_symbols) == 1):
                raise KeyError
            k = PurePoly(k, expand=False)
        return self._dict[k]

    def __setitem__(self, k, v):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(k, PurePoly):
            if not (isinstance(k, Expr) and len(k.free_symbols) == 1):
                raise ValueError('expecting univariate expression')
            k = PurePoly(k, expand=False)
        self._dict[k] = v

    def __contains__(self, k):
        if False:
            print('Hello World!')
        try:
            self[k]
            return True
        except KeyError:
            return False
_reals_cache = _pure_key_dict()
_complexes_cache = _pure_key_dict()

def _pure_factors(poly):
    if False:
        while True:
            i = 10
    (_, factors) = poly.factor_list()
    return [(PurePoly(f, expand=False), m) for (f, m) in factors]

def _imag_count_of_factor(f):
    if False:
        print('Hello World!')
    'Return the number of imaginary roots for irreducible\n    univariate polynomial ``f``.\n    '
    terms = [(i, j) for ((i,), j) in f.terms()]
    if any((i % 2 for (i, j) in terms)):
        return 0
    even = [(i, I ** i * j) for (i, j) in terms]
    even = Poly.from_dict(dict(even), Dummy('x'))
    return int(even.count_roots(-oo, oo))

@public
def rootof(f, x, index=None, radicals=True, expand=True):
    if False:
        print('Hello World!')
    'An indexed root of a univariate polynomial.\n\n    Returns either a :obj:`ComplexRootOf` object or an explicit\n    expression involving radicals.\n\n    Parameters\n    ==========\n\n    f : Expr\n        Univariate polynomial.\n    x : Symbol, optional\n        Generator for ``f``.\n    index : int or Integer\n    radicals : bool\n               Return a radical expression if possible.\n    expand : bool\n             Expand ``f``.\n    '
    return CRootOf(f, x, index=index, radicals=radicals, expand=expand)

@public
class RootOf(Expr):
    """Represents a root of a univariate polynomial.

    Base class for roots of different kinds of polynomials.
    Only complex roots are currently supported.
    """
    __slots__ = ('poly',)

    def __new__(cls, f, x, index=None, radicals=True, expand=True):
        if False:
            i = 10
            return i + 15
        'Construct a new ``CRootOf`` object for ``k``-th root of ``f``.'
        return rootof(f, x, index=index, radicals=radicals, expand=expand)

@public
class ComplexRootOf(RootOf):
    """Represents an indexed complex root of a polynomial.

    Roots of a univariate polynomial separated into disjoint
    real or complex intervals and indexed in a fixed order:

    * real roots come first and are sorted in increasing order;
    * complex roots come next and are sorted primarily by increasing
      real part, secondarily by increasing imaginary part.

    Currently only rational coefficients are allowed.
    Can be imported as ``CRootOf``. To avoid confusion, the
    generator must be a Symbol.


    Examples
    ========

    >>> from sympy import CRootOf, rootof
    >>> from sympy.abc import x

    CRootOf is a way to reference a particular root of a
    polynomial. If there is a rational root, it will be returned:

    >>> CRootOf.clear_cache()  # for doctest reproducibility
    >>> CRootOf(x**2 - 4, 0)
    -2

    Whether roots involving radicals are returned or not
    depends on whether the ``radicals`` flag is true (which is
    set to True with rootof):

    >>> CRootOf(x**2 - 3, 0)
    CRootOf(x**2 - 3, 0)
    >>> CRootOf(x**2 - 3, 0, radicals=True)
    -sqrt(3)
    >>> rootof(x**2 - 3, 0)
    -sqrt(3)

    The following cannot be expressed in terms of radicals:

    >>> r = rootof(4*x**5 + 16*x**3 + 12*x**2 + 7, 0); r
    CRootOf(4*x**5 + 16*x**3 + 12*x**2 + 7, 0)

    The root bounds can be seen, however, and they are used by the
    evaluation methods to get numerical approximations for the root.

    >>> interval = r._get_interval(); interval
    (-1, 0)
    >>> r.evalf(2)
    -0.98

    The evalf method refines the width of the root bounds until it
    guarantees that any decimal approximation within those bounds
    will satisfy the desired precision. It then stores the refined
    interval so subsequent requests at or below the requested
    precision will not have to recompute the root bounds and will
    return very quickly.

    Before evaluation above, the interval was

    >>> interval
    (-1, 0)

    After evaluation it is now

    >>> r._get_interval() # doctest: +SKIP
    (-165/169, -206/211)

    To reset all intervals for a given polynomial, the :meth:`_reset` method
    can be called from any CRootOf instance of the polynomial:

    >>> r._reset()
    >>> r._get_interval()
    (-1, 0)

    The :meth:`eval_approx` method will also find the root to a given
    precision but the interval is not modified unless the search
    for the root fails to converge within the root bounds. And
    the secant method is used to find the root. (The ``evalf``
    method uses bisection and will always update the interval.)

    >>> r.eval_approx(2)
    -0.98

    The interval needed to be slightly updated to find that root:

    >>> r._get_interval()
    (-1, -1/2)

    The ``evalf_rational`` will compute a rational approximation
    of the root to the desired accuracy or precision.

    >>> r.eval_rational(n=2)
    -69629/71318

    >>> t = CRootOf(x**3 + 10*x + 1, 1)
    >>> t.eval_rational(1e-1)
    15/256 - 805*I/256
    >>> t.eval_rational(1e-1, 1e-4)
    3275/65536 - 414645*I/131072
    >>> t.eval_rational(1e-4, 1e-4)
    6545/131072 - 414645*I/131072
    >>> t.eval_rational(n=2)
    104755/2097152 - 6634255*I/2097152

    Notes
    =====

    Although a PurePoly can be constructed from a non-symbol generator
    RootOf instances of non-symbols are disallowed to avoid confusion
    over what root is being represented.

    >>> from sympy import exp, PurePoly
    >>> PurePoly(x) == PurePoly(exp(x))
    True
    >>> CRootOf(x - 1, 0)
    1
    >>> CRootOf(exp(x) - 1, 0)  # would correspond to x == 0
    Traceback (most recent call last):
    ...
    sympy.polys.polyerrors.PolynomialError: generator must be a Symbol

    See Also
    ========

    eval_approx
    eval_rational

    """
    __slots__ = ('index',)
    is_complex = True
    is_number = True
    is_finite = True

    def __new__(cls, f, x, index=None, radicals=False, expand=True):
        if False:
            return 10
        ' Construct an indexed complex root of a polynomial.\n\n        See ``rootof`` for the parameters.\n\n        The default value of ``radicals`` is ``False`` to satisfy\n        ``eval(srepr(expr) == expr``.\n        '
        x = sympify(x)
        if index is None and x.is_Integer:
            (x, index) = (None, x)
        else:
            index = sympify(index)
        if index is not None and index.is_Integer:
            index = int(index)
        else:
            raise ValueError('expected an integer root index, got %s' % index)
        poly = PurePoly(f, x, greedy=False, expand=expand)
        if not poly.is_univariate:
            raise PolynomialError('only univariate polynomials are allowed')
        if not poly.gen.is_Symbol:
            raise PolynomialError('generator must be a Symbol')
        degree = poly.degree()
        if degree <= 0:
            raise PolynomialError('Cannot construct CRootOf object for %s' % f)
        if index < -degree or index >= degree:
            raise IndexError('root index out of [%d, %d] range, got %d' % (-degree, degree - 1, index))
        elif index < 0:
            index += degree
        dom = poly.get_domain()
        if not dom.is_Exact:
            poly = poly.to_exact()
        roots = cls._roots_trivial(poly, radicals)
        if roots is not None:
            return roots[index]
        (coeff, poly) = preprocess_roots(poly)
        dom = poly.get_domain()
        if not dom.is_ZZ:
            raise NotImplementedError('CRootOf is not supported over %s' % dom)
        root = cls._indexed_root(poly, index, lazy=True)
        return coeff * cls._postprocess_root(root, radicals)

    @classmethod
    def _new(cls, poly, index):
        if False:
            return 10
        'Construct new ``CRootOf`` object from raw data. '
        obj = Expr.__new__(cls)
        obj.poly = PurePoly(poly)
        obj.index = index
        try:
            _reals_cache[obj.poly] = _reals_cache[poly]
            _complexes_cache[obj.poly] = _complexes_cache[poly]
        except KeyError:
            pass
        return obj

    def _hashable_content(self):
        if False:
            return 10
        return (self.poly, self.index)

    @property
    def expr(self):
        if False:
            print('Hello World!')
        return self.poly.as_expr()

    @property
    def args(self):
        if False:
            while True:
                i = 10
        return (self.expr, Integer(self.index))

    @property
    def free_symbols(self):
        if False:
            print('Hello World!')
        return set()

    def _eval_is_real(self):
        if False:
            for i in range(10):
                print('nop')
        'Return ``True`` if the root is real. '
        self._ensure_reals_init()
        return self.index < len(_reals_cache[self.poly])

    def _eval_is_imaginary(self):
        if False:
            print('Hello World!')
        'Return ``True`` if the root is imaginary. '
        self._ensure_reals_init()
        if self.index >= len(_reals_cache[self.poly]):
            ivl = self._get_interval()
            return ivl.ax * ivl.bx <= 0
        return False

    @classmethod
    def real_roots(cls, poly, radicals=True):
        if False:
            return 10
        'Get real roots of a polynomial. '
        return cls._get_roots('_real_roots', poly, radicals)

    @classmethod
    def all_roots(cls, poly, radicals=True):
        if False:
            print('Hello World!')
        'Get real and complex roots of a polynomial. '
        return cls._get_roots('_all_roots', poly, radicals)

    @classmethod
    def _get_reals_sqf(cls, currentfactor, use_cache=True):
        if False:
            print('Hello World!')
        'Get real root isolating intervals for a square-free factor.'
        if use_cache and currentfactor in _reals_cache:
            real_part = _reals_cache[currentfactor]
        else:
            _reals_cache[currentfactor] = real_part = dup_isolate_real_roots_sqf(currentfactor.rep.to_list(), currentfactor.rep.dom, blackbox=True)
        return real_part

    @classmethod
    def _get_complexes_sqf(cls, currentfactor, use_cache=True):
        if False:
            print('Hello World!')
        'Get complex root isolating intervals for a square-free factor.'
        if use_cache and currentfactor in _complexes_cache:
            complex_part = _complexes_cache[currentfactor]
        else:
            _complexes_cache[currentfactor] = complex_part = dup_isolate_complex_roots_sqf(currentfactor.rep.to_list(), currentfactor.rep.dom, blackbox=True)
        return complex_part

    @classmethod
    def _get_reals(cls, factors, use_cache=True):
        if False:
            i = 10
            return i + 15
        'Compute real root isolating intervals for a list of factors. '
        reals = []
        for (currentfactor, k) in factors:
            try:
                if not use_cache:
                    raise KeyError
                r = _reals_cache[currentfactor]
                reals.extend([(i, currentfactor, k) for i in r])
            except KeyError:
                real_part = cls._get_reals_sqf(currentfactor, use_cache)
                new = [(root, currentfactor, k) for root in real_part]
                reals.extend(new)
        reals = cls._reals_sorted(reals)
        return reals

    @classmethod
    def _get_complexes(cls, factors, use_cache=True):
        if False:
            for i in range(10):
                print('nop')
        'Compute complex root isolating intervals for a list of factors. '
        complexes = []
        for (currentfactor, k) in ordered(factors):
            try:
                if not use_cache:
                    raise KeyError
                c = _complexes_cache[currentfactor]
                complexes.extend([(i, currentfactor, k) for i in c])
            except KeyError:
                complex_part = cls._get_complexes_sqf(currentfactor, use_cache)
                new = [(root, currentfactor, k) for root in complex_part]
                complexes.extend(new)
        complexes = cls._complexes_sorted(complexes)
        return complexes

    @classmethod
    def _reals_sorted(cls, reals):
        if False:
            return 10
        'Make real isolating intervals disjoint and sort roots. '
        cache = {}
        for (i, (u, f, k)) in enumerate(reals):
            for (j, (v, g, m)) in enumerate(reals[i + 1:]):
                (u, v) = u.refine_disjoint(v)
                reals[i + j + 1] = (v, g, m)
            reals[i] = (u, f, k)
        reals = sorted(reals, key=lambda r: r[0].a)
        for (root, currentfactor, _) in reals:
            if currentfactor in cache:
                cache[currentfactor].append(root)
            else:
                cache[currentfactor] = [root]
        for (currentfactor, root) in cache.items():
            _reals_cache[currentfactor] = root
        return reals

    @classmethod
    def _refine_imaginary(cls, complexes):
        if False:
            i = 10
            return i + 15
        sifted = sift(complexes, lambda c: c[1])
        complexes = []
        for f in ordered(sifted):
            nimag = _imag_count_of_factor(f)
            if nimag == 0:
                for (u, f, k) in sifted[f]:
                    while u.ax * u.bx <= 0:
                        u = u._inner_refine()
                    complexes.append((u, f, k))
            else:
                potential_imag = list(range(len(sifted[f])))
                while True:
                    assert len(potential_imag) > 1
                    for i in list(potential_imag):
                        (u, f, k) = sifted[f][i]
                        if u.ax * u.bx > 0:
                            potential_imag.remove(i)
                        elif u.ax != u.bx:
                            u = u._inner_refine()
                            sifted[f][i] = (u, f, k)
                    if len(potential_imag) == nimag:
                        break
                complexes.extend(sifted[f])
        return complexes

    @classmethod
    def _refine_complexes(cls, complexes):
        if False:
            print('Hello World!')
        'return complexes such that no bounding rectangles of non-conjugate\n        roots would intersect. In addition, assure that neither ay nor by is\n        0 to guarantee that non-real roots are distinct from real roots in\n        terms of the y-bounds.\n        '
        for (i, (u, f, k)) in enumerate(complexes):
            for (j, (v, g, m)) in enumerate(complexes[i + 1:]):
                (u, v) = u.refine_disjoint(v)
                complexes[i + j + 1] = (v, g, m)
            complexes[i] = (u, f, k)
        complexes = cls._refine_imaginary(complexes)
        for (i, (u, f, k)) in enumerate(complexes):
            while u.ay * u.by <= 0:
                u = u.refine()
            complexes[i] = (u, f, k)
        return complexes

    @classmethod
    def _complexes_sorted(cls, complexes):
        if False:
            i = 10
            return i + 15
        'Make complex isolating intervals disjoint and sort roots. '
        complexes = cls._refine_complexes(complexes)
        (C, F) = (0, 1)
        fs = {i[F] for i in complexes}
        for i in range(1, len(complexes)):
            if complexes[i][F] != complexes[i - 1][F]:
                fs.remove(complexes[i - 1][F])
        for (i, cmplx) in enumerate(complexes):
            assert cmplx[C].conj is (i % 2 == 0)
        cache = {}
        for (root, currentfactor, _) in complexes:
            cache.setdefault(currentfactor, []).append(root)
        for (currentfactor, root) in cache.items():
            _complexes_cache[currentfactor] = root
        return complexes

    @classmethod
    def _reals_index(cls, reals, index):
        if False:
            return 10
        '\n        Map initial real root index to an index in a factor where\n        the root belongs.\n        '
        i = 0
        for (j, (_, currentfactor, k)) in enumerate(reals):
            if index < i + k:
                (poly, index) = (currentfactor, 0)
                for (_, currentfactor, _) in reals[:j]:
                    if currentfactor == poly:
                        index += 1
                return (poly, index)
            else:
                i += k

    @classmethod
    def _complexes_index(cls, complexes, index):
        if False:
            print('Hello World!')
        '\n        Map initial complex root index to an index in a factor where\n        the root belongs.\n        '
        i = 0
        for (j, (_, currentfactor, k)) in enumerate(complexes):
            if index < i + k:
                (poly, index) = (currentfactor, 0)
                for (_, currentfactor, _) in complexes[:j]:
                    if currentfactor == poly:
                        index += 1
                index += len(_reals_cache[poly])
                return (poly, index)
            else:
                i += k

    @classmethod
    def _count_roots(cls, roots):
        if False:
            print('Hello World!')
        'Count the number of real or complex roots with multiplicities.'
        return sum([k for (_, _, k) in roots])

    @classmethod
    def _indexed_root(cls, poly, index, lazy=False):
        if False:
            return 10
        'Get a root of a composite polynomial by index. '
        factors = _pure_factors(poly)
        if lazy and len(factors) == 1 and (factors[0][1] == 1):
            return (factors[0][0], index)
        reals = cls._get_reals(factors)
        reals_count = cls._count_roots(reals)
        if index < reals_count:
            return cls._reals_index(reals, index)
        else:
            complexes = cls._get_complexes(factors)
            return cls._complexes_index(complexes, index - reals_count)

    def _ensure_reals_init(self):
        if False:
            while True:
                i = 10
        'Ensure that our poly has entries in the reals cache. '
        if self.poly not in _reals_cache:
            self._indexed_root(self.poly, self.index)

    def _ensure_complexes_init(self):
        if False:
            return 10
        'Ensure that our poly has entries in the complexes cache. '
        if self.poly not in _complexes_cache:
            self._indexed_root(self.poly, self.index)

    @classmethod
    def _real_roots(cls, poly):
        if False:
            i = 10
            return i + 15
        'Get real roots of a composite polynomial. '
        factors = _pure_factors(poly)
        reals = cls._get_reals(factors)
        reals_count = cls._count_roots(reals)
        roots = []
        for index in range(0, reals_count):
            roots.append(cls._reals_index(reals, index))
        return roots

    def _reset(self):
        if False:
            print('Hello World!')
        '\n        Reset all intervals\n        '
        self._all_roots(self.poly, use_cache=False)

    @classmethod
    def _all_roots(cls, poly, use_cache=True):
        if False:
            i = 10
            return i + 15
        'Get real and complex roots of a composite polynomial. '
        factors = _pure_factors(poly)
        reals = cls._get_reals(factors, use_cache=use_cache)
        reals_count = cls._count_roots(reals)
        roots = []
        for index in range(0, reals_count):
            roots.append(cls._reals_index(reals, index))
        complexes = cls._get_complexes(factors, use_cache=use_cache)
        complexes_count = cls._count_roots(complexes)
        for index in range(0, complexes_count):
            roots.append(cls._complexes_index(complexes, index))
        return roots

    @classmethod
    @cacheit
    def _roots_trivial(cls, poly, radicals):
        if False:
            while True:
                i = 10
        'Compute roots in linear, quadratic and binomial cases. '
        if poly.degree() == 1:
            return roots_linear(poly)
        if not radicals:
            return None
        if poly.degree() == 2:
            return roots_quadratic(poly)
        elif poly.length() == 2 and poly.TC():
            return roots_binomial(poly)
        else:
            return None

    @classmethod
    def _preprocess_roots(cls, poly):
        if False:
            print('Hello World!')
        'Take heroic measures to make ``poly`` compatible with ``CRootOf``.'
        dom = poly.get_domain()
        if not dom.is_Exact:
            poly = poly.to_exact()
        (coeff, poly) = preprocess_roots(poly)
        dom = poly.get_domain()
        if not dom.is_ZZ:
            raise NotImplementedError('sorted roots not supported over %s' % dom)
        return (coeff, poly)

    @classmethod
    def _postprocess_root(cls, root, radicals):
        if False:
            i = 10
            return i + 15
        'Return the root if it is trivial or a ``CRootOf`` object. '
        (poly, index) = root
        roots = cls._roots_trivial(poly, radicals)
        if roots is not None:
            return roots[index]
        else:
            return cls._new(poly, index)

    @classmethod
    def _get_roots(cls, method, poly, radicals):
        if False:
            print('Hello World!')
        'Return postprocessed roots of specified kind. '
        if not poly.is_univariate:
            raise PolynomialError('only univariate polynomials are allowed')
        d = Dummy()
        poly = poly.subs(poly.gen, d)
        x = symbols('x')
        free_names = {str(i) for i in poly.free_symbols}
        for x in chain((symbols('x'),), numbered_symbols('x')):
            if x.name not in free_names:
                poly = poly.xreplace({d: x})
                break
        (coeff, poly) = cls._preprocess_roots(poly)
        roots = []
        for root in getattr(cls, method)(poly):
            roots.append(coeff * cls._postprocess_root(root, radicals))
        return roots

    @classmethod
    def clear_cache(cls):
        if False:
            i = 10
            return i + 15
        'Reset cache for reals and complexes.\n\n        The intervals used to approximate a root instance are updated\n        as needed. When a request is made to see the intervals, the\n        most current values are shown. `clear_cache` will reset all\n        CRootOf instances back to their original state.\n\n        See Also\n        ========\n\n        _reset\n        '
        global _reals_cache, _complexes_cache
        _reals_cache = _pure_key_dict()
        _complexes_cache = _pure_key_dict()

    def _get_interval(self):
        if False:
            i = 10
            return i + 15
        'Internal function for retrieving isolation interval from cache. '
        self._ensure_reals_init()
        if self.is_real:
            return _reals_cache[self.poly][self.index]
        else:
            reals_count = len(_reals_cache[self.poly])
            self._ensure_complexes_init()
            return _complexes_cache[self.poly][self.index - reals_count]

    def _set_interval(self, interval):
        if False:
            return 10
        'Internal function for updating isolation interval in cache. '
        self._ensure_reals_init()
        if self.is_real:
            _reals_cache[self.poly][self.index] = interval
        else:
            reals_count = len(_reals_cache[self.poly])
            self._ensure_complexes_init()
            _complexes_cache[self.poly][self.index - reals_count] = interval

    def _eval_subs(self, old, new):
        if False:
            return 10
        return self

    def _eval_conjugate(self):
        if False:
            return 10
        if self.is_real:
            return self
        (expr, i) = self.args
        return self.func(expr, i + (1 if self._get_interval().conj else -1))

    def eval_approx(self, n, return_mpmath=False):
        if False:
            print('Hello World!')
        'Evaluate this complex root to the given precision.\n\n        This uses secant method and root bounds are used to both\n        generate an initial guess and to check that the root\n        returned is valid. If ever the method converges outside the\n        root bounds, the bounds will be made smaller and updated.\n        '
        prec = dps_to_prec(n)
        with workprec(prec):
            g = self.poly.gen
            if not g.is_Symbol:
                d = Dummy('x')
                if self.is_imaginary:
                    d *= I
                func = lambdify(d, self.expr.subs(g, d))
            else:
                expr = self.expr
                if self.is_imaginary:
                    expr = self.expr.subs(g, I * g)
                func = lambdify(g, expr)
            interval = self._get_interval()
            while True:
                if self.is_real:
                    a = mpf(str(interval.a))
                    b = mpf(str(interval.b))
                    if a == b:
                        root = a
                        break
                    x0 = mpf(str(interval.center))
                    x1 = x0 + mpf(str(interval.dx)) / 4
                elif self.is_imaginary:
                    a = mpf(str(interval.ay))
                    b = mpf(str(interval.by))
                    if a == b:
                        root = mpc(mpf('0'), a)
                        break
                    x0 = mpf(str(interval.center[1]))
                    x1 = x0 + mpf(str(interval.dy)) / 4
                else:
                    ax = mpf(str(interval.ax))
                    bx = mpf(str(interval.bx))
                    ay = mpf(str(interval.ay))
                    by = mpf(str(interval.by))
                    if ax == bx and ay == by:
                        root = mpc(ax, ay)
                        break
                    x0 = mpc(*map(str, interval.center))
                    x1 = x0 + mpc(*map(str, (interval.dx, interval.dy))) / 4
                try:
                    root = findroot(func, (x0, x1))
                    if self.is_real or self.is_imaginary:
                        if not bool(root.imag) == self.is_real and a <= root <= b:
                            if self.is_imaginary:
                                root = mpc(mpf('0'), root.real)
                            break
                    elif ax <= root.real <= bx and ay <= root.imag <= by:
                        break
                except (UnboundLocalError, ValueError):
                    pass
                interval = interval.refine()
        self._set_interval(interval)
        if return_mpmath:
            return root
        return Float._new(root.real._mpf_, prec) + I * Float._new(root.imag._mpf_, prec)

    def _eval_evalf(self, prec, **kwargs):
        if False:
            i = 10
            return i + 15
        'Evaluate this complex root to the given precision.'
        return self.eval_rational(n=prec_to_dps(prec))._evalf(prec)

    def eval_rational(self, dx=None, dy=None, n=15):
        if False:
            print('Hello World!')
        '\n        Return a Rational approximation of ``self`` that has real\n        and imaginary component approximations that are within ``dx``\n        and ``dy`` of the true values, respectively. Alternatively,\n        ``n`` digits of precision can be specified.\n\n        The interval is refined with bisection and is sure to\n        converge. The root bounds are updated when the refinement\n        is complete so recalculation at the same or lesser precision\n        will not have to repeat the refinement and should be much\n        faster.\n\n        The following example first obtains Rational approximation to\n        1e-8 accuracy for all roots of the 4-th order Legendre\n        polynomial. Since the roots are all less than 1, this will\n        ensure the decimal representation of the approximation will be\n        correct (including rounding) to 6 digits:\n\n        >>> from sympy import legendre_poly, Symbol\n        >>> x = Symbol("x")\n        >>> p = legendre_poly(4, x, polys=True)\n        >>> r = p.real_roots()[-1]\n        >>> r.eval_rational(10**-8).n(6)\n        0.861136\n\n        It is not necessary to a two-step calculation, however: the\n        decimal representation can be computed directly:\n\n        >>> r.evalf(17)\n        0.86113631159405258\n\n        '
        dy = dy or dx
        if dx:
            rtol = None
            dx = dx if isinstance(dx, Rational) else Rational(str(dx))
            dy = dy if isinstance(dy, Rational) else Rational(str(dy))
        else:
            rtol = S(10) ** (-(n + 2))
        interval = self._get_interval()
        while True:
            if self.is_real:
                if rtol:
                    dx = abs(interval.center * rtol)
                interval = interval.refine_size(dx=dx)
                c = interval.center
                real = Rational(c)
                imag = S.Zero
                if not rtol or interval.dx < abs(c * rtol):
                    break
            elif self.is_imaginary:
                if rtol:
                    dy = abs(interval.center[1] * rtol)
                    dx = 1
                interval = interval.refine_size(dx=dx, dy=dy)
                c = interval.center[1]
                imag = Rational(c)
                real = S.Zero
                if not rtol or interval.dy < abs(c * rtol):
                    break
            else:
                if rtol:
                    dx = abs(interval.center[0] * rtol)
                    dy = abs(interval.center[1] * rtol)
                interval = interval.refine_size(dx, dy)
                c = interval.center
                (real, imag) = map(Rational, c)
                if not rtol or (interval.dx < abs(c[0] * rtol) and interval.dy < abs(c[1] * rtol)):
                    break
        self._set_interval(interval)
        return real + I * imag
CRootOf = ComplexRootOf

@dispatch(ComplexRootOf, ComplexRootOf)
def _eval_is_eq(lhs, rhs):
    if False:
        return 10
    return lhs == rhs

@dispatch(ComplexRootOf, Basic)
def _eval_is_eq(lhs, rhs):
    if False:
        i = 10
        return i + 15
    if not rhs.is_number:
        return None
    if not rhs.is_finite:
        return False
    z = lhs.expr.subs(lhs.expr.free_symbols.pop(), rhs).is_zero
    if z is False:
        return False
    o = (rhs.is_real, rhs.is_imaginary)
    s = (lhs.is_real, lhs.is_imaginary)
    assert None not in s
    if o != s and None not in o:
        return False
    (re, im) = rhs.as_real_imag()
    if lhs.is_real:
        if im:
            return False
        i = lhs._get_interval()
        (a, b) = [Rational(str(_)) for _ in (i.a, i.b)]
        return sympify(a <= rhs and rhs <= b)
    i = lhs._get_interval()
    (r1, r2, i1, i2) = [Rational(str(j)) for j in (i.ax, i.bx, i.ay, i.by)]
    return is_le(r1, re) and is_le(re, r2) and is_le(i1, im) and is_le(im, i2)

@public
class RootSum(Expr):
    """Represents a sum of all roots of a univariate polynomial. """
    __slots__ = ('poly', 'fun', 'auto')

    def __new__(cls, expr, func=None, x=None, auto=True, quadratic=False):
        if False:
            while True:
                i = 10
        'Construct a new ``RootSum`` instance of roots of a polynomial.'
        (coeff, poly) = cls._transform(expr, x)
        if not poly.is_univariate:
            raise MultivariatePolynomialError('only univariate polynomials are allowed')
        if func is None:
            func = Lambda(poly.gen, poly.gen)
        else:
            is_func = getattr(func, 'is_Function', False)
            if is_func and 1 in func.nargs:
                if not isinstance(func, Lambda):
                    func = Lambda(poly.gen, func(poly.gen))
            else:
                raise ValueError('expected a univariate function, got %s' % func)
        (var, expr) = (func.variables[0], func.expr)
        if coeff is not S.One:
            expr = expr.subs(var, coeff * var)
        deg = poly.degree()
        if not expr.has(var):
            return deg * expr
        if expr.is_Add:
            (add_const, expr) = expr.as_independent(var)
        else:
            add_const = S.Zero
        if expr.is_Mul:
            (mul_const, expr) = expr.as_independent(var)
        else:
            mul_const = S.One
        func = Lambda(var, expr)
        rational = cls._is_func_rational(poly, func)
        (factors, terms) = (_pure_factors(poly), [])
        for (poly, k) in factors:
            if poly.is_linear:
                term = func(roots_linear(poly)[0])
            elif quadratic and poly.is_quadratic:
                term = sum(map(func, roots_quadratic(poly)))
            elif not rational or not auto:
                term = cls._new(poly, func, auto)
            else:
                term = cls._rational_case(poly, func)
            terms.append(k * term)
        return mul_const * Add(*terms) + deg * add_const

    @classmethod
    def _new(cls, poly, func, auto=True):
        if False:
            print('Hello World!')
        'Construct new raw ``RootSum`` instance. '
        obj = Expr.__new__(cls)
        obj.poly = poly
        obj.fun = func
        obj.auto = auto
        return obj

    @classmethod
    def new(cls, poly, func, auto=True):
        if False:
            print('Hello World!')
        'Construct new ``RootSum`` instance. '
        if not func.expr.has(*func.variables):
            return func.expr
        rational = cls._is_func_rational(poly, func)
        if not rational or not auto:
            return cls._new(poly, func, auto)
        else:
            return cls._rational_case(poly, func)

    @classmethod
    def _transform(cls, expr, x):
        if False:
            for i in range(10):
                print('nop')
        'Transform an expression to a polynomial. '
        poly = PurePoly(expr, x, greedy=False)
        return preprocess_roots(poly)

    @classmethod
    def _is_func_rational(cls, poly, func):
        if False:
            return 10
        'Check if a lambda is a rational function. '
        (var, expr) = (func.variables[0], func.expr)
        return expr.is_rational_function(var)

    @classmethod
    def _rational_case(cls, poly, func):
        if False:
            while True:
                i = 10
        'Handle the rational function case. '
        roots = symbols('r:%d' % poly.degree())
        (var, expr) = (func.variables[0], func.expr)
        f = sum((expr.subs(var, r) for r in roots))
        (p, q) = together(f).as_numer_denom()
        domain = QQ[roots]
        p = p.expand()
        q = q.expand()
        try:
            p = Poly(p, domain=domain, expand=False)
        except GeneratorsNeeded:
            (p, p_coeff) = (None, (p,))
        else:
            (p_monom, p_coeff) = zip(*p.terms())
        try:
            q = Poly(q, domain=domain, expand=False)
        except GeneratorsNeeded:
            (q, q_coeff) = (None, (q,))
        else:
            (q_monom, q_coeff) = zip(*q.terms())
        (coeffs, mapping) = symmetrize(p_coeff + q_coeff, formal=True)
        (formulas, values) = (viete(poly, roots), [])
        for ((sym, _), (_, val)) in zip(mapping, formulas):
            values.append((sym, val))
        for (i, (coeff, _)) in enumerate(coeffs):
            coeffs[i] = coeff.subs(values)
        n = len(p_coeff)
        p_coeff = coeffs[:n]
        q_coeff = coeffs[n:]
        if p is not None:
            p = Poly(dict(zip(p_monom, p_coeff)), *p.gens).as_expr()
        else:
            (p,) = p_coeff
        if q is not None:
            q = Poly(dict(zip(q_monom, q_coeff)), *q.gens).as_expr()
        else:
            (q,) = q_coeff
        return factor(p / q)

    def _hashable_content(self):
        if False:
            return 10
        return (self.poly, self.fun)

    @property
    def expr(self):
        if False:
            i = 10
            return i + 15
        return self.poly.as_expr()

    @property
    def args(self):
        if False:
            while True:
                i = 10
        return (self.expr, self.fun, self.poly.gen)

    @property
    def free_symbols(self):
        if False:
            i = 10
            return i + 15
        return self.poly.free_symbols | self.fun.free_symbols

    @property
    def is_commutative(self):
        if False:
            while True:
                i = 10
        return True

    def doit(self, **hints):
        if False:
            return 10
        if not hints.get('roots', True):
            return self
        _roots = roots(self.poly, multiple=True)
        if len(_roots) < self.poly.degree():
            return self
        else:
            return Add(*[self.fun(r) for r in _roots])

    def _eval_evalf(self, prec):
        if False:
            i = 10
            return i + 15
        try:
            _roots = self.poly.nroots(n=prec_to_dps(prec))
        except (DomainError, PolynomialError):
            return self
        else:
            return Add(*[self.fun(r) for r in _roots])

    def _eval_derivative(self, x):
        if False:
            while True:
                i = 10
        (var, expr) = self.fun.args
        func = Lambda(var, expr.diff(x))
        return self.new(self.poly, func, self.auto)