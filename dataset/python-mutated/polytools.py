"""User-friendly public interface to polynomial functions. """
from functools import wraps, reduce
from operator import mul
from typing import Optional
from sympy.core import S, Expr, Add, Tuple
from sympy.core.basic import Basic
from sympy.core.decorators import _sympifyit
from sympy.core.exprtools import Factors, factor_nc, factor_terms
from sympy.core.evalf import pure_complex, evalf, fastlog, _evalf_with_bounded_error, quad_to_mpmath
from sympy.core.function import Derivative
from sympy.core.mul import Mul, _keep_coeff
from sympy.core.intfunc import ilcm
from sympy.core.numbers import I, Integer, equal_valued
from sympy.core.relational import Relational, Equality
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify, _sympify
from sympy.core.traversal import preorder_traversal, bottom_up
from sympy.logic.boolalg import BooleanAtom
from sympy.polys import polyoptions as options
from sympy.polys.constructor import construct_domain
from sympy.polys.domains import FF, QQ, ZZ
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.fglmtools import matrix_fglm
from sympy.polys.groebnertools import groebner as _groebner
from sympy.polys.monomials import Monomial
from sympy.polys.orderings import monomial_key
from sympy.polys.polyclasses import DMP, DMF, ANP
from sympy.polys.polyerrors import OperationNotSupported, DomainError, CoercionFailed, UnificationFailed, GeneratorsNeeded, PolynomialError, MultivariatePolynomialError, ExactQuotientFailed, PolificationFailed, ComputationFailed, GeneratorsError
from sympy.polys.polyutils import basic_from_dict, _sort_gens, _unify_gens, _dict_reorder, _dict_from_expr, _parallel_dict_from_expr
from sympy.polys.rationaltools import together
from sympy.polys.rootisolation import dup_isolate_real_roots_list
from sympy.utilities import group, public, filldedent
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import iterable, sift
import sympy.polys
import mpmath
from mpmath.libmp.libhyper import NoConvergence

def _polifyit(func):
    if False:
        return 10

    @wraps(func)
    def wrapper(f, g):
        if False:
            while True:
                i = 10
        g = _sympify(g)
        if isinstance(g, Poly):
            return func(f, g)
        elif isinstance(g, Integer):
            g = f.from_expr(g, *f.gens, domain=f.domain)
            return func(f, g)
        elif isinstance(g, Expr):
            try:
                g = f.from_expr(g, *f.gens)
            except PolynomialError:
                if g.is_Matrix:
                    return NotImplemented
                expr_method = getattr(f.as_expr(), func.__name__)
                result = expr_method(g)
                if result is not NotImplemented:
                    sympy_deprecation_warning('\n                        Mixing Poly with non-polynomial expressions in binary\n                        operations is deprecated. Either explicitly convert\n                        the non-Poly operand to a Poly with as_poly() or\n                        convert the Poly to an Expr with as_expr().\n                        ', deprecated_since_version='1.6', active_deprecations_target='deprecated-poly-nonpoly-binary-operations')
                return result
            else:
                return func(f, g)
        else:
            return NotImplemented
    return wrapper

@public
class Poly(Basic):
    """
    Generic class for representing and operating on polynomial expressions.

    See :ref:`polys-docs` for general documentation.

    Poly is a subclass of Basic rather than Expr but instances can be
    converted to Expr with the :py:meth:`~.Poly.as_expr` method.

    .. deprecated:: 1.6

       Combining Poly with non-Poly objects in binary operations is
       deprecated. Explicitly convert both objects to either Poly or Expr
       first. See :ref:`deprecated-poly-nonpoly-binary-operations`.

    Examples
    ========

    >>> from sympy import Poly
    >>> from sympy.abc import x, y

    Create a univariate polynomial:

    >>> Poly(x*(x**2 + x - 1)**2)
    Poly(x**5 + 2*x**4 - x**3 - 2*x**2 + x, x, domain='ZZ')

    Create a univariate polynomial with specific domain:

    >>> from sympy import sqrt
    >>> Poly(x**2 + 2*x + sqrt(3), domain='R')
    Poly(1.0*x**2 + 2.0*x + 1.73205080756888, x, domain='RR')

    Create a multivariate polynomial:

    >>> Poly(y*x**2 + x*y + 1)
    Poly(x**2*y + x*y + 1, x, y, domain='ZZ')

    Create a univariate polynomial, where y is a constant:

    >>> Poly(y*x**2 + x*y + 1,x)
    Poly(y*x**2 + y*x + 1, x, domain='ZZ[y]')

    You can evaluate the above polynomial as a function of y:

    >>> Poly(y*x**2 + x*y + 1,x).eval(2)
    6*y + 1

    See Also
    ========

    sympy.core.expr.Expr

    """
    __slots__ = ('rep', 'gens')
    is_commutative = True
    is_Poly = True
    _op_priority = 10.001

    def __new__(cls, rep, *gens, **args):
        if False:
            while True:
                i = 10
        'Create a new polynomial instance out of something useful. '
        opt = options.build_options(gens, args)
        if 'order' in opt:
            raise NotImplementedError("'order' keyword is not implemented yet")
        if isinstance(rep, (DMP, DMF, ANP, DomainElement)):
            return cls._from_domain_element(rep, opt)
        elif iterable(rep, exclude=str):
            if isinstance(rep, dict):
                return cls._from_dict(rep, opt)
            else:
                return cls._from_list(list(rep), opt)
        else:
            rep = sympify(rep, evaluate=type(rep) is not str)
            if rep.is_Poly:
                return cls._from_poly(rep, opt)
            else:
                return cls._from_expr(rep, opt)

    @classmethod
    def new(cls, rep, *gens):
        if False:
            i = 10
            return i + 15
        'Construct :class:`Poly` instance from raw representation. '
        if not isinstance(rep, DMP):
            raise PolynomialError('invalid polynomial representation: %s' % rep)
        elif rep.lev != len(gens) - 1:
            raise PolynomialError('invalid arguments: %s, %s' % (rep, gens))
        obj = Basic.__new__(cls)
        obj.rep = rep
        obj.gens = gens
        return obj

    @property
    def expr(self):
        if False:
            i = 10
            return i + 15
        return basic_from_dict(self.rep.to_sympy_dict(), *self.gens)

    @property
    def args(self):
        if False:
            i = 10
            return i + 15
        return (self.expr,) + self.gens

    def _hashable_content(self):
        if False:
            for i in range(10):
                print('nop')
        return (self.rep,) + self.gens

    @classmethod
    def from_dict(cls, rep, *gens, **args):
        if False:
            return 10
        'Construct a polynomial from a ``dict``. '
        opt = options.build_options(gens, args)
        return cls._from_dict(rep, opt)

    @classmethod
    def from_list(cls, rep, *gens, **args):
        if False:
            while True:
                i = 10
        'Construct a polynomial from a ``list``. '
        opt = options.build_options(gens, args)
        return cls._from_list(rep, opt)

    @classmethod
    def from_poly(cls, rep, *gens, **args):
        if False:
            while True:
                i = 10
        'Construct a polynomial from a polynomial. '
        opt = options.build_options(gens, args)
        return cls._from_poly(rep, opt)

    @classmethod
    def from_expr(cls, rep, *gens, **args):
        if False:
            i = 10
            return i + 15
        'Construct a polynomial from an expression. '
        opt = options.build_options(gens, args)
        return cls._from_expr(rep, opt)

    @classmethod
    def _from_dict(cls, rep, opt):
        if False:
            print('Hello World!')
        'Construct a polynomial from a ``dict``. '
        gens = opt.gens
        if not gens:
            raise GeneratorsNeeded("Cannot initialize from 'dict' without generators")
        level = len(gens) - 1
        domain = opt.domain
        if domain is None:
            (domain, rep) = construct_domain(rep, opt=opt)
        else:
            for (monom, coeff) in rep.items():
                rep[monom] = domain.convert(coeff)
        return cls.new(DMP.from_dict(rep, level, domain), *gens)

    @classmethod
    def _from_list(cls, rep, opt):
        if False:
            i = 10
            return i + 15
        'Construct a polynomial from a ``list``. '
        gens = opt.gens
        if not gens:
            raise GeneratorsNeeded("Cannot initialize from 'list' without generators")
        elif len(gens) != 1:
            raise MultivariatePolynomialError("'list' representation not supported")
        level = len(gens) - 1
        domain = opt.domain
        if domain is None:
            (domain, rep) = construct_domain(rep, opt=opt)
        else:
            rep = list(map(domain.convert, rep))
        return cls.new(DMP.from_list(rep, level, domain), *gens)

    @classmethod
    def _from_poly(cls, rep, opt):
        if False:
            for i in range(10):
                print('nop')
        'Construct a polynomial from a polynomial. '
        if cls != rep.__class__:
            rep = cls.new(rep.rep, *rep.gens)
        gens = opt.gens
        field = opt.field
        domain = opt.domain
        if gens and rep.gens != gens:
            if set(rep.gens) != set(gens):
                return cls._from_expr(rep.as_expr(), opt)
            else:
                rep = rep.reorder(*gens)
        if 'domain' in opt and domain:
            rep = rep.set_domain(domain)
        elif field is True:
            rep = rep.to_field()
        return rep

    @classmethod
    def _from_expr(cls, rep, opt):
        if False:
            return 10
        'Construct a polynomial from an expression. '
        (rep, opt) = _dict_from_expr(rep, opt)
        return cls._from_dict(rep, opt)

    @classmethod
    def _from_domain_element(cls, rep, opt):
        if False:
            while True:
                i = 10
        gens = opt.gens
        domain = opt.domain
        level = len(gens) - 1
        rep = [domain.convert(rep)]
        return cls.new(DMP.from_list(rep, level, domain), *gens)

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return super().__hash__()

    @property
    def free_symbols(self):
        if False:
            print('Hello World!')
        '\n        Free symbols of a polynomial expression.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x, y, z\n\n        >>> Poly(x**2 + 1).free_symbols\n        {x}\n        >>> Poly(x**2 + y).free_symbols\n        {x, y}\n        >>> Poly(x**2 + y, x).free_symbols\n        {x, y}\n        >>> Poly(x**2 + y, x, z).free_symbols\n        {x, y}\n\n        '
        symbols = set()
        gens = self.gens
        for i in range(len(gens)):
            for monom in self.monoms():
                if monom[i]:
                    symbols |= gens[i].free_symbols
                    break
        return symbols | self.free_symbols_in_domain

    @property
    def free_symbols_in_domain(self):
        if False:
            i = 10
            return i + 15
        '\n        Free symbols of the domain of ``self``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x, y\n\n        >>> Poly(x**2 + 1).free_symbols_in_domain\n        set()\n        >>> Poly(x**2 + y).free_symbols_in_domain\n        set()\n        >>> Poly(x**2 + y, x).free_symbols_in_domain\n        {y}\n\n        '
        (domain, symbols) = (self.rep.dom, set())
        if domain.is_Composite:
            for gen in domain.symbols:
                symbols |= gen.free_symbols
        elif domain.is_EX:
            for coeff in self.coeffs():
                symbols |= coeff.free_symbols
        return symbols

    @property
    def gen(self):
        if False:
            return 10
        '\n        Return the principal generator.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x**2 + 1, x).gen\n        x\n\n        '
        return self.gens[0]

    @property
    def domain(self):
        if False:
            print('Hello World!')
        "Get the ground domain of a :py:class:`~.Poly`\n\n        Returns\n        =======\n\n        :py:class:`~.Domain`:\n            Ground domain of the :py:class:`~.Poly`.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly, Symbol\n        >>> x = Symbol('x')\n        >>> p = Poly(x**2 + x)\n        >>> p\n        Poly(x**2 + x, x, domain='ZZ')\n        >>> p.domain\n        ZZ\n        "
        return self.get_domain()

    @property
    def zero(self):
        if False:
            print('Hello World!')
        "Return zero polynomial with ``self``'s properties. "
        return self.new(self.rep.zero(self.rep.lev, self.rep.dom), *self.gens)

    @property
    def one(self):
        if False:
            return 10
        "Return one polynomial with ``self``'s properties. "
        return self.new(self.rep.one(self.rep.lev, self.rep.dom), *self.gens)

    @property
    def unit(self):
        if False:
            while True:
                i = 10
        "Return unit polynomial with ``self``'s properties. "
        return self.new(self.rep.unit(self.rep.lev, self.rep.dom), *self.gens)

    def unify(f, g):
        if False:
            for i in range(10):
                print('nop')
        "\n        Make ``f`` and ``g`` belong to the same domain.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> f, g = Poly(x/2 + 1), Poly(2*x + 1)\n\n        >>> f\n        Poly(1/2*x + 1, x, domain='QQ')\n        >>> g\n        Poly(2*x + 1, x, domain='ZZ')\n\n        >>> F, G = f.unify(g)\n\n        >>> F\n        Poly(1/2*x + 1, x, domain='QQ')\n        >>> G\n        Poly(2*x + 1, x, domain='QQ')\n\n        "
        (_, per, F, G) = f._unify(g)
        return (per(F), per(G))

    def _unify(f, g):
        if False:
            return 10
        g = sympify(g)
        if not g.is_Poly:
            try:
                g_coeff = f.rep.dom.from_sympy(g)
            except CoercionFailed:
                raise UnificationFailed('Cannot unify %s with %s' % (f, g))
            else:
                return (f.rep.dom, f.per, f.rep, f.rep.ground_new(g_coeff))
        if isinstance(f.rep, DMP) and isinstance(g.rep, DMP):
            gens = _unify_gens(f.gens, g.gens)
            (dom, lev) = (f.rep.dom.unify(g.rep.dom, gens), len(gens) - 1)
            if f.gens != gens:
                (f_monoms, f_coeffs) = _dict_reorder(f.rep.to_dict(), f.gens, gens)
                if f.rep.dom != dom:
                    f_coeffs = [dom.convert(c, f.rep.dom) for c in f_coeffs]
                F = DMP.from_dict(dict(list(zip(f_monoms, f_coeffs))), lev, dom)
            else:
                F = f.rep.convert(dom)
            if g.gens != gens:
                (g_monoms, g_coeffs) = _dict_reorder(g.rep.to_dict(), g.gens, gens)
                if g.rep.dom != dom:
                    g_coeffs = [dom.convert(c, g.rep.dom) for c in g_coeffs]
                G = DMP.from_dict(dict(list(zip(g_monoms, g_coeffs))), lev, dom)
            else:
                G = g.rep.convert(dom)
        else:
            raise UnificationFailed('Cannot unify %s with %s' % (f, g))
        cls = f.__class__

        def per(rep, dom=dom, gens=gens, remove=None):
            if False:
                i = 10
                return i + 15
            if remove is not None:
                gens = gens[:remove] + gens[remove + 1:]
                if not gens:
                    return dom.to_sympy(rep)
            return cls.new(rep, *gens)
        return (dom, per, F, G)

    def per(f, rep, gens=None, remove=None):
        if False:
            i = 10
            return i + 15
        "\n        Create a Poly out of the given representation.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly, ZZ\n        >>> from sympy.abc import x, y\n\n        >>> from sympy.polys.polyclasses import DMP\n\n        >>> a = Poly(x**2 + 1)\n\n        >>> a.per(DMP([ZZ(1), ZZ(1)], ZZ), gens=[y])\n        Poly(y + 1, y, domain='ZZ')\n\n        "
        if gens is None:
            gens = f.gens
        if remove is not None:
            gens = gens[:remove] + gens[remove + 1:]
            if not gens:
                return f.rep.dom.to_sympy(rep)
        return f.__class__.new(rep, *gens)

    def set_domain(f, domain):
        if False:
            print('Hello World!')
        'Set the ground domain of ``f``. '
        opt = options.build_options(f.gens, {'domain': domain})
        return f.per(f.rep.convert(opt.domain))

    def get_domain(f):
        if False:
            while True:
                i = 10
        'Get the ground domain of ``f``. '
        return f.rep.dom

    def set_modulus(f, modulus):
        if False:
            return 10
        '\n        Set the modulus of ``f``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(5*x**2 + 2*x - 1, x).set_modulus(2)\n        Poly(x**2 + 1, x, modulus=2)\n\n        '
        modulus = options.Modulus.preprocess(modulus)
        return f.set_domain(FF(modulus))

    def get_modulus(f):
        if False:
            print('Hello World!')
        '\n        Get the modulus of ``f``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x**2 + 1, modulus=2).get_modulus()\n        2\n\n        '
        domain = f.get_domain()
        if domain.is_FiniteField:
            return Integer(domain.characteristic())
        else:
            raise PolynomialError('not a polynomial over a Galois field')

    def _eval_subs(f, old, new):
        if False:
            while True:
                i = 10
        'Internal implementation of :func:`subs`. '
        if old in f.gens:
            if new.is_number:
                return f.eval(old, new)
            else:
                try:
                    return f.replace(old, new)
                except PolynomialError:
                    pass
        return f.as_expr().subs(old, new)

    def exclude(f):
        if False:
            print('Hello World!')
        "\n        Remove unnecessary generators from ``f``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import a, b, c, d, x\n\n        >>> Poly(a + x, a, b, c, d, x).exclude()\n        Poly(a + x, a, x, domain='ZZ')\n\n        "
        (J, new) = f.rep.exclude()
        gens = [gen for (j, gen) in enumerate(f.gens) if j not in J]
        return f.per(new, gens=gens)

    def replace(f, x, y=None, **_ignore):
        if False:
            return 10
        "\n        Replace ``x`` with ``y`` in generators list.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x, y\n\n        >>> Poly(x**2 + 1, x).replace(x, y)\n        Poly(y**2 + 1, y, domain='ZZ')\n\n        "
        if y is None:
            if f.is_univariate:
                (x, y) = (f.gen, x)
            else:
                raise PolynomialError('syntax supported only in univariate case')
        if x == y or x not in f.gens:
            return f
        if x in f.gens and y not in f.gens:
            dom = f.get_domain()
            if not dom.is_Composite or y not in dom.symbols:
                gens = list(f.gens)
                gens[gens.index(x)] = y
                return f.per(f.rep, gens=gens)
        raise PolynomialError('Cannot replace %s with %s in %s' % (x, y, f))

    def match(f, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Match expression from Poly. See Basic.match()'
        return f.as_expr().match(*args, **kwargs)

    def reorder(f, *gens, **args):
        if False:
            i = 10
            return i + 15
        "\n        Efficiently apply new order of generators.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x, y\n\n        >>> Poly(x**2 + x*y**2, x, y).reorder(y, x)\n        Poly(y**2*x + x**2, y, x, domain='ZZ')\n\n        "
        opt = options.Options((), args)
        if not gens:
            gens = _sort_gens(f.gens, opt=opt)
        elif set(f.gens) != set(gens):
            raise PolynomialError('generators list can differ only up to order of elements')
        rep = dict(list(zip(*_dict_reorder(f.rep.to_dict(), f.gens, gens))))
        return f.per(DMP.from_dict(rep, len(gens) - 1, f.rep.dom), gens=gens)

    def ltrim(f, gen):
        if False:
            print('Hello World!')
        "\n        Remove dummy generators from ``f`` that are to the left of\n        specified ``gen`` in the generators as ordered. When ``gen``\n        is an integer, it refers to the generator located at that\n        position within the tuple of generators of ``f``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x, y, z\n\n        >>> Poly(y**2 + y*z**2, x, y, z).ltrim(y)\n        Poly(y**2 + y*z**2, y, z, domain='ZZ')\n        >>> Poly(z, x, y, z).ltrim(-1)\n        Poly(z, z, domain='ZZ')\n\n        "
        rep = f.as_dict(native=True)
        j = f._gen_to_level(gen)
        terms = {}
        for (monom, coeff) in rep.items():
            if any(monom[:j]):
                raise PolynomialError('Cannot left trim %s' % f)
            terms[monom[j:]] = coeff
        gens = f.gens[j:]
        return f.new(DMP.from_dict(terms, len(gens) - 1, f.rep.dom), *gens)

    def has_only_gens(f, *gens):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return ``True`` if ``Poly(f, *gens)`` retains ground domain.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x, y, z\n\n        >>> Poly(x*y + 1, x, y, z).has_only_gens(x, y)\n        True\n        >>> Poly(x*y + z, x, y, z).has_only_gens(x, y)\n        False\n\n        '
        indices = set()
        for gen in gens:
            try:
                index = f.gens.index(gen)
            except ValueError:
                raise GeneratorsError("%s doesn't have %s as generator" % (f, gen))
            else:
                indices.add(index)
        for monom in f.monoms():
            for (i, elt) in enumerate(monom):
                if i not in indices and elt:
                    return False
        return True

    def to_ring(f):
        if False:
            return 10
        "\n        Make the ground domain a ring.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly, QQ\n        >>> from sympy.abc import x\n\n        >>> Poly(x**2 + 1, domain=QQ).to_ring()\n        Poly(x**2 + 1, x, domain='ZZ')\n\n        "
        if hasattr(f.rep, 'to_ring'):
            result = f.rep.to_ring()
        else:
            raise OperationNotSupported(f, 'to_ring')
        return f.per(result)

    def to_field(f):
        if False:
            return 10
        "\n        Make the ground domain a field.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly, ZZ\n        >>> from sympy.abc import x\n\n        >>> Poly(x**2 + 1, x, domain=ZZ).to_field()\n        Poly(x**2 + 1, x, domain='QQ')\n\n        "
        if hasattr(f.rep, 'to_field'):
            result = f.rep.to_field()
        else:
            raise OperationNotSupported(f, 'to_field')
        return f.per(result)

    def to_exact(f):
        if False:
            print('Hello World!')
        "\n        Make the ground domain exact.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly, RR\n        >>> from sympy.abc import x\n\n        >>> Poly(x**2 + 1.0, x, domain=RR).to_exact()\n        Poly(x**2 + 1, x, domain='QQ')\n\n        "
        if hasattr(f.rep, 'to_exact'):
            result = f.rep.to_exact()
        else:
            raise OperationNotSupported(f, 'to_exact')
        return f.per(result)

    def retract(f, field=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Recalculate the ground domain of a polynomial.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> f = Poly(x**2 + 1, x, domain='QQ[y]')\n        >>> f\n        Poly(x**2 + 1, x, domain='QQ[y]')\n\n        >>> f.retract()\n        Poly(x**2 + 1, x, domain='ZZ')\n        >>> f.retract(field=True)\n        Poly(x**2 + 1, x, domain='QQ')\n\n        "
        (dom, rep) = construct_domain(f.as_dict(zero=True), field=field, composite=f.domain.is_Composite or None)
        return f.from_dict(rep, f.gens, domain=dom)

    def slice(f, x, m, n=None):
        if False:
            i = 10
            return i + 15
        'Take a continuous subsequence of terms of ``f``. '
        if n is None:
            (j, m, n) = (0, x, m)
        else:
            j = f._gen_to_level(x)
        (m, n) = (int(m), int(n))
        if hasattr(f.rep, 'slice'):
            result = f.rep.slice(m, n, j)
        else:
            raise OperationNotSupported(f, 'slice')
        return f.per(result)

    def coeffs(f, order=None):
        if False:
            print('Hello World!')
        '\n        Returns all non-zero coefficients from ``f`` in lex order.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x**3 + 2*x + 3, x).coeffs()\n        [1, 2, 3]\n\n        See Also\n        ========\n        all_coeffs\n        coeff_monomial\n        nth\n\n        '
        return [f.rep.dom.to_sympy(c) for c in f.rep.coeffs(order=order)]

    def monoms(f, order=None):
        if False:
            return 10
        '\n        Returns all non-zero monomials from ``f`` in lex order.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x, y\n\n        >>> Poly(x**2 + 2*x*y**2 + x*y + 3*y, x, y).monoms()\n        [(2, 0), (1, 2), (1, 1), (0, 1)]\n\n        See Also\n        ========\n        all_monoms\n\n        '
        return f.rep.monoms(order=order)

    def terms(f, order=None):
        if False:
            return 10
        '\n        Returns all non-zero terms from ``f`` in lex order.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x, y\n\n        >>> Poly(x**2 + 2*x*y**2 + x*y + 3*y, x, y).terms()\n        [((2, 0), 1), ((1, 2), 2), ((1, 1), 1), ((0, 1), 3)]\n\n        See Also\n        ========\n        all_terms\n\n        '
        return [(m, f.rep.dom.to_sympy(c)) for (m, c) in f.rep.terms(order=order)]

    def all_coeffs(f):
        if False:
            print('Hello World!')
        '\n        Returns all coefficients from a univariate polynomial ``f``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x**3 + 2*x - 1, x).all_coeffs()\n        [1, 0, 2, -1]\n\n        '
        return [f.rep.dom.to_sympy(c) for c in f.rep.all_coeffs()]

    def all_monoms(f):
        if False:
            return 10
        '\n        Returns all monomials from a univariate polynomial ``f``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x**3 + 2*x - 1, x).all_monoms()\n        [(3,), (2,), (1,), (0,)]\n\n        See Also\n        ========\n        all_terms\n\n        '
        return f.rep.all_monoms()

    def all_terms(f):
        if False:
            while True:
                i = 10
        '\n        Returns all terms from a univariate polynomial ``f``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x**3 + 2*x - 1, x).all_terms()\n        [((3,), 1), ((2,), 0), ((1,), 2), ((0,), -1)]\n\n        '
        return [(m, f.rep.dom.to_sympy(c)) for (m, c) in f.rep.all_terms()]

    def termwise(f, func, *gens, **args):
        if False:
            return 10
        "\n        Apply a function to all terms of ``f``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> def func(k, coeff):\n        ...     k = k[0]\n        ...     return coeff//10**(2-k)\n\n        >>> Poly(x**2 + 20*x + 400).termwise(func)\n        Poly(x**2 + 2*x + 4, x, domain='ZZ')\n\n        "
        terms = {}
        for (monom, coeff) in f.terms():
            result = func(monom, coeff)
            if isinstance(result, tuple):
                (monom, coeff) = result
            else:
                coeff = result
            if coeff:
                if monom not in terms:
                    terms[monom] = coeff
                else:
                    raise PolynomialError('%s monomial was generated twice' % monom)
        return f.from_dict(terms, *(gens or f.gens), **args)

    def length(f):
        if False:
            i = 10
            return i + 15
        '\n        Returns the number of non-zero terms in ``f``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x**2 + 2*x - 1).length()\n        3\n\n        '
        return len(f.as_dict())

    def as_dict(f, native=False, zero=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Switch to a ``dict`` representation.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x, y\n\n        >>> Poly(x**2 + 2*x*y**2 - y, x, y).as_dict()\n        {(0, 1): -1, (1, 2): 2, (2, 0): 1}\n\n        '
        if native:
            return f.rep.to_dict(zero=zero)
        else:
            return f.rep.to_sympy_dict(zero=zero)

    def as_list(f, native=False):
        if False:
            while True:
                i = 10
        'Switch to a ``list`` representation. '
        if native:
            return f.rep.to_list()
        else:
            return f.rep.to_sympy_list()

    def as_expr(f, *gens):
        if False:
            return 10
        '\n        Convert a Poly instance to an Expr instance.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x, y\n\n        >>> f = Poly(x**2 + 2*x*y**2 - y, x, y)\n\n        >>> f.as_expr()\n        x**2 + 2*x*y**2 - y\n        >>> f.as_expr({x: 5})\n        10*y**2 - y + 25\n        >>> f.as_expr(5, 6)\n        379\n\n        '
        if not gens:
            return f.expr
        if len(gens) == 1 and isinstance(gens[0], dict):
            mapping = gens[0]
            gens = list(f.gens)
            for (gen, value) in mapping.items():
                try:
                    index = gens.index(gen)
                except ValueError:
                    raise GeneratorsError("%s doesn't have %s as generator" % (f, gen))
                else:
                    gens[index] = value
        return basic_from_dict(f.rep.to_sympy_dict(), *gens)

    def as_poly(self, *gens, **args):
        if False:
            for i in range(10):
                print('nop')
        "Converts ``self`` to a polynomial or returns ``None``.\n\n        >>> from sympy import sin\n        >>> from sympy.abc import x, y\n\n        >>> print((x**2 + x*y).as_poly())\n        Poly(x**2 + x*y, x, y, domain='ZZ')\n\n        >>> print((x**2 + x*y).as_poly(x, y))\n        Poly(x**2 + x*y, x, y, domain='ZZ')\n\n        >>> print((x**2 + sin(y)).as_poly(x, y))\n        None\n\n        "
        try:
            poly = Poly(self, *gens, **args)
            if not poly.is_Poly:
                return None
            else:
                return poly
        except PolynomialError:
            return None

    def lift(f):
        if False:
            return 10
        "\n        Convert algebraic coefficients to rationals.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly, I\n        >>> from sympy.abc import x\n\n        >>> Poly(x**2 + I*x + 1, x, extension=I).lift()\n        Poly(x**4 + 3*x**2 + 1, x, domain='QQ')\n\n        "
        if hasattr(f.rep, 'lift'):
            result = f.rep.lift()
        else:
            raise OperationNotSupported(f, 'lift')
        return f.per(result)

    def deflate(f):
        if False:
            while True:
                i = 10
        "\n        Reduce degree of ``f`` by mapping ``x_i**m`` to ``y_i``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x, y\n\n        >>> Poly(x**6*y**2 + x**3 + 1, x, y).deflate()\n        ((3, 2), Poly(x**2*y + x + 1, x, y, domain='ZZ'))\n\n        "
        if hasattr(f.rep, 'deflate'):
            (J, result) = f.rep.deflate()
        else:
            raise OperationNotSupported(f, 'deflate')
        return (J, f.per(result))

    def inject(f, front=False):
        if False:
            return 10
        "\n        Inject ground domain generators into ``f``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x, y\n\n        >>> f = Poly(x**2*y + x*y**3 + x*y + 1, x)\n\n        >>> f.inject()\n        Poly(x**2*y + x*y**3 + x*y + 1, x, y, domain='ZZ')\n        >>> f.inject(front=True)\n        Poly(y**3*x + y*x**2 + y*x + 1, y, x, domain='ZZ')\n\n        "
        dom = f.rep.dom
        if dom.is_Numerical:
            return f
        elif not dom.is_Poly:
            raise DomainError('Cannot inject generators over %s' % dom)
        if hasattr(f.rep, 'inject'):
            result = f.rep.inject(front=front)
        else:
            raise OperationNotSupported(f, 'inject')
        if front:
            gens = dom.symbols + f.gens
        else:
            gens = f.gens + dom.symbols
        return f.new(result, *gens)

    def eject(f, *gens):
        if False:
            print('Hello World!')
        "\n        Eject selected generators into the ground domain.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x, y\n\n        >>> f = Poly(x**2*y + x*y**3 + x*y + 1, x, y)\n\n        >>> f.eject(x)\n        Poly(x*y**3 + (x**2 + x)*y + 1, y, domain='ZZ[x]')\n        >>> f.eject(y)\n        Poly(y*x**2 + (y**3 + y)*x + 1, x, domain='ZZ[y]')\n\n        "
        dom = f.rep.dom
        if not dom.is_Numerical:
            raise DomainError('Cannot eject generators over %s' % dom)
        k = len(gens)
        if f.gens[:k] == gens:
            (_gens, front) = (f.gens[k:], True)
        elif f.gens[-k:] == gens:
            (_gens, front) = (f.gens[:-k], False)
        else:
            raise NotImplementedError('can only eject front or back generators')
        dom = dom.inject(*gens)
        if hasattr(f.rep, 'eject'):
            result = f.rep.eject(dom, front=front)
        else:
            raise OperationNotSupported(f, 'eject')
        return f.new(result, *_gens)

    def terms_gcd(f):
        if False:
            for i in range(10):
                print('nop')
        "\n        Remove GCD of terms from the polynomial ``f``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x, y\n\n        >>> Poly(x**6*y**2 + x**3*y, x, y).terms_gcd()\n        ((3, 1), Poly(x**3*y + 1, x, y, domain='ZZ'))\n\n        "
        if hasattr(f.rep, 'terms_gcd'):
            (J, result) = f.rep.terms_gcd()
        else:
            raise OperationNotSupported(f, 'terms_gcd')
        return (J, f.per(result))

    def add_ground(f, coeff):
        if False:
            print('Hello World!')
        "\n        Add an element of the ground domain to ``f``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x + 1).add_ground(2)\n        Poly(x + 3, x, domain='ZZ')\n\n        "
        if hasattr(f.rep, 'add_ground'):
            result = f.rep.add_ground(coeff)
        else:
            raise OperationNotSupported(f, 'add_ground')
        return f.per(result)

    def sub_ground(f, coeff):
        if False:
            for i in range(10):
                print('nop')
        "\n        Subtract an element of the ground domain from ``f``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x + 1).sub_ground(2)\n        Poly(x - 1, x, domain='ZZ')\n\n        "
        if hasattr(f.rep, 'sub_ground'):
            result = f.rep.sub_ground(coeff)
        else:
            raise OperationNotSupported(f, 'sub_ground')
        return f.per(result)

    def mul_ground(f, coeff):
        if False:
            print('Hello World!')
        "\n        Multiply ``f`` by a an element of the ground domain.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x + 1).mul_ground(2)\n        Poly(2*x + 2, x, domain='ZZ')\n\n        "
        if hasattr(f.rep, 'mul_ground'):
            result = f.rep.mul_ground(coeff)
        else:
            raise OperationNotSupported(f, 'mul_ground')
        return f.per(result)

    def quo_ground(f, coeff):
        if False:
            return 10
        "\n        Quotient of ``f`` by a an element of the ground domain.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(2*x + 4).quo_ground(2)\n        Poly(x + 2, x, domain='ZZ')\n\n        >>> Poly(2*x + 3).quo_ground(2)\n        Poly(x + 1, x, domain='ZZ')\n\n        "
        if hasattr(f.rep, 'quo_ground'):
            result = f.rep.quo_ground(coeff)
        else:
            raise OperationNotSupported(f, 'quo_ground')
        return f.per(result)

    def exquo_ground(f, coeff):
        if False:
            print('Hello World!')
        "\n        Exact quotient of ``f`` by a an element of the ground domain.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(2*x + 4).exquo_ground(2)\n        Poly(x + 2, x, domain='ZZ')\n\n        >>> Poly(2*x + 3).exquo_ground(2)\n        Traceback (most recent call last):\n        ...\n        ExactQuotientFailed: 2 does not divide 3 in ZZ\n\n        "
        if hasattr(f.rep, 'exquo_ground'):
            result = f.rep.exquo_ground(coeff)
        else:
            raise OperationNotSupported(f, 'exquo_ground')
        return f.per(result)

    def abs(f):
        if False:
            print('Hello World!')
        "\n        Make all coefficients in ``f`` positive.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x**2 - 1, x).abs()\n        Poly(x**2 + 1, x, domain='ZZ')\n\n        "
        if hasattr(f.rep, 'abs'):
            result = f.rep.abs()
        else:
            raise OperationNotSupported(f, 'abs')
        return f.per(result)

    def neg(f):
        if False:
            for i in range(10):
                print('nop')
        "\n        Negate all coefficients in ``f``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x**2 - 1, x).neg()\n        Poly(-x**2 + 1, x, domain='ZZ')\n\n        >>> -Poly(x**2 - 1, x)\n        Poly(-x**2 + 1, x, domain='ZZ')\n\n        "
        if hasattr(f.rep, 'neg'):
            result = f.rep.neg()
        else:
            raise OperationNotSupported(f, 'neg')
        return f.per(result)

    def add(f, g):
        if False:
            while True:
                i = 10
        "\n        Add two polynomials ``f`` and ``g``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x**2 + 1, x).add(Poly(x - 2, x))\n        Poly(x**2 + x - 1, x, domain='ZZ')\n\n        >>> Poly(x**2 + 1, x) + Poly(x - 2, x)\n        Poly(x**2 + x - 1, x, domain='ZZ')\n\n        "
        g = sympify(g)
        if not g.is_Poly:
            return f.add_ground(g)
        (_, per, F, G) = f._unify(g)
        if hasattr(f.rep, 'add'):
            result = F.add(G)
        else:
            raise OperationNotSupported(f, 'add')
        return per(result)

    def sub(f, g):
        if False:
            return 10
        "\n        Subtract two polynomials ``f`` and ``g``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x**2 + 1, x).sub(Poly(x - 2, x))\n        Poly(x**2 - x + 3, x, domain='ZZ')\n\n        >>> Poly(x**2 + 1, x) - Poly(x - 2, x)\n        Poly(x**2 - x + 3, x, domain='ZZ')\n\n        "
        g = sympify(g)
        if not g.is_Poly:
            return f.sub_ground(g)
        (_, per, F, G) = f._unify(g)
        if hasattr(f.rep, 'sub'):
            result = F.sub(G)
        else:
            raise OperationNotSupported(f, 'sub')
        return per(result)

    def mul(f, g):
        if False:
            print('Hello World!')
        "\n        Multiply two polynomials ``f`` and ``g``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x**2 + 1, x).mul(Poly(x - 2, x))\n        Poly(x**3 - 2*x**2 + x - 2, x, domain='ZZ')\n\n        >>> Poly(x**2 + 1, x)*Poly(x - 2, x)\n        Poly(x**3 - 2*x**2 + x - 2, x, domain='ZZ')\n\n        "
        g = sympify(g)
        if not g.is_Poly:
            return f.mul_ground(g)
        (_, per, F, G) = f._unify(g)
        if hasattr(f.rep, 'mul'):
            result = F.mul(G)
        else:
            raise OperationNotSupported(f, 'mul')
        return per(result)

    def sqr(f):
        if False:
            for i in range(10):
                print('nop')
        "\n        Square a polynomial ``f``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x - 2, x).sqr()\n        Poly(x**2 - 4*x + 4, x, domain='ZZ')\n\n        >>> Poly(x - 2, x)**2\n        Poly(x**2 - 4*x + 4, x, domain='ZZ')\n\n        "
        if hasattr(f.rep, 'sqr'):
            result = f.rep.sqr()
        else:
            raise OperationNotSupported(f, 'sqr')
        return f.per(result)

    def pow(f, n):
        if False:
            for i in range(10):
                print('nop')
        "\n        Raise ``f`` to a non-negative power ``n``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x - 2, x).pow(3)\n        Poly(x**3 - 6*x**2 + 12*x - 8, x, domain='ZZ')\n\n        >>> Poly(x - 2, x)**3\n        Poly(x**3 - 6*x**2 + 12*x - 8, x, domain='ZZ')\n\n        "
        n = int(n)
        if hasattr(f.rep, 'pow'):
            result = f.rep.pow(n)
        else:
            raise OperationNotSupported(f, 'pow')
        return f.per(result)

    def pdiv(f, g):
        if False:
            while True:
                i = 10
        "\n        Polynomial pseudo-division of ``f`` by ``g``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x**2 + 1, x).pdiv(Poly(2*x - 4, x))\n        (Poly(2*x + 4, x, domain='ZZ'), Poly(20, x, domain='ZZ'))\n\n        "
        (_, per, F, G) = f._unify(g)
        if hasattr(f.rep, 'pdiv'):
            (q, r) = F.pdiv(G)
        else:
            raise OperationNotSupported(f, 'pdiv')
        return (per(q), per(r))

    def prem(f, g):
        if False:
            return 10
        "\n        Polynomial pseudo-remainder of ``f`` by ``g``.\n\n        Caveat: The function prem(f, g, x) can be safely used to compute\n          in Z[x] _only_ subresultant polynomial remainder sequences (prs's).\n\n          To safely compute Euclidean and Sturmian prs's in Z[x]\n          employ anyone of the corresponding functions found in\n          the module sympy.polys.subresultants_qq_zz. The functions\n          in the module with suffix _pg compute prs's in Z[x] employing\n          rem(f, g, x), whereas the functions with suffix _amv\n          compute prs's in Z[x] employing rem_z(f, g, x).\n\n          The function rem_z(f, g, x) differs from prem(f, g, x) in that\n          to compute the remainder polynomials in Z[x] it premultiplies\n          the divident times the absolute value of the leading coefficient\n          of the divisor raised to the power degree(f, x) - degree(g, x) + 1.\n\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x**2 + 1, x).prem(Poly(2*x - 4, x))\n        Poly(20, x, domain='ZZ')\n\n        "
        (_, per, F, G) = f._unify(g)
        if hasattr(f.rep, 'prem'):
            result = F.prem(G)
        else:
            raise OperationNotSupported(f, 'prem')
        return per(result)

    def pquo(f, g):
        if False:
            for i in range(10):
                print('nop')
        "\n        Polynomial pseudo-quotient of ``f`` by ``g``.\n\n        See the Caveat note in the function prem(f, g).\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x**2 + 1, x).pquo(Poly(2*x - 4, x))\n        Poly(2*x + 4, x, domain='ZZ')\n\n        >>> Poly(x**2 - 1, x).pquo(Poly(2*x - 2, x))\n        Poly(2*x + 2, x, domain='ZZ')\n\n        "
        (_, per, F, G) = f._unify(g)
        if hasattr(f.rep, 'pquo'):
            result = F.pquo(G)
        else:
            raise OperationNotSupported(f, 'pquo')
        return per(result)

    def pexquo(f, g):
        if False:
            return 10
        "\n        Polynomial exact pseudo-quotient of ``f`` by ``g``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x**2 - 1, x).pexquo(Poly(2*x - 2, x))\n        Poly(2*x + 2, x, domain='ZZ')\n\n        >>> Poly(x**2 + 1, x).pexquo(Poly(2*x - 4, x))\n        Traceback (most recent call last):\n        ...\n        ExactQuotientFailed: 2*x - 4 does not divide x**2 + 1\n\n        "
        (_, per, F, G) = f._unify(g)
        if hasattr(f.rep, 'pexquo'):
            try:
                result = F.pexquo(G)
            except ExactQuotientFailed as exc:
                raise exc.new(f.as_expr(), g.as_expr())
        else:
            raise OperationNotSupported(f, 'pexquo')
        return per(result)

    def div(f, g, auto=True):
        if False:
            while True:
                i = 10
        "\n        Polynomial division with remainder of ``f`` by ``g``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x**2 + 1, x).div(Poly(2*x - 4, x))\n        (Poly(1/2*x + 1, x, domain='QQ'), Poly(5, x, domain='QQ'))\n\n        >>> Poly(x**2 + 1, x).div(Poly(2*x - 4, x), auto=False)\n        (Poly(0, x, domain='ZZ'), Poly(x**2 + 1, x, domain='ZZ'))\n\n        "
        (dom, per, F, G) = f._unify(g)
        retract = False
        if auto and dom.is_Ring and (not dom.is_Field):
            (F, G) = (F.to_field(), G.to_field())
            retract = True
        if hasattr(f.rep, 'div'):
            (q, r) = F.div(G)
        else:
            raise OperationNotSupported(f, 'div')
        if retract:
            try:
                (Q, R) = (q.to_ring(), r.to_ring())
            except CoercionFailed:
                pass
            else:
                (q, r) = (Q, R)
        return (per(q), per(r))

    def rem(f, g, auto=True):
        if False:
            print('Hello World!')
        "\n        Computes the polynomial remainder of ``f`` by ``g``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x**2 + 1, x).rem(Poly(2*x - 4, x))\n        Poly(5, x, domain='ZZ')\n\n        >>> Poly(x**2 + 1, x).rem(Poly(2*x - 4, x), auto=False)\n        Poly(x**2 + 1, x, domain='ZZ')\n\n        "
        (dom, per, F, G) = f._unify(g)
        retract = False
        if auto and dom.is_Ring and (not dom.is_Field):
            (F, G) = (F.to_field(), G.to_field())
            retract = True
        if hasattr(f.rep, 'rem'):
            r = F.rem(G)
        else:
            raise OperationNotSupported(f, 'rem')
        if retract:
            try:
                r = r.to_ring()
            except CoercionFailed:
                pass
        return per(r)

    def quo(f, g, auto=True):
        if False:
            while True:
                i = 10
        "\n        Computes polynomial quotient of ``f`` by ``g``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x**2 + 1, x).quo(Poly(2*x - 4, x))\n        Poly(1/2*x + 1, x, domain='QQ')\n\n        >>> Poly(x**2 - 1, x).quo(Poly(x - 1, x))\n        Poly(x + 1, x, domain='ZZ')\n\n        "
        (dom, per, F, G) = f._unify(g)
        retract = False
        if auto and dom.is_Ring and (not dom.is_Field):
            (F, G) = (F.to_field(), G.to_field())
            retract = True
        if hasattr(f.rep, 'quo'):
            q = F.quo(G)
        else:
            raise OperationNotSupported(f, 'quo')
        if retract:
            try:
                q = q.to_ring()
            except CoercionFailed:
                pass
        return per(q)

    def exquo(f, g, auto=True):
        if False:
            while True:
                i = 10
        "\n        Computes polynomial exact quotient of ``f`` by ``g``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x**2 - 1, x).exquo(Poly(x - 1, x))\n        Poly(x + 1, x, domain='ZZ')\n\n        >>> Poly(x**2 + 1, x).exquo(Poly(2*x - 4, x))\n        Traceback (most recent call last):\n        ...\n        ExactQuotientFailed: 2*x - 4 does not divide x**2 + 1\n\n        "
        (dom, per, F, G) = f._unify(g)
        retract = False
        if auto and dom.is_Ring and (not dom.is_Field):
            (F, G) = (F.to_field(), G.to_field())
            retract = True
        if hasattr(f.rep, 'exquo'):
            try:
                q = F.exquo(G)
            except ExactQuotientFailed as exc:
                raise exc.new(f.as_expr(), g.as_expr())
        else:
            raise OperationNotSupported(f, 'exquo')
        if retract:
            try:
                q = q.to_ring()
            except CoercionFailed:
                pass
        return per(q)

    def _gen_to_level(f, gen):
        if False:
            for i in range(10):
                print('nop')
        'Returns level associated with the given generator. '
        if isinstance(gen, int):
            length = len(f.gens)
            if -length <= gen < length:
                if gen < 0:
                    return length + gen
                else:
                    return gen
            else:
                raise PolynomialError('-%s <= gen < %s expected, got %s' % (length, length, gen))
        else:
            try:
                return f.gens.index(sympify(gen))
            except ValueError:
                raise PolynomialError('a valid generator expected, got %s' % gen)

    def degree(f, gen=0):
        if False:
            while True:
                i = 10
        '\n        Returns degree of ``f`` in ``x_j``.\n\n        The degree of 0 is negative infinity.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x, y\n\n        >>> Poly(x**2 + y*x + 1, x, y).degree()\n        2\n        >>> Poly(x**2 + y*x + y, x, y).degree(y)\n        1\n        >>> Poly(0, x).degree()\n        -oo\n\n        '
        j = f._gen_to_level(gen)
        if hasattr(f.rep, 'degree'):
            d = f.rep.degree(j)
            if d < 0:
                d = S.NegativeInfinity
            return d
        else:
            raise OperationNotSupported(f, 'degree')

    def degree_list(f):
        if False:
            i = 10
            return i + 15
        '\n        Returns a list of degrees of ``f``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x, y\n\n        >>> Poly(x**2 + y*x + 1, x, y).degree_list()\n        (2, 1)\n\n        '
        if hasattr(f.rep, 'degree_list'):
            return f.rep.degree_list()
        else:
            raise OperationNotSupported(f, 'degree_list')

    def total_degree(f):
        if False:
            print('Hello World!')
        '\n        Returns the total degree of ``f``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x, y\n\n        >>> Poly(x**2 + y*x + 1, x, y).total_degree()\n        2\n        >>> Poly(x + y**5, x, y).total_degree()\n        5\n\n        '
        if hasattr(f.rep, 'total_degree'):
            return f.rep.total_degree()
        else:
            raise OperationNotSupported(f, 'total_degree')

    def homogenize(f, s):
        if False:
            while True:
                i = 10
        "\n        Returns the homogeneous polynomial of ``f``.\n\n        A homogeneous polynomial is a polynomial whose all monomials with\n        non-zero coefficients have the same total degree. If you only\n        want to check if a polynomial is homogeneous, then use\n        :func:`Poly.is_homogeneous`. If you want not only to check if a\n        polynomial is homogeneous but also compute its homogeneous order,\n        then use :func:`Poly.homogeneous_order`.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x, y, z\n\n        >>> f = Poly(x**5 + 2*x**2*y**2 + 9*x*y**3)\n        >>> f.homogenize(z)\n        Poly(x**5 + 2*x**2*y**2*z + 9*x*y**3*z, x, y, z, domain='ZZ')\n\n        "
        if not isinstance(s, Symbol):
            raise TypeError('``Symbol`` expected, got %s' % type(s))
        if s in f.gens:
            i = f.gens.index(s)
            gens = f.gens
        else:
            i = len(f.gens)
            gens = f.gens + (s,)
        if hasattr(f.rep, 'homogenize'):
            return f.per(f.rep.homogenize(i), gens=gens)
        raise OperationNotSupported(f, 'homogeneous_order')

    def homogeneous_order(f):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the homogeneous order of ``f``.\n\n        A homogeneous polynomial is a polynomial whose all monomials with\n        non-zero coefficients have the same total degree. This degree is\n        the homogeneous order of ``f``. If you only want to check if a\n        polynomial is homogeneous, then use :func:`Poly.is_homogeneous`.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x, y\n\n        >>> f = Poly(x**5 + 2*x**3*y**2 + 9*x*y**4)\n        >>> f.homogeneous_order()\n        5\n\n        '
        if hasattr(f.rep, 'homogeneous_order'):
            return f.rep.homogeneous_order()
        else:
            raise OperationNotSupported(f, 'homogeneous_order')

    def LC(f, order=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the leading coefficient of ``f``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(4*x**3 + 2*x**2 + 3*x, x).LC()\n        4\n\n        '
        if order is not None:
            return f.coeffs(order)[0]
        if hasattr(f.rep, 'LC'):
            result = f.rep.LC()
        else:
            raise OperationNotSupported(f, 'LC')
        return f.rep.dom.to_sympy(result)

    def TC(f):
        if False:
            return 10
        '\n        Returns the trailing coefficient of ``f``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x**3 + 2*x**2 + 3*x, x).TC()\n        0\n\n        '
        if hasattr(f.rep, 'TC'):
            result = f.rep.TC()
        else:
            raise OperationNotSupported(f, 'TC')
        return f.rep.dom.to_sympy(result)

    def EC(f, order=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the last non-zero coefficient of ``f``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x**3 + 2*x**2 + 3*x, x).EC()\n        3\n\n        '
        if hasattr(f.rep, 'coeffs'):
            return f.coeffs(order)[-1]
        else:
            raise OperationNotSupported(f, 'EC')

    def coeff_monomial(f, monom):
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns the coefficient of ``monom`` in ``f`` if there, else None.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly, exp\n        >>> from sympy.abc import x, y\n\n        >>> p = Poly(24*x*y*exp(8) + 23*x, x, y)\n\n        >>> p.coeff_monomial(x)\n        23\n        >>> p.coeff_monomial(y)\n        0\n        >>> p.coeff_monomial(x*y)\n        24*exp(8)\n\n        Note that ``Expr.coeff()`` behaves differently, collecting terms\n        if possible; the Poly must be converted to an Expr to use that\n        method, however:\n\n        >>> p.as_expr().coeff(x)\n        24*y*exp(8) + 23\n        >>> p.as_expr().coeff(y)\n        24*x*exp(8)\n        >>> p.as_expr().coeff(x*y)\n        24*exp(8)\n\n        See Also\n        ========\n        nth: more efficient query using exponents of the monomial's generators\n\n        "
        return f.nth(*Monomial(monom, f.gens).exponents)

    def nth(f, *N):
        if False:
            while True:
                i = 10
        "\n        Returns the ``n``-th coefficient of ``f`` where ``N`` are the\n        exponents of the generators in the term of interest.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly, sqrt\n        >>> from sympy.abc import x, y\n\n        >>> Poly(x**3 + 2*x**2 + 3*x, x).nth(2)\n        2\n        >>> Poly(x**3 + 2*x*y**2 + y**2, x, y).nth(1, 2)\n        2\n        >>> Poly(4*sqrt(x)*y)\n        Poly(4*y*(sqrt(x)), y, sqrt(x), domain='ZZ')\n        >>> _.nth(1, 1)\n        4\n\n        See Also\n        ========\n        coeff_monomial\n\n        "
        if hasattr(f.rep, 'nth'):
            if len(N) != len(f.gens):
                raise ValueError('exponent of each generator must be specified')
            result = f.rep.nth(*list(map(int, N)))
        else:
            raise OperationNotSupported(f, 'nth')
        return f.rep.dom.to_sympy(result)

    def coeff(f, x, n=1, right=False):
        if False:
            print('Hello World!')
        raise NotImplementedError("Either convert to Expr with `as_expr` method to use Expr's coeff method or else use the `coeff_monomial` method of Polys.")

    def LM(f, order=None):
        if False:
            while True:
                i = 10
        '\n        Returns the leading monomial of ``f``.\n\n        The Leading monomial signifies the monomial having\n        the highest power of the principal generator in the\n        expression f.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x, y\n\n        >>> Poly(4*x**2 + 2*x*y**2 + x*y + 3*y, x, y).LM()\n        x**2*y**0\n\n        '
        return Monomial(f.monoms(order)[0], f.gens)

    def EM(f, order=None):
        if False:
            return 10
        '\n        Returns the last non-zero monomial of ``f``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x, y\n\n        >>> Poly(4*x**2 + 2*x*y**2 + x*y + 3*y, x, y).EM()\n        x**0*y**1\n\n        '
        return Monomial(f.monoms(order)[-1], f.gens)

    def LT(f, order=None):
        if False:
            while True:
                i = 10
        '\n        Returns the leading term of ``f``.\n\n        The Leading term signifies the term having\n        the highest power of the principal generator in the\n        expression f along with its coefficient.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x, y\n\n        >>> Poly(4*x**2 + 2*x*y**2 + x*y + 3*y, x, y).LT()\n        (x**2*y**0, 4)\n\n        '
        (monom, coeff) = f.terms(order)[0]
        return (Monomial(monom, f.gens), coeff)

    def ET(f, order=None):
        if False:
            print('Hello World!')
        '\n        Returns the last non-zero term of ``f``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x, y\n\n        >>> Poly(4*x**2 + 2*x*y**2 + x*y + 3*y, x, y).ET()\n        (x**0*y**1, 3)\n\n        '
        (monom, coeff) = f.terms(order)[-1]
        return (Monomial(monom, f.gens), coeff)

    def max_norm(f):
        if False:
            while True:
                i = 10
        '\n        Returns maximum norm of ``f``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(-x**2 + 2*x - 3, x).max_norm()\n        3\n\n        '
        if hasattr(f.rep, 'max_norm'):
            result = f.rep.max_norm()
        else:
            raise OperationNotSupported(f, 'max_norm')
        return f.rep.dom.to_sympy(result)

    def l1_norm(f):
        if False:
            while True:
                i = 10
        '\n        Returns l1 norm of ``f``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(-x**2 + 2*x - 3, x).l1_norm()\n        6\n\n        '
        if hasattr(f.rep, 'l1_norm'):
            result = f.rep.l1_norm()
        else:
            raise OperationNotSupported(f, 'l1_norm')
        return f.rep.dom.to_sympy(result)

    def clear_denoms(self, convert=False):
        if False:
            for i in range(10):
                print('nop')
        "\n        Clear denominators, but keep the ground domain.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly, S, QQ\n        >>> from sympy.abc import x\n\n        >>> f = Poly(x/2 + S(1)/3, x, domain=QQ)\n\n        >>> f.clear_denoms()\n        (6, Poly(3*x + 2, x, domain='QQ'))\n        >>> f.clear_denoms(convert=True)\n        (6, Poly(3*x + 2, x, domain='ZZ'))\n\n        "
        f = self
        if not f.rep.dom.is_Field:
            return (S.One, f)
        dom = f.get_domain()
        if dom.has_assoc_Ring:
            dom = f.rep.dom.get_ring()
        if hasattr(f.rep, 'clear_denoms'):
            (coeff, result) = f.rep.clear_denoms()
        else:
            raise OperationNotSupported(f, 'clear_denoms')
        (coeff, f) = (dom.to_sympy(coeff), f.per(result))
        if not convert or not dom.has_assoc_Ring:
            return (coeff, f)
        else:
            return (coeff, f.to_ring())

    def rat_clear_denoms(self, g):
        if False:
            print('Hello World!')
        "\n        Clear denominators in a rational function ``f/g``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x, y\n\n        >>> f = Poly(x**2/y + 1, x)\n        >>> g = Poly(x**3 + y, x)\n\n        >>> p, q = f.rat_clear_denoms(g)\n\n        >>> p\n        Poly(x**2 + y, x, domain='ZZ[y]')\n        >>> q\n        Poly(y*x**3 + y**2, x, domain='ZZ[y]')\n\n        "
        f = self
        (dom, per, f, g) = f._unify(g)
        f = per(f)
        g = per(g)
        if not (dom.is_Field and dom.has_assoc_Ring):
            return (f, g)
        (a, f) = f.clear_denoms(convert=True)
        (b, g) = g.clear_denoms(convert=True)
        f = f.mul_ground(b)
        g = g.mul_ground(a)
        return (f, g)

    def integrate(self, *specs, **args):
        if False:
            return 10
        "\n        Computes indefinite integral of ``f``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x, y\n\n        >>> Poly(x**2 + 2*x + 1, x).integrate()\n        Poly(1/3*x**3 + x**2 + x, x, domain='QQ')\n\n        >>> Poly(x*y**2 + x, x, y).integrate((0, 1), (1, 0))\n        Poly(1/2*x**2*y**2 + 1/2*x**2, x, y, domain='QQ')\n\n        "
        f = self
        if args.get('auto', True) and f.rep.dom.is_Ring:
            f = f.to_field()
        if hasattr(f.rep, 'integrate'):
            if not specs:
                return f.per(f.rep.integrate(m=1))
            rep = f.rep
            for spec in specs:
                if isinstance(spec, tuple):
                    (gen, m) = spec
                else:
                    (gen, m) = (spec, 1)
                rep = rep.integrate(int(m), f._gen_to_level(gen))
            return f.per(rep)
        else:
            raise OperationNotSupported(f, 'integrate')

    def diff(f, *specs, **kwargs):
        if False:
            while True:
                i = 10
        "\n        Computes partial derivative of ``f``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x, y\n\n        >>> Poly(x**2 + 2*x + 1, x).diff()\n        Poly(2*x + 2, x, domain='ZZ')\n\n        >>> Poly(x*y**2 + x, x, y).diff((0, 0), (1, 1))\n        Poly(2*x*y, x, y, domain='ZZ')\n\n        "
        if not kwargs.get('evaluate', True):
            return Derivative(f, *specs, **kwargs)
        if hasattr(f.rep, 'diff'):
            if not specs:
                return f.per(f.rep.diff(m=1))
            rep = f.rep
            for spec in specs:
                if isinstance(spec, tuple):
                    (gen, m) = spec
                else:
                    (gen, m) = (spec, 1)
                rep = rep.diff(int(m), f._gen_to_level(gen))
            return f.per(rep)
        else:
            raise OperationNotSupported(f, 'diff')
    _eval_derivative = diff

    def eval(self, x, a=None, auto=True):
        if False:
            return 10
        "\n        Evaluate ``f`` at ``a`` in the given variable.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x, y, z\n\n        >>> Poly(x**2 + 2*x + 3, x).eval(2)\n        11\n\n        >>> Poly(2*x*y + 3*x + y + 2, x, y).eval(x, 2)\n        Poly(5*y + 8, y, domain='ZZ')\n\n        >>> f = Poly(2*x*y + 3*x + y + 2*z, x, y, z)\n\n        >>> f.eval({x: 2})\n        Poly(5*y + 2*z + 6, y, z, domain='ZZ')\n        >>> f.eval({x: 2, y: 5})\n        Poly(2*z + 31, z, domain='ZZ')\n        >>> f.eval({x: 2, y: 5, z: 7})\n        45\n\n        >>> f.eval((2, 5))\n        Poly(2*z + 31, z, domain='ZZ')\n        >>> f(2, 5)\n        Poly(2*z + 31, z, domain='ZZ')\n\n        "
        f = self
        if a is None:
            if isinstance(x, dict):
                mapping = x
                for (gen, value) in mapping.items():
                    f = f.eval(gen, value)
                return f
            elif isinstance(x, (tuple, list)):
                values = x
                if len(values) > len(f.gens):
                    raise ValueError('too many values provided')
                for (gen, value) in zip(f.gens, values):
                    f = f.eval(gen, value)
                return f
            else:
                (j, a) = (0, x)
        else:
            j = f._gen_to_level(x)
        if not hasattr(f.rep, 'eval'):
            raise OperationNotSupported(f, 'eval')
        try:
            result = f.rep.eval(a, j)
        except CoercionFailed:
            if not auto:
                raise DomainError('Cannot evaluate at %s in %s' % (a, f.rep.dom))
            else:
                (a_domain, [a]) = construct_domain([a])
                new_domain = f.get_domain().unify_with_symbols(a_domain, f.gens)
                f = f.set_domain(new_domain)
                a = new_domain.convert(a, a_domain)
                result = f.rep.eval(a, j)
        return f.per(result, remove=j)

    def __call__(f, *values):
        if False:
            while True:
                i = 10
        "\n        Evaluate ``f`` at the give values.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x, y, z\n\n        >>> f = Poly(2*x*y + 3*x + y + 2*z, x, y, z)\n\n        >>> f(2)\n        Poly(5*y + 2*z + 6, y, z, domain='ZZ')\n        >>> f(2, 5)\n        Poly(2*z + 31, z, domain='ZZ')\n        >>> f(2, 5, 7)\n        45\n\n        "
        return f.eval(values)

    def half_gcdex(f, g, auto=True):
        if False:
            print('Hello World!')
        "\n        Half extended Euclidean algorithm of ``f`` and ``g``.\n\n        Returns ``(s, h)`` such that ``h = gcd(f, g)`` and ``s*f = h (mod g)``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> f = x**4 - 2*x**3 - 6*x**2 + 12*x + 15\n        >>> g = x**3 + x**2 - 4*x - 4\n\n        >>> Poly(f).half_gcdex(Poly(g))\n        (Poly(-1/5*x + 3/5, x, domain='QQ'), Poly(x + 1, x, domain='QQ'))\n\n        "
        (dom, per, F, G) = f._unify(g)
        if auto and dom.is_Ring:
            (F, G) = (F.to_field(), G.to_field())
        if hasattr(f.rep, 'half_gcdex'):
            (s, h) = F.half_gcdex(G)
        else:
            raise OperationNotSupported(f, 'half_gcdex')
        return (per(s), per(h))

    def gcdex(f, g, auto=True):
        if False:
            while True:
                i = 10
        "\n        Extended Euclidean algorithm of ``f`` and ``g``.\n\n        Returns ``(s, t, h)`` such that ``h = gcd(f, g)`` and ``s*f + t*g = h``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> f = x**4 - 2*x**3 - 6*x**2 + 12*x + 15\n        >>> g = x**3 + x**2 - 4*x - 4\n\n        >>> Poly(f).gcdex(Poly(g))\n        (Poly(-1/5*x + 3/5, x, domain='QQ'),\n         Poly(1/5*x**2 - 6/5*x + 2, x, domain='QQ'),\n         Poly(x + 1, x, domain='QQ'))\n\n        "
        (dom, per, F, G) = f._unify(g)
        if auto and dom.is_Ring:
            (F, G) = (F.to_field(), G.to_field())
        if hasattr(f.rep, 'gcdex'):
            (s, t, h) = F.gcdex(G)
        else:
            raise OperationNotSupported(f, 'gcdex')
        return (per(s), per(t), per(h))

    def invert(f, g, auto=True):
        if False:
            print('Hello World!')
        "\n        Invert ``f`` modulo ``g`` when possible.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x**2 - 1, x).invert(Poly(2*x - 1, x))\n        Poly(-4/3, x, domain='QQ')\n\n        >>> Poly(x**2 - 1, x).invert(Poly(x - 1, x))\n        Traceback (most recent call last):\n        ...\n        NotInvertible: zero divisor\n\n        "
        (dom, per, F, G) = f._unify(g)
        if auto and dom.is_Ring:
            (F, G) = (F.to_field(), G.to_field())
        if hasattr(f.rep, 'invert'):
            result = F.invert(G)
        else:
            raise OperationNotSupported(f, 'invert')
        return per(result)

    def revert(f, n):
        if False:
            print('Hello World!')
        "\n        Compute ``f**(-1)`` mod ``x**n``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(1, x).revert(2)\n        Poly(1, x, domain='ZZ')\n\n        >>> Poly(1 + x, x).revert(1)\n        Poly(1, x, domain='ZZ')\n\n        >>> Poly(x**2 - 2, x).revert(2)\n        Traceback (most recent call last):\n        ...\n        NotReversible: only units are reversible in a ring\n\n        >>> Poly(1/x, x).revert(1)\n        Traceback (most recent call last):\n        ...\n        PolynomialError: 1/x contains an element of the generators set\n\n        "
        if hasattr(f.rep, 'revert'):
            result = f.rep.revert(int(n))
        else:
            raise OperationNotSupported(f, 'revert')
        return f.per(result)

    def subresultants(f, g):
        if False:
            while True:
                i = 10
        "\n        Computes the subresultant PRS of ``f`` and ``g``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x**2 + 1, x).subresultants(Poly(x**2 - 1, x))\n        [Poly(x**2 + 1, x, domain='ZZ'),\n         Poly(x**2 - 1, x, domain='ZZ'),\n         Poly(-2, x, domain='ZZ')]\n\n        "
        (_, per, F, G) = f._unify(g)
        if hasattr(f.rep, 'subresultants'):
            result = F.subresultants(G)
        else:
            raise OperationNotSupported(f, 'subresultants')
        return list(map(per, result))

    def resultant(f, g, includePRS=False):
        if False:
            print('Hello World!')
        "\n        Computes the resultant of ``f`` and ``g`` via PRS.\n\n        If includePRS=True, it includes the subresultant PRS in the result.\n        Because the PRS is used to calculate the resultant, this is more\n        efficient than calling :func:`subresultants` separately.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> f = Poly(x**2 + 1, x)\n\n        >>> f.resultant(Poly(x**2 - 1, x))\n        4\n        >>> f.resultant(Poly(x**2 - 1, x), includePRS=True)\n        (4, [Poly(x**2 + 1, x, domain='ZZ'), Poly(x**2 - 1, x, domain='ZZ'),\n             Poly(-2, x, domain='ZZ')])\n\n        "
        (_, per, F, G) = f._unify(g)
        if hasattr(f.rep, 'resultant'):
            if includePRS:
                (result, R) = F.resultant(G, includePRS=includePRS)
            else:
                result = F.resultant(G)
        else:
            raise OperationNotSupported(f, 'resultant')
        if includePRS:
            return (per(result, remove=0), list(map(per, R)))
        return per(result, remove=0)

    def discriminant(f):
        if False:
            i = 10
            return i + 15
        '\n        Computes the discriminant of ``f``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x**2 + 2*x + 3, x).discriminant()\n        -8\n\n        '
        if hasattr(f.rep, 'discriminant'):
            result = f.rep.discriminant()
        else:
            raise OperationNotSupported(f, 'discriminant')
        return f.per(result, remove=0)

    def dispersionset(f, g=None):
        if False:
            print('Hello World!')
        "Compute the *dispersion set* of two polynomials.\n\n        For two polynomials `f(x)` and `g(x)` with `\\deg f > 0`\n        and `\\deg g > 0` the dispersion set `\\operatorname{J}(f, g)` is defined as:\n\n        .. math::\n            \\operatorname{J}(f, g)\n            & := \\{a \\in \\mathbb{N}_0 | \\gcd(f(x), g(x+a)) \\neq 1\\} \\\\\n            &  = \\{a \\in \\mathbb{N}_0 | \\deg \\gcd(f(x), g(x+a)) \\geq 1\\}\n\n        For a single polynomial one defines `\\operatorname{J}(f) := \\operatorname{J}(f, f)`.\n\n        Examples\n        ========\n\n        >>> from sympy import poly\n        >>> from sympy.polys.dispersion import dispersion, dispersionset\n        >>> from sympy.abc import x\n\n        Dispersion set and dispersion of a simple polynomial:\n\n        >>> fp = poly((x - 3)*(x + 3), x)\n        >>> sorted(dispersionset(fp))\n        [0, 6]\n        >>> dispersion(fp)\n        6\n\n        Note that the definition of the dispersion is not symmetric:\n\n        >>> fp = poly(x**4 - 3*x**2 + 1, x)\n        >>> gp = fp.shift(-3)\n        >>> sorted(dispersionset(fp, gp))\n        [2, 3, 4]\n        >>> dispersion(fp, gp)\n        4\n        >>> sorted(dispersionset(gp, fp))\n        []\n        >>> dispersion(gp, fp)\n        -oo\n\n        Computing the dispersion also works over field extensions:\n\n        >>> from sympy import sqrt\n        >>> fp = poly(x**2 + sqrt(5)*x - 1, x, domain='QQ<sqrt(5)>')\n        >>> gp = poly(x**2 + (2 + sqrt(5))*x + sqrt(5), x, domain='QQ<sqrt(5)>')\n        >>> sorted(dispersionset(fp, gp))\n        [2]\n        >>> sorted(dispersionset(gp, fp))\n        [1, 4]\n\n        We can even perform the computations for polynomials\n        having symbolic coefficients:\n\n        >>> from sympy.abc import a\n        >>> fp = poly(4*x**4 + (4*a + 8)*x**3 + (a**2 + 6*a + 4)*x**2 + (a**2 + 2*a)*x, x)\n        >>> sorted(dispersionset(fp))\n        [0, 1]\n\n        See Also\n        ========\n\n        dispersion\n\n        References\n        ==========\n\n        1. [ManWright94]_\n        2. [Koepf98]_\n        3. [Abramov71]_\n        4. [Man93]_\n        "
        from sympy.polys.dispersion import dispersionset
        return dispersionset(f, g)

    def dispersion(f, g=None):
        if False:
            i = 10
            return i + 15
        "Compute the *dispersion* of polynomials.\n\n        For two polynomials `f(x)` and `g(x)` with `\\deg f > 0`\n        and `\\deg g > 0` the dispersion `\\operatorname{dis}(f, g)` is defined as:\n\n        .. math::\n            \\operatorname{dis}(f, g)\n            & := \\max\\{ J(f,g) \\cup \\{0\\} \\} \\\\\n            &  = \\max\\{ \\{a \\in \\mathbb{N} | \\gcd(f(x), g(x+a)) \\neq 1\\} \\cup \\{0\\} \\}\n\n        and for a single polynomial `\\operatorname{dis}(f) := \\operatorname{dis}(f, f)`.\n\n        Examples\n        ========\n\n        >>> from sympy import poly\n        >>> from sympy.polys.dispersion import dispersion, dispersionset\n        >>> from sympy.abc import x\n\n        Dispersion set and dispersion of a simple polynomial:\n\n        >>> fp = poly((x - 3)*(x + 3), x)\n        >>> sorted(dispersionset(fp))\n        [0, 6]\n        >>> dispersion(fp)\n        6\n\n        Note that the definition of the dispersion is not symmetric:\n\n        >>> fp = poly(x**4 - 3*x**2 + 1, x)\n        >>> gp = fp.shift(-3)\n        >>> sorted(dispersionset(fp, gp))\n        [2, 3, 4]\n        >>> dispersion(fp, gp)\n        4\n        >>> sorted(dispersionset(gp, fp))\n        []\n        >>> dispersion(gp, fp)\n        -oo\n\n        Computing the dispersion also works over field extensions:\n\n        >>> from sympy import sqrt\n        >>> fp = poly(x**2 + sqrt(5)*x - 1, x, domain='QQ<sqrt(5)>')\n        >>> gp = poly(x**2 + (2 + sqrt(5))*x + sqrt(5), x, domain='QQ<sqrt(5)>')\n        >>> sorted(dispersionset(fp, gp))\n        [2]\n        >>> sorted(dispersionset(gp, fp))\n        [1, 4]\n\n        We can even perform the computations for polynomials\n        having symbolic coefficients:\n\n        >>> from sympy.abc import a\n        >>> fp = poly(4*x**4 + (4*a + 8)*x**3 + (a**2 + 6*a + 4)*x**2 + (a**2 + 2*a)*x, x)\n        >>> sorted(dispersionset(fp))\n        [0, 1]\n\n        See Also\n        ========\n\n        dispersionset\n\n        References\n        ==========\n\n        1. [ManWright94]_\n        2. [Koepf98]_\n        3. [Abramov71]_\n        4. [Man93]_\n        "
        from sympy.polys.dispersion import dispersion
        return dispersion(f, g)

    def cofactors(f, g):
        if False:
            i = 10
            return i + 15
        "\n        Returns the GCD of ``f`` and ``g`` and their cofactors.\n\n        Returns polynomials ``(h, cff, cfg)`` such that ``h = gcd(f, g)``, and\n        ``cff = quo(f, h)`` and ``cfg = quo(g, h)`` are, so called, cofactors\n        of ``f`` and ``g``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x**2 - 1, x).cofactors(Poly(x**2 - 3*x + 2, x))\n        (Poly(x - 1, x, domain='ZZ'),\n         Poly(x + 1, x, domain='ZZ'),\n         Poly(x - 2, x, domain='ZZ'))\n\n        "
        (_, per, F, G) = f._unify(g)
        if hasattr(f.rep, 'cofactors'):
            (h, cff, cfg) = F.cofactors(G)
        else:
            raise OperationNotSupported(f, 'cofactors')
        return (per(h), per(cff), per(cfg))

    def gcd(f, g):
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns the polynomial GCD of ``f`` and ``g``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x**2 - 1, x).gcd(Poly(x**2 - 3*x + 2, x))\n        Poly(x - 1, x, domain='ZZ')\n\n        "
        (_, per, F, G) = f._unify(g)
        if hasattr(f.rep, 'gcd'):
            result = F.gcd(G)
        else:
            raise OperationNotSupported(f, 'gcd')
        return per(result)

    def lcm(f, g):
        if False:
            while True:
                i = 10
        "\n        Returns polynomial LCM of ``f`` and ``g``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x**2 - 1, x).lcm(Poly(x**2 - 3*x + 2, x))\n        Poly(x**3 - 2*x**2 - x + 2, x, domain='ZZ')\n\n        "
        (_, per, F, G) = f._unify(g)
        if hasattr(f.rep, 'lcm'):
            result = F.lcm(G)
        else:
            raise OperationNotSupported(f, 'lcm')
        return per(result)

    def trunc(f, p):
        if False:
            while True:
                i = 10
        "\n        Reduce ``f`` modulo a constant ``p``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(2*x**3 + 3*x**2 + 5*x + 7, x).trunc(3)\n        Poly(-x**3 - x + 1, x, domain='ZZ')\n\n        "
        p = f.rep.dom.convert(p)
        if hasattr(f.rep, 'trunc'):
            result = f.rep.trunc(p)
        else:
            raise OperationNotSupported(f, 'trunc')
        return f.per(result)

    def monic(self, auto=True):
        if False:
            while True:
                i = 10
        "\n        Divides all coefficients by ``LC(f)``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly, ZZ\n        >>> from sympy.abc import x\n\n        >>> Poly(3*x**2 + 6*x + 9, x, domain=ZZ).monic()\n        Poly(x**2 + 2*x + 3, x, domain='QQ')\n\n        >>> Poly(3*x**2 + 4*x + 2, x, domain=ZZ).monic()\n        Poly(x**2 + 4/3*x + 2/3, x, domain='QQ')\n\n        "
        f = self
        if auto and f.rep.dom.is_Ring:
            f = f.to_field()
        if hasattr(f.rep, 'monic'):
            result = f.rep.monic()
        else:
            raise OperationNotSupported(f, 'monic')
        return f.per(result)

    def content(f):
        if False:
            return 10
        '\n        Returns the GCD of polynomial coefficients.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(6*x**2 + 8*x + 12, x).content()\n        2\n\n        '
        if hasattr(f.rep, 'content'):
            result = f.rep.content()
        else:
            raise OperationNotSupported(f, 'content')
        return f.rep.dom.to_sympy(result)

    def primitive(f):
        if False:
            while True:
                i = 10
        "\n        Returns the content and a primitive form of ``f``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(2*x**2 + 8*x + 12, x).primitive()\n        (2, Poly(x**2 + 4*x + 6, x, domain='ZZ'))\n\n        "
        if hasattr(f.rep, 'primitive'):
            (cont, result) = f.rep.primitive()
        else:
            raise OperationNotSupported(f, 'primitive')
        return (f.rep.dom.to_sympy(cont), f.per(result))

    def compose(f, g):
        if False:
            print('Hello World!')
        "\n        Computes the functional composition of ``f`` and ``g``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x**2 + x, x).compose(Poly(x - 1, x))\n        Poly(x**2 - x, x, domain='ZZ')\n\n        "
        (_, per, F, G) = f._unify(g)
        if hasattr(f.rep, 'compose'):
            result = F.compose(G)
        else:
            raise OperationNotSupported(f, 'compose')
        return per(result)

    def decompose(f):
        if False:
            while True:
                i = 10
        "\n        Computes a functional decomposition of ``f``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x**4 + 2*x**3 - x - 1, x, domain='ZZ').decompose()\n        [Poly(x**2 - x - 1, x, domain='ZZ'), Poly(x**2 + x, x, domain='ZZ')]\n\n        "
        if hasattr(f.rep, 'decompose'):
            result = f.rep.decompose()
        else:
            raise OperationNotSupported(f, 'decompose')
        return list(map(f.per, result))

    def shift(f, a):
        if False:
            return 10
        "\n        Efficiently compute Taylor shift ``f(x + a)``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x**2 - 2*x + 1, x).shift(2)\n        Poly(x**2 + 2*x + 1, x, domain='ZZ')\n\n        "
        if hasattr(f.rep, 'shift'):
            result = f.rep.shift(a)
        else:
            raise OperationNotSupported(f, 'shift')
        return f.per(result)

    def transform(f, p, q):
        if False:
            while True:
                i = 10
        "\n        Efficiently evaluate the functional transformation ``q**n * f(p/q)``.\n\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x**2 - 2*x + 1, x).transform(Poly(x + 1, x), Poly(x - 1, x))\n        Poly(4, x, domain='ZZ')\n\n        "
        (P, Q) = p.unify(q)
        (F, P) = f.unify(P)
        (F, Q) = F.unify(Q)
        if hasattr(F.rep, 'transform'):
            result = F.rep.transform(P.rep, Q.rep)
        else:
            raise OperationNotSupported(F, 'transform')
        return F.per(result)

    def sturm(self, auto=True):
        if False:
            print('Hello World!')
        "\n        Computes the Sturm sequence of ``f``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x**3 - 2*x**2 + x - 3, x).sturm()\n        [Poly(x**3 - 2*x**2 + x - 3, x, domain='QQ'),\n         Poly(3*x**2 - 4*x + 1, x, domain='QQ'),\n         Poly(2/9*x + 25/9, x, domain='QQ'),\n         Poly(-2079/4, x, domain='QQ')]\n\n        "
        f = self
        if auto and f.rep.dom.is_Ring:
            f = f.to_field()
        if hasattr(f.rep, 'sturm'):
            result = f.rep.sturm()
        else:
            raise OperationNotSupported(f, 'sturm')
        return list(map(f.per, result))

    def gff_list(f):
        if False:
            print('Hello World!')
        "\n        Computes greatest factorial factorization of ``f``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> f = x**5 + 2*x**4 - x**3 - 2*x**2\n\n        >>> Poly(f).gff_list()\n        [(Poly(x, x, domain='ZZ'), 1), (Poly(x + 2, x, domain='ZZ'), 4)]\n\n        "
        if hasattr(f.rep, 'gff_list'):
            result = f.rep.gff_list()
        else:
            raise OperationNotSupported(f, 'gff_list')
        return [(f.per(g), k) for (g, k) in result]

    def norm(f):
        if False:
            return 10
        "\n        Computes the product, ``Norm(f)``, of the conjugates of\n        a polynomial ``f`` defined over a number field ``K``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly, sqrt\n        >>> from sympy.abc import x\n\n        >>> a, b = sqrt(2), sqrt(3)\n\n        A polynomial over a quadratic extension.\n        Two conjugates x - a and x + a.\n\n        >>> f = Poly(x - a, x, extension=a)\n        >>> f.norm()\n        Poly(x**2 - 2, x, domain='QQ')\n\n        A polynomial over a quartic extension.\n        Four conjugates x - a, x - a, x + a and x + a.\n\n        >>> f = Poly(x - a, x, extension=(a, b))\n        >>> f.norm()\n        Poly(x**4 - 4*x**2 + 4, x, domain='QQ')\n\n        "
        if hasattr(f.rep, 'norm'):
            r = f.rep.norm()
        else:
            raise OperationNotSupported(f, 'norm')
        return f.per(r)

    def sqf_norm(f):
        if False:
            return 10
        "\n        Computes square-free norm of ``f``.\n\n        Returns ``s``, ``f``, ``r``, such that ``g(x) = f(x-sa)`` and\n        ``r(x) = Norm(g(x))`` is a square-free polynomial over ``K``,\n        where ``a`` is the algebraic extension of the ground domain.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly, sqrt\n        >>> from sympy.abc import x\n\n        >>> s, f, r = Poly(x**2 + 1, x, extension=[sqrt(3)]).sqf_norm()\n\n        >>> s\n        1\n        >>> f\n        Poly(x**2 - 2*sqrt(3)*x + 4, x, domain='QQ<sqrt(3)>')\n        >>> r\n        Poly(x**4 - 4*x**2 + 16, x, domain='QQ')\n\n        "
        if hasattr(f.rep, 'sqf_norm'):
            (s, g, r) = f.rep.sqf_norm()
        else:
            raise OperationNotSupported(f, 'sqf_norm')
        return (s, f.per(g), f.per(r))

    def sqf_part(f):
        if False:
            print('Hello World!')
        "\n        Computes square-free part of ``f``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x**3 - 3*x - 2, x).sqf_part()\n        Poly(x**2 - x - 2, x, domain='ZZ')\n\n        "
        if hasattr(f.rep, 'sqf_part'):
            result = f.rep.sqf_part()
        else:
            raise OperationNotSupported(f, 'sqf_part')
        return f.per(result)

    def sqf_list(f, all=False):
        if False:
            while True:
                i = 10
        "\n        Returns a list of square-free factors of ``f``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> f = 2*x**5 + 16*x**4 + 50*x**3 + 76*x**2 + 56*x + 16\n\n        >>> Poly(f).sqf_list()\n        (2, [(Poly(x + 1, x, domain='ZZ'), 2),\n             (Poly(x + 2, x, domain='ZZ'), 3)])\n\n        >>> Poly(f).sqf_list(all=True)\n        (2, [(Poly(1, x, domain='ZZ'), 1),\n             (Poly(x + 1, x, domain='ZZ'), 2),\n             (Poly(x + 2, x, domain='ZZ'), 3)])\n\n        "
        if hasattr(f.rep, 'sqf_list'):
            (coeff, factors) = f.rep.sqf_list(all)
        else:
            raise OperationNotSupported(f, 'sqf_list')
        return (f.rep.dom.to_sympy(coeff), [(f.per(g), k) for (g, k) in factors])

    def sqf_list_include(f, all=False):
        if False:
            return 10
        "\n        Returns a list of square-free factors of ``f``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly, expand\n        >>> from sympy.abc import x\n\n        >>> f = expand(2*(x + 1)**3*x**4)\n        >>> f\n        2*x**7 + 6*x**6 + 6*x**5 + 2*x**4\n\n        >>> Poly(f).sqf_list_include()\n        [(Poly(2, x, domain='ZZ'), 1),\n         (Poly(x + 1, x, domain='ZZ'), 3),\n         (Poly(x, x, domain='ZZ'), 4)]\n\n        >>> Poly(f).sqf_list_include(all=True)\n        [(Poly(2, x, domain='ZZ'), 1),\n         (Poly(1, x, domain='ZZ'), 2),\n         (Poly(x + 1, x, domain='ZZ'), 3),\n         (Poly(x, x, domain='ZZ'), 4)]\n\n        "
        if hasattr(f.rep, 'sqf_list_include'):
            factors = f.rep.sqf_list_include(all)
        else:
            raise OperationNotSupported(f, 'sqf_list_include')
        return [(f.per(g), k) for (g, k) in factors]

    def factor_list(f):
        if False:
            while True:
                i = 10
        "\n        Returns a list of irreducible factors of ``f``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x, y\n\n        >>> f = 2*x**5 + 2*x**4*y + 4*x**3 + 4*x**2*y + 2*x + 2*y\n\n        >>> Poly(f).factor_list()\n        (2, [(Poly(x + y, x, y, domain='ZZ'), 1),\n             (Poly(x**2 + 1, x, y, domain='ZZ'), 2)])\n\n        "
        if hasattr(f.rep, 'factor_list'):
            try:
                (coeff, factors) = f.rep.factor_list()
            except DomainError:
                if f.degree() == 0:
                    return (f.as_expr(), [])
                else:
                    return (S.One, [(f, 1)])
        else:
            raise OperationNotSupported(f, 'factor_list')
        return (f.rep.dom.to_sympy(coeff), [(f.per(g), k) for (g, k) in factors])

    def factor_list_include(f):
        if False:
            print('Hello World!')
        "\n        Returns a list of irreducible factors of ``f``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x, y\n\n        >>> f = 2*x**5 + 2*x**4*y + 4*x**3 + 4*x**2*y + 2*x + 2*y\n\n        >>> Poly(f).factor_list_include()\n        [(Poly(2*x + 2*y, x, y, domain='ZZ'), 1),\n         (Poly(x**2 + 1, x, y, domain='ZZ'), 2)]\n\n        "
        if hasattr(f.rep, 'factor_list_include'):
            try:
                factors = f.rep.factor_list_include()
            except DomainError:
                return [(f, 1)]
        else:
            raise OperationNotSupported(f, 'factor_list_include')
        return [(f.per(g), k) for (g, k) in factors]

    def intervals(f, all=False, eps=None, inf=None, sup=None, fast=False, sqf=False):
        if False:
            while True:
                i = 10
        '\n        Compute isolating intervals for roots of ``f``.\n\n        For real roots the Vincent-Akritas-Strzebonski (VAS) continued fractions method is used.\n\n        References\n        ==========\n        .. [#] Alkiviadis G. Akritas and Adam W. Strzebonski: A Comparative Study of Two Real Root\n            Isolation Methods . Nonlinear Analysis: Modelling and Control, Vol. 10, No. 4, 297-304, 2005.\n        .. [#] Alkiviadis G. Akritas, Adam W. Strzebonski and Panagiotis S. Vigklas: Improving the\n            Performance of the Continued Fractions Method Using new Bounds of Positive Roots. Nonlinear\n            Analysis: Modelling and Control, Vol. 13, No. 3, 265-279, 2008.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x**2 - 3, x).intervals()\n        [((-2, -1), 1), ((1, 2), 1)]\n        >>> Poly(x**2 - 3, x).intervals(eps=1e-2)\n        [((-26/15, -19/11), 1), ((19/11, 26/15), 1)]\n\n        '
        if eps is not None:
            eps = QQ.convert(eps)
            if eps <= 0:
                raise ValueError("'eps' must be a positive rational")
        if inf is not None:
            inf = QQ.convert(inf)
        if sup is not None:
            sup = QQ.convert(sup)
        if hasattr(f.rep, 'intervals'):
            result = f.rep.intervals(all=all, eps=eps, inf=inf, sup=sup, fast=fast, sqf=sqf)
        else:
            raise OperationNotSupported(f, 'intervals')
        if sqf:

            def _real(interval):
                if False:
                    i = 10
                    return i + 15
                (s, t) = interval
                return (QQ.to_sympy(s), QQ.to_sympy(t))
            if not all:
                return list(map(_real, result))

            def _complex(rectangle):
                if False:
                    while True:
                        i = 10
                ((u, v), (s, t)) = rectangle
                return (QQ.to_sympy(u) + I * QQ.to_sympy(v), QQ.to_sympy(s) + I * QQ.to_sympy(t))
            (real_part, complex_part) = result
            return (list(map(_real, real_part)), list(map(_complex, complex_part)))
        else:

            def _real(interval):
                if False:
                    i = 10
                    return i + 15
                ((s, t), k) = interval
                return ((QQ.to_sympy(s), QQ.to_sympy(t)), k)
            if not all:
                return list(map(_real, result))

            def _complex(rectangle):
                if False:
                    print('Hello World!')
                (((u, v), (s, t)), k) = rectangle
                return ((QQ.to_sympy(u) + I * QQ.to_sympy(v), QQ.to_sympy(s) + I * QQ.to_sympy(t)), k)
            (real_part, complex_part) = result
            return (list(map(_real, real_part)), list(map(_complex, complex_part)))

    def refine_root(f, s, t, eps=None, steps=None, fast=False, check_sqf=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Refine an isolating interval of a root to the given precision.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x**2 - 3, x).refine_root(1, 2, eps=1e-2)\n        (19/11, 26/15)\n\n        '
        if check_sqf and (not f.is_sqf):
            raise PolynomialError('only square-free polynomials supported')
        (s, t) = (QQ.convert(s), QQ.convert(t))
        if eps is not None:
            eps = QQ.convert(eps)
            if eps <= 0:
                raise ValueError("'eps' must be a positive rational")
        if steps is not None:
            steps = int(steps)
        elif eps is None:
            steps = 1
        if hasattr(f.rep, 'refine_root'):
            (S, T) = f.rep.refine_root(s, t, eps=eps, steps=steps, fast=fast)
        else:
            raise OperationNotSupported(f, 'refine_root')
        return (QQ.to_sympy(S), QQ.to_sympy(T))

    def count_roots(f, inf=None, sup=None):
        if False:
            while True:
                i = 10
        '\n        Return the number of roots of ``f`` in ``[inf, sup]`` interval.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly, I\n        >>> from sympy.abc import x\n\n        >>> Poly(x**4 - 4, x).count_roots(-3, 3)\n        2\n        >>> Poly(x**4 - 4, x).count_roots(0, 1 + 3*I)\n        1\n\n        '
        (inf_real, sup_real) = (True, True)
        if inf is not None:
            inf = sympify(inf)
            if inf is S.NegativeInfinity:
                inf = None
            else:
                (re, im) = inf.as_real_imag()
                if not im:
                    inf = QQ.convert(inf)
                else:
                    (inf, inf_real) = (list(map(QQ.convert, (re, im))), False)
        if sup is not None:
            sup = sympify(sup)
            if sup is S.Infinity:
                sup = None
            else:
                (re, im) = sup.as_real_imag()
                if not im:
                    sup = QQ.convert(sup)
                else:
                    (sup, sup_real) = (list(map(QQ.convert, (re, im))), False)
        if inf_real and sup_real:
            if hasattr(f.rep, 'count_real_roots'):
                count = f.rep.count_real_roots(inf=inf, sup=sup)
            else:
                raise OperationNotSupported(f, 'count_real_roots')
        else:
            if inf_real and inf is not None:
                inf = (inf, QQ.zero)
            if sup_real and sup is not None:
                sup = (sup, QQ.zero)
            if hasattr(f.rep, 'count_complex_roots'):
                count = f.rep.count_complex_roots(inf=inf, sup=sup)
            else:
                raise OperationNotSupported(f, 'count_complex_roots')
        return Integer(count)

    def root(f, index, radicals=True):
        if False:
            print('Hello World!')
        '\n        Get an indexed root of a polynomial.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> f = Poly(2*x**3 - 7*x**2 + 4*x + 4)\n\n        >>> f.root(0)\n        -1/2\n        >>> f.root(1)\n        2\n        >>> f.root(2)\n        2\n        >>> f.root(3)\n        Traceback (most recent call last):\n        ...\n        IndexError: root index out of [-3, 2] range, got 3\n\n        >>> Poly(x**5 + x + 1).root(0)\n        CRootOf(x**3 - x**2 + 1, 0)\n\n        '
        return sympy.polys.rootoftools.rootof(f, index, radicals=radicals)

    def real_roots(f, multiple=True, radicals=True):
        if False:
            while True:
                i = 10
        '\n        Return a list of real roots with multiplicities.\n\n        See :func:`real_roots` for more explanation.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(2*x**3 - 7*x**2 + 4*x + 4).real_roots()\n        [-1/2, 2, 2]\n        >>> Poly(x**3 + x + 1).real_roots()\n        [CRootOf(x**3 + x + 1, 0)]\n        '
        reals = sympy.polys.rootoftools.CRootOf.real_roots(f, radicals=radicals)
        if multiple:
            return reals
        else:
            return group(reals, multiple=False)

    def all_roots(f, multiple=True, radicals=True):
        if False:
            while True:
                i = 10
        '\n        Return a list of real and complex roots with multiplicities.\n\n        See :func:`all_roots` for more explanation.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(2*x**3 - 7*x**2 + 4*x + 4).all_roots()\n        [-1/2, 2, 2]\n        >>> Poly(x**3 + x + 1).all_roots()\n        [CRootOf(x**3 + x + 1, 0),\n         CRootOf(x**3 + x + 1, 1),\n         CRootOf(x**3 + x + 1, 2)]\n\n        '
        roots = sympy.polys.rootoftools.CRootOf.all_roots(f, radicals=radicals)
        if multiple:
            return roots
        else:
            return group(roots, multiple=False)

    def nroots(f, n=15, maxsteps=50, cleanup=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Compute numerical approximations of roots of ``f``.\n\n        Parameters\n        ==========\n\n        n ... the number of digits to calculate\n        maxsteps ... the maximum number of iterations to do\n\n        If the accuracy `n` cannot be reached in `maxsteps`, it will raise an\n        exception. You need to rerun with higher maxsteps.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x**2 - 3).nroots(n=15)\n        [-1.73205080756888, 1.73205080756888]\n        >>> Poly(x**2 - 3).nroots(n=30)\n        [-1.73205080756887729352744634151, 1.73205080756887729352744634151]\n\n        '
        if f.is_multivariate:
            raise MultivariatePolynomialError('Cannot compute numerical roots of %s' % f)
        if f.degree() <= 0:
            return []
        if f.rep.dom is ZZ:
            coeffs = [int(coeff) for coeff in f.all_coeffs()]
        elif f.rep.dom is QQ:
            denoms = [coeff.q for coeff in f.all_coeffs()]
            fac = ilcm(*denoms)
            coeffs = [int(coeff * fac) for coeff in f.all_coeffs()]
        else:
            coeffs = [coeff.evalf(n=n).as_real_imag() for coeff in f.all_coeffs()]
            try:
                coeffs = [mpmath.mpc(*coeff) for coeff in coeffs]
            except TypeError:
                raise DomainError('Numerical domain expected, got %s' % f.rep.dom)
        dps = mpmath.mp.dps
        mpmath.mp.dps = n
        from sympy.functions.elementary.complexes import sign
        try:
            roots = mpmath.polyroots(coeffs, maxsteps=maxsteps, cleanup=cleanup, error=False, extraprec=f.degree() * 10)
            roots = list(map(sympify, sorted(roots, key=lambda r: (1 if r.imag else 0, r.real, abs(r.imag), sign(r.imag)))))
        except NoConvergence:
            try:
                roots = mpmath.polyroots(coeffs, maxsteps=maxsteps, cleanup=cleanup, error=False, extraprec=f.degree() * 15)
                roots = list(map(sympify, sorted(roots, key=lambda r: (1 if r.imag else 0, r.real, abs(r.imag), sign(r.imag)))))
            except NoConvergence:
                raise NoConvergence('convergence to root failed; try n < %s or maxsteps > %s' % (n, maxsteps))
        finally:
            mpmath.mp.dps = dps
        return roots

    def ground_roots(f):
        if False:
            i = 10
            return i + 15
        '\n        Compute roots of ``f`` by factorization in the ground domain.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x**6 - 4*x**4 + 4*x**3 - x**2).ground_roots()\n        {0: 2, 1: 2}\n\n        '
        if f.is_multivariate:
            raise MultivariatePolynomialError('Cannot compute ground roots of %s' % f)
        roots = {}
        for (factor, k) in f.factor_list()[1]:
            if factor.is_linear:
                (a, b) = factor.all_coeffs()
                roots[-b / a] = k
        return roots

    def nth_power_roots_poly(f, n):
        if False:
            return 10
        "\n        Construct a polynomial with n-th powers of roots of ``f``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> f = Poly(x**4 - x**2 + 1)\n\n        >>> f.nth_power_roots_poly(2)\n        Poly(x**4 - 2*x**3 + 3*x**2 - 2*x + 1, x, domain='ZZ')\n        >>> f.nth_power_roots_poly(3)\n        Poly(x**4 + 2*x**2 + 1, x, domain='ZZ')\n        >>> f.nth_power_roots_poly(4)\n        Poly(x**4 + 2*x**3 + 3*x**2 + 2*x + 1, x, domain='ZZ')\n        >>> f.nth_power_roots_poly(12)\n        Poly(x**4 - 4*x**3 + 6*x**2 - 4*x + 1, x, domain='ZZ')\n\n        "
        if f.is_multivariate:
            raise MultivariatePolynomialError('must be a univariate polynomial')
        N = sympify(n)
        if N.is_Integer and N >= 1:
            n = int(N)
        else:
            raise ValueError("'n' must an integer and n >= 1, got %s" % n)
        x = f.gen
        t = Dummy('t')
        r = f.resultant(f.__class__.from_expr(x ** n - t, x, t))
        return r.replace(t, x)

    def same_root(f, a, b):
        if False:
            i = 10
            return i + 15
        '\n        Decide whether two roots of this polynomial are equal.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly, cyclotomic_poly, exp, I, pi\n        >>> f = Poly(cyclotomic_poly(5))\n        >>> r0 = exp(2*I*pi/5)\n        >>> indices = [i for i, r in enumerate(f.all_roots()) if f.same_root(r, r0)]\n        >>> print(indices)\n        [3]\n\n        Raises\n        ======\n\n        DomainError\n            If the domain of the polynomial is not :ref:`ZZ`, :ref:`QQ`,\n            :ref:`RR`, or :ref:`CC`.\n        MultivariatePolynomialError\n            If the polynomial is not univariate.\n        PolynomialError\n            If the polynomial is of degree < 2.\n\n        '
        if f.is_multivariate:
            raise MultivariatePolynomialError('Must be a univariate polynomial')
        dom_delta_sq = f.rep.mignotte_sep_bound_squared()
        delta_sq = f.domain.get_field().to_sympy(dom_delta_sq)
        eps_sq = delta_sq / 9
        (r, _, _, _) = evalf(1 / eps_sq, 1, {})
        n = fastlog(r)
        m = n // 2 + n % 2
        ev = lambda x: quad_to_mpmath(_evalf_with_bounded_error(x, m=m))
        (A, B) = (ev(a), ev(b))
        return (A.real - B.real) ** 2 + (A.imag - B.imag) ** 2 < eps_sq

    def cancel(f, g, include=False):
        if False:
            i = 10
            return i + 15
        "\n        Cancel common factors in a rational function ``f/g``.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(2*x**2 - 2, x).cancel(Poly(x**2 - 2*x + 1, x))\n        (1, Poly(2*x + 2, x, domain='ZZ'), Poly(x - 1, x, domain='ZZ'))\n\n        >>> Poly(2*x**2 - 2, x).cancel(Poly(x**2 - 2*x + 1, x), include=True)\n        (Poly(2*x + 2, x, domain='ZZ'), Poly(x - 1, x, domain='ZZ'))\n\n        "
        (dom, per, F, G) = f._unify(g)
        if hasattr(F, 'cancel'):
            result = F.cancel(G, include=include)
        else:
            raise OperationNotSupported(f, 'cancel')
        if not include:
            if dom.has_assoc_Ring:
                dom = dom.get_ring()
            (cp, cq, p, q) = result
            cp = dom.to_sympy(cp)
            cq = dom.to_sympy(cq)
            return (cp / cq, per(p), per(q))
        else:
            return tuple(map(per, result))

    def make_monic_over_integers_by_scaling_roots(f):
        if False:
            while True:
                i = 10
        "\n        Turn any univariate polynomial over :ref:`QQ` or :ref:`ZZ` into a monic\n        polynomial over :ref:`ZZ`, by scaling the roots as necessary.\n\n        Explanation\n        ===========\n\n        This operation can be performed whether or not *f* is irreducible; when\n        it is, this can be understood as determining an algebraic integer\n        generating the same field as a root of *f*.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly, S\n        >>> from sympy.abc import x\n        >>> f = Poly(x**2/2 + S(1)/4 * x + S(1)/8, x, domain='QQ')\n        >>> f.make_monic_over_integers_by_scaling_roots()\n        (Poly(x**2 + 2*x + 4, x, domain='ZZ'), 4)\n\n        Returns\n        =======\n\n        Pair ``(g, c)``\n            g is the polynomial\n\n            c is the integer by which the roots had to be scaled\n\n        "
        if not f.is_univariate or f.domain not in [ZZ, QQ]:
            raise ValueError('Polynomial must be univariate over ZZ or QQ.')
        if f.is_monic and f.domain == ZZ:
            return (f, ZZ.one)
        else:
            fm = f.monic()
            (c, _) = fm.clear_denoms()
            return (fm.transform(Poly(fm.gen), c).to_ring(), c)

    def galois_group(f, by_name=False, max_tries=30, randomize=False):
        if False:
            while True:
                i = 10
        '\n        Compute the Galois group of this polynomial.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n        >>> f = Poly(x**4 - 2)\n        >>> G, _ = f.galois_group(by_name=True)\n        >>> print(G)\n        S4TransitiveSubgroups.D4\n\n        See Also\n        ========\n\n        sympy.polys.numberfields.galoisgroups.galois_group\n\n        '
        from sympy.polys.numberfields.galoisgroups import _galois_group_degree_3, _galois_group_degree_4_lookup, _galois_group_degree_5_lookup_ext_factor, _galois_group_degree_6_lookup
        if not f.is_univariate or not f.is_irreducible or f.domain not in [ZZ, QQ]:
            raise ValueError('Polynomial must be irreducible and univariate over ZZ or QQ.')
        gg = {3: _galois_group_degree_3, 4: _galois_group_degree_4_lookup, 5: _galois_group_degree_5_lookup_ext_factor, 6: _galois_group_degree_6_lookup}
        max_supported = max(gg.keys())
        n = f.degree()
        if n > max_supported:
            raise ValueError(f'Only polynomials up to degree {max_supported} are supported.')
        elif n < 1:
            raise ValueError('Constant polynomial has no Galois group.')
        elif n == 1:
            from sympy.combinatorics.galois import S1TransitiveSubgroups
            (name, alt) = (S1TransitiveSubgroups.S1, True)
        elif n == 2:
            from sympy.combinatorics.galois import S2TransitiveSubgroups
            (name, alt) = (S2TransitiveSubgroups.S2, False)
        else:
            (g, _) = f.make_monic_over_integers_by_scaling_roots()
            (name, alt) = gg[n](g, max_tries=max_tries, randomize=randomize)
        G = name if by_name else name.get_perm_group()
        return (G, alt)

    @property
    def is_zero(f):
        if False:
            print('Hello World!')
        '\n        Returns ``True`` if ``f`` is a zero polynomial.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(0, x).is_zero\n        True\n        >>> Poly(1, x).is_zero\n        False\n\n        '
        return f.rep.is_zero

    @property
    def is_one(f):
        if False:
            while True:
                i = 10
        '\n        Returns ``True`` if ``f`` is a unit polynomial.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(0, x).is_one\n        False\n        >>> Poly(1, x).is_one\n        True\n\n        '
        return f.rep.is_one

    @property
    def is_sqf(f):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns ``True`` if ``f`` is a square-free polynomial.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x**2 - 2*x + 1, x).is_sqf\n        False\n        >>> Poly(x**2 - 1, x).is_sqf\n        True\n\n        '
        return f.rep.is_sqf

    @property
    def is_monic(f):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns ``True`` if the leading coefficient of ``f`` is one.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x + 2, x).is_monic\n        True\n        >>> Poly(2*x + 2, x).is_monic\n        False\n\n        '
        return f.rep.is_monic

    @property
    def is_primitive(f):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns ``True`` if GCD of the coefficients of ``f`` is one.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(2*x**2 + 6*x + 12, x).is_primitive\n        False\n        >>> Poly(x**2 + 3*x + 6, x).is_primitive\n        True\n\n        '
        return f.rep.is_primitive

    @property
    def is_ground(f):
        if False:
            i = 10
            return i + 15
        '\n        Returns ``True`` if ``f`` is an element of the ground domain.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x, y\n\n        >>> Poly(x, x).is_ground\n        False\n        >>> Poly(2, x).is_ground\n        True\n        >>> Poly(y, x).is_ground\n        True\n\n        '
        return f.rep.is_ground

    @property
    def is_linear(f):
        if False:
            print('Hello World!')
        '\n        Returns ``True`` if ``f`` is linear in all its variables.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x, y\n\n        >>> Poly(x + y + 2, x, y).is_linear\n        True\n        >>> Poly(x*y + 2, x, y).is_linear\n        False\n\n        '
        return f.rep.is_linear

    @property
    def is_quadratic(f):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns ``True`` if ``f`` is quadratic in all its variables.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x, y\n\n        >>> Poly(x*y + 2, x, y).is_quadratic\n        True\n        >>> Poly(x*y**2 + 2, x, y).is_quadratic\n        False\n\n        '
        return f.rep.is_quadratic

    @property
    def is_monomial(f):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns ``True`` if ``f`` is zero or has only one term.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(3*x**2, x).is_monomial\n        True\n        >>> Poly(3*x**2 + 1, x).is_monomial\n        False\n\n        '
        return f.rep.is_monomial

    @property
    def is_homogeneous(f):
        if False:
            i = 10
            return i + 15
        '\n        Returns ``True`` if ``f`` is a homogeneous polynomial.\n\n        A homogeneous polynomial is a polynomial whose all monomials with\n        non-zero coefficients have the same total degree. If you want not\n        only to check if a polynomial is homogeneous but also compute its\n        homogeneous order, then use :func:`Poly.homogeneous_order`.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x, y\n\n        >>> Poly(x**2 + x*y, x, y).is_homogeneous\n        True\n        >>> Poly(x**3 + x*y, x, y).is_homogeneous\n        False\n\n        '
        return f.rep.is_homogeneous

    @property
    def is_irreducible(f):
        if False:
            print('Hello World!')
        '\n        Returns ``True`` if ``f`` has no factors over its domain.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> Poly(x**2 + x + 1, x, modulus=2).is_irreducible\n        True\n        >>> Poly(x**2 + 1, x, modulus=2).is_irreducible\n        False\n\n        '
        return f.rep.is_irreducible

    @property
    def is_univariate(f):
        if False:
            while True:
                i = 10
        '\n        Returns ``True`` if ``f`` is a univariate polynomial.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x, y\n\n        >>> Poly(x**2 + x + 1, x).is_univariate\n        True\n        >>> Poly(x*y**2 + x*y + 1, x, y).is_univariate\n        False\n        >>> Poly(x*y**2 + x*y + 1, x).is_univariate\n        True\n        >>> Poly(x**2 + x + 1, x, y).is_univariate\n        False\n\n        '
        return len(f.gens) == 1

    @property
    def is_multivariate(f):
        if False:
            return 10
        '\n        Returns ``True`` if ``f`` is a multivariate polynomial.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x, y\n\n        >>> Poly(x**2 + x + 1, x).is_multivariate\n        False\n        >>> Poly(x*y**2 + x*y + 1, x, y).is_multivariate\n        True\n        >>> Poly(x*y**2 + x*y + 1, x).is_multivariate\n        False\n        >>> Poly(x**2 + x + 1, x, y).is_multivariate\n        True\n\n        '
        return len(f.gens) != 1

    @property
    def is_cyclotomic(f):
        if False:
            while True:
                i = 10
        '\n        Returns ``True`` if ``f`` is a cyclotomic polnomial.\n\n        Examples\n        ========\n\n        >>> from sympy import Poly\n        >>> from sympy.abc import x\n\n        >>> f = x**16 + x**14 - x**10 + x**8 - x**6 + x**2 + 1\n\n        >>> Poly(f).is_cyclotomic\n        False\n\n        >>> g = x**16 + x**14 - x**10 - x**8 - x**6 + x**2 + 1\n\n        >>> Poly(g).is_cyclotomic\n        True\n\n        '
        return f.rep.is_cyclotomic

    def __abs__(f):
        if False:
            while True:
                i = 10
        return f.abs()

    def __neg__(f):
        if False:
            i = 10
            return i + 15
        return f.neg()

    @_polifyit
    def __add__(f, g):
        if False:
            return 10
        return f.add(g)

    @_polifyit
    def __radd__(f, g):
        if False:
            print('Hello World!')
        return g.add(f)

    @_polifyit
    def __sub__(f, g):
        if False:
            for i in range(10):
                print('nop')
        return f.sub(g)

    @_polifyit
    def __rsub__(f, g):
        if False:
            print('Hello World!')
        return g.sub(f)

    @_polifyit
    def __mul__(f, g):
        if False:
            while True:
                i = 10
        return f.mul(g)

    @_polifyit
    def __rmul__(f, g):
        if False:
            for i in range(10):
                print('nop')
        return g.mul(f)

    @_sympifyit('n', NotImplemented)
    def __pow__(f, n):
        if False:
            i = 10
            return i + 15
        if n.is_Integer and n >= 0:
            return f.pow(n)
        else:
            return NotImplemented

    @_polifyit
    def __divmod__(f, g):
        if False:
            return 10
        return f.div(g)

    @_polifyit
    def __rdivmod__(f, g):
        if False:
            return 10
        return g.div(f)

    @_polifyit
    def __mod__(f, g):
        if False:
            print('Hello World!')
        return f.rem(g)

    @_polifyit
    def __rmod__(f, g):
        if False:
            i = 10
            return i + 15
        return g.rem(f)

    @_polifyit
    def __floordiv__(f, g):
        if False:
            return 10
        return f.quo(g)

    @_polifyit
    def __rfloordiv__(f, g):
        if False:
            i = 10
            return i + 15
        return g.quo(f)

    @_sympifyit('g', NotImplemented)
    def __truediv__(f, g):
        if False:
            for i in range(10):
                print('nop')
        return f.as_expr() / g.as_expr()

    @_sympifyit('g', NotImplemented)
    def __rtruediv__(f, g):
        if False:
            print('Hello World!')
        return g.as_expr() / f.as_expr()

    @_sympifyit('other', NotImplemented)
    def __eq__(self, other):
        if False:
            while True:
                i = 10
        (f, g) = (self, other)
        if not g.is_Poly:
            try:
                g = f.__class__(g, f.gens, domain=f.get_domain())
            except (PolynomialError, DomainError, CoercionFailed):
                return False
        if f.gens != g.gens:
            return False
        if f.rep.dom != g.rep.dom:
            return False
        return f.rep == g.rep

    @_sympifyit('g', NotImplemented)
    def __ne__(f, g):
        if False:
            for i in range(10):
                print('nop')
        return not f == g

    def __bool__(f):
        if False:
            while True:
                i = 10
        return not f.is_zero

    def eq(f, g, strict=False):
        if False:
            print('Hello World!')
        if not strict:
            return f == g
        else:
            return f._strict_eq(sympify(g))

    def ne(f, g, strict=False):
        if False:
            for i in range(10):
                print('nop')
        return not f.eq(g, strict=strict)

    def _strict_eq(f, g):
        if False:
            print('Hello World!')
        return isinstance(g, f.__class__) and f.gens == g.gens and f.rep.eq(g.rep, strict=True)

@public
class PurePoly(Poly):
    """Class for representing pure polynomials. """

    def _hashable_content(self):
        if False:
            i = 10
            return i + 15
        'Allow SymPy to hash Poly instances. '
        return (self.rep,)

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return super().__hash__()

    @property
    def free_symbols(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Free symbols of a polynomial.\n\n        Examples\n        ========\n\n        >>> from sympy import PurePoly\n        >>> from sympy.abc import x, y\n\n        >>> PurePoly(x**2 + 1).free_symbols\n        set()\n        >>> PurePoly(x**2 + y).free_symbols\n        set()\n        >>> PurePoly(x**2 + y, x).free_symbols\n        {y}\n\n        '
        return self.free_symbols_in_domain

    @_sympifyit('other', NotImplemented)
    def __eq__(self, other):
        if False:
            while True:
                i = 10
        (f, g) = (self, other)
        if not g.is_Poly:
            try:
                g = f.__class__(g, f.gens, domain=f.get_domain())
            except (PolynomialError, DomainError, CoercionFailed):
                return False
        if len(f.gens) != len(g.gens):
            return False
        if f.rep.dom != g.rep.dom:
            try:
                dom = f.rep.dom.unify(g.rep.dom, f.gens)
            except UnificationFailed:
                return False
            f = f.set_domain(dom)
            g = g.set_domain(dom)
        return f.rep == g.rep

    def _strict_eq(f, g):
        if False:
            return 10
        return isinstance(g, f.__class__) and f.rep.eq(g.rep, strict=True)

    def _unify(f, g):
        if False:
            print('Hello World!')
        g = sympify(g)
        if not g.is_Poly:
            try:
                return (f.rep.dom, f.per, f.rep, f.rep.per(f.rep.dom.from_sympy(g)))
            except CoercionFailed:
                raise UnificationFailed('Cannot unify %s with %s' % (f, g))
        if len(f.gens) != len(g.gens):
            raise UnificationFailed('Cannot unify %s with %s' % (f, g))
        if not (isinstance(f.rep, DMP) and isinstance(g.rep, DMP)):
            raise UnificationFailed('Cannot unify %s with %s' % (f, g))
        cls = f.__class__
        gens = f.gens
        dom = f.rep.dom.unify(g.rep.dom, gens)
        F = f.rep.convert(dom)
        G = g.rep.convert(dom)

        def per(rep, dom=dom, gens=gens, remove=None):
            if False:
                for i in range(10):
                    print('nop')
            if remove is not None:
                gens = gens[:remove] + gens[remove + 1:]
                if not gens:
                    return dom.to_sympy(rep)
            return cls.new(rep, *gens)
        return (dom, per, F, G)

@public
def poly_from_expr(expr, *gens, **args):
    if False:
        print('Hello World!')
    'Construct a polynomial from an expression. '
    opt = options.build_options(gens, args)
    return _poly_from_expr(expr, opt)

def _poly_from_expr(expr, opt):
    if False:
        return 10
    'Construct a polynomial from an expression. '
    (orig, expr) = (expr, sympify(expr))
    if not isinstance(expr, Basic):
        raise PolificationFailed(opt, orig, expr)
    elif expr.is_Poly:
        poly = expr.__class__._from_poly(expr, opt)
        opt.gens = poly.gens
        opt.domain = poly.domain
        if opt.polys is None:
            opt.polys = True
        return (poly, opt)
    elif opt.expand:
        expr = expr.expand()
    (rep, opt) = _dict_from_expr(expr, opt)
    if not opt.gens:
        raise PolificationFailed(opt, orig, expr)
    (monoms, coeffs) = list(zip(*list(rep.items())))
    domain = opt.domain
    if domain is None:
        (opt.domain, coeffs) = construct_domain(coeffs, opt=opt)
    else:
        coeffs = list(map(domain.from_sympy, coeffs))
    rep = dict(list(zip(monoms, coeffs)))
    poly = Poly._from_dict(rep, opt)
    if opt.polys is None:
        opt.polys = False
    return (poly, opt)

@public
def parallel_poly_from_expr(exprs, *gens, **args):
    if False:
        while True:
            i = 10
    'Construct polynomials from expressions. '
    opt = options.build_options(gens, args)
    return _parallel_poly_from_expr(exprs, opt)

def _parallel_poly_from_expr(exprs, opt):
    if False:
        while True:
            i = 10
    'Construct polynomials from expressions. '
    if len(exprs) == 2:
        (f, g) = exprs
        if isinstance(f, Poly) and isinstance(g, Poly):
            f = f.__class__._from_poly(f, opt)
            g = g.__class__._from_poly(g, opt)
            (f, g) = f.unify(g)
            opt.gens = f.gens
            opt.domain = f.domain
            if opt.polys is None:
                opt.polys = True
            return ([f, g], opt)
    (origs, exprs) = (list(exprs), [])
    (_exprs, _polys) = ([], [])
    failed = False
    for (i, expr) in enumerate(origs):
        expr = sympify(expr)
        if isinstance(expr, Basic):
            if expr.is_Poly:
                _polys.append(i)
            else:
                _exprs.append(i)
                if opt.expand:
                    expr = expr.expand()
        else:
            failed = True
        exprs.append(expr)
    if failed:
        raise PolificationFailed(opt, origs, exprs, True)
    if _polys:
        for i in _polys:
            exprs[i] = exprs[i].as_expr()
    (reps, opt) = _parallel_dict_from_expr(exprs, opt)
    if not opt.gens:
        raise PolificationFailed(opt, origs, exprs, True)
    from sympy.functions.elementary.piecewise import Piecewise
    for k in opt.gens:
        if isinstance(k, Piecewise):
            raise PolynomialError('Piecewise generators do not make sense')
    (coeffs_list, lengths) = ([], [])
    all_monoms = []
    all_coeffs = []
    for rep in reps:
        (monoms, coeffs) = list(zip(*list(rep.items())))
        coeffs_list.extend(coeffs)
        all_monoms.append(monoms)
        lengths.append(len(coeffs))
    domain = opt.domain
    if domain is None:
        (opt.domain, coeffs_list) = construct_domain(coeffs_list, opt=opt)
    else:
        coeffs_list = list(map(domain.from_sympy, coeffs_list))
    for k in lengths:
        all_coeffs.append(coeffs_list[:k])
        coeffs_list = coeffs_list[k:]
    polys = []
    for (monoms, coeffs) in zip(all_monoms, all_coeffs):
        rep = dict(list(zip(monoms, coeffs)))
        poly = Poly._from_dict(rep, opt)
        polys.append(poly)
    if opt.polys is None:
        opt.polys = bool(_polys)
    return (polys, opt)

def _update_args(args, key, value):
    if False:
        i = 10
        return i + 15
    'Add a new ``(key, value)`` pair to arguments ``dict``. '
    args = dict(args)
    if key not in args:
        args[key] = value
    return args

@public
def degree(f, gen=0):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the degree of ``f`` in the given variable.\n\n    The degree of 0 is negative infinity.\n\n    Examples\n    ========\n\n    >>> from sympy import degree\n    >>> from sympy.abc import x, y\n\n    >>> degree(x**2 + y*x + 1, gen=x)\n    2\n    >>> degree(x**2 + y*x + 1, gen=y)\n    1\n    >>> degree(0, x)\n    -oo\n\n    See also\n    ========\n\n    sympy.polys.polytools.Poly.total_degree\n    degree_list\n    '
    f = sympify(f, strict=True)
    gen_is_Num = sympify(gen, strict=True).is_Number
    if f.is_Poly:
        p = f
        isNum = p.as_expr().is_Number
    else:
        isNum = f.is_Number
        if not isNum:
            if gen_is_Num:
                (p, _) = poly_from_expr(f)
            else:
                (p, _) = poly_from_expr(f, gen)
    if isNum:
        return S.Zero if f else S.NegativeInfinity
    if not gen_is_Num:
        if f.is_Poly and gen not in p.gens:
            (p, _) = poly_from_expr(f.as_expr())
        if gen not in p.gens:
            return S.Zero
    elif not f.is_Poly and len(f.free_symbols) > 1:
        raise TypeError(filldedent('\n         A symbolic generator of interest is required for a multivariate\n         expression like func = %s, e.g. degree(func, gen = %s) instead of\n         degree(func, gen = %s).\n        ' % (f, next(ordered(f.free_symbols)), gen)))
    result = p.degree(gen)
    return Integer(result) if isinstance(result, int) else S.NegativeInfinity

@public
def total_degree(f, *gens):
    if False:
        return 10
    '\n    Return the total_degree of ``f`` in the given variables.\n\n    Examples\n    ========\n    >>> from sympy import total_degree, Poly\n    >>> from sympy.abc import x, y\n\n    >>> total_degree(1)\n    0\n    >>> total_degree(x + x*y)\n    2\n    >>> total_degree(x + x*y, x)\n    1\n\n    If the expression is a Poly and no variables are given\n    then the generators of the Poly will be used:\n\n    >>> p = Poly(x + x*y, y)\n    >>> total_degree(p)\n    1\n\n    To deal with the underlying expression of the Poly, convert\n    it to an Expr:\n\n    >>> total_degree(p.as_expr())\n    2\n\n    This is done automatically if any variables are given:\n\n    >>> total_degree(p, x)\n    1\n\n    See also\n    ========\n    degree\n    '
    p = sympify(f)
    if p.is_Poly:
        p = p.as_expr()
    if p.is_Number:
        rv = 0
    else:
        if f.is_Poly:
            gens = gens or f.gens
        rv = Poly(p, gens).total_degree()
    return Integer(rv)

@public
def degree_list(f, *gens, **args):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a list of degrees of ``f`` in all variables.\n\n    Examples\n    ========\n\n    >>> from sympy import degree_list\n    >>> from sympy.abc import x, y\n\n    >>> degree_list(x**2 + y*x + 1)\n    (2, 1)\n\n    '
    options.allowed_flags(args, ['polys'])
    try:
        (F, opt) = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        raise ComputationFailed('degree_list', 1, exc)
    degrees = F.degree_list()
    return tuple(map(Integer, degrees))

@public
def LC(f, *gens, **args):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the leading coefficient of ``f``.\n\n    Examples\n    ========\n\n    >>> from sympy import LC\n    >>> from sympy.abc import x, y\n\n    >>> LC(4*x**2 + 2*x*y**2 + x*y + 3*y)\n    4\n\n    '
    options.allowed_flags(args, ['polys'])
    try:
        (F, opt) = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        raise ComputationFailed('LC', 1, exc)
    return F.LC(order=opt.order)

@public
def LM(f, *gens, **args):
    if False:
        print('Hello World!')
    '\n    Return the leading monomial of ``f``.\n\n    Examples\n    ========\n\n    >>> from sympy import LM\n    >>> from sympy.abc import x, y\n\n    >>> LM(4*x**2 + 2*x*y**2 + x*y + 3*y)\n    x**2\n\n    '
    options.allowed_flags(args, ['polys'])
    try:
        (F, opt) = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        raise ComputationFailed('LM', 1, exc)
    monom = F.LM(order=opt.order)
    return monom.as_expr()

@public
def LT(f, *gens, **args):
    if False:
        while True:
            i = 10
    '\n    Return the leading term of ``f``.\n\n    Examples\n    ========\n\n    >>> from sympy import LT\n    >>> from sympy.abc import x, y\n\n    >>> LT(4*x**2 + 2*x*y**2 + x*y + 3*y)\n    4*x**2\n\n    '
    options.allowed_flags(args, ['polys'])
    try:
        (F, opt) = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        raise ComputationFailed('LT', 1, exc)
    (monom, coeff) = F.LT(order=opt.order)
    return coeff * monom.as_expr()

@public
def pdiv(f, g, *gens, **args):
    if False:
        while True:
            i = 10
    '\n    Compute polynomial pseudo-division of ``f`` and ``g``.\n\n    Examples\n    ========\n\n    >>> from sympy import pdiv\n    >>> from sympy.abc import x\n\n    >>> pdiv(x**2 + 1, 2*x - 4)\n    (2*x + 4, 20)\n\n    '
    options.allowed_flags(args, ['polys'])
    try:
        ((F, G), opt) = parallel_poly_from_expr((f, g), *gens, **args)
    except PolificationFailed as exc:
        raise ComputationFailed('pdiv', 2, exc)
    (q, r) = F.pdiv(G)
    if not opt.polys:
        return (q.as_expr(), r.as_expr())
    else:
        return (q, r)

@public
def prem(f, g, *gens, **args):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute polynomial pseudo-remainder of ``f`` and ``g``.\n\n    Examples\n    ========\n\n    >>> from sympy import prem\n    >>> from sympy.abc import x\n\n    >>> prem(x**2 + 1, 2*x - 4)\n    20\n\n    '
    options.allowed_flags(args, ['polys'])
    try:
        ((F, G), opt) = parallel_poly_from_expr((f, g), *gens, **args)
    except PolificationFailed as exc:
        raise ComputationFailed('prem', 2, exc)
    r = F.prem(G)
    if not opt.polys:
        return r.as_expr()
    else:
        return r

@public
def pquo(f, g, *gens, **args):
    if False:
        i = 10
        return i + 15
    '\n    Compute polynomial pseudo-quotient of ``f`` and ``g``.\n\n    Examples\n    ========\n\n    >>> from sympy import pquo\n    >>> from sympy.abc import x\n\n    >>> pquo(x**2 + 1, 2*x - 4)\n    2*x + 4\n    >>> pquo(x**2 - 1, 2*x - 1)\n    2*x + 1\n\n    '
    options.allowed_flags(args, ['polys'])
    try:
        ((F, G), opt) = parallel_poly_from_expr((f, g), *gens, **args)
    except PolificationFailed as exc:
        raise ComputationFailed('pquo', 2, exc)
    try:
        q = F.pquo(G)
    except ExactQuotientFailed:
        raise ExactQuotientFailed(f, g)
    if not opt.polys:
        return q.as_expr()
    else:
        return q

@public
def pexquo(f, g, *gens, **args):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute polynomial exact pseudo-quotient of ``f`` and ``g``.\n\n    Examples\n    ========\n\n    >>> from sympy import pexquo\n    >>> from sympy.abc import x\n\n    >>> pexquo(x**2 - 1, 2*x - 2)\n    2*x + 2\n\n    >>> pexquo(x**2 + 1, 2*x - 4)\n    Traceback (most recent call last):\n    ...\n    ExactQuotientFailed: 2*x - 4 does not divide x**2 + 1\n\n    '
    options.allowed_flags(args, ['polys'])
    try:
        ((F, G), opt) = parallel_poly_from_expr((f, g), *gens, **args)
    except PolificationFailed as exc:
        raise ComputationFailed('pexquo', 2, exc)
    q = F.pexquo(G)
    if not opt.polys:
        return q.as_expr()
    else:
        return q

@public
def div(f, g, *gens, **args):
    if False:
        print('Hello World!')
    '\n    Compute polynomial division of ``f`` and ``g``.\n\n    Examples\n    ========\n\n    >>> from sympy import div, ZZ, QQ\n    >>> from sympy.abc import x\n\n    >>> div(x**2 + 1, 2*x - 4, domain=ZZ)\n    (0, x**2 + 1)\n    >>> div(x**2 + 1, 2*x - 4, domain=QQ)\n    (x/2 + 1, 5)\n\n    '
    options.allowed_flags(args, ['auto', 'polys'])
    try:
        ((F, G), opt) = parallel_poly_from_expr((f, g), *gens, **args)
    except PolificationFailed as exc:
        raise ComputationFailed('div', 2, exc)
    (q, r) = F.div(G, auto=opt.auto)
    if not opt.polys:
        return (q.as_expr(), r.as_expr())
    else:
        return (q, r)

@public
def rem(f, g, *gens, **args):
    if False:
        while True:
            i = 10
    '\n    Compute polynomial remainder of ``f`` and ``g``.\n\n    Examples\n    ========\n\n    >>> from sympy import rem, ZZ, QQ\n    >>> from sympy.abc import x\n\n    >>> rem(x**2 + 1, 2*x - 4, domain=ZZ)\n    x**2 + 1\n    >>> rem(x**2 + 1, 2*x - 4, domain=QQ)\n    5\n\n    '
    options.allowed_flags(args, ['auto', 'polys'])
    try:
        ((F, G), opt) = parallel_poly_from_expr((f, g), *gens, **args)
    except PolificationFailed as exc:
        raise ComputationFailed('rem', 2, exc)
    r = F.rem(G, auto=opt.auto)
    if not opt.polys:
        return r.as_expr()
    else:
        return r

@public
def quo(f, g, *gens, **args):
    if False:
        while True:
            i = 10
    '\n    Compute polynomial quotient of ``f`` and ``g``.\n\n    Examples\n    ========\n\n    >>> from sympy import quo\n    >>> from sympy.abc import x\n\n    >>> quo(x**2 + 1, 2*x - 4)\n    x/2 + 1\n    >>> quo(x**2 - 1, x - 1)\n    x + 1\n\n    '
    options.allowed_flags(args, ['auto', 'polys'])
    try:
        ((F, G), opt) = parallel_poly_from_expr((f, g), *gens, **args)
    except PolificationFailed as exc:
        raise ComputationFailed('quo', 2, exc)
    q = F.quo(G, auto=opt.auto)
    if not opt.polys:
        return q.as_expr()
    else:
        return q

@public
def exquo(f, g, *gens, **args):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute polynomial exact quotient of ``f`` and ``g``.\n\n    Examples\n    ========\n\n    >>> from sympy import exquo\n    >>> from sympy.abc import x\n\n    >>> exquo(x**2 - 1, x - 1)\n    x + 1\n\n    >>> exquo(x**2 + 1, 2*x - 4)\n    Traceback (most recent call last):\n    ...\n    ExactQuotientFailed: 2*x - 4 does not divide x**2 + 1\n\n    '
    options.allowed_flags(args, ['auto', 'polys'])
    try:
        ((F, G), opt) = parallel_poly_from_expr((f, g), *gens, **args)
    except PolificationFailed as exc:
        raise ComputationFailed('exquo', 2, exc)
    q = F.exquo(G, auto=opt.auto)
    if not opt.polys:
        return q.as_expr()
    else:
        return q

@public
def half_gcdex(f, g, *gens, **args):
    if False:
        return 10
    '\n    Half extended Euclidean algorithm of ``f`` and ``g``.\n\n    Returns ``(s, h)`` such that ``h = gcd(f, g)`` and ``s*f = h (mod g)``.\n\n    Examples\n    ========\n\n    >>> from sympy import half_gcdex\n    >>> from sympy.abc import x\n\n    >>> half_gcdex(x**4 - 2*x**3 - 6*x**2 + 12*x + 15, x**3 + x**2 - 4*x - 4)\n    (3/5 - x/5, x + 1)\n\n    '
    options.allowed_flags(args, ['auto', 'polys'])
    try:
        ((F, G), opt) = parallel_poly_from_expr((f, g), *gens, **args)
    except PolificationFailed as exc:
        (domain, (a, b)) = construct_domain(exc.exprs)
        try:
            (s, h) = domain.half_gcdex(a, b)
        except NotImplementedError:
            raise ComputationFailed('half_gcdex', 2, exc)
        else:
            return (domain.to_sympy(s), domain.to_sympy(h))
    (s, h) = F.half_gcdex(G, auto=opt.auto)
    if not opt.polys:
        return (s.as_expr(), h.as_expr())
    else:
        return (s, h)

@public
def gcdex(f, g, *gens, **args):
    if False:
        i = 10
        return i + 15
    '\n    Extended Euclidean algorithm of ``f`` and ``g``.\n\n    Returns ``(s, t, h)`` such that ``h = gcd(f, g)`` and ``s*f + t*g = h``.\n\n    Examples\n    ========\n\n    >>> from sympy import gcdex\n    >>> from sympy.abc import x\n\n    >>> gcdex(x**4 - 2*x**3 - 6*x**2 + 12*x + 15, x**3 + x**2 - 4*x - 4)\n    (3/5 - x/5, x**2/5 - 6*x/5 + 2, x + 1)\n\n    '
    options.allowed_flags(args, ['auto', 'polys'])
    try:
        ((F, G), opt) = parallel_poly_from_expr((f, g), *gens, **args)
    except PolificationFailed as exc:
        (domain, (a, b)) = construct_domain(exc.exprs)
        try:
            (s, t, h) = domain.gcdex(a, b)
        except NotImplementedError:
            raise ComputationFailed('gcdex', 2, exc)
        else:
            return (domain.to_sympy(s), domain.to_sympy(t), domain.to_sympy(h))
    (s, t, h) = F.gcdex(G, auto=opt.auto)
    if not opt.polys:
        return (s.as_expr(), t.as_expr(), h.as_expr())
    else:
        return (s, t, h)

@public
def invert(f, g, *gens, **args):
    if False:
        i = 10
        return i + 15
    '\n    Invert ``f`` modulo ``g`` when possible.\n\n    Examples\n    ========\n\n    >>> from sympy import invert, S, mod_inverse\n    >>> from sympy.abc import x\n\n    >>> invert(x**2 - 1, 2*x - 1)\n    -4/3\n\n    >>> invert(x**2 - 1, x - 1)\n    Traceback (most recent call last):\n    ...\n    NotInvertible: zero divisor\n\n    For more efficient inversion of Rationals,\n    use the :obj:`sympy.core.intfunc.mod_inverse` function:\n\n    >>> mod_inverse(3, 5)\n    2\n    >>> (S(2)/5).invert(S(7)/3)\n    5/2\n\n    See Also\n    ========\n    sympy.core.intfunc.mod_inverse\n\n    '
    options.allowed_flags(args, ['auto', 'polys'])
    try:
        ((F, G), opt) = parallel_poly_from_expr((f, g), *gens, **args)
    except PolificationFailed as exc:
        (domain, (a, b)) = construct_domain(exc.exprs)
        try:
            return domain.to_sympy(domain.invert(a, b))
        except NotImplementedError:
            raise ComputationFailed('invert', 2, exc)
    h = F.invert(G, auto=opt.auto)
    if not opt.polys:
        return h.as_expr()
    else:
        return h

@public
def subresultants(f, g, *gens, **args):
    if False:
        i = 10
        return i + 15
    '\n    Compute subresultant PRS of ``f`` and ``g``.\n\n    Examples\n    ========\n\n    >>> from sympy import subresultants\n    >>> from sympy.abc import x\n\n    >>> subresultants(x**2 + 1, x**2 - 1)\n    [x**2 + 1, x**2 - 1, -2]\n\n    '
    options.allowed_flags(args, ['polys'])
    try:
        ((F, G), opt) = parallel_poly_from_expr((f, g), *gens, **args)
    except PolificationFailed as exc:
        raise ComputationFailed('subresultants', 2, exc)
    result = F.subresultants(G)
    if not opt.polys:
        return [r.as_expr() for r in result]
    else:
        return result

@public
def resultant(f, g, *gens, includePRS=False, **args):
    if False:
        print('Hello World!')
    '\n    Compute resultant of ``f`` and ``g``.\n\n    Examples\n    ========\n\n    >>> from sympy import resultant\n    >>> from sympy.abc import x\n\n    >>> resultant(x**2 + 1, x**2 - 1)\n    4\n\n    '
    options.allowed_flags(args, ['polys'])
    try:
        ((F, G), opt) = parallel_poly_from_expr((f, g), *gens, **args)
    except PolificationFailed as exc:
        raise ComputationFailed('resultant', 2, exc)
    if includePRS:
        (result, R) = F.resultant(G, includePRS=includePRS)
    else:
        result = F.resultant(G)
    if not opt.polys:
        if includePRS:
            return (result.as_expr(), [r.as_expr() for r in R])
        return result.as_expr()
    else:
        if includePRS:
            return (result, R)
        return result

@public
def discriminant(f, *gens, **args):
    if False:
        return 10
    '\n    Compute discriminant of ``f``.\n\n    Examples\n    ========\n\n    >>> from sympy import discriminant\n    >>> from sympy.abc import x\n\n    >>> discriminant(x**2 + 2*x + 3)\n    -8\n\n    '
    options.allowed_flags(args, ['polys'])
    try:
        (F, opt) = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        raise ComputationFailed('discriminant', 1, exc)
    result = F.discriminant()
    if not opt.polys:
        return result.as_expr()
    else:
        return result

@public
def cofactors(f, g, *gens, **args):
    if False:
        i = 10
        return i + 15
    '\n    Compute GCD and cofactors of ``f`` and ``g``.\n\n    Returns polynomials ``(h, cff, cfg)`` such that ``h = gcd(f, g)``, and\n    ``cff = quo(f, h)`` and ``cfg = quo(g, h)`` are, so called, cofactors\n    of ``f`` and ``g``.\n\n    Examples\n    ========\n\n    >>> from sympy import cofactors\n    >>> from sympy.abc import x\n\n    >>> cofactors(x**2 - 1, x**2 - 3*x + 2)\n    (x - 1, x + 1, x - 2)\n\n    '
    options.allowed_flags(args, ['polys'])
    try:
        ((F, G), opt) = parallel_poly_from_expr((f, g), *gens, **args)
    except PolificationFailed as exc:
        (domain, (a, b)) = construct_domain(exc.exprs)
        try:
            (h, cff, cfg) = domain.cofactors(a, b)
        except NotImplementedError:
            raise ComputationFailed('cofactors', 2, exc)
        else:
            return (domain.to_sympy(h), domain.to_sympy(cff), domain.to_sympy(cfg))
    (h, cff, cfg) = F.cofactors(G)
    if not opt.polys:
        return (h.as_expr(), cff.as_expr(), cfg.as_expr())
    else:
        return (h, cff, cfg)

@public
def gcd_list(seq, *gens, **args):
    if False:
        return 10
    '\n    Compute GCD of a list of polynomials.\n\n    Examples\n    ========\n\n    >>> from sympy import gcd_list\n    >>> from sympy.abc import x\n\n    >>> gcd_list([x**3 - 1, x**2 - 1, x**2 - 3*x + 2])\n    x - 1\n\n    '
    seq = sympify(seq)

    def try_non_polynomial_gcd(seq):
        if False:
            for i in range(10):
                print('nop')
        if not gens and (not args):
            (domain, numbers) = construct_domain(seq)
            if not numbers:
                return domain.zero
            elif domain.is_Numerical:
                (result, numbers) = (numbers[0], numbers[1:])
                for number in numbers:
                    result = domain.gcd(result, number)
                    if domain.is_one(result):
                        break
                return domain.to_sympy(result)
        return None
    result = try_non_polynomial_gcd(seq)
    if result is not None:
        return result
    options.allowed_flags(args, ['polys'])
    try:
        (polys, opt) = parallel_poly_from_expr(seq, *gens, **args)
        if len(seq) > 1 and all((elt.is_algebraic and elt.is_irrational for elt in seq)):
            a = seq[-1]
            lst = [(a / elt).ratsimp() for elt in seq[:-1]]
            if all((frc.is_rational for frc in lst)):
                lc = 1
                for frc in lst:
                    lc = lcm(lc, frc.as_numer_denom()[0])
                return abs(a / lc)
    except PolificationFailed as exc:
        result = try_non_polynomial_gcd(exc.exprs)
        if result is not None:
            return result
        else:
            raise ComputationFailed('gcd_list', len(seq), exc)
    if not polys:
        if not opt.polys:
            return S.Zero
        else:
            return Poly(0, opt=opt)
    (result, polys) = (polys[0], polys[1:])
    for poly in polys:
        result = result.gcd(poly)
        if result.is_one:
            break
    if not opt.polys:
        return result.as_expr()
    else:
        return result

@public
def gcd(f, g=None, *gens, **args):
    if False:
        i = 10
        return i + 15
    '\n    Compute GCD of ``f`` and ``g``.\n\n    Examples\n    ========\n\n    >>> from sympy import gcd\n    >>> from sympy.abc import x\n\n    >>> gcd(x**2 - 1, x**2 - 3*x + 2)\n    x - 1\n\n    '
    if hasattr(f, '__iter__'):
        if g is not None:
            gens = (g,) + gens
        return gcd_list(f, *gens, **args)
    elif g is None:
        raise TypeError('gcd() takes 2 arguments or a sequence of arguments')
    options.allowed_flags(args, ['polys'])
    try:
        ((F, G), opt) = parallel_poly_from_expr((f, g), *gens, **args)
        (a, b) = map(sympify, (f, g))
        if a.is_algebraic and a.is_irrational and b.is_algebraic and b.is_irrational:
            frc = (a / b).ratsimp()
            if frc.is_rational:
                return abs(a / frc.as_numer_denom()[0])
    except PolificationFailed as exc:
        (domain, (a, b)) = construct_domain(exc.exprs)
        try:
            return domain.to_sympy(domain.gcd(a, b))
        except NotImplementedError:
            raise ComputationFailed('gcd', 2, exc)
    result = F.gcd(G)
    if not opt.polys:
        return result.as_expr()
    else:
        return result

@public
def lcm_list(seq, *gens, **args):
    if False:
        while True:
            i = 10
    '\n    Compute LCM of a list of polynomials.\n\n    Examples\n    ========\n\n    >>> from sympy import lcm_list\n    >>> from sympy.abc import x\n\n    >>> lcm_list([x**3 - 1, x**2 - 1, x**2 - 3*x + 2])\n    x**5 - x**4 - 2*x**3 - x**2 + x + 2\n\n    '
    seq = sympify(seq)

    def try_non_polynomial_lcm(seq) -> Optional[Expr]:
        if False:
            i = 10
            return i + 15
        if not gens and (not args):
            (domain, numbers) = construct_domain(seq)
            if not numbers:
                return domain.to_sympy(domain.one)
            elif domain.is_Numerical:
                (result, numbers) = (numbers[0], numbers[1:])
                for number in numbers:
                    result = domain.lcm(result, number)
                return domain.to_sympy(result)
        return None
    result = try_non_polynomial_lcm(seq)
    if result is not None:
        return result
    options.allowed_flags(args, ['polys'])
    try:
        (polys, opt) = parallel_poly_from_expr(seq, *gens, **args)
        if len(seq) > 1 and all((elt.is_algebraic and elt.is_irrational for elt in seq)):
            a = seq[-1]
            lst = [(a / elt).ratsimp() for elt in seq[:-1]]
            if all((frc.is_rational for frc in lst)):
                lc = 1
                for frc in lst:
                    lc = lcm(lc, frc.as_numer_denom()[1])
                return a * lc
    except PolificationFailed as exc:
        result = try_non_polynomial_lcm(exc.exprs)
        if result is not None:
            return result
        else:
            raise ComputationFailed('lcm_list', len(seq), exc)
    if not polys:
        if not opt.polys:
            return S.One
        else:
            return Poly(1, opt=opt)
    (result, polys) = (polys[0], polys[1:])
    for poly in polys:
        result = result.lcm(poly)
    if not opt.polys:
        return result.as_expr()
    else:
        return result

@public
def lcm(f, g=None, *gens, **args):
    if False:
        print('Hello World!')
    '\n    Compute LCM of ``f`` and ``g``.\n\n    Examples\n    ========\n\n    >>> from sympy import lcm\n    >>> from sympy.abc import x\n\n    >>> lcm(x**2 - 1, x**2 - 3*x + 2)\n    x**3 - 2*x**2 - x + 2\n\n    '
    if hasattr(f, '__iter__'):
        if g is not None:
            gens = (g,) + gens
        return lcm_list(f, *gens, **args)
    elif g is None:
        raise TypeError('lcm() takes 2 arguments or a sequence of arguments')
    options.allowed_flags(args, ['polys'])
    try:
        ((F, G), opt) = parallel_poly_from_expr((f, g), *gens, **args)
        (a, b) = map(sympify, (f, g))
        if a.is_algebraic and a.is_irrational and b.is_algebraic and b.is_irrational:
            frc = (a / b).ratsimp()
            if frc.is_rational:
                return a * frc.as_numer_denom()[1]
    except PolificationFailed as exc:
        (domain, (a, b)) = construct_domain(exc.exprs)
        try:
            return domain.to_sympy(domain.lcm(a, b))
        except NotImplementedError:
            raise ComputationFailed('lcm', 2, exc)
    result = F.lcm(G)
    if not opt.polys:
        return result.as_expr()
    else:
        return result

@public
def terms_gcd(f, *gens, **args):
    if False:
        for i in range(10):
            print('nop')
    '\n    Remove GCD of terms from ``f``.\n\n    If the ``deep`` flag is True, then the arguments of ``f`` will have\n    terms_gcd applied to them.\n\n    If a fraction is factored out of ``f`` and ``f`` is an Add, then\n    an unevaluated Mul will be returned so that automatic simplification\n    does not redistribute it. The hint ``clear``, when set to False, can be\n    used to prevent such factoring when all coefficients are not fractions.\n\n    Examples\n    ========\n\n    >>> from sympy import terms_gcd, cos\n    >>> from sympy.abc import x, y\n    >>> terms_gcd(x**6*y**2 + x**3*y, x, y)\n    x**3*y*(x**3*y + 1)\n\n    The default action of polys routines is to expand the expression\n    given to them. terms_gcd follows this behavior:\n\n    >>> terms_gcd((3+3*x)*(x+x*y))\n    3*x*(x*y + x + y + 1)\n\n    If this is not desired then the hint ``expand`` can be set to False.\n    In this case the expression will be treated as though it were comprised\n    of one or more terms:\n\n    >>> terms_gcd((3+3*x)*(x+x*y), expand=False)\n    (3*x + 3)*(x*y + x)\n\n    In order to traverse factors of a Mul or the arguments of other\n    functions, the ``deep`` hint can be used:\n\n    >>> terms_gcd((3 + 3*x)*(x + x*y), expand=False, deep=True)\n    3*x*(x + 1)*(y + 1)\n    >>> terms_gcd(cos(x + x*y), deep=True)\n    cos(x*(y + 1))\n\n    Rationals are factored out by default:\n\n    >>> terms_gcd(x + y/2)\n    (2*x + y)/2\n\n    Only the y-term had a coefficient that was a fraction; if one\n    does not want to factor out the 1/2 in cases like this, the\n    flag ``clear`` can be set to False:\n\n    >>> terms_gcd(x + y/2, clear=False)\n    x + y/2\n    >>> terms_gcd(x*y/2 + y**2, clear=False)\n    y*(x/2 + y)\n\n    The ``clear`` flag is ignored if all coefficients are fractions:\n\n    >>> terms_gcd(x/3 + y/2, clear=False)\n    (2*x + 3*y)/6\n\n    See Also\n    ========\n    sympy.core.exprtools.gcd_terms, sympy.core.exprtools.factor_terms\n\n    '
    orig = sympify(f)
    if isinstance(f, Equality):
        return Equality(*(terms_gcd(s, *gens, **args) for s in [f.lhs, f.rhs]))
    elif isinstance(f, Relational):
        raise TypeError('Inequalities cannot be used with terms_gcd. Found: %s' % (f,))
    if not isinstance(f, Expr) or f.is_Atom:
        return orig
    if args.get('deep', False):
        new = f.func(*[terms_gcd(a, *gens, **args) for a in f.args])
        args.pop('deep')
        args['expand'] = False
        return terms_gcd(new, *gens, **args)
    clear = args.pop('clear', True)
    options.allowed_flags(args, ['polys'])
    try:
        (F, opt) = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        return exc.expr
    (J, f) = F.terms_gcd()
    if opt.domain.is_Ring:
        if opt.domain.is_Field:
            (denom, f) = f.clear_denoms(convert=True)
        (coeff, f) = f.primitive()
        if opt.domain.is_Field:
            coeff /= denom
    else:
        coeff = S.One
    term = Mul(*[x ** j for (x, j) in zip(f.gens, J)])
    if equal_valued(coeff, 1):
        coeff = S.One
        if term == 1:
            return orig
    if clear:
        return _keep_coeff(coeff, term * f.as_expr())
    (coeff, f) = _keep_coeff(coeff, f.as_expr(), clear=False).as_coeff_Mul()
    return _keep_coeff(coeff, term * f, clear=False)

@public
def trunc(f, p, *gens, **args):
    if False:
        while True:
            i = 10
    '\n    Reduce ``f`` modulo a constant ``p``.\n\n    Examples\n    ========\n\n    >>> from sympy import trunc\n    >>> from sympy.abc import x\n\n    >>> trunc(2*x**3 + 3*x**2 + 5*x + 7, 3)\n    -x**3 - x + 1\n\n    '
    options.allowed_flags(args, ['auto', 'polys'])
    try:
        (F, opt) = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        raise ComputationFailed('trunc', 1, exc)
    result = F.trunc(sympify(p))
    if not opt.polys:
        return result.as_expr()
    else:
        return result

@public
def monic(f, *gens, **args):
    if False:
        while True:
            i = 10
    '\n    Divide all coefficients of ``f`` by ``LC(f)``.\n\n    Examples\n    ========\n\n    >>> from sympy import monic\n    >>> from sympy.abc import x\n\n    >>> monic(3*x**2 + 4*x + 2)\n    x**2 + 4*x/3 + 2/3\n\n    '
    options.allowed_flags(args, ['auto', 'polys'])
    try:
        (F, opt) = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        raise ComputationFailed('monic', 1, exc)
    result = F.monic(auto=opt.auto)
    if not opt.polys:
        return result.as_expr()
    else:
        return result

@public
def content(f, *gens, **args):
    if False:
        while True:
            i = 10
    '\n    Compute GCD of coefficients of ``f``.\n\n    Examples\n    ========\n\n    >>> from sympy import content\n    >>> from sympy.abc import x\n\n    >>> content(6*x**2 + 8*x + 12)\n    2\n\n    '
    options.allowed_flags(args, ['polys'])
    try:
        (F, opt) = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        raise ComputationFailed('content', 1, exc)
    return F.content()

@public
def primitive(f, *gens, **args):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute content and the primitive form of ``f``.\n\n    Examples\n    ========\n\n    >>> from sympy.polys.polytools import primitive\n    >>> from sympy.abc import x\n\n    >>> primitive(6*x**2 + 8*x + 12)\n    (2, 3*x**2 + 4*x + 6)\n\n    >>> eq = (2 + 2*x)*x + 2\n\n    Expansion is performed by default:\n\n    >>> primitive(eq)\n    (2, x**2 + x + 1)\n\n    Set ``expand`` to False to shut this off. Note that the\n    extraction will not be recursive; use the as_content_primitive method\n    for recursive, non-destructive Rational extraction.\n\n    >>> primitive(eq, expand=False)\n    (1, x*(2*x + 2) + 2)\n\n    >>> eq.as_content_primitive()\n    (2, x*(x + 1) + 1)\n\n    '
    options.allowed_flags(args, ['polys'])
    try:
        (F, opt) = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        raise ComputationFailed('primitive', 1, exc)
    (cont, result) = F.primitive()
    if not opt.polys:
        return (cont, result.as_expr())
    else:
        return (cont, result)

@public
def compose(f, g, *gens, **args):
    if False:
        while True:
            i = 10
    '\n    Compute functional composition ``f(g)``.\n\n    Examples\n    ========\n\n    >>> from sympy import compose\n    >>> from sympy.abc import x\n\n    >>> compose(x**2 + x, x - 1)\n    x**2 - x\n\n    '
    options.allowed_flags(args, ['polys'])
    try:
        ((F, G), opt) = parallel_poly_from_expr((f, g), *gens, **args)
    except PolificationFailed as exc:
        raise ComputationFailed('compose', 2, exc)
    result = F.compose(G)
    if not opt.polys:
        return result.as_expr()
    else:
        return result

@public
def decompose(f, *gens, **args):
    if False:
        i = 10
        return i + 15
    '\n    Compute functional decomposition of ``f``.\n\n    Examples\n    ========\n\n    >>> from sympy import decompose\n    >>> from sympy.abc import x\n\n    >>> decompose(x**4 + 2*x**3 - x - 1)\n    [x**2 - x - 1, x**2 + x]\n\n    '
    options.allowed_flags(args, ['polys'])
    try:
        (F, opt) = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        raise ComputationFailed('decompose', 1, exc)
    result = F.decompose()
    if not opt.polys:
        return [r.as_expr() for r in result]
    else:
        return result

@public
def sturm(f, *gens, **args):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute Sturm sequence of ``f``.\n\n    Examples\n    ========\n\n    >>> from sympy import sturm\n    >>> from sympy.abc import x\n\n    >>> sturm(x**3 - 2*x**2 + x - 3)\n    [x**3 - 2*x**2 + x - 3, 3*x**2 - 4*x + 1, 2*x/9 + 25/9, -2079/4]\n\n    '
    options.allowed_flags(args, ['auto', 'polys'])
    try:
        (F, opt) = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        raise ComputationFailed('sturm', 1, exc)
    result = F.sturm(auto=opt.auto)
    if not opt.polys:
        return [r.as_expr() for r in result]
    else:
        return result

@public
def gff_list(f, *gens, **args):
    if False:
        while True:
            i = 10
    "\n    Compute a list of greatest factorial factors of ``f``.\n\n    Note that the input to ff() and rf() should be Poly instances to use the\n    definitions here.\n\n    Examples\n    ========\n\n    >>> from sympy import gff_list, ff, Poly\n    >>> from sympy.abc import x\n\n    >>> f = Poly(x**5 + 2*x**4 - x**3 - 2*x**2, x)\n\n    >>> gff_list(f)\n    [(Poly(x, x, domain='ZZ'), 1), (Poly(x + 2, x, domain='ZZ'), 4)]\n\n    >>> (ff(Poly(x), 1)*ff(Poly(x + 2), 4)) == f\n    True\n\n    >>> f = Poly(x**12 + 6*x**11 - 11*x**10 - 56*x**9 + 220*x**8 + 208*x**7 -         1401*x**6 + 1090*x**5 + 2715*x**4 - 6720*x**3 - 1092*x**2 + 5040*x, x)\n\n    >>> gff_list(f)\n    [(Poly(x**3 + 7, x, domain='ZZ'), 2), (Poly(x**2 + 5*x, x, domain='ZZ'), 3)]\n\n    >>> ff(Poly(x**3 + 7, x), 2)*ff(Poly(x**2 + 5*x, x), 3) == f\n    True\n\n    "
    options.allowed_flags(args, ['polys'])
    try:
        (F, opt) = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        raise ComputationFailed('gff_list', 1, exc)
    factors = F.gff_list()
    if not opt.polys:
        return [(g.as_expr(), k) for (g, k) in factors]
    else:
        return factors

@public
def gff(f, *gens, **args):
    if False:
        i = 10
        return i + 15
    'Compute greatest factorial factorization of ``f``. '
    raise NotImplementedError('symbolic falling factorial')

@public
def sqf_norm(f, *gens, **args):
    if False:
        i = 10
        return i + 15
    '\n    Compute square-free norm of ``f``.\n\n    Returns ``s``, ``f``, ``r``, such that ``g(x) = f(x-sa)`` and\n    ``r(x) = Norm(g(x))`` is a square-free polynomial over ``K``,\n    where ``a`` is the algebraic extension of the ground domain.\n\n    Examples\n    ========\n\n    >>> from sympy import sqf_norm, sqrt\n    >>> from sympy.abc import x\n\n    >>> sqf_norm(x**2 + 1, extension=[sqrt(3)])\n    (1, x**2 - 2*sqrt(3)*x + 4, x**4 - 4*x**2 + 16)\n\n    '
    options.allowed_flags(args, ['polys'])
    try:
        (F, opt) = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        raise ComputationFailed('sqf_norm', 1, exc)
    (s, g, r) = F.sqf_norm()
    if not opt.polys:
        return (Integer(s), g.as_expr(), r.as_expr())
    else:
        return (Integer(s), g, r)

@public
def sqf_part(f, *gens, **args):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute square-free part of ``f``.\n\n    Examples\n    ========\n\n    >>> from sympy import sqf_part\n    >>> from sympy.abc import x\n\n    >>> sqf_part(x**3 - 3*x - 2)\n    x**2 - x - 2\n\n    '
    options.allowed_flags(args, ['polys'])
    try:
        (F, opt) = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        raise ComputationFailed('sqf_part', 1, exc)
    result = F.sqf_part()
    if not opt.polys:
        return result.as_expr()
    else:
        return result

def _sorted_factors(factors, method):
    if False:
        return 10
    'Sort a list of ``(expr, exp)`` pairs. '
    if method == 'sqf':

        def key(obj):
            if False:
                while True:
                    i = 10
            (poly, exp) = obj
            rep = poly.rep.to_list()
            return (exp, len(rep), len(poly.gens), str(poly.domain), rep)
    else:

        def key(obj):
            if False:
                return 10
            (poly, exp) = obj
            rep = poly.rep.to_list()
            return (len(rep), len(poly.gens), exp, str(poly.domain), rep)
    return sorted(factors, key=key)

def _factors_product(factors):
    if False:
        while True:
            i = 10
    'Multiply a list of ``(expr, exp)`` pairs. '
    return Mul(*[f.as_expr() ** k for (f, k) in factors])

def _symbolic_factor_list(expr, opt, method):
    if False:
        i = 10
        return i + 15
    'Helper function for :func:`_symbolic_factor`. '
    (coeff, factors) = (S.One, [])
    args = [i._eval_factor() if hasattr(i, '_eval_factor') else i for i in Mul.make_args(expr)]
    for arg in args:
        if arg.is_Number or (isinstance(arg, Expr) and pure_complex(arg)):
            coeff *= arg
            continue
        elif arg.is_Pow and arg.base != S.Exp1:
            (base, exp) = arg.args
            if base.is_Number and exp.is_Number:
                coeff *= arg
                continue
            if base.is_Number:
                factors.append((base, exp))
                continue
        else:
            (base, exp) = (arg, S.One)
        try:
            (poly, _) = _poly_from_expr(base, opt)
        except PolificationFailed as exc:
            factors.append((exc.expr, exp))
        else:
            func = getattr(poly, method + '_list')
            (_coeff, _factors) = func()
            if _coeff is not S.One:
                if exp.is_Integer:
                    coeff *= _coeff ** exp
                elif _coeff.is_positive:
                    factors.append((_coeff, exp))
                else:
                    _factors.append((_coeff, S.One))
            if exp is S.One:
                factors.extend(_factors)
            elif exp.is_integer:
                factors.extend([(f, k * exp) for (f, k) in _factors])
            else:
                other = []
                for (f, k) in _factors:
                    if f.as_expr().is_positive:
                        factors.append((f, k * exp))
                    else:
                        other.append((f, k))
                factors.append((_factors_product(other), exp))
    if method == 'sqf':
        factors = [(reduce(mul, (f for (f, _) in factors if _ == k)), k) for k in {i for (_, i) in factors}]
    return (coeff, factors)

def _symbolic_factor(expr, opt, method):
    if False:
        print('Hello World!')
    'Helper function for :func:`_factor`. '
    if isinstance(expr, Expr):
        if hasattr(expr, '_eval_factor'):
            return expr._eval_factor()
        (coeff, factors) = _symbolic_factor_list(together(expr, fraction=opt['fraction']), opt, method)
        return _keep_coeff(coeff, _factors_product(factors))
    elif hasattr(expr, 'args'):
        return expr.func(*[_symbolic_factor(arg, opt, method) for arg in expr.args])
    elif hasattr(expr, '__iter__'):
        return expr.__class__([_symbolic_factor(arg, opt, method) for arg in expr])
    else:
        return expr

def _generic_factor_list(expr, gens, args, method):
    if False:
        for i in range(10):
            print('nop')
    'Helper function for :func:`sqf_list` and :func:`factor_list`. '
    options.allowed_flags(args, ['frac', 'polys'])
    opt = options.build_options(gens, args)
    expr = sympify(expr)
    if isinstance(expr, (Expr, Poly)):
        if isinstance(expr, Poly):
            (numer, denom) = (expr, 1)
        else:
            (numer, denom) = together(expr).as_numer_denom()
        (cp, fp) = _symbolic_factor_list(numer, opt, method)
        (cq, fq) = _symbolic_factor_list(denom, opt, method)
        if fq and (not opt.frac):
            raise PolynomialError('a polynomial expected, got %s' % expr)
        _opt = opt.clone({'expand': True})
        for factors in (fp, fq):
            for (i, (f, k)) in enumerate(factors):
                if not f.is_Poly:
                    (f, _) = _poly_from_expr(f, _opt)
                    factors[i] = (f, k)
        fp = _sorted_factors(fp, method)
        fq = _sorted_factors(fq, method)
        if not opt.polys:
            fp = [(f.as_expr(), k) for (f, k) in fp]
            fq = [(f.as_expr(), k) for (f, k) in fq]
        coeff = cp / cq
        if not opt.frac:
            return (coeff, fp)
        else:
            return (coeff, fp, fq)
    else:
        raise PolynomialError('a polynomial expected, got %s' % expr)

def _generic_factor(expr, gens, args, method):
    if False:
        i = 10
        return i + 15
    'Helper function for :func:`sqf` and :func:`factor`. '
    fraction = args.pop('fraction', True)
    options.allowed_flags(args, [])
    opt = options.build_options(gens, args)
    opt['fraction'] = fraction
    return _symbolic_factor(sympify(expr), opt, method)

def to_rational_coeffs(f):
    if False:
        while True:
            i = 10
    "\n    try to transform a polynomial to have rational coefficients\n\n    try to find a transformation ``x = alpha*y``\n\n    ``f(x) = lc*alpha**n * g(y)`` where ``g`` is a polynomial with\n    rational coefficients, ``lc`` the leading coefficient.\n\n    If this fails, try ``x = y + beta``\n    ``f(x) = g(y)``\n\n    Returns ``None`` if ``g`` not found;\n    ``(lc, alpha, None, g)`` in case of rescaling\n    ``(None, None, beta, g)`` in case of translation\n\n    Notes\n    =====\n\n    Currently it transforms only polynomials without roots larger than 2.\n\n    Examples\n    ========\n\n    >>> from sympy import sqrt, Poly, simplify\n    >>> from sympy.polys.polytools import to_rational_coeffs\n    >>> from sympy.abc import x\n    >>> p = Poly(((x**2-1)*(x-2)).subs({x:x*(1 + sqrt(2))}), x, domain='EX')\n    >>> lc, r, _, g = to_rational_coeffs(p)\n    >>> lc, r\n    (7 + 5*sqrt(2), 2 - 2*sqrt(2))\n    >>> g\n    Poly(x**3 + x**2 - 1/4*x - 1/4, x, domain='QQ')\n    >>> r1 = simplify(1/r)\n    >>> Poly(lc*r**3*(g.as_expr()).subs({x:x*r1}), x, domain='EX') == p\n    True\n\n    "
    from sympy.simplify.simplify import simplify

    def _try_rescale(f, f1=None):
        if False:
            print('Hello World!')
        '\n        try rescaling ``x -> alpha*x`` to convert f to a polynomial\n        with rational coefficients.\n        Returns ``alpha, f``; if the rescaling is successful,\n        ``alpha`` is the rescaling factor, and ``f`` is the rescaled\n        polynomial; else ``alpha`` is ``None``.\n        '
        if not len(f.gens) == 1 or not f.gens[0].is_Atom:
            return (None, f)
        n = f.degree()
        lc = f.LC()
        f1 = f1 or f1.monic()
        coeffs = f1.all_coeffs()[1:]
        coeffs = [simplify(coeffx) for coeffx in coeffs]
        if len(coeffs) > 1 and coeffs[-2]:
            rescale1_x = simplify(coeffs[-2] / coeffs[-1])
            coeffs1 = []
            for i in range(len(coeffs)):
                coeffx = simplify(coeffs[i] * rescale1_x ** (i + 1))
                if not coeffx.is_rational:
                    break
                coeffs1.append(coeffx)
            else:
                rescale_x = simplify(1 / rescale1_x)
                x = f.gens[0]
                v = [x ** n]
                for i in range(1, n + 1):
                    v.append(coeffs1[i - 1] * x ** (n - i))
                f = Add(*v)
                f = Poly(f)
                return (lc, rescale_x, f)
        return None

    def _try_translate(f, f1=None):
        if False:
            while True:
                i = 10
        '\n        try translating ``x -> x + alpha`` to convert f to a polynomial\n        with rational coefficients.\n        Returns ``alpha, f``; if the translating is successful,\n        ``alpha`` is the translating factor, and ``f`` is the shifted\n        polynomial; else ``alpha`` is ``None``.\n        '
        if not len(f.gens) == 1 or not f.gens[0].is_Atom:
            return (None, f)
        n = f.degree()
        f1 = f1 or f1.monic()
        coeffs = f1.all_coeffs()[1:]
        c = simplify(coeffs[0])
        if c.is_Add and (not c.is_rational):
            (rat, nonrat) = sift(c.args, lambda z: z.is_rational is True, binary=True)
            alpha = -c.func(*nonrat) / n
            f2 = f1.shift(alpha)
            return (alpha, f2)
        return None

    def _has_square_roots(p):
        if False:
            i = 10
            return i + 15
        '\n        Return True if ``f`` is a sum with square roots but no other root\n        '
        coeffs = p.coeffs()
        has_sq = False
        for y in coeffs:
            for x in Add.make_args(y):
                f = Factors(x).factors
                r = [wx.q for (b, wx) in f.items() if b.is_number and wx.is_Rational and (wx.q >= 2)]
                if not r:
                    continue
                if min(r) == 2:
                    has_sq = True
                if max(r) > 2:
                    return False
        return has_sq
    if f.get_domain().is_EX and _has_square_roots(f):
        f1 = f.monic()
        r = _try_rescale(f, f1)
        if r:
            return (r[0], r[1], None, r[2])
        else:
            r = _try_translate(f, f1)
            if r:
                return (None, None, r[0], r[1])
    return None

def _torational_factor_list(p, x):
    if False:
        while True:
            i = 10
    '\n    helper function to factor polynomial using to_rational_coeffs\n\n    Examples\n    ========\n\n    >>> from sympy.polys.polytools import _torational_factor_list\n    >>> from sympy.abc import x\n    >>> from sympy import sqrt, expand, Mul\n    >>> p = expand(((x**2-1)*(x-2)).subs({x:x*(1 + sqrt(2))}))\n    >>> factors = _torational_factor_list(p, x); factors\n    (-2, [(-x*(1 + sqrt(2))/2 + 1, 1), (-x*(1 + sqrt(2)) - 1, 1), (-x*(1 + sqrt(2)) + 1, 1)])\n    >>> expand(factors[0]*Mul(*[z[0] for z in factors[1]])) == p\n    True\n    >>> p = expand(((x**2-1)*(x-2)).subs({x:x + sqrt(2)}))\n    >>> factors = _torational_factor_list(p, x); factors\n    (1, [(x - 2 + sqrt(2), 1), (x - 1 + sqrt(2), 1), (x + 1 + sqrt(2), 1)])\n    >>> expand(factors[0]*Mul(*[z[0] for z in factors[1]])) == p\n    True\n\n    '
    from sympy.simplify.simplify import simplify
    p1 = Poly(p, x, domain='EX')
    n = p1.degree()
    res = to_rational_coeffs(p1)
    if not res:
        return None
    (lc, r, t, g) = res
    factors = factor_list(g.as_expr())
    if lc:
        c = simplify(factors[0] * lc * r ** n)
        r1 = simplify(1 / r)
        a = []
        for z in factors[1:][0]:
            a.append((simplify(z[0].subs({x: x * r1})), z[1]))
    else:
        c = factors[0]
        a = []
        for z in factors[1:][0]:
            a.append((z[0].subs({x: x - t}), z[1]))
    return (c, a)

@public
def sqf_list(f, *gens, **args):
    if False:
        i = 10
        return i + 15
    '\n    Compute a list of square-free factors of ``f``.\n\n    Examples\n    ========\n\n    >>> from sympy import sqf_list\n    >>> from sympy.abc import x\n\n    >>> sqf_list(2*x**5 + 16*x**4 + 50*x**3 + 76*x**2 + 56*x + 16)\n    (2, [(x + 1, 2), (x + 2, 3)])\n\n    '
    return _generic_factor_list(f, gens, args, method='sqf')

@public
def sqf(f, *gens, **args):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute square-free factorization of ``f``.\n\n    Examples\n    ========\n\n    >>> from sympy import sqf\n    >>> from sympy.abc import x\n\n    >>> sqf(2*x**5 + 16*x**4 + 50*x**3 + 76*x**2 + 56*x + 16)\n    2*(x + 1)**2*(x + 2)**3\n\n    '
    return _generic_factor(f, gens, args, method='sqf')

@public
def factor_list(f, *gens, **args):
    if False:
        return 10
    '\n    Compute a list of irreducible factors of ``f``.\n\n    Examples\n    ========\n\n    >>> from sympy import factor_list\n    >>> from sympy.abc import x, y\n\n    >>> factor_list(2*x**5 + 2*x**4*y + 4*x**3 + 4*x**2*y + 2*x + 2*y)\n    (2, [(x + y, 1), (x**2 + 1, 2)])\n\n    '
    return _generic_factor_list(f, gens, args, method='factor')

@public
def factor(f, *gens, deep=False, **args):
    if False:
        i = 10
        return i + 15
    '\n    Compute the factorization of expression, ``f``, into irreducibles. (To\n    factor an integer into primes, use ``factorint``.)\n\n    There two modes implemented: symbolic and formal. If ``f`` is not an\n    instance of :class:`Poly` and generators are not specified, then the\n    former mode is used. Otherwise, the formal mode is used.\n\n    In symbolic mode, :func:`factor` will traverse the expression tree and\n    factor its components without any prior expansion, unless an instance\n    of :class:`~.Add` is encountered (in this case formal factorization is\n    used). This way :func:`factor` can handle large or symbolic exponents.\n\n    By default, the factorization is computed over the rationals. To factor\n    over other domain, e.g. an algebraic or finite field, use appropriate\n    options: ``extension``, ``modulus`` or ``domain``.\n\n    Examples\n    ========\n\n    >>> from sympy import factor, sqrt, exp\n    >>> from sympy.abc import x, y\n\n    >>> factor(2*x**5 + 2*x**4*y + 4*x**3 + 4*x**2*y + 2*x + 2*y)\n    2*(x + y)*(x**2 + 1)**2\n\n    >>> factor(x**2 + 1)\n    x**2 + 1\n    >>> factor(x**2 + 1, modulus=2)\n    (x + 1)**2\n    >>> factor(x**2 + 1, gaussian=True)\n    (x - I)*(x + I)\n\n    >>> factor(x**2 - 2, extension=sqrt(2))\n    (x - sqrt(2))*(x + sqrt(2))\n\n    >>> factor((x**2 - 1)/(x**2 + 4*x + 4))\n    (x - 1)*(x + 1)/(x + 2)**2\n    >>> factor((x**2 + 4*x + 4)**10000000*(x**2 + 1))\n    (x + 2)**20000000*(x**2 + 1)\n\n    By default, factor deals with an expression as a whole:\n\n    >>> eq = 2**(x**2 + 2*x + 1)\n    >>> factor(eq)\n    2**(x**2 + 2*x + 1)\n\n    If the ``deep`` flag is True then subexpressions will\n    be factored:\n\n    >>> factor(eq, deep=True)\n    2**((x + 1)**2)\n\n    If the ``fraction`` flag is False then rational expressions\n    will not be combined. By default it is True.\n\n    >>> factor(5*x + 3*exp(2 - 7*x), deep=True)\n    (5*x*exp(7*x) + 3*exp(2))*exp(-7*x)\n    >>> factor(5*x + 3*exp(2 - 7*x), deep=True, fraction=False)\n    5*x + 3*exp(2)*exp(-7*x)\n\n    See Also\n    ========\n    sympy.ntheory.factor_.factorint\n\n    '
    f = sympify(f)
    if deep:

        def _try_factor(expr):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Factor, but avoid changing the expression when unable to.\n            '
            fac = factor(expr, *gens, **args)
            if fac.is_Mul or fac.is_Pow:
                return fac
            return expr
        f = bottom_up(f, _try_factor)
        partials = {}
        muladd = f.atoms(Mul, Add)
        for p in muladd:
            fac = factor(p, *gens, **args)
            if (fac.is_Mul or fac.is_Pow) and fac != p:
                partials[p] = fac
        return f.xreplace(partials)
    try:
        return _generic_factor(f, gens, args, method='factor')
    except PolynomialError as msg:
        if not f.is_commutative:
            return factor_nc(f)
        else:
            raise PolynomialError(msg)

@public
def intervals(F, all=False, eps=None, inf=None, sup=None, strict=False, fast=False, sqf=False):
    if False:
        return 10
    '\n    Compute isolating intervals for roots of ``f``.\n\n    Examples\n    ========\n\n    >>> from sympy import intervals\n    >>> from sympy.abc import x\n\n    >>> intervals(x**2 - 3)\n    [((-2, -1), 1), ((1, 2), 1)]\n    >>> intervals(x**2 - 3, eps=1e-2)\n    [((-26/15, -19/11), 1), ((19/11, 26/15), 1)]\n\n    '
    if not hasattr(F, '__iter__'):
        try:
            F = Poly(F)
        except GeneratorsNeeded:
            return []
        return F.intervals(all=all, eps=eps, inf=inf, sup=sup, fast=fast, sqf=sqf)
    else:
        (polys, opt) = parallel_poly_from_expr(F, domain='QQ')
        if len(opt.gens) > 1:
            raise MultivariatePolynomialError
        for (i, poly) in enumerate(polys):
            polys[i] = poly.rep.to_list()
        if eps is not None:
            eps = opt.domain.convert(eps)
            if eps <= 0:
                raise ValueError("'eps' must be a positive rational")
        if inf is not None:
            inf = opt.domain.convert(inf)
        if sup is not None:
            sup = opt.domain.convert(sup)
        intervals = dup_isolate_real_roots_list(polys, opt.domain, eps=eps, inf=inf, sup=sup, strict=strict, fast=fast)
        result = []
        for ((s, t), indices) in intervals:
            (s, t) = (opt.domain.to_sympy(s), opt.domain.to_sympy(t))
            result.append(((s, t), indices))
        return result

@public
def refine_root(f, s, t, eps=None, steps=None, fast=False, check_sqf=False):
    if False:
        return 10
    '\n    Refine an isolating interval of a root to the given precision.\n\n    Examples\n    ========\n\n    >>> from sympy import refine_root\n    >>> from sympy.abc import x\n\n    >>> refine_root(x**2 - 3, 1, 2, eps=1e-2)\n    (19/11, 26/15)\n\n    '
    try:
        F = Poly(f)
        if not isinstance(f, Poly) and (not F.gen.is_Symbol):
            raise PolynomialError('generator must be a Symbol')
    except GeneratorsNeeded:
        raise PolynomialError('Cannot refine a root of %s, not a polynomial' % f)
    return F.refine_root(s, t, eps=eps, steps=steps, fast=fast, check_sqf=check_sqf)

@public
def count_roots(f, inf=None, sup=None):
    if False:
        print('Hello World!')
    '\n    Return the number of roots of ``f`` in ``[inf, sup]`` interval.\n\n    If one of ``inf`` or ``sup`` is complex, it will return the number of roots\n    in the complex rectangle with corners at ``inf`` and ``sup``.\n\n    Examples\n    ========\n\n    >>> from sympy import count_roots, I\n    >>> from sympy.abc import x\n\n    >>> count_roots(x**4 - 4, -3, 3)\n    2\n    >>> count_roots(x**4 - 4, 0, 1 + 3*I)\n    1\n\n    '
    try:
        F = Poly(f, greedy=False)
        if not isinstance(f, Poly) and (not F.gen.is_Symbol):
            raise PolynomialError('generator must be a Symbol')
    except GeneratorsNeeded:
        raise PolynomialError('Cannot count roots of %s, not a polynomial' % f)
    return F.count_roots(inf=inf, sup=sup)

@public
def all_roots(f, multiple=True, radicals=True):
    if False:
        while True:
            i = 10
    '\n    Returns the real and complex roots of ``f`` with multiplicities.\n\n    Explanation\n    ===========\n\n    Finds all real and complex roots of a univariate polynomial with rational\n    coefficients of any degree exactly. The roots are represented in the form\n    given by :func:`~.rootof`. This is equivalent to using :func:`~.rootof` to\n    find each of the indexed roots.\n\n    Examples\n    ========\n\n    >>> from sympy import all_roots\n    >>> from sympy.abc import x, y\n\n    >>> print(all_roots(x**3 + 1))\n    [-1, 1/2 - sqrt(3)*I/2, 1/2 + sqrt(3)*I/2]\n\n    Simple radical formulae are used in some cases but the cubic and quartic\n    formulae are avoided. Instead most non-rational roots will be represented\n    as :class:`~.ComplexRootOf`:\n\n    >>> print(all_roots(x**3 + x + 1))\n    [CRootOf(x**3 + x + 1, 0), CRootOf(x**3 + x + 1, 1), CRootOf(x**3 + x + 1, 2)]\n\n    All roots of any polynomial with rational coefficients of any degree can be\n    represented using :py:class:`~.ComplexRootOf`. The use of\n    :py:class:`~.ComplexRootOf` bypasses limitations on the availability of\n    radical formulae for quintic and higher degree polynomials _[1]:\n\n    >>> p = x**5 - x - 1\n    >>> for r in all_roots(p): print(r)\n    CRootOf(x**5 - x - 1, 0)\n    CRootOf(x**5 - x - 1, 1)\n    CRootOf(x**5 - x - 1, 2)\n    CRootOf(x**5 - x - 1, 3)\n    CRootOf(x**5 - x - 1, 4)\n    >>> [r.evalf(3) for r in all_roots(p)]\n    [1.17, -0.765 - 0.352*I, -0.765 + 0.352*I, 0.181 - 1.08*I, 0.181 + 1.08*I]\n\n    Irrational algebraic or transcendental coefficients cannot currently be\n    handled by :func:`all_roots` (or :func:`~.rootof` more generally):\n\n    >>> from sympy import sqrt, expand\n    >>> p = expand((x - sqrt(2))*(x - sqrt(3)))\n    >>> print(p)\n    x**2 - sqrt(3)*x - sqrt(2)*x + sqrt(6)\n    >>> all_roots(p)\n    Traceback (most recent call last):\n    ...\n    NotImplementedError: sorted roots not supported over EX\n\n    In the case of algebraic or transcendental coefficients\n    :func:`~.ground_roots` might be able to find some roots by factorisation:\n\n    >>> from sympy import ground_roots\n    >>> ground_roots(p, x, extension=True)\n    {sqrt(2): 1, sqrt(3): 1}\n\n    If the coefficients are numeric then :func:`~.nroots` can be used to find\n    all roots approximately:\n\n    >>> from sympy import nroots\n    >>> nroots(p, 5)\n    [1.4142, 1.732]\n\n    If the coefficients are symbolic then :func:`sympy.polys.polyroots.roots`\n    or :func:`~.ground_roots` should be used instead:\n\n    >>> from sympy import roots, ground_roots\n    >>> p = x**2 - 3*x*y + 2*y**2\n    >>> roots(p, x)\n    {y: 1, 2*y: 1}\n    >>> ground_roots(p, x)\n    {y: 1, 2*y: 1}\n\n    Parameters\n    ==========\n\n    f : :class:`~.Expr` or :class:`~.Poly`\n        A univariate polynomial with rational (or ``Float``) coefficients.\n    multiple : ``bool`` (default ``True``).\n        Whether to return a ``list`` of roots or a list of root/multiplicity\n        pairs.\n    radicals : ``bool`` (default ``True``)\n        Use simple radical formulae rather than :py:class:`~.ComplexRootOf` for\n        some irrational roots.\n\n    Returns\n    =======\n\n    A list of :class:`~.Expr` (usually :class:`~.ComplexRootOf`) representing\n    the roots is returned with each root repeated according to its multiplicity\n    as a root of ``f``. The roots are always uniquely ordered with real roots\n    coming before complex roots. The real roots are in increasing order.\n    Complex roots are ordered by increasing real part and then increasing\n    imaginary part.\n\n    If ``multiple=False`` is passed then a list of root/multiplicity pairs is\n    returned instead.\n\n    If ``radicals=False`` is passed then all roots will be represented as\n    either rational numbers or :class:`~.ComplexRootOf`.\n\n    See also\n    ========\n\n    Poly.all_roots:\n        The underlying :class:`Poly` method used by :func:`~.all_roots`.\n    rootof:\n        Compute a single numbered root of a univariate polynomial.\n    real_roots:\n        Compute all the real roots using :func:`~.rootof`.\n    ground_roots:\n        Compute some roots in the ground domain by factorisation.\n    nroots:\n        Compute all roots using approximate numerical techniques.\n    sympy.polys.polyroots.roots:\n        Compute symbolic expressions for roots using radical formulae.\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Abel%E2%80%93Ruffini_theorem\n    '
    try:
        F = Poly(f, greedy=False)
        if not isinstance(f, Poly) and (not F.gen.is_Symbol):
            raise PolynomialError('generator must be a Symbol')
    except GeneratorsNeeded:
        raise PolynomialError('Cannot compute real roots of %s, not a polynomial' % f)
    return F.all_roots(multiple=multiple, radicals=radicals)

@public
def real_roots(f, multiple=True, radicals=True):
    if False:
        while True:
            i = 10
    '\n    Returns the real roots of ``f`` with multiplicities.\n\n    Explanation\n    ===========\n\n    Finds all real roots of a univariate polynomial with rational coefficients\n    of any degree exactly. The roots are represented in the form given by\n    :func:`~.rootof`. This is equivalent to using :func:`~.rootof` or\n    :func:`~.all_roots` and filtering out only the real roots. However if only\n    the real roots are needed then :func:`real_roots` is more efficient than\n    :func:`~.all_roots` because it computes only the real roots and avoids\n    costly complex root isolation routines.\n\n    Examples\n    ========\n\n    >>> from sympy import real_roots\n    >>> from sympy.abc import x, y\n\n    >>> real_roots(2*x**3 - 7*x**2 + 4*x + 4)\n    [-1/2, 2, 2]\n    >>> real_roots(2*x**3 - 7*x**2 + 4*x + 4, multiple=False)\n    [(-1/2, 1), (2, 2)]\n\n    Real roots of any polynomial with rational coefficients of any degree can\n    be represented using :py:class:`~.ComplexRootOf`:\n\n    >>> p = x**9 + 2*x + 2\n    >>> print(real_roots(p))\n    [CRootOf(x**9 + 2*x + 2, 0)]\n    >>> [r.evalf(3) for r in real_roots(p)]\n    [-0.865]\n\n    All rational roots will be returned as rational numbers. Roots of some\n    simple factors will be expressed using radical or other formulae (unless\n    ``radicals=False`` is passed). All other roots will be expressed as\n    :class:`~.ComplexRootOf`.\n\n    >>> p = (x + 7)*(x**2 - 2)*(x**3 + x + 1)\n    >>> print(real_roots(p))\n    [-7, -sqrt(2), CRootOf(x**3 + x + 1, 0), sqrt(2)]\n    >>> print(real_roots(p, radicals=False))\n    [-7, CRootOf(x**2 - 2, 0), CRootOf(x**3 + x + 1, 0), CRootOf(x**2 - 2, 1)]\n\n    All returned root expressions will numerically evaluate to real numbers\n    with no imaginary part. This is in contrast to the expressions generated by\n    the cubic or quartic formulae as used by :func:`~.roots` which suffer from\n    casus irreducibilis [1]_:\n\n    >>> from sympy import roots\n    >>> p = 2*x**3 - 9*x**2 - 6*x + 3\n    >>> [r.evalf(5) for r in roots(p, multiple=True)]\n    [5.0365 - 0.e-11*I, 0.33984 + 0.e-13*I, -0.87636 + 0.e-10*I]\n    >>> [r.evalf(5) for r in real_roots(p, x)]\n    [-0.87636, 0.33984, 5.0365]\n    >>> [r.is_real for r in roots(p, multiple=True)]\n    [None, None, None]\n    >>> [r.is_real for r in real_roots(p)]\n    [True, True, True]\n\n    Using :func:`real_roots` is equivalent to using :func:`~.all_roots` (or\n    :func:`~.rootof`) and filtering out only the real roots:\n\n    >>> from sympy import all_roots\n    >>> r = [r for r in all_roots(p) if r.is_real]\n    >>> real_roots(p) == r\n    True\n\n    If only the real roots are wanted then using :func:`real_roots` is faster\n    than using :func:`~.all_roots`. Using :func:`real_roots` avoids complex root\n    isolation which can be a lot slower than real root isolation especially for\n    polynomials of high degree which typically have many more complex roots\n    than real roots.\n\n    Irrational algebraic or transcendental coefficients cannot be handled by\n    :func:`real_roots` (or :func:`~.rootof` more generally):\n\n    >>> from sympy import sqrt, expand\n    >>> p = expand((x - sqrt(2))*(x - sqrt(3)))\n    >>> print(p)\n    x**2 - sqrt(3)*x - sqrt(2)*x + sqrt(6)\n    >>> real_roots(p)\n    Traceback (most recent call last):\n    ...\n    NotImplementedError: sorted roots not supported over EX\n\n    In the case of algebraic or transcendental coefficients\n    :func:`~.ground_roots` might be able to find some roots by factorisation:\n\n    >>> from sympy import ground_roots\n    >>> ground_roots(p, x, extension=True)\n    {sqrt(2): 1, sqrt(3): 1}\n\n    If the coefficients are numeric then :func:`~.nroots` can be used to find\n    all roots approximately:\n\n    >>> from sympy import nroots\n    >>> nroots(p, 5)\n    [1.4142, 1.732]\n\n    If the coefficients are symbolic then :func:`sympy.polys.polyroots.roots`\n    or :func:`~.ground_roots` should be used instead.\n\n    >>> from sympy import roots, ground_roots\n    >>> p = x**2 - 3*x*y + 2*y**2\n    >>> roots(p, x)\n    {y: 1, 2*y: 1}\n    >>> ground_roots(p, x)\n    {y: 1, 2*y: 1}\n\n    Parameters\n    ==========\n\n    f : :class:`~.Expr` or :class:`~.Poly`\n        A univariate polynomial with rational (or ``Float``) coefficients.\n    multiple : ``bool`` (default ``True``).\n        Whether to return a ``list`` of roots or a list of root/multiplicity\n        pairs.\n    radicals : ``bool`` (default ``True``)\n        Use simple radical formulae rather than :py:class:`~.ComplexRootOf` for\n        some irrational roots.\n\n    Returns\n    =======\n\n    A list of :class:`~.Expr` (usually :class:`~.ComplexRootOf`) representing\n    the real roots is returned. The roots are arranged in increasing order and\n    are repeated according to their multiplicities as roots of ``f``.\n\n    If ``multiple=False`` is passed then a list of root/multiplicity pairs is\n    returned instead.\n\n    If ``radicals=False`` is passed then all roots will be represented as\n    either rational numbers or :class:`~.ComplexRootOf`.\n\n    See also\n    ========\n\n    Poly.real_roots:\n        The underlying :class:`Poly` method used by :func:`real_roots`.\n    rootof:\n        Compute a single numbered root of a univariate polynomial.\n    all_roots:\n        Compute all real and non-real roots using :func:`~.rootof`.\n    ground_roots:\n        Compute some roots in the ground domain by factorisation.\n    nroots:\n        Compute all roots using approximate numerical techniques.\n    sympy.polys.polyroots.roots:\n        Compute symbolic expressions for roots using radical formulae.\n\n    References\n    ==========\n\n    .. [1] https://en.wikipedia.org/wiki/Casus_irreducibilis\n    '
    try:
        F = Poly(f, greedy=False)
        if not isinstance(f, Poly) and (not F.gen.is_Symbol):
            raise PolynomialError('generator must be a Symbol')
    except GeneratorsNeeded:
        raise PolynomialError('Cannot compute real roots of %s, not a polynomial' % f)
    return F.real_roots(multiple=multiple, radicals=radicals)

@public
def nroots(f, n=15, maxsteps=50, cleanup=True):
    if False:
        while True:
            i = 10
    '\n    Compute numerical approximations of roots of ``f``.\n\n    Examples\n    ========\n\n    >>> from sympy import nroots\n    >>> from sympy.abc import x\n\n    >>> nroots(x**2 - 3, n=15)\n    [-1.73205080756888, 1.73205080756888]\n    >>> nroots(x**2 - 3, n=30)\n    [-1.73205080756887729352744634151, 1.73205080756887729352744634151]\n\n    '
    try:
        F = Poly(f, greedy=False)
        if not isinstance(f, Poly) and (not F.gen.is_Symbol):
            raise PolynomialError('generator must be a Symbol')
    except GeneratorsNeeded:
        raise PolynomialError('Cannot compute numerical roots of %s, not a polynomial' % f)
    return F.nroots(n=n, maxsteps=maxsteps, cleanup=cleanup)

@public
def ground_roots(f, *gens, **args):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute roots of ``f`` by factorization in the ground domain.\n\n    Examples\n    ========\n\n    >>> from sympy import ground_roots\n    >>> from sympy.abc import x\n\n    >>> ground_roots(x**6 - 4*x**4 + 4*x**3 - x**2)\n    {0: 2, 1: 2}\n\n    '
    options.allowed_flags(args, [])
    try:
        (F, opt) = poly_from_expr(f, *gens, **args)
        if not isinstance(f, Poly) and (not F.gen.is_Symbol):
            raise PolynomialError('generator must be a Symbol')
    except PolificationFailed as exc:
        raise ComputationFailed('ground_roots', 1, exc)
    return F.ground_roots()

@public
def nth_power_roots_poly(f, n, *gens, **args):
    if False:
        print('Hello World!')
    '\n    Construct a polynomial with n-th powers of roots of ``f``.\n\n    Examples\n    ========\n\n    >>> from sympy import nth_power_roots_poly, factor, roots\n    >>> from sympy.abc import x\n\n    >>> f = x**4 - x**2 + 1\n    >>> g = factor(nth_power_roots_poly(f, 2))\n\n    >>> g\n    (x**2 - x + 1)**2\n\n    >>> R_f = [ (r**2).expand() for r in roots(f) ]\n    >>> R_g = roots(g).keys()\n\n    >>> set(R_f) == set(R_g)\n    True\n\n    '
    options.allowed_flags(args, [])
    try:
        (F, opt) = poly_from_expr(f, *gens, **args)
        if not isinstance(f, Poly) and (not F.gen.is_Symbol):
            raise PolynomialError('generator must be a Symbol')
    except PolificationFailed as exc:
        raise ComputationFailed('nth_power_roots_poly', 1, exc)
    result = F.nth_power_roots_poly(n)
    if not opt.polys:
        return result.as_expr()
    else:
        return result

@public
def cancel(f, *gens, _signsimp=True, **args):
    if False:
        print('Hello World!')
    "\n    Cancel common factors in a rational function ``f``.\n\n    Examples\n    ========\n\n    >>> from sympy import cancel, sqrt, Symbol, together\n    >>> from sympy.abc import x\n    >>> A = Symbol('A', commutative=False)\n\n    >>> cancel((2*x**2 - 2)/(x**2 - 2*x + 1))\n    (2*x + 2)/(x - 1)\n    >>> cancel((sqrt(3) + sqrt(15)*A)/(sqrt(2) + sqrt(10)*A))\n    sqrt(6)/2\n\n    Note: due to automatic distribution of Rationals, a sum divided by an integer\n    will appear as a sum. To recover a rational form use `together` on the result:\n\n    >>> cancel(x/2 + 1)\n    x/2 + 1\n    >>> together(_)\n    (x + 2)/2\n    "
    from sympy.simplify.simplify import signsimp
    from sympy.polys.rings import sring
    options.allowed_flags(args, ['polys'])
    f = sympify(f)
    if _signsimp:
        f = signsimp(f)
    opt = {}
    if 'polys' in args:
        opt['polys'] = args['polys']
    if not isinstance(f, (tuple, Tuple)):
        if f.is_Number or isinstance(f, Relational) or (not isinstance(f, Expr)):
            return f
        f = factor_terms(f, radical=True)
        (p, q) = f.as_numer_denom()
    elif len(f) == 2:
        (p, q) = f
        if isinstance(p, Poly) and isinstance(q, Poly):
            opt['gens'] = p.gens
            opt['domain'] = p.domain
            opt['polys'] = opt.get('polys', True)
        (p, q) = (p.as_expr(), q.as_expr())
    elif isinstance(f, Tuple):
        return factor_terms(f)
    else:
        raise ValueError('unexpected argument: %s' % f)
    from sympy.functions.elementary.piecewise import Piecewise
    try:
        if f.has(Piecewise):
            raise PolynomialError()
        (R, (F, G)) = sring((p, q), *gens, **args)
        if not R.ngens:
            if not isinstance(f, (tuple, Tuple)):
                return f.expand()
            else:
                return (S.One, p, q)
    except PolynomialError as msg:
        if f.is_commutative and (not f.has(Piecewise)):
            raise PolynomialError(msg)
        if f.is_Add or f.is_Mul:
            (c, nc) = sift(f.args, lambda x: x.is_commutative is True and (not x.has(Piecewise)), binary=True)
            nc = [cancel(i) for i in nc]
            return f.func(cancel(f.func(*c)), *nc)
        else:
            reps = []
            pot = preorder_traversal(f)
            next(pot)
            for e in pot:
                if isinstance(e, (tuple, Tuple, BooleanAtom)):
                    continue
                try:
                    reps.append((e, cancel(e)))
                    pot.skip()
                except NotImplementedError:
                    pass
            return f.xreplace(dict(reps))
    (c, (P, Q)) = (1, F.cancel(G))
    if opt.get('polys', False) and 'gens' not in opt:
        opt['gens'] = R.symbols
    if not isinstance(f, (tuple, Tuple)):
        return c * (P.as_expr() / Q.as_expr())
    else:
        (P, Q) = (P.as_expr(), Q.as_expr())
        if not opt.get('polys', False):
            return (c, P, Q)
        else:
            return (c, Poly(P, *gens, **opt), Poly(Q, *gens, **opt))

@public
def reduced(f, G, *gens, **args):
    if False:
        for i in range(10):
            print('nop')
    '\n    Reduces a polynomial ``f`` modulo a set of polynomials ``G``.\n\n    Given a polynomial ``f`` and a set of polynomials ``G = (g_1, ..., g_n)``,\n    computes a set of quotients ``q = (q_1, ..., q_n)`` and the remainder ``r``\n    such that ``f = q_1*g_1 + ... + q_n*g_n + r``, where ``r`` vanishes or ``r``\n    is a completely reduced polynomial with respect to ``G``.\n\n    Examples\n    ========\n\n    >>> from sympy import reduced\n    >>> from sympy.abc import x, y\n\n    >>> reduced(2*x**4 + y**2 - x**2 + y**3, [x**3 - x, y**3 - y])\n    ([2*x, 1], x**2 + y**2 + y)\n\n    '
    options.allowed_flags(args, ['polys', 'auto'])
    try:
        (polys, opt) = parallel_poly_from_expr([f] + list(G), *gens, **args)
    except PolificationFailed as exc:
        raise ComputationFailed('reduced', 0, exc)
    domain = opt.domain
    retract = False
    if opt.auto and domain.is_Ring and (not domain.is_Field):
        opt = opt.clone({'domain': domain.get_field()})
        retract = True
    from sympy.polys.rings import xring
    (_ring, _) = xring(opt.gens, opt.domain, opt.order)
    for (i, poly) in enumerate(polys):
        poly = poly.set_domain(opt.domain).rep.to_dict()
        polys[i] = _ring.from_dict(poly)
    (Q, r) = polys[0].div(polys[1:])
    Q = [Poly._from_dict(dict(q), opt) for q in Q]
    r = Poly._from_dict(dict(r), opt)
    if retract:
        try:
            (_Q, _r) = ([q.to_ring() for q in Q], r.to_ring())
        except CoercionFailed:
            pass
        else:
            (Q, r) = (_Q, _r)
    if not opt.polys:
        return ([q.as_expr() for q in Q], r.as_expr())
    else:
        return (Q, r)

@public
def groebner(F, *gens, **args):
    if False:
        i = 10
        return i + 15
    "\n    Computes the reduced Groebner basis for a set of polynomials.\n\n    Use the ``order`` argument to set the monomial ordering that will be\n    used to compute the basis. Allowed orders are ``lex``, ``grlex`` and\n    ``grevlex``. If no order is specified, it defaults to ``lex``.\n\n    For more information on Groebner bases, see the references and the docstring\n    of :func:`~.solve_poly_system`.\n\n    Examples\n    ========\n\n    Example taken from [1].\n\n    >>> from sympy import groebner\n    >>> from sympy.abc import x, y\n\n    >>> F = [x*y - 2*y, 2*y**2 - x**2]\n\n    >>> groebner(F, x, y, order='lex')\n    GroebnerBasis([x**2 - 2*y**2, x*y - 2*y, y**3 - 2*y], x, y,\n                  domain='ZZ', order='lex')\n    >>> groebner(F, x, y, order='grlex')\n    GroebnerBasis([y**3 - 2*y, x**2 - 2*y**2, x*y - 2*y], x, y,\n                  domain='ZZ', order='grlex')\n    >>> groebner(F, x, y, order='grevlex')\n    GroebnerBasis([y**3 - 2*y, x**2 - 2*y**2, x*y - 2*y], x, y,\n                  domain='ZZ', order='grevlex')\n\n    By default, an improved implementation of the Buchberger algorithm is\n    used. Optionally, an implementation of the F5B algorithm can be used. The\n    algorithm can be set using the ``method`` flag or with the\n    :func:`sympy.polys.polyconfig.setup` function.\n\n    >>> F = [x**2 - x - 1, (2*x - 1) * y - (x**10 - (1 - x)**10)]\n\n    >>> groebner(F, x, y, method='buchberger')\n    GroebnerBasis([x**2 - x - 1, y - 55], x, y, domain='ZZ', order='lex')\n    >>> groebner(F, x, y, method='f5b')\n    GroebnerBasis([x**2 - x - 1, y - 55], x, y, domain='ZZ', order='lex')\n\n    References\n    ==========\n\n    1. [Buchberger01]_\n    2. [Cox97]_\n\n    "
    return GroebnerBasis(F, *gens, **args)

@public
def is_zero_dimensional(F, *gens, **args):
    if False:
        return 10
    "\n    Checks if the ideal generated by a Groebner basis is zero-dimensional.\n\n    The algorithm checks if the set of monomials not divisible by the\n    leading monomial of any element of ``F`` is bounded.\n\n    References\n    ==========\n\n    David A. Cox, John B. Little, Donal O'Shea. Ideals, Varieties and\n    Algorithms, 3rd edition, p. 230\n\n    "
    return GroebnerBasis(F, *gens, **args).is_zero_dimensional

@public
class GroebnerBasis(Basic):
    """Represents a reduced Groebner basis. """

    def __new__(cls, F, *gens, **args):
        if False:
            for i in range(10):
                print('nop')
        'Compute a reduced Groebner basis for a system of polynomials. '
        options.allowed_flags(args, ['polys', 'method'])
        try:
            (polys, opt) = parallel_poly_from_expr(F, *gens, **args)
        except PolificationFailed as exc:
            raise ComputationFailed('groebner', len(F), exc)
        from sympy.polys.rings import PolyRing
        ring = PolyRing(opt.gens, opt.domain, opt.order)
        polys = [ring.from_dict(poly.rep.to_dict()) for poly in polys if poly]
        G = _groebner(polys, ring, method=opt.method)
        G = [Poly._from_dict(g, opt) for g in G]
        return cls._new(G, opt)

    @classmethod
    def _new(cls, basis, options):
        if False:
            i = 10
            return i + 15
        obj = Basic.__new__(cls)
        obj._basis = tuple(basis)
        obj._options = options
        return obj

    @property
    def args(self):
        if False:
            return 10
        basis = (p.as_expr() for p in self._basis)
        return (Tuple(*basis), Tuple(*self._options.gens))

    @property
    def exprs(self):
        if False:
            i = 10
            return i + 15
        return [poly.as_expr() for poly in self._basis]

    @property
    def polys(self):
        if False:
            print('Hello World!')
        return list(self._basis)

    @property
    def gens(self):
        if False:
            print('Hello World!')
        return self._options.gens

    @property
    def domain(self):
        if False:
            for i in range(10):
                print('nop')
        return self._options.domain

    @property
    def order(self):
        if False:
            while True:
                i = 10
        return self._options.order

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self._basis)

    def __iter__(self):
        if False:
            while True:
                i = 10
        if self._options.polys:
            return iter(self.polys)
        else:
            return iter(self.exprs)

    def __getitem__(self, item):
        if False:
            return 10
        if self._options.polys:
            basis = self.polys
        else:
            basis = self.exprs
        return basis[item]

    def __hash__(self):
        if False:
            return 10
        return hash((self._basis, tuple(self._options.items())))

    def __eq__(self, other):
        if False:
            return 10
        if isinstance(other, self.__class__):
            return self._basis == other._basis and self._options == other._options
        elif iterable(other):
            return self.polys == list(other) or self.exprs == list(other)
        else:
            return False

    def __ne__(self, other):
        if False:
            i = 10
            return i + 15
        return not self == other

    @property
    def is_zero_dimensional(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Checks if the ideal generated by a Groebner basis is zero-dimensional.\n\n        The algorithm checks if the set of monomials not divisible by the\n        leading monomial of any element of ``F`` is bounded.\n\n        References\n        ==========\n\n        David A. Cox, John B. Little, Donal O'Shea. Ideals, Varieties and\n        Algorithms, 3rd edition, p. 230\n\n        "

        def single_var(monomial):
            if False:
                i = 10
                return i + 15
            return sum(map(bool, monomial)) == 1
        exponents = Monomial([0] * len(self.gens))
        order = self._options.order
        for poly in self.polys:
            monomial = poly.LM(order=order)
            if single_var(monomial):
                exponents *= monomial
        return all(exponents)

    def fglm(self, order):
        if False:
            while True:
                i = 10
        "\n        Convert a Groebner basis from one ordering to another.\n\n        The FGLM algorithm converts reduced Groebner bases of zero-dimensional\n        ideals from one ordering to another. This method is often used when it\n        is infeasible to compute a Groebner basis with respect to a particular\n        ordering directly.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import x, y\n        >>> from sympy import groebner\n\n        >>> F = [x**2 - 3*y - x + 1, y**2 - 2*x + y - 1]\n        >>> G = groebner(F, x, y, order='grlex')\n\n        >>> list(G.fglm('lex'))\n        [2*x - y**2 - y + 1, y**4 + 2*y**3 - 3*y**2 - 16*y + 7]\n        >>> list(groebner(F, x, y, order='lex'))\n        [2*x - y**2 - y + 1, y**4 + 2*y**3 - 3*y**2 - 16*y + 7]\n\n        References\n        ==========\n\n        .. [1] J.C. Faugere, P. Gianni, D. Lazard, T. Mora (1994). Efficient\n               Computation of Zero-dimensional Groebner Bases by Change of\n               Ordering\n\n        "
        opt = self._options
        src_order = opt.order
        dst_order = monomial_key(order)
        if src_order == dst_order:
            return self
        if not self.is_zero_dimensional:
            raise NotImplementedError('Cannot convert Groebner bases of ideals with positive dimension')
        polys = list(self._basis)
        domain = opt.domain
        opt = opt.clone({'domain': domain.get_field(), 'order': dst_order})
        from sympy.polys.rings import xring
        (_ring, _) = xring(opt.gens, opt.domain, src_order)
        for (i, poly) in enumerate(polys):
            poly = poly.set_domain(opt.domain).rep.to_dict()
            polys[i] = _ring.from_dict(poly)
        G = matrix_fglm(polys, _ring, dst_order)
        G = [Poly._from_dict(dict(g), opt) for g in G]
        if not domain.is_Field:
            G = [g.clear_denoms(convert=True)[1] for g in G]
            opt.domain = domain
        return self._new(G, opt)

    def reduce(self, expr, auto=True):
        if False:
            print('Hello World!')
        '\n        Reduces a polynomial modulo a Groebner basis.\n\n        Given a polynomial ``f`` and a set of polynomials ``G = (g_1, ..., g_n)``,\n        computes a set of quotients ``q = (q_1, ..., q_n)`` and the remainder ``r``\n        such that ``f = q_1*f_1 + ... + q_n*f_n + r``, where ``r`` vanishes or ``r``\n        is a completely reduced polynomial with respect to ``G``.\n\n        Examples\n        ========\n\n        >>> from sympy import groebner, expand\n        >>> from sympy.abc import x, y\n\n        >>> f = 2*x**4 - x**2 + y**3 + y**2\n        >>> G = groebner([x**3 - x, y**3 - y])\n\n        >>> G.reduce(f)\n        ([2*x, 1], x**2 + y**2 + y)\n        >>> Q, r = _\n\n        >>> expand(sum(q*g for q, g in zip(Q, G)) + r)\n        2*x**4 - x**2 + y**3 + y**2\n        >>> _ == f\n        True\n\n        '
        poly = Poly._from_expr(expr, self._options)
        polys = [poly] + list(self._basis)
        opt = self._options
        domain = opt.domain
        retract = False
        if auto and domain.is_Ring and (not domain.is_Field):
            opt = opt.clone({'domain': domain.get_field()})
            retract = True
        from sympy.polys.rings import xring
        (_ring, _) = xring(opt.gens, opt.domain, opt.order)
        for (i, poly) in enumerate(polys):
            poly = poly.set_domain(opt.domain).rep.to_dict()
            polys[i] = _ring.from_dict(poly)
        (Q, r) = polys[0].div(polys[1:])
        Q = [Poly._from_dict(dict(q), opt) for q in Q]
        r = Poly._from_dict(dict(r), opt)
        if retract:
            try:
                (_Q, _r) = ([q.to_ring() for q in Q], r.to_ring())
            except CoercionFailed:
                pass
            else:
                (Q, r) = (_Q, _r)
        if not opt.polys:
            return ([q.as_expr() for q in Q], r.as_expr())
        else:
            return (Q, r)

    def contains(self, poly):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check if ``poly`` belongs the ideal generated by ``self``.\n\n        Examples\n        ========\n\n        >>> from sympy import groebner\n        >>> from sympy.abc import x, y\n\n        >>> f = 2*x**3 + y**3 + 3*y\n        >>> G = groebner([x**2 + y**2 - 1, x*y - 2])\n\n        >>> G.contains(f)\n        True\n        >>> G.contains(f + 1)\n        False\n\n        '
        return self.reduce(poly)[1] == 0

@public
def poly(expr, *gens, **args):
    if False:
        while True:
            i = 10
    "\n    Efficiently transform an expression into a polynomial.\n\n    Examples\n    ========\n\n    >>> from sympy import poly\n    >>> from sympy.abc import x\n\n    >>> poly(x*(x**2 + x - 1)**2)\n    Poly(x**5 + 2*x**4 - x**3 - 2*x**2 + x, x, domain='ZZ')\n\n    "
    options.allowed_flags(args, [])

    def _poly(expr, opt):
        if False:
            i = 10
            return i + 15
        (terms, poly_terms) = ([], [])
        for term in Add.make_args(expr):
            (factors, poly_factors) = ([], [])
            for factor in Mul.make_args(term):
                if factor.is_Add:
                    poly_factors.append(_poly(factor, opt))
                elif factor.is_Pow and factor.base.is_Add and factor.exp.is_Integer and (factor.exp >= 0):
                    poly_factors.append(_poly(factor.base, opt).pow(factor.exp))
                else:
                    factors.append(factor)
            if not poly_factors:
                terms.append(term)
            else:
                product = poly_factors[0]
                for factor in poly_factors[1:]:
                    product = product.mul(factor)
                if factors:
                    factor = Mul(*factors)
                    if factor.is_Number:
                        product = product.mul(factor)
                    else:
                        product = product.mul(Poly._from_expr(factor, opt))
                poly_terms.append(product)
        if not poly_terms:
            result = Poly._from_expr(expr, opt)
        else:
            result = poly_terms[0]
            for term in poly_terms[1:]:
                result = result.add(term)
            if terms:
                term = Add(*terms)
                if term.is_Number:
                    result = result.add(term)
                else:
                    result = result.add(Poly._from_expr(term, opt))
        return result.reorder(*opt.get('gens', ()), **args)
    expr = sympify(expr)
    if expr.is_Poly:
        return Poly(expr, *gens, **args)
    if 'expand' not in args:
        args['expand'] = False
    opt = options.build_options(gens, args)
    return _poly(expr, opt)

def named_poly(n, f, K, name, x, polys):
    if False:
        while True:
            i = 10
    'Common interface to the low-level polynomial generating functions\n    in orthopolys and appellseqs.\n\n    Parameters\n    ==========\n\n    n : int\n        Index of the polynomial, which may or may not equal its degree.\n    f : callable\n        Low-level generating function to use.\n    K : Domain or None\n        Domain in which to perform the computations. If None, use the smallest\n        field containing the rationals and the extra parameters of x (see below).\n    name : str\n        Name of an arbitrary individual polynomial in the sequence generated\n        by f, only used in the error message for invalid n.\n    x : seq\n        The first element of this argument is the main variable of all\n        polynomials in this sequence. Any further elements are extra\n        parameters required by f.\n    polys : bool, optional\n        If True, return a Poly, otherwise (default) return an expression.\n    '
    if n < 0:
        raise ValueError('Cannot generate %s of index %s' % (name, n))
    (head, tail) = (x[0], x[1:])
    if K is None:
        (K, tail) = construct_domain(tail, field=True)
    poly = DMP(f(int(n), *tail, K), K)
    if head is None:
        poly = PurePoly.new(poly, Dummy('x'))
    else:
        poly = Poly.new(poly, head)
    return poly if polys else poly.as_expr()