"""Implementation of :class:`Domain` class. """
from __future__ import annotations
from typing import Any
from sympy.core.numbers import AlgebraicNumber
from sympy.core import Basic, sympify
from sympy.core.sorting import ordered
from sympy.external.gmpy import GROUND_TYPES
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.orderings import lex
from sympy.polys.polyerrors import UnificationFailed, CoercionFailed, DomainError
from sympy.polys.polyutils import _unify_gens, _not_a_coeff
from sympy.utilities import public
from sympy.utilities.iterables import is_sequence

@public
class Domain:
    """Superclass for all domains in the polys domains system.

    See :ref:`polys-domainsintro` for an introductory explanation of the
    domains system.

    The :py:class:`~.Domain` class is an abstract base class for all of the
    concrete domain types. There are many different :py:class:`~.Domain`
    subclasses each of which has an associated ``dtype`` which is a class
    representing the elements of the domain. The coefficients of a
    :py:class:`~.Poly` are elements of a domain which must be a subclass of
    :py:class:`~.Domain`.

    Examples
    ========

    The most common example domains are the integers :ref:`ZZ` and the
    rationals :ref:`QQ`.

    >>> from sympy import Poly, symbols, Domain
    >>> x, y = symbols('x, y')
    >>> p = Poly(x**2 + y)
    >>> p
    Poly(x**2 + y, x, y, domain='ZZ')
    >>> p.domain
    ZZ
    >>> isinstance(p.domain, Domain)
    True
    >>> Poly(x**2 + y/2)
    Poly(x**2 + 1/2*y, x, y, domain='QQ')

    The domains can be used directly in which case the domain object e.g.
    (:ref:`ZZ` or :ref:`QQ`) can be used as a constructor for elements of
    ``dtype``.

    >>> from sympy import ZZ, QQ
    >>> ZZ(2)
    2
    >>> ZZ.dtype  # doctest: +SKIP
    <class 'int'>
    >>> type(ZZ(2))  # doctest: +SKIP
    <class 'int'>
    >>> QQ(1, 2)
    1/2
    >>> type(QQ(1, 2))  # doctest: +SKIP
    <class 'sympy.polys.domains.pythonrational.PythonRational'>

    The corresponding domain elements can be used with the arithmetic
    operations ``+,-,*,**`` and depending on the domain some combination of
    ``/,//,%`` might be usable. For example in :ref:`ZZ` both ``//`` (floor
    division) and ``%`` (modulo division) can be used but ``/`` (true
    division) cannot. Since :ref:`QQ` is a :py:class:`~.Field` its elements
    can be used with ``/`` but ``//`` and ``%`` should not be used. Some
    domains have a :py:meth:`~.Domain.gcd` method.

    >>> ZZ(2) + ZZ(3)
    5
    >>> ZZ(5) // ZZ(2)
    2
    >>> ZZ(5) % ZZ(2)
    1
    >>> QQ(1, 2) / QQ(2, 3)
    3/4
    >>> ZZ.gcd(ZZ(4), ZZ(2))
    2
    >>> QQ.gcd(QQ(2,7), QQ(5,3))
    1/21
    >>> ZZ.is_Field
    False
    >>> QQ.is_Field
    True

    There are also many other domains including:

        1. :ref:`GF(p)` for finite fields of prime order.
        2. :ref:`RR` for real (floating point) numbers.
        3. :ref:`CC` for complex (floating point) numbers.
        4. :ref:`QQ(a)` for algebraic number fields.
        5. :ref:`K[x]` for polynomial rings.
        6. :ref:`K(x)` for rational function fields.
        7. :ref:`EX` for arbitrary expressions.

    Each domain is represented by a domain object and also an implementation
    class (``dtype``) for the elements of the domain. For example the
    :ref:`K[x]` domains are represented by a domain object which is an
    instance of :py:class:`~.PolynomialRing` and the elements are always
    instances of :py:class:`~.PolyElement`. The implementation class
    represents particular types of mathematical expressions in a way that is
    more efficient than a normal SymPy expression which is of type
    :py:class:`~.Expr`. The domain methods :py:meth:`~.Domain.from_sympy` and
    :py:meth:`~.Domain.to_sympy` are used to convert from :py:class:`~.Expr`
    to a domain element and vice versa.

    >>> from sympy import Symbol, ZZ, Expr
    >>> x = Symbol('x')
    >>> K = ZZ[x]           # polynomial ring domain
    >>> K
    ZZ[x]
    >>> type(K)             # class of the domain
    <class 'sympy.polys.domains.polynomialring.PolynomialRing'>
    >>> K.dtype             # class of the elements
    <class 'sympy.polys.rings.PolyElement'>
    >>> p_expr = x**2 + 1   # Expr
    >>> p_expr
    x**2 + 1
    >>> type(p_expr)
    <class 'sympy.core.add.Add'>
    >>> isinstance(p_expr, Expr)
    True
    >>> p_domain = K.from_sympy(p_expr)
    >>> p_domain            # domain element
    x**2 + 1
    >>> type(p_domain)
    <class 'sympy.polys.rings.PolyElement'>
    >>> K.to_sympy(p_domain) == p_expr
    True

    The :py:meth:`~.Domain.convert_from` method is used to convert domain
    elements from one domain to another.

    >>> from sympy import ZZ, QQ
    >>> ez = ZZ(2)
    >>> eq = QQ.convert_from(ez, ZZ)
    >>> type(ez)  # doctest: +SKIP
    <class 'int'>
    >>> type(eq)  # doctest: +SKIP
    <class 'sympy.polys.domains.pythonrational.PythonRational'>

    Elements from different domains should not be mixed in arithmetic or other
    operations: they should be converted to a common domain first.  The domain
    method :py:meth:`~.Domain.unify` is used to find a domain that can
    represent all the elements of two given domains.

    >>> from sympy import ZZ, QQ, symbols
    >>> x, y = symbols('x, y')
    >>> ZZ.unify(QQ)
    QQ
    >>> ZZ[x].unify(QQ)
    QQ[x]
    >>> ZZ[x].unify(QQ[y])
    QQ[x,y]

    If a domain is a :py:class:`~.Ring` then is might have an associated
    :py:class:`~.Field` and vice versa. The :py:meth:`~.Domain.get_field` and
    :py:meth:`~.Domain.get_ring` methods will find or create the associated
    domain.

    >>> from sympy import ZZ, QQ, Symbol
    >>> x = Symbol('x')
    >>> ZZ.has_assoc_Field
    True
    >>> ZZ.get_field()
    QQ
    >>> QQ.has_assoc_Ring
    True
    >>> QQ.get_ring()
    ZZ
    >>> K = QQ[x]
    >>> K
    QQ[x]
    >>> K.get_field()
    QQ(x)

    See also
    ========

    DomainElement: abstract base class for domain elements
    construct_domain: construct a minimal domain for some expressions

    """
    dtype: type | None = None
    'The type (class) of the elements of this :py:class:`~.Domain`:\n\n    >>> from sympy import ZZ, QQ, Symbol\n    >>> ZZ.dtype\n    <class \'int\'>\n    >>> z = ZZ(2)\n    >>> z\n    2\n    >>> type(z)\n    <class \'int\'>\n    >>> type(z) == ZZ.dtype\n    True\n\n    Every domain has an associated **dtype** ("datatype") which is the\n    class of the associated domain elements.\n\n    See also\n    ========\n\n    of_type\n    '
    zero: Any = None
    'The zero element of the :py:class:`~.Domain`:\n\n    >>> from sympy import QQ\n    >>> QQ.zero\n    0\n    >>> QQ.of_type(QQ.zero)\n    True\n\n    See also\n    ========\n\n    of_type\n    one\n    '
    one: Any = None
    'The one element of the :py:class:`~.Domain`:\n\n    >>> from sympy import QQ\n    >>> QQ.one\n    1\n    >>> QQ.of_type(QQ.one)\n    True\n\n    See also\n    ========\n\n    of_type\n    zero\n    '
    is_Ring = False
    'Boolean flag indicating if the domain is a :py:class:`~.Ring`.\n\n    >>> from sympy import ZZ\n    >>> ZZ.is_Ring\n    True\n\n    Basically every :py:class:`~.Domain` represents a ring so this flag is\n    not that useful.\n\n    See also\n    ========\n\n    is_PID\n    is_Field\n    get_ring\n    has_assoc_Ring\n    '
    is_Field = False
    'Boolean flag indicating if the domain is a :py:class:`~.Field`.\n\n    >>> from sympy import ZZ, QQ\n    >>> ZZ.is_Field\n    False\n    >>> QQ.is_Field\n    True\n\n    See also\n    ========\n\n    is_PID\n    is_Ring\n    get_field\n    has_assoc_Field\n    '
    has_assoc_Ring = False
    'Boolean flag indicating if the domain has an associated\n    :py:class:`~.Ring`.\n\n    >>> from sympy import QQ\n    >>> QQ.has_assoc_Ring\n    True\n    >>> QQ.get_ring()\n    ZZ\n\n    See also\n    ========\n\n    is_Field\n    get_ring\n    '
    has_assoc_Field = False
    'Boolean flag indicating if the domain has an associated\n    :py:class:`~.Field`.\n\n    >>> from sympy import ZZ\n    >>> ZZ.has_assoc_Field\n    True\n    >>> ZZ.get_field()\n    QQ\n\n    See also\n    ========\n\n    is_Field\n    get_field\n    '
    is_FiniteField = is_FF = False
    is_IntegerRing = is_ZZ = False
    is_RationalField = is_QQ = False
    is_GaussianRing = is_ZZ_I = False
    is_GaussianField = is_QQ_I = False
    is_RealField = is_RR = False
    is_ComplexField = is_CC = False
    is_AlgebraicField = is_Algebraic = False
    is_PolynomialRing = is_Poly = False
    is_FractionField = is_Frac = False
    is_SymbolicDomain = is_EX = False
    is_SymbolicRawDomain = is_EXRAW = False
    is_FiniteExtension = False
    is_Exact = True
    is_Numerical = False
    is_Simple = False
    is_Composite = False
    is_PID = False
    'Boolean flag indicating if the domain is a `principal ideal domain`_.\n\n    >>> from sympy import ZZ\n    >>> ZZ.has_assoc_Field\n    True\n    >>> ZZ.get_field()\n    QQ\n\n    .. _principal ideal domain: https://en.wikipedia.org/wiki/Principal_ideal_domain\n\n    See also\n    ========\n\n    is_Field\n    get_field\n    '
    has_CharacteristicZero = False
    rep: str | None = None
    alias: str | None = None

    def __init__(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def __str__(self):
        if False:
            print('Hello World!')
        return self.rep

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return str(self)

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash((self.__class__.__name__, self.dtype))

    def new(self, *args):
        if False:
            return 10
        return self.dtype(*args)

    @property
    def tp(self):
        if False:
            while True:
                i = 10
        'Alias for :py:attr:`~.Domain.dtype`'
        return self.dtype

    def __call__(self, *args):
        if False:
            print('Hello World!')
        'Construct an element of ``self`` domain from ``args``. '
        return self.new(*args)

    def normal(self, *args):
        if False:
            while True:
                i = 10
        return self.dtype(*args)

    def convert_from(self, element, base):
        if False:
            for i in range(10):
                print('nop')
        'Convert ``element`` to ``self.dtype`` given the base domain. '
        if base.alias is not None:
            method = 'from_' + base.alias
        else:
            method = 'from_' + base.__class__.__name__
        _convert = getattr(self, method)
        if _convert is not None:
            result = _convert(element, base)
            if result is not None:
                return result
        raise CoercionFailed('Cannot convert %s of type %s from %s to %s' % (element, type(element), base, self))

    def convert(self, element, base=None):
        if False:
            i = 10
            return i + 15
        'Convert ``element`` to ``self.dtype``. '
        if base is not None:
            if _not_a_coeff(element):
                raise CoercionFailed('%s is not in any domain' % element)
            return self.convert_from(element, base)
        if self.of_type(element):
            return element
        if _not_a_coeff(element):
            raise CoercionFailed('%s is not in any domain' % element)
        from sympy.polys.domains import ZZ, QQ, RealField, ComplexField
        if ZZ.of_type(element):
            return self.convert_from(element, ZZ)
        if isinstance(element, int):
            return self.convert_from(ZZ(element), ZZ)
        if GROUND_TYPES != 'python':
            if isinstance(element, ZZ.tp):
                return self.convert_from(element, ZZ)
            if isinstance(element, QQ.tp):
                return self.convert_from(element, QQ)
        if isinstance(element, float):
            parent = RealField(tol=False)
            return self.convert_from(parent(element), parent)
        if isinstance(element, complex):
            parent = ComplexField(tol=False)
            return self.convert_from(parent(element), parent)
        if isinstance(element, DomainElement):
            return self.convert_from(element, element.parent())
        if self.is_Numerical and getattr(element, 'is_ground', False):
            return self.convert(element.LC())
        if isinstance(element, Basic):
            try:
                return self.from_sympy(element)
            except (TypeError, ValueError):
                pass
        elif not is_sequence(element):
            try:
                element = sympify(element, strict=True)
                if isinstance(element, Basic):
                    return self.from_sympy(element)
            except (TypeError, ValueError):
                pass
        raise CoercionFailed('Cannot convert %s of type %s to %s' % (element, type(element), self))

    def of_type(self, element):
        if False:
            print('Hello World!')
        'Check if ``a`` is of type ``dtype``. '
        return isinstance(element, self.tp)

    def __contains__(self, a):
        if False:
            return 10
        'Check if ``a`` belongs to this domain. '
        try:
            if _not_a_coeff(a):
                raise CoercionFailed
            self.convert(a)
        except CoercionFailed:
            return False
        return True

    def to_sympy(self, a):
        if False:
            while True:
                i = 10
        "Convert domain element *a* to a SymPy expression (Expr).\n\n        Explanation\n        ===========\n\n        Convert a :py:class:`~.Domain` element *a* to :py:class:`~.Expr`. Most\n        public SymPy functions work with objects of type :py:class:`~.Expr`.\n        The elements of a :py:class:`~.Domain` have a different internal\n        representation. It is not possible to mix domain elements with\n        :py:class:`~.Expr` so each domain has :py:meth:`~.Domain.to_sympy` and\n        :py:meth:`~.Domain.from_sympy` methods to convert its domain elements\n        to and from :py:class:`~.Expr`.\n\n        Parameters\n        ==========\n\n        a: domain element\n            An element of this :py:class:`~.Domain`.\n\n        Returns\n        =======\n\n        expr: Expr\n            A normal SymPy expression of type :py:class:`~.Expr`.\n\n        Examples\n        ========\n\n        Construct an element of the :ref:`QQ` domain and then convert it to\n        :py:class:`~.Expr`.\n\n        >>> from sympy import QQ, Expr\n        >>> q_domain = QQ(2)\n        >>> q_domain\n        2\n        >>> q_expr = QQ.to_sympy(q_domain)\n        >>> q_expr\n        2\n\n        Although the printed forms look similar these objects are not of the\n        same type.\n\n        >>> isinstance(q_domain, Expr)\n        False\n        >>> isinstance(q_expr, Expr)\n        True\n\n        Construct an element of :ref:`K[x]` and convert to\n        :py:class:`~.Expr`.\n\n        >>> from sympy import Symbol\n        >>> x = Symbol('x')\n        >>> K = QQ[x]\n        >>> x_domain = K.gens[0]  # generator x as a domain element\n        >>> p_domain = x_domain**2/3 + 1\n        >>> p_domain\n        1/3*x**2 + 1\n        >>> p_expr = K.to_sympy(p_domain)\n        >>> p_expr\n        x**2/3 + 1\n\n        The :py:meth:`~.Domain.from_sympy` method is used for the opposite\n        conversion from a normal SymPy expression to a domain element.\n\n        >>> p_domain == p_expr\n        False\n        >>> K.from_sympy(p_expr) == p_domain\n        True\n        >>> K.to_sympy(p_domain) == p_expr\n        True\n        >>> K.from_sympy(K.to_sympy(p_domain)) == p_domain\n        True\n        >>> K.to_sympy(K.from_sympy(p_expr)) == p_expr\n        True\n\n        The :py:meth:`~.Domain.from_sympy` method makes it easier to construct\n        domain elements interactively.\n\n        >>> from sympy import Symbol\n        >>> x = Symbol('x')\n        >>> K = QQ[x]\n        >>> K.from_sympy(x**2/3 + 1)\n        1/3*x**2 + 1\n\n        See also\n        ========\n\n        from_sympy\n        convert_from\n        "
        raise NotImplementedError

    def from_sympy(self, a):
        if False:
            print('Hello World!')
        'Convert a SymPy expression to an element of this domain.\n\n        Explanation\n        ===========\n\n        See :py:meth:`~.Domain.to_sympy` for explanation and examples.\n\n        Parameters\n        ==========\n\n        expr: Expr\n            A normal SymPy expression of type :py:class:`~.Expr`.\n\n        Returns\n        =======\n\n        a: domain element\n            An element of this :py:class:`~.Domain`.\n\n        See also\n        ========\n\n        to_sympy\n        convert_from\n        '
        raise NotImplementedError

    def sum(self, args):
        if False:
            for i in range(10):
                print('nop')
        return sum(args, start=self.zero)

    def from_FF(K1, a, K0):
        if False:
            print('Hello World!')
        'Convert ``ModularInteger(int)`` to ``dtype``. '
        return None

    def from_FF_python(K1, a, K0):
        if False:
            i = 10
            return i + 15
        'Convert ``ModularInteger(int)`` to ``dtype``. '
        return None

    def from_ZZ_python(K1, a, K0):
        if False:
            print('Hello World!')
        'Convert a Python ``int`` object to ``dtype``. '
        return None

    def from_QQ_python(K1, a, K0):
        if False:
            print('Hello World!')
        'Convert a Python ``Fraction`` object to ``dtype``. '
        return None

    def from_FF_gmpy(K1, a, K0):
        if False:
            print('Hello World!')
        'Convert ``ModularInteger(mpz)`` to ``dtype``. '
        return None

    def from_ZZ_gmpy(K1, a, K0):
        if False:
            print('Hello World!')
        'Convert a GMPY ``mpz`` object to ``dtype``. '
        return None

    def from_QQ_gmpy(K1, a, K0):
        if False:
            return 10
        'Convert a GMPY ``mpq`` object to ``dtype``. '
        return None

    def from_RealField(K1, a, K0):
        if False:
            for i in range(10):
                print('nop')
        'Convert a real element object to ``dtype``. '
        return None

    def from_ComplexField(K1, a, K0):
        if False:
            for i in range(10):
                print('nop')
        'Convert a complex element to ``dtype``. '
        return None

    def from_AlgebraicField(K1, a, K0):
        if False:
            return 10
        'Convert an algebraic number to ``dtype``. '
        return None

    def from_PolynomialRing(K1, a, K0):
        if False:
            print('Hello World!')
        'Convert a polynomial to ``dtype``. '
        if a.is_ground:
            return K1.convert(a.LC, K0.dom)

    def from_FractionField(K1, a, K0):
        if False:
            print('Hello World!')
        'Convert a rational function to ``dtype``. '
        return None

    def from_MonogenicFiniteExtension(K1, a, K0):
        if False:
            while True:
                i = 10
        'Convert an ``ExtensionElement`` to ``dtype``. '
        return K1.convert_from(a.rep, K0.ring)

    def from_ExpressionDomain(K1, a, K0):
        if False:
            for i in range(10):
                print('nop')
        'Convert a ``EX`` object to ``dtype``. '
        return K1.from_sympy(a.ex)

    def from_ExpressionRawDomain(K1, a, K0):
        if False:
            return 10
        'Convert a ``EX`` object to ``dtype``. '
        return K1.from_sympy(a)

    def from_GlobalPolynomialRing(K1, a, K0):
        if False:
            i = 10
            return i + 15
        'Convert a polynomial to ``dtype``. '
        if a.degree() <= 0:
            return K1.convert(a.LC(), K0.dom)

    def from_GeneralizedPolynomialRing(K1, a, K0):
        if False:
            return 10
        return K1.from_FractionField(a, K0)

    def unify_with_symbols(K0, K1, symbols):
        if False:
            return 10
        if K0.is_Composite and set(K0.symbols) & set(symbols) or (K1.is_Composite and set(K1.symbols) & set(symbols)):
            raise UnificationFailed('Cannot unify %s with %s, given %s generators' % (K0, K1, tuple(symbols)))
        return K0.unify(K1)

    def unify_composite(K0, K1):
        if False:
            return 10
        'Unify two domains where at least one is composite.'
        K0_ground = K0.dom if K0.is_Composite else K0
        K1_ground = K1.dom if K1.is_Composite else K1
        K0_symbols = K0.symbols if K0.is_Composite else ()
        K1_symbols = K1.symbols if K1.is_Composite else ()
        domain = K0_ground.unify(K1_ground)
        symbols = _unify_gens(K0_symbols, K1_symbols)
        order = K0.order if K0.is_Composite else K1.order
        if (K0.is_FractionField and K1.is_PolynomialRing or (K1.is_FractionField and K0.is_PolynomialRing)) and (not K0_ground.is_Field or not K1_ground.is_Field) and domain.is_Field and domain.has_assoc_Ring:
            domain = domain.get_ring()
        if K0.is_Composite and (not K1.is_Composite or K0.is_FractionField or K1.is_PolynomialRing):
            cls = K0.__class__
        else:
            cls = K1.__class__
        from sympy.polys.domains.old_polynomialring import GlobalPolynomialRing
        if cls == GlobalPolynomialRing:
            return cls(domain, symbols)
        return cls(domain, symbols, order)

    def unify(K0, K1, symbols=None):
        if False:
            return 10
        '\n        Construct a minimal domain that contains elements of ``K0`` and ``K1``.\n\n        Known domains (from smallest to largest):\n\n        - ``GF(p)``\n        - ``ZZ``\n        - ``QQ``\n        - ``RR(prec, tol)``\n        - ``CC(prec, tol)``\n        - ``ALG(a, b, c)``\n        - ``K[x, y, z]``\n        - ``K(x, y, z)``\n        - ``EX``\n\n        '
        if symbols is not None:
            return K0.unify_with_symbols(K1, symbols)
        if K0 == K1:
            return K0
        if not (K0.has_CharacteristicZero and K1.has_CharacteristicZero):
            if K0.characteristic() != K1.characteristic():
                raise UnificationFailed('Cannot unify %s with %s' % (K0, K1))
            return K0.unify_composite(K1)
        if K0.is_EXRAW:
            return K0
        if K1.is_EXRAW:
            return K1
        if K0.is_EX:
            return K0
        if K1.is_EX:
            return K1
        if K0.is_FiniteExtension or K1.is_FiniteExtension:
            if K1.is_FiniteExtension:
                (K0, K1) = (K1, K0)
            if K1.is_FiniteExtension:
                if list(ordered([K0.modulus, K1.modulus]))[1] == K0.modulus:
                    (K0, K1) = (K1, K0)
                return K1.set_domain(K0)
            else:
                K1 = K1.drop(K0.symbol)
                K1 = K0.domain.unify(K1)
                return K0.set_domain(K1)
        if K0.is_Composite or K1.is_Composite:
            return K0.unify_composite(K1)

        def mkinexact(cls, K0, K1):
            if False:
                i = 10
                return i + 15
            prec = max(K0.precision, K1.precision)
            tol = max(K0.tolerance, K1.tolerance)
            return cls(prec=prec, tol=tol)
        if K1.is_ComplexField:
            (K0, K1) = (K1, K0)
        if K0.is_ComplexField:
            if K1.is_ComplexField or K1.is_RealField:
                return mkinexact(K0.__class__, K0, K1)
            else:
                return K0
        if K1.is_RealField:
            (K0, K1) = (K1, K0)
        if K0.is_RealField:
            if K1.is_RealField:
                return mkinexact(K0.__class__, K0, K1)
            elif K1.is_GaussianRing or K1.is_GaussianField:
                from sympy.polys.domains.complexfield import ComplexField
                return ComplexField(prec=K0.precision, tol=K0.tolerance)
            else:
                return K0
        if K1.is_AlgebraicField:
            (K0, K1) = (K1, K0)
        if K0.is_AlgebraicField:
            if K1.is_GaussianRing:
                K1 = K1.get_field()
            if K1.is_GaussianField:
                K1 = K1.as_AlgebraicField()
            if K1.is_AlgebraicField:
                return K0.__class__(K0.dom.unify(K1.dom), *_unify_gens(K0.orig_ext, K1.orig_ext))
            else:
                return K0
        if K0.is_GaussianField:
            return K0
        if K1.is_GaussianField:
            return K1
        if K0.is_GaussianRing:
            if K1.is_RationalField:
                K0 = K0.get_field()
            return K0
        if K1.is_GaussianRing:
            if K0.is_RationalField:
                K1 = K1.get_field()
            return K1
        if K0.is_RationalField:
            return K0
        if K1.is_RationalField:
            return K1
        if K0.is_IntegerRing:
            return K0
        if K1.is_IntegerRing:
            return K1
        from sympy.polys.domains import EX
        return EX

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        'Returns ``True`` if two domains are equivalent. '
        return isinstance(other, Domain) and self.dtype == other.dtype

    def __ne__(self, other):
        if False:
            i = 10
            return i + 15
        'Returns ``False`` if two domains are equivalent. '
        return not self == other

    def map(self, seq):
        if False:
            i = 10
            return i + 15
        'Rersively apply ``self`` to all elements of ``seq``. '
        result = []
        for elt in seq:
            if isinstance(elt, list):
                result.append(self.map(elt))
            else:
                result.append(self(elt))
        return result

    def get_ring(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns a ring associated with ``self``. '
        raise DomainError('there is no ring associated with %s' % self)

    def get_field(self):
        if False:
            while True:
                i = 10
        'Returns a field associated with ``self``. '
        raise DomainError('there is no field associated with %s' % self)

    def get_exact(self):
        if False:
            print('Hello World!')
        'Returns an exact domain associated with ``self``. '
        return self

    def __getitem__(self, symbols):
        if False:
            return 10
        'The mathematical way to make a polynomial ring. '
        if hasattr(symbols, '__iter__'):
            return self.poly_ring(*symbols)
        else:
            return self.poly_ring(symbols)

    def poly_ring(self, *symbols, order=lex):
        if False:
            while True:
                i = 10
        'Returns a polynomial ring, i.e. `K[X]`. '
        from sympy.polys.domains.polynomialring import PolynomialRing
        return PolynomialRing(self, symbols, order)

    def frac_field(self, *symbols, order=lex):
        if False:
            while True:
                i = 10
        'Returns a fraction field, i.e. `K(X)`. '
        from sympy.polys.domains.fractionfield import FractionField
        return FractionField(self, symbols, order)

    def old_poly_ring(self, *symbols, **kwargs):
        if False:
            i = 10
            return i + 15
        'Returns a polynomial ring, i.e. `K[X]`. '
        from sympy.polys.domains.old_polynomialring import PolynomialRing
        return PolynomialRing(self, *symbols, **kwargs)

    def old_frac_field(self, *symbols, **kwargs):
        if False:
            return 10
        'Returns a fraction field, i.e. `K(X)`. '
        from sympy.polys.domains.old_fractionfield import FractionField
        return FractionField(self, *symbols, **kwargs)

    def algebraic_field(self, *extension, alias=None):
        if False:
            while True:
                i = 10
        'Returns an algebraic field, i.e. `K(\\alpha, \\ldots)`. '
        raise DomainError('Cannot create algebraic field over %s' % self)

    def alg_field_from_poly(self, poly, alias=None, root_index=-1):
        if False:
            for i in range(10):
                print('nop')
        '\n        Convenience method to construct an algebraic extension on a root of a\n        polynomial, chosen by root index.\n\n        Parameters\n        ==========\n\n        poly : :py:class:`~.Poly`\n            The polynomial whose root generates the extension.\n        alias : str, optional (default=None)\n            Symbol name for the generator of the extension.\n            E.g. "alpha" or "theta".\n        root_index : int, optional (default=-1)\n            Specifies which root of the polynomial is desired. The ordering is\n            as defined by the :py:class:`~.ComplexRootOf` class. The default of\n            ``-1`` selects the most natural choice in the common cases of\n            quadratic and cyclotomic fields (the square root on the positive\n            real or imaginary axis, resp. $\\mathrm{e}^{2\\pi i/n}$).\n\n        Examples\n        ========\n\n        >>> from sympy import QQ, Poly\n        >>> from sympy.abc import x\n        >>> f = Poly(x**2 - 2)\n        >>> K = QQ.alg_field_from_poly(f)\n        >>> K.ext.minpoly == f\n        True\n        >>> g = Poly(8*x**3 - 6*x - 1)\n        >>> L = QQ.alg_field_from_poly(g, "alpha")\n        >>> L.ext.minpoly == g\n        True\n        >>> L.to_sympy(L([1, 1, 1]))\n        alpha**2 + alpha + 1\n\n        '
        from sympy.polys.rootoftools import CRootOf
        root = CRootOf(poly, root_index)
        alpha = AlgebraicNumber(root, alias=alias)
        return self.algebraic_field(alpha, alias=alias)

    def cyclotomic_field(self, n, ss=False, alias='zeta', gen=None, root_index=-1):
        if False:
            i = 10
            return i + 15
        '\n        Convenience method to construct a cyclotomic field.\n\n        Parameters\n        ==========\n\n        n : int\n            Construct the nth cyclotomic field.\n        ss : boolean, optional (default=False)\n            If True, append *n* as a subscript on the alias string.\n        alias : str, optional (default="zeta")\n            Symbol name for the generator.\n        gen : :py:class:`~.Symbol`, optional (default=None)\n            Desired variable for the cyclotomic polynomial that defines the\n            field. If ``None``, a dummy variable will be used.\n        root_index : int, optional (default=-1)\n            Specifies which root of the polynomial is desired. The ordering is\n            as defined by the :py:class:`~.ComplexRootOf` class. The default of\n            ``-1`` selects the root $\\mathrm{e}^{2\\pi i/n}$.\n\n        Examples\n        ========\n\n        >>> from sympy import QQ, latex\n        >>> K = QQ.cyclotomic_field(5)\n        >>> K.to_sympy(K([-1, 1]))\n        1 - zeta\n        >>> L = QQ.cyclotomic_field(7, True)\n        >>> a = L.to_sympy(L([-1, 1]))\n        >>> print(a)\n        1 - zeta7\n        >>> print(latex(a))\n        1 - \\zeta_{7}\n\n        '
        from sympy.polys.specialpolys import cyclotomic_poly
        if ss:
            alias += str(n)
        return self.alg_field_from_poly(cyclotomic_poly(n, gen), alias=alias, root_index=root_index)

    def inject(self, *symbols):
        if False:
            i = 10
            return i + 15
        'Inject generators into this domain. '
        raise NotImplementedError

    def drop(self, *symbols):
        if False:
            for i in range(10):
                print('nop')
        'Drop generators from this domain. '
        if self.is_Simple:
            return self
        raise NotImplementedError

    def is_zero(self, a):
        if False:
            print('Hello World!')
        'Returns True if ``a`` is zero. '
        return not a

    def is_one(self, a):
        if False:
            return 10
        'Returns True if ``a`` is one. '
        return a == self.one

    def is_positive(self, a):
        if False:
            i = 10
            return i + 15
        'Returns True if ``a`` is positive. '
        return a > 0

    def is_negative(self, a):
        if False:
            print('Hello World!')
        'Returns True if ``a`` is negative. '
        return a < 0

    def is_nonpositive(self, a):
        if False:
            while True:
                i = 10
        'Returns True if ``a`` is non-positive. '
        return a <= 0

    def is_nonnegative(self, a):
        if False:
            i = 10
            return i + 15
        'Returns True if ``a`` is non-negative. '
        return a >= 0

    def canonical_unit(self, a):
        if False:
            print('Hello World!')
        if self.is_negative(a):
            return -self.one
        else:
            return self.one

    def abs(self, a):
        if False:
            return 10
        'Absolute value of ``a``, implies ``__abs__``. '
        return abs(a)

    def neg(self, a):
        if False:
            print('Hello World!')
        'Returns ``a`` negated, implies ``__neg__``. '
        return -a

    def pos(self, a):
        if False:
            i = 10
            return i + 15
        'Returns ``a`` positive, implies ``__pos__``. '
        return +a

    def add(self, a, b):
        if False:
            i = 10
            return i + 15
        'Sum of ``a`` and ``b``, implies ``__add__``.  '
        return a + b

    def sub(self, a, b):
        if False:
            i = 10
            return i + 15
        'Difference of ``a`` and ``b``, implies ``__sub__``.  '
        return a - b

    def mul(self, a, b):
        if False:
            for i in range(10):
                print('nop')
        'Product of ``a`` and ``b``, implies ``__mul__``.  '
        return a * b

    def pow(self, a, b):
        if False:
            for i in range(10):
                print('nop')
        'Raise ``a`` to power ``b``, implies ``__pow__``.  '
        return a ** b

    def exquo(self, a, b):
        if False:
            while True:
                i = 10
        "Exact quotient of *a* and *b*. Analogue of ``a / b``.\n\n        Explanation\n        ===========\n\n        This is essentially the same as ``a / b`` except that an error will be\n        raised if the division is inexact (if there is any remainder) and the\n        result will always be a domain element. When working in a\n        :py:class:`~.Domain` that is not a :py:class:`~.Field` (e.g. :ref:`ZZ`\n        or :ref:`K[x]`) ``exquo`` should be used instead of ``/``.\n\n        The key invariant is that if ``q = K.exquo(a, b)`` (and ``exquo`` does\n        not raise an exception) then ``a == b*q``.\n\n        Examples\n        ========\n\n        We can use ``K.exquo`` instead of ``/`` for exact division.\n\n        >>> from sympy import ZZ\n        >>> ZZ.exquo(ZZ(4), ZZ(2))\n        2\n        >>> ZZ.exquo(ZZ(5), ZZ(2))\n        Traceback (most recent call last):\n            ...\n        ExactQuotientFailed: 2 does not divide 5 in ZZ\n\n        Over a :py:class:`~.Field` such as :ref:`QQ`, division (with nonzero\n        divisor) is always exact so in that case ``/`` can be used instead of\n        :py:meth:`~.Domain.exquo`.\n\n        >>> from sympy import QQ\n        >>> QQ.exquo(QQ(5), QQ(2))\n        5/2\n        >>> QQ(5) / QQ(2)\n        5/2\n\n        Parameters\n        ==========\n\n        a: domain element\n            The dividend\n        b: domain element\n            The divisor\n\n        Returns\n        =======\n\n        q: domain element\n            The exact quotient\n\n        Raises\n        ======\n\n        ExactQuotientFailed: if exact division is not possible.\n        ZeroDivisionError: when the divisor is zero.\n\n        See also\n        ========\n\n        quo: Analogue of ``a // b``\n        rem: Analogue of ``a % b``\n        div: Analogue of ``divmod(a, b)``\n\n        Notes\n        =====\n\n        Since the default :py:attr:`~.Domain.dtype` for :ref:`ZZ` is ``int``\n        (or ``mpz``) division as ``a / b`` should not be used as it would give\n        a ``float`` which is not a domain element.\n\n        >>> ZZ(4) / ZZ(2) # doctest: +SKIP\n        2.0\n        >>> ZZ(5) / ZZ(2) # doctest: +SKIP\n        2.5\n\n        On the other hand with `SYMPY_GROUND_TYPES=flint` elements of :ref:`ZZ`\n        are ``flint.fmpz`` and division would raise an exception:\n\n        >>> ZZ(4) / ZZ(2) # doctest: +SKIP\n        Traceback (most recent call last):\n        ...\n        TypeError: unsupported operand type(s) for /: 'fmpz' and 'fmpz'\n\n        Using ``/`` with :ref:`ZZ` will lead to incorrect results so\n        :py:meth:`~.Domain.exquo` should be used instead.\n\n        "
        raise NotImplementedError

    def quo(self, a, b):
        if False:
            print('Hello World!')
        'Quotient of *a* and *b*. Analogue of ``a // b``.\n\n        ``K.quo(a, b)`` is equivalent to ``K.div(a, b)[0]``. See\n        :py:meth:`~.Domain.div` for more explanation.\n\n        See also\n        ========\n\n        rem: Analogue of ``a % b``\n        div: Analogue of ``divmod(a, b)``\n        exquo: Analogue of ``a / b``\n        '
        raise NotImplementedError

    def rem(self, a, b):
        if False:
            print('Hello World!')
        'Modulo division of *a* and *b*. Analogue of ``a % b``.\n\n        ``K.rem(a, b)`` is equivalent to ``K.div(a, b)[1]``. See\n        :py:meth:`~.Domain.div` for more explanation.\n\n        See also\n        ========\n\n        quo: Analogue of ``a // b``\n        div: Analogue of ``divmod(a, b)``\n        exquo: Analogue of ``a / b``\n        '
        raise NotImplementedError

    def div(self, a, b):
        if False:
            for i in range(10):
                print('nop')
        'Quotient and remainder for *a* and *b*. Analogue of ``divmod(a, b)``\n\n        Explanation\n        ===========\n\n        This is essentially the same as ``divmod(a, b)`` except that is more\n        consistent when working over some :py:class:`~.Field` domains such as\n        :ref:`QQ`. When working over an arbitrary :py:class:`~.Domain` the\n        :py:meth:`~.Domain.div` method should be used instead of ``divmod``.\n\n        The key invariant is that if ``q, r = K.div(a, b)`` then\n        ``a == b*q + r``.\n\n        The result of ``K.div(a, b)`` is the same as the tuple\n        ``(K.quo(a, b), K.rem(a, b))`` except that if both quotient and\n        remainder are needed then it is more efficient to use\n        :py:meth:`~.Domain.div`.\n\n        Examples\n        ========\n\n        We can use ``K.div`` instead of ``divmod`` for floor division and\n        remainder.\n\n        >>> from sympy import ZZ, QQ\n        >>> ZZ.div(ZZ(5), ZZ(2))\n        (2, 1)\n\n        If ``K`` is a :py:class:`~.Field` then the division is always exact\n        with a remainder of :py:attr:`~.Domain.zero`.\n\n        >>> QQ.div(QQ(5), QQ(2))\n        (5/2, 0)\n\n        Parameters\n        ==========\n\n        a: domain element\n            The dividend\n        b: domain element\n            The divisor\n\n        Returns\n        =======\n\n        (q, r): tuple of domain elements\n            The quotient and remainder\n\n        Raises\n        ======\n\n        ZeroDivisionError: when the divisor is zero.\n\n        See also\n        ========\n\n        quo: Analogue of ``a // b``\n        rem: Analogue of ``a % b``\n        exquo: Analogue of ``a / b``\n\n        Notes\n        =====\n\n        If ``gmpy`` is installed then the ``gmpy.mpq`` type will be used as\n        the :py:attr:`~.Domain.dtype` for :ref:`QQ`. The ``gmpy.mpq`` type\n        defines ``divmod`` in a way that is undesirable so\n        :py:meth:`~.Domain.div` should be used instead of ``divmod``.\n\n        >>> a = QQ(1)\n        >>> b = QQ(3, 2)\n        >>> a               # doctest: +SKIP\n        mpq(1,1)\n        >>> b               # doctest: +SKIP\n        mpq(3,2)\n        >>> divmod(a, b)    # doctest: +SKIP\n        (mpz(0), mpq(1,1))\n        >>> QQ.div(a, b)    # doctest: +SKIP\n        (mpq(2,3), mpq(0,1))\n\n        Using ``//`` or ``%`` with :ref:`QQ` will lead to incorrect results so\n        :py:meth:`~.Domain.div` should be used instead.\n\n        '
        raise NotImplementedError

    def invert(self, a, b):
        if False:
            print('Hello World!')
        'Returns inversion of ``a mod b``, implies something. '
        raise NotImplementedError

    def revert(self, a):
        if False:
            while True:
                i = 10
        'Returns ``a**(-1)`` if possible. '
        raise NotImplementedError

    def numer(self, a):
        if False:
            i = 10
            return i + 15
        'Returns numerator of ``a``. '
        raise NotImplementedError

    def denom(self, a):
        if False:
            for i in range(10):
                print('nop')
        'Returns denominator of ``a``. '
        raise NotImplementedError

    def half_gcdex(self, a, b):
        if False:
            return 10
        'Half extended GCD of ``a`` and ``b``. '
        (s, t, h) = self.gcdex(a, b)
        return (s, h)

    def gcdex(self, a, b):
        if False:
            for i in range(10):
                print('nop')
        'Extended GCD of ``a`` and ``b``. '
        raise NotImplementedError

    def cofactors(self, a, b):
        if False:
            while True:
                i = 10
        'Returns GCD and cofactors of ``a`` and ``b``. '
        gcd = self.gcd(a, b)
        cfa = self.quo(a, gcd)
        cfb = self.quo(b, gcd)
        return (gcd, cfa, cfb)

    def gcd(self, a, b):
        if False:
            i = 10
            return i + 15
        'Returns GCD of ``a`` and ``b``. '
        raise NotImplementedError

    def lcm(self, a, b):
        if False:
            i = 10
            return i + 15
        'Returns LCM of ``a`` and ``b``. '
        raise NotImplementedError

    def log(self, a, b):
        if False:
            while True:
                i = 10
        'Returns b-base logarithm of ``a``. '
        raise NotImplementedError

    def sqrt(self, a):
        if False:
            for i in range(10):
                print('nop')
        'Returns a (possibly inexact) square root of ``a``.\n\n        Explanation\n        ===========\n        There is no universal definition of "inexact square root" for all\n        domains. It is not recommended to implement this method for domains\n        other then :ref:`ZZ`.\n\n        See also\n        ========\n        exsqrt\n        '
        raise NotImplementedError

    def is_square(self, a):
        if False:
            while True:
                i = 10
        'Returns whether ``a`` is a square in the domain.\n\n        Explanation\n        ===========\n        Returns ``True`` if there is an element ``b`` in the domain such that\n        ``b * b == a``, otherwise returns ``False``. For inexact domains like\n        :ref:`RR` and :ref:`CC`, a tiny difference in this equality can be\n        tolerated.\n\n        See also\n        ========\n        exsqrt\n        '
        raise NotImplementedError

    def exsqrt(self, a):
        if False:
            i = 10
            return i + 15
        'Principal square root of a within the domain if ``a`` is square.\n\n        Explanation\n        ===========\n        The implementation of this method should return an element ``b`` in the\n        domain such that ``b * b == a``, or ``None`` if there is no such ``b``.\n        For inexact domains like :ref:`RR` and :ref:`CC`, a tiny difference in\n        this equality can be tolerated. The choice of a "principal" square root\n        should follow a consistent rule whenever possible.\n\n        See also\n        ========\n        sqrt, is_square\n        '
        raise NotImplementedError

    def evalf(self, a, prec=None, **options):
        if False:
            while True:
                i = 10
        'Returns numerical approximation of ``a``. '
        return self.to_sympy(a).evalf(prec, **options)
    n = evalf

    def real(self, a):
        if False:
            print('Hello World!')
        return a

    def imag(self, a):
        if False:
            return 10
        return self.zero

    def almosteq(self, a, b, tolerance=None):
        if False:
            return 10
        'Check if ``a`` and ``b`` are almost equal. '
        return a == b

    def characteristic(self):
        if False:
            while True:
                i = 10
        'Return the characteristic of this domain. '
        raise NotImplementedError('characteristic()')
__all__ = ['Domain']