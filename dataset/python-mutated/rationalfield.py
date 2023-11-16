"""Implementation of :class:`RationalField` class. """
from sympy.external.gmpy import MPQ
from sympy.polys.domains.groundtypes import SymPyRational, is_square, sqrtrem
from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.field import Field
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public

@public
class RationalField(Field, CharacteristicZero, SimpleDomain):
    """Abstract base class for the domain :ref:`QQ`.

    The :py:class:`RationalField` class represents the field of rational
    numbers $\\mathbb{Q}$ as a :py:class:`~.Domain` in the domain system.
    :py:class:`RationalField` is a superclass of
    :py:class:`PythonRationalField` and :py:class:`GMPYRationalField` one of
    which will be the implementation for :ref:`QQ` depending on whether either
    of ``gmpy`` or ``gmpy2`` is installed or not.

    See also
    ========

    Domain
    """
    rep = 'QQ'
    alias = 'QQ'
    is_RationalField = is_QQ = True
    is_Numerical = True
    has_assoc_Ring = True
    has_assoc_Field = True
    dtype = MPQ
    zero = dtype(0)
    one = dtype(1)
    tp = type(one)

    def __init__(self):
        if False:
            while True:
                i = 10
        pass

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Returns ``True`` if two domains are equivalent. '
        if isinstance(other, RationalField):
            return True
        else:
            return NotImplemented

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns hash code of ``self``. '
        return hash('QQ')

    def get_ring(self):
        if False:
            i = 10
            return i + 15
        'Returns ring associated with ``self``. '
        from sympy.polys.domains import ZZ
        return ZZ

    def to_sympy(self, a):
        if False:
            while True:
                i = 10
        'Convert ``a`` to a SymPy object. '
        return SymPyRational(int(a.numerator), int(a.denominator))

    def from_sympy(self, a):
        if False:
            i = 10
            return i + 15
        "Convert SymPy's Integer to ``dtype``. "
        if a.is_Rational:
            return MPQ(a.p, a.q)
        elif a.is_Float:
            from sympy.polys.domains import RR
            return MPQ(*map(int, RR.to_rational(a)))
        else:
            raise CoercionFailed('expected `Rational` object, got %s' % a)

    def algebraic_field(self, *extension, alias=None):
        if False:
            return 10
        'Returns an algebraic field, i.e. `\\mathbb{Q}(\\alpha, \\ldots)`.\n\n        Parameters\n        ==========\n\n        *extension : One or more :py:class:`~.Expr`\n            Generators of the extension. These should be expressions that are\n            algebraic over `\\mathbb{Q}`.\n\n        alias : str, :py:class:`~.Symbol`, None, optional (default=None)\n            If provided, this will be used as the alias symbol for the\n            primitive element of the returned :py:class:`~.AlgebraicField`.\n\n        Returns\n        =======\n\n        :py:class:`~.AlgebraicField`\n            A :py:class:`~.Domain` representing the algebraic field extension.\n\n        Examples\n        ========\n\n        >>> from sympy import QQ, sqrt\n        >>> QQ.algebraic_field(sqrt(2))\n        QQ<sqrt(2)>\n        '
        from sympy.polys.domains import AlgebraicField
        return AlgebraicField(self, *extension, alias=alias)

    def from_AlgebraicField(K1, a, K0):
        if False:
            i = 10
            return i + 15
        'Convert a :py:class:`~.ANP` object to :ref:`QQ`.\n\n        See :py:meth:`~.Domain.convert`\n        '
        if a.is_ground:
            return K1.convert(a.LC(), K0.dom)

    def from_ZZ(K1, a, K0):
        if False:
            for i in range(10):
                print('nop')
        'Convert a Python ``int`` object to ``dtype``. '
        return MPQ(a)

    def from_ZZ_python(K1, a, K0):
        if False:
            return 10
        'Convert a Python ``int`` object to ``dtype``. '
        return MPQ(a)

    def from_QQ(K1, a, K0):
        if False:
            return 10
        'Convert a Python ``Fraction`` object to ``dtype``. '
        return MPQ(a.numerator, a.denominator)

    def from_QQ_python(K1, a, K0):
        if False:
            for i in range(10):
                print('nop')
        'Convert a Python ``Fraction`` object to ``dtype``. '
        return MPQ(a.numerator, a.denominator)

    def from_ZZ_gmpy(K1, a, K0):
        if False:
            for i in range(10):
                print('nop')
        'Convert a GMPY ``mpz`` object to ``dtype``. '
        return MPQ(a)

    def from_QQ_gmpy(K1, a, K0):
        if False:
            for i in range(10):
                print('nop')
        'Convert a GMPY ``mpq`` object to ``dtype``. '
        return a

    def from_GaussianRationalField(K1, a, K0):
        if False:
            print('Hello World!')
        'Convert a ``GaussianElement`` object to ``dtype``. '
        if a.y == 0:
            return MPQ(a.x)

    def from_RealField(K1, a, K0):
        if False:
            return 10
        'Convert a mpmath ``mpf`` object to ``dtype``. '
        return MPQ(*map(int, K0.to_rational(a)))

    def exquo(self, a, b):
        if False:
            while True:
                i = 10
        'Exact quotient of ``a`` and ``b``, implies ``__truediv__``.  '
        return MPQ(a) / MPQ(b)

    def quo(self, a, b):
        if False:
            i = 10
            return i + 15
        'Quotient of ``a`` and ``b``, implies ``__truediv__``. '
        return MPQ(a) / MPQ(b)

    def rem(self, a, b):
        if False:
            print('Hello World!')
        'Remainder of ``a`` and ``b``, implies nothing.  '
        return self.zero

    def div(self, a, b):
        if False:
            print('Hello World!')
        'Division of ``a`` and ``b``, implies ``__truediv__``. '
        return (MPQ(a) / MPQ(b), self.zero)

    def numer(self, a):
        if False:
            for i in range(10):
                print('nop')
        'Returns numerator of ``a``. '
        return a.numerator

    def denom(self, a):
        if False:
            for i in range(10):
                print('nop')
        'Returns denominator of ``a``. '
        return a.denominator

    def is_square(self, a):
        if False:
            while True:
                i = 10
        'Return ``True`` if ``a`` is a square.\n\n        Explanation\n        ===========\n        A rational number is a square if and only if there exists\n        a rational number ``b`` such that ``b * b == a``.\n        '
        return is_square(a.numerator) and is_square(a.denominator)

    def exsqrt(self, a):
        if False:
            print('Hello World!')
        'Non-negative square root of ``a`` if ``a`` is a square.\n\n        See also\n        ========\n        is_square\n        '
        if a.numerator < 0:
            return None
        (p_sqrt, p_rem) = sqrtrem(a.numerator)
        if p_rem != 0:
            return None
        (q_sqrt, q_rem) = sqrtrem(a.denominator)
        if q_rem != 0:
            return None
        return MPQ(p_sqrt, q_sqrt)
QQ = RationalField()