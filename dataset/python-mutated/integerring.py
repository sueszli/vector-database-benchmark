"""Implementation of :class:`IntegerRing` class. """
from sympy.external.gmpy import MPZ, GROUND_TYPES
from sympy.core.numbers import int_valued
from sympy.polys.domains.groundtypes import SymPyInteger, factorial, gcdex, gcd, lcm, sqrt, is_square, sqrtrem
from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.ring import Ring
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public
import math

@public
class IntegerRing(Ring, CharacteristicZero, SimpleDomain):
    """The domain ``ZZ`` representing the integers `\\mathbb{Z}`.

    The :py:class:`IntegerRing` class represents the ring of integers as a
    :py:class:`~.Domain` in the domain system. :py:class:`IntegerRing` is a
    super class of :py:class:`PythonIntegerRing` and
    :py:class:`GMPYIntegerRing` one of which will be the implementation for
    :ref:`ZZ` depending on whether or not ``gmpy`` or ``gmpy2`` is installed.

    See also
    ========

    Domain
    """
    rep = 'ZZ'
    alias = 'ZZ'
    dtype = MPZ
    zero = dtype(0)
    one = dtype(1)
    tp = type(one)
    is_IntegerRing = is_ZZ = True
    is_Numerical = True
    is_PID = True
    has_assoc_Ring = True
    has_assoc_Field = True

    def __init__(self):
        if False:
            return 10
        'Allow instantiation of this domain. '

    def __eq__(self, other):
        if False:
            return 10
        'Returns ``True`` if two domains are equivalent. '
        if isinstance(other, IntegerRing):
            return True
        else:
            return NotImplemented

    def __hash__(self):
        if False:
            print('Hello World!')
        'Compute a hash value for this domain. '
        return hash('ZZ')

    def to_sympy(self, a):
        if False:
            print('Hello World!')
        'Convert ``a`` to a SymPy object. '
        return SymPyInteger(int(a))

    def from_sympy(self, a):
        if False:
            for i in range(10):
                print('nop')
        "Convert SymPy's Integer to ``dtype``. "
        if a.is_Integer:
            return MPZ(a.p)
        elif int_valued(a):
            return MPZ(int(a))
        else:
            raise CoercionFailed('expected an integer, got %s' % a)

    def get_field(self):
        if False:
            return 10
        'Return the associated field of fractions :ref:`QQ`\n\n        Returns\n        =======\n\n        :ref:`QQ`:\n            The associated field of fractions :ref:`QQ`, a\n            :py:class:`~.Domain` representing the rational numbers\n            `\\mathbb{Q}`.\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> ZZ.get_field()\n        QQ\n        '
        from sympy.polys.domains import QQ
        return QQ

    def algebraic_field(self, *extension, alias=None):
        if False:
            while True:
                i = 10
        'Returns an algebraic field, i.e. `\\mathbb{Q}(\\alpha, \\ldots)`.\n\n        Parameters\n        ==========\n\n        *extension : One or more :py:class:`~.Expr`.\n            Generators of the extension. These should be expressions that are\n            algebraic over `\\mathbb{Q}`.\n\n        alias : str, :py:class:`~.Symbol`, None, optional (default=None)\n            If provided, this will be used as the alias symbol for the\n            primitive element of the returned :py:class:`~.AlgebraicField`.\n\n        Returns\n        =======\n\n        :py:class:`~.AlgebraicField`\n            A :py:class:`~.Domain` representing the algebraic field extension.\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ, sqrt\n        >>> ZZ.algebraic_field(sqrt(2))\n        QQ<sqrt(2)>\n        '
        return self.get_field().algebraic_field(*extension, alias=alias)

    def from_AlgebraicField(K1, a, K0):
        if False:
            while True:
                i = 10
        'Convert a :py:class:`~.ANP` object to :ref:`ZZ`.\n\n        See :py:meth:`~.Domain.convert`.\n        '
        if a.is_ground:
            return K1.convert(a.LC(), K0.dom)

    def log(self, a, b):
        if False:
            return 10
        'Logarithm of *a* to the base *b*.\n\n        Parameters\n        ==========\n\n        a: number\n        b: number\n\n        Returns\n        =======\n\n        $\\\\lfloor\\log(a, b)\\\\rfloor$:\n            Floor of the logarithm of *a* to the base *b*\n\n        Examples\n        ========\n\n        >>> from sympy import ZZ\n        >>> ZZ.log(ZZ(8), ZZ(2))\n        3\n        >>> ZZ.log(ZZ(9), ZZ(2))\n        3\n\n        Notes\n        =====\n\n        This function uses ``math.log`` which is based on ``float`` so it will\n        fail for large integer arguments.\n        '
        return self.dtype(int(math.log(int(a), b)))

    def from_FF(K1, a, K0):
        if False:
            for i in range(10):
                print('nop')
        "Convert ``ModularInteger(int)`` to GMPY's ``mpz``. "
        return MPZ(K0.to_int(a))

    def from_FF_python(K1, a, K0):
        if False:
            return 10
        "Convert ``ModularInteger(int)`` to GMPY's ``mpz``. "
        return MPZ(K0.to_int(a))

    def from_ZZ(K1, a, K0):
        if False:
            return 10
        "Convert Python's ``int`` to GMPY's ``mpz``. "
        return MPZ(a)

    def from_ZZ_python(K1, a, K0):
        if False:
            print('Hello World!')
        "Convert Python's ``int`` to GMPY's ``mpz``. "
        return MPZ(a)

    def from_QQ(K1, a, K0):
        if False:
            i = 10
            return i + 15
        "Convert Python's ``Fraction`` to GMPY's ``mpz``. "
        if a.denominator == 1:
            return MPZ(a.numerator)

    def from_QQ_python(K1, a, K0):
        if False:
            return 10
        "Convert Python's ``Fraction`` to GMPY's ``mpz``. "
        if a.denominator == 1:
            return MPZ(a.numerator)

    def from_FF_gmpy(K1, a, K0):
        if False:
            print('Hello World!')
        "Convert ``ModularInteger(mpz)`` to GMPY's ``mpz``. "
        return MPZ(K0.to_int(a))

    def from_ZZ_gmpy(K1, a, K0):
        if False:
            i = 10
            return i + 15
        "Convert GMPY's ``mpz`` to GMPY's ``mpz``. "
        return a

    def from_QQ_gmpy(K1, a, K0):
        if False:
            return 10
        "Convert GMPY ``mpq`` to GMPY's ``mpz``. "
        if a.denominator == 1:
            return a.numerator

    def from_RealField(K1, a, K0):
        if False:
            for i in range(10):
                print('nop')
        "Convert mpmath's ``mpf`` to GMPY's ``mpz``. "
        (p, q) = K0.to_rational(a)
        if q == 1:
            return MPZ(int(p))

    def from_GaussianIntegerRing(K1, a, K0):
        if False:
            while True:
                i = 10
        if a.y == 0:
            return a.x

    def from_EX(K1, a, K0):
        if False:
            print('Hello World!')
        "Convert ``Expression`` to GMPY's ``mpz``. "
        if a.is_Integer:
            return K1.from_sympy(a)

    def gcdex(self, a, b):
        if False:
            for i in range(10):
                print('nop')
        'Compute extended GCD of ``a`` and ``b``. '
        (h, s, t) = gcdex(a, b)
        if GROUND_TYPES == 'gmpy':
            return (s, t, h)
        else:
            return (h, s, t)

    def gcd(self, a, b):
        if False:
            for i in range(10):
                print('nop')
        'Compute GCD of ``a`` and ``b``. '
        return gcd(a, b)

    def lcm(self, a, b):
        if False:
            return 10
        'Compute LCM of ``a`` and ``b``. '
        return lcm(a, b)

    def sqrt(self, a):
        if False:
            i = 10
            return i + 15
        'Compute square root of ``a``. '
        return sqrt(a)

    def is_square(self, a):
        if False:
            i = 10
            return i + 15
        'Return ``True`` if ``a`` is a square.\n\n        Explanation\n        ===========\n        An integer is a square if and only if there exists an integer\n        ``b`` such that ``b * b == a``.\n        '
        return is_square(a)

    def exsqrt(self, a):
        if False:
            i = 10
            return i + 15
        'Non-negative square root of ``a`` if ``a`` is a square.\n\n        See also\n        ========\n        is_square\n        '
        if a < 0:
            return None
        (root, rem) = sqrtrem(a)
        if rem != 0:
            return None
        return root

    def factorial(self, a):
        if False:
            for i in range(10):
                print('nop')
        'Compute factorial of ``a``. '
        return factorial(a)
ZZ = IntegerRing()