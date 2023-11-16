"""Implementation of :class:`GMPYIntegerRing` class. """
from sympy.polys.domains.groundtypes import GMPYInteger, SymPyInteger, factorial as gmpy_factorial, gmpy_gcdex, gmpy_gcd, gmpy_lcm, sqrt as gmpy_sqrt
from sympy.core.numbers import int_valued
from sympy.polys.domains.integerring import IntegerRing
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public

@public
class GMPYIntegerRing(IntegerRing):
    """Integer ring based on GMPY's ``mpz`` type.

    This will be the implementation of :ref:`ZZ` if ``gmpy`` or ``gmpy2`` is
    installed. Elements will be of type ``gmpy.mpz``.
    """
    dtype = GMPYInteger
    zero = dtype(0)
    one = dtype(1)
    tp = type(one)
    alias = 'ZZ_gmpy'

    def __init__(self):
        if False:
            while True:
                i = 10
        'Allow instantiation of this domain. '

    def to_sympy(self, a):
        if False:
            i = 10
            return i + 15
        'Convert ``a`` to a SymPy object. '
        return SymPyInteger(int(a))

    def from_sympy(self, a):
        if False:
            while True:
                i = 10
        "Convert SymPy's Integer to ``dtype``. "
        if a.is_Integer:
            return GMPYInteger(a.p)
        elif int_valued(a):
            return GMPYInteger(int(a))
        else:
            raise CoercionFailed('expected an integer, got %s' % a)

    def from_FF_python(K1, a, K0):
        if False:
            return 10
        "Convert ``ModularInteger(int)`` to GMPY's ``mpz``. "
        return K0.to_int(a)

    def from_ZZ_python(K1, a, K0):
        if False:
            i = 10
            return i + 15
        "Convert Python's ``int`` to GMPY's ``mpz``. "
        return GMPYInteger(a)

    def from_QQ(K1, a, K0):
        if False:
            while True:
                i = 10
        "Convert Python's ``Fraction`` to GMPY's ``mpz``. "
        if a.denominator == 1:
            return GMPYInteger(a.numerator)

    def from_QQ_python(K1, a, K0):
        if False:
            return 10
        "Convert Python's ``Fraction`` to GMPY's ``mpz``. "
        if a.denominator == 1:
            return GMPYInteger(a.numerator)

    def from_FF_gmpy(K1, a, K0):
        if False:
            for i in range(10):
                print('nop')
        "Convert ``ModularInteger(mpz)`` to GMPY's ``mpz``. "
        return K0.to_int(a)

    def from_ZZ_gmpy(K1, a, K0):
        if False:
            for i in range(10):
                print('nop')
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
            return GMPYInteger(p)

    def from_GaussianIntegerRing(K1, a, K0):
        if False:
            i = 10
            return i + 15
        if a.y == 0:
            return a.x

    def gcdex(self, a, b):
        if False:
            i = 10
            return i + 15
        'Compute extended GCD of ``a`` and ``b``. '
        (h, s, t) = gmpy_gcdex(a, b)
        return (s, t, h)

    def gcd(self, a, b):
        if False:
            while True:
                i = 10
        'Compute GCD of ``a`` and ``b``. '
        return gmpy_gcd(a, b)

    def lcm(self, a, b):
        if False:
            for i in range(10):
                print('nop')
        'Compute LCM of ``a`` and ``b``. '
        return gmpy_lcm(a, b)

    def sqrt(self, a):
        if False:
            while True:
                i = 10
        'Compute square root of ``a``. '
        return gmpy_sqrt(a)

    def factorial(self, a):
        if False:
            for i in range(10):
                print('nop')
        'Compute factorial of ``a``. '
        return gmpy_factorial(a)