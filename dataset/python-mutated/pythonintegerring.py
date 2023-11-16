"""Implementation of :class:`PythonIntegerRing` class. """
from sympy.core.numbers import int_valued
from sympy.polys.domains.groundtypes import PythonInteger, SymPyInteger, sqrt as python_sqrt, factorial as python_factorial, python_gcdex, python_gcd, python_lcm
from sympy.polys.domains.integerring import IntegerRing
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public

@public
class PythonIntegerRing(IntegerRing):
    """Integer ring based on Python's ``int`` type.

    This will be used as :ref:`ZZ` if ``gmpy`` and ``gmpy2`` are not
    installed. Elements are instances of the standard Python ``int`` type.
    """
    dtype = PythonInteger
    zero = dtype(0)
    one = dtype(1)
    alias = 'ZZ_python'

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        'Allow instantiation of this domain. '

    def to_sympy(self, a):
        if False:
            for i in range(10):
                print('nop')
        'Convert ``a`` to a SymPy object. '
        return SymPyInteger(a)

    def from_sympy(self, a):
        if False:
            return 10
        "Convert SymPy's Integer to ``dtype``. "
        if a.is_Integer:
            return PythonInteger(a.p)
        elif int_valued(a):
            return PythonInteger(int(a))
        else:
            raise CoercionFailed('expected an integer, got %s' % a)

    def from_FF_python(K1, a, K0):
        if False:
            print('Hello World!')
        "Convert ``ModularInteger(int)`` to Python's ``int``. "
        return K0.to_int(a)

    def from_ZZ_python(K1, a, K0):
        if False:
            for i in range(10):
                print('nop')
        "Convert Python's ``int`` to Python's ``int``. "
        return a

    def from_QQ(K1, a, K0):
        if False:
            return 10
        "Convert Python's ``Fraction`` to Python's ``int``. "
        if a.denominator == 1:
            return a.numerator

    def from_QQ_python(K1, a, K0):
        if False:
            i = 10
            return i + 15
        "Convert Python's ``Fraction`` to Python's ``int``. "
        if a.denominator == 1:
            return a.numerator

    def from_FF_gmpy(K1, a, K0):
        if False:
            for i in range(10):
                print('nop')
        "Convert ``ModularInteger(mpz)`` to Python's ``int``. "
        return PythonInteger(K0.to_int(a))

    def from_ZZ_gmpy(K1, a, K0):
        if False:
            return 10
        "Convert GMPY's ``mpz`` to Python's ``int``. "
        return PythonInteger(a)

    def from_QQ_gmpy(K1, a, K0):
        if False:
            for i in range(10):
                print('nop')
        "Convert GMPY's ``mpq`` to Python's ``int``. "
        if a.denom() == 1:
            return PythonInteger(a.numer())

    def from_RealField(K1, a, K0):
        if False:
            return 10
        "Convert mpmath's ``mpf`` to Python's ``int``. "
        (p, q) = K0.to_rational(a)
        if q == 1:
            return PythonInteger(p)

    def gcdex(self, a, b):
        if False:
            while True:
                i = 10
        'Compute extended GCD of ``a`` and ``b``. '
        return python_gcdex(a, b)

    def gcd(self, a, b):
        if False:
            print('Hello World!')
        'Compute GCD of ``a`` and ``b``. '
        return python_gcd(a, b)

    def lcm(self, a, b):
        if False:
            i = 10
            return i + 15
        'Compute LCM of ``a`` and ``b``. '
        return python_lcm(a, b)

    def sqrt(self, a):
        if False:
            while True:
                i = 10
        'Compute square root of ``a``. '
        return python_sqrt(a)

    def factorial(self, a):
        if False:
            while True:
                i = 10
        'Compute factorial of ``a``. '
        return python_factorial(a)