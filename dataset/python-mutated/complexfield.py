"""Implementation of :class:`ComplexField` class. """
from sympy.external.gmpy import SYMPY_INTS
from sympy.core.numbers import Float, I
from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.field import Field
from sympy.polys.domains.mpelements import MPContext
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.polyerrors import DomainError, CoercionFailed
from sympy.utilities import public

@public
class ComplexField(Field, CharacteristicZero, SimpleDomain):
    """Complex numbers up to the given precision. """
    rep = 'CC'
    is_ComplexField = is_CC = True
    is_Exact = False
    is_Numerical = True
    has_assoc_Ring = False
    has_assoc_Field = True
    _default_precision = 53

    @property
    def has_default_precision(self):
        if False:
            i = 10
            return i + 15
        return self.precision == self._default_precision

    @property
    def precision(self):
        if False:
            while True:
                i = 10
        return self._context.prec

    @property
    def dps(self):
        if False:
            for i in range(10):
                print('nop')
        return self._context.dps

    @property
    def tolerance(self):
        if False:
            for i in range(10):
                print('nop')
        return self._context.tolerance

    def __init__(self, prec=_default_precision, dps=None, tol=None):
        if False:
            i = 10
            return i + 15
        context = MPContext(prec, dps, tol, False)
        context._parent = self
        self._context = context
        self._dtype = context.mpc
        self.zero = self.dtype(0)
        self.one = self.dtype(1)

    @property
    def tp(self):
        if False:
            for i in range(10):
                print('nop')
        return self._dtype

    def dtype(self, x, y=0):
        if False:
            return 10
        if isinstance(x, SYMPY_INTS):
            x = int(x)
        if isinstance(y, SYMPY_INTS):
            y = int(y)
        return self._dtype(x, y)

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return isinstance(other, ComplexField) and self.precision == other.precision and (self.tolerance == other.tolerance)

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash((self.__class__.__name__, self._dtype, self.precision, self.tolerance))

    def to_sympy(self, element):
        if False:
            return 10
        'Convert ``element`` to SymPy number. '
        return Float(element.real, self.dps) + I * Float(element.imag, self.dps)

    def from_sympy(self, expr):
        if False:
            return 10
        "Convert SymPy's number to ``dtype``. "
        number = expr.evalf(n=self.dps)
        (real, imag) = number.as_real_imag()
        if real.is_Number and imag.is_Number:
            return self.dtype(real, imag)
        else:
            raise CoercionFailed('expected complex number, got %s' % expr)

    def from_ZZ(self, element, base):
        if False:
            return 10
        return self.dtype(element)

    def from_ZZ_gmpy(self, element, base):
        if False:
            print('Hello World!')
        return self.dtype(int(element))

    def from_ZZ_python(self, element, base):
        if False:
            for i in range(10):
                print('nop')
        return self.dtype(element)

    def from_QQ(self, element, base):
        if False:
            return 10
        return self.dtype(int(element.numerator)) / int(element.denominator)

    def from_QQ_python(self, element, base):
        if False:
            for i in range(10):
                print('nop')
        return self.dtype(element.numerator) / element.denominator

    def from_QQ_gmpy(self, element, base):
        if False:
            for i in range(10):
                print('nop')
        return self.dtype(int(element.numerator)) / int(element.denominator)

    def from_GaussianIntegerRing(self, element, base):
        if False:
            while True:
                i = 10
        return self.dtype(int(element.x), int(element.y))

    def from_GaussianRationalField(self, element, base):
        if False:
            for i in range(10):
                print('nop')
        x = element.x
        y = element.y
        return self.dtype(int(x.numerator)) / int(x.denominator) + self.dtype(0, int(y.numerator)) / int(y.denominator)

    def from_AlgebraicField(self, element, base):
        if False:
            i = 10
            return i + 15
        return self.from_sympy(base.to_sympy(element).evalf(self.dps))

    def from_RealField(self, element, base):
        if False:
            while True:
                i = 10
        return self.dtype(element)

    def from_ComplexField(self, element, base):
        if False:
            return 10
        if self == base:
            return element
        else:
            return self.dtype(element)

    def get_ring(self):
        if False:
            print('Hello World!')
        'Returns a ring associated with ``self``. '
        raise DomainError('there is no ring associated with %s' % self)

    def get_exact(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns an exact domain associated with ``self``. '
        raise DomainError('there is no exact domain associated with %s' % self)

    def is_negative(self, element):
        if False:
            return 10
        'Returns ``False`` for any ``ComplexElement``. '
        return False

    def is_positive(self, element):
        if False:
            while True:
                i = 10
        'Returns ``False`` for any ``ComplexElement``. '
        return False

    def is_nonnegative(self, element):
        if False:
            i = 10
            return i + 15
        'Returns ``False`` for any ``ComplexElement``. '
        return False

    def is_nonpositive(self, element):
        if False:
            for i in range(10):
                print('nop')
        'Returns ``False`` for any ``ComplexElement``. '
        return False

    def gcd(self, a, b):
        if False:
            print('Hello World!')
        'Returns GCD of ``a`` and ``b``. '
        return self.one

    def lcm(self, a, b):
        if False:
            return 10
        'Returns LCM of ``a`` and ``b``. '
        return a * b

    def almosteq(self, a, b, tolerance=None):
        if False:
            for i in range(10):
                print('nop')
        'Check if ``a`` and ``b`` are almost equal. '
        return self._context.almosteq(a, b, tolerance)

    def is_square(self, a):
        if False:
            print('Hello World!')
        'Returns ``True``. Every complex number has a complex square root.'
        return True

    def exsqrt(self, a):
        if False:
            i = 10
            return i + 15
        'Returns the principal complex square root of ``a``.\n\n        Explanation\n        ===========\n        The argument of the principal square root is always within\n        $(-\\frac{\\pi}{2}, \\frac{\\pi}{2}]$. The square root may be\n        slightly inaccurate due to floating point rounding error.\n        '
        return a ** 0.5
CC = ComplexField()