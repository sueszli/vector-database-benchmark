"""Implementation of :class:`RealField` class. """
from sympy.external.gmpy import SYMPY_INTS
from sympy.core.numbers import Float
from sympy.polys.domains.field import Field
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.mpelements import MPContext
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public

@public
class RealField(Field, CharacteristicZero, SimpleDomain):
    """Real numbers up to the given precision. """
    rep = 'RR'
    is_RealField = is_RR = True
    is_Exact = False
    is_Numerical = True
    is_PID = False
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
            print('Hello World!')
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
            print('Hello World!')
        return self._context.tolerance

    def __init__(self, prec=_default_precision, dps=None, tol=None):
        if False:
            for i in range(10):
                print('nop')
        context = MPContext(prec, dps, tol, True)
        context._parent = self
        self._context = context
        self._dtype = context.mpf
        self.zero = self.dtype(0)
        self.one = self.dtype(1)

    @property
    def tp(self):
        if False:
            for i in range(10):
                print('nop')
        return self._dtype

    def dtype(self, arg):
        if False:
            while True:
                i = 10
        if isinstance(arg, SYMPY_INTS):
            arg = int(arg)
        return self._dtype(arg)

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return isinstance(other, RealField) and self.precision == other.precision and (self.tolerance == other.tolerance)

    def __hash__(self):
        if False:
            print('Hello World!')
        return hash((self.__class__.__name__, self._dtype, self.precision, self.tolerance))

    def to_sympy(self, element):
        if False:
            for i in range(10):
                print('nop')
        'Convert ``element`` to SymPy number. '
        return Float(element, self.dps)

    def from_sympy(self, expr):
        if False:
            return 10
        "Convert SymPy's number to ``dtype``. "
        number = expr.evalf(n=self.dps)
        if number.is_Number:
            return self.dtype(number)
        else:
            raise CoercionFailed('expected real number, got %s' % expr)

    def from_ZZ(self, element, base):
        if False:
            i = 10
            return i + 15
        return self.dtype(element)

    def from_ZZ_python(self, element, base):
        if False:
            while True:
                i = 10
        return self.dtype(element)

    def from_ZZ_gmpy(self, element, base):
        if False:
            while True:
                i = 10
        return self.dtype(int(element))

    def from_QQ(self, element, base):
        if False:
            print('Hello World!')
        return self.dtype(element.numerator) / int(element.denominator)

    def from_QQ_python(self, element, base):
        if False:
            for i in range(10):
                print('nop')
        return self.dtype(element.numerator) / int(element.denominator)

    def from_QQ_gmpy(self, element, base):
        if False:
            print('Hello World!')
        return self.dtype(int(element.numerator)) / int(element.denominator)

    def from_AlgebraicField(self, element, base):
        if False:
            while True:
                i = 10
        return self.from_sympy(base.to_sympy(element).evalf(self.dps))

    def from_RealField(self, element, base):
        if False:
            i = 10
            return i + 15
        if self == base:
            return element
        else:
            return self.dtype(element)

    def from_ComplexField(self, element, base):
        if False:
            while True:
                i = 10
        if not element.imag:
            return self.dtype(element.real)

    def to_rational(self, element, limit=True):
        if False:
            return 10
        'Convert a real number to rational number. '
        return self._context.to_rational(element, limit)

    def get_ring(self):
        if False:
            i = 10
            return i + 15
        'Returns a ring associated with ``self``. '
        return self

    def get_exact(self):
        if False:
            while True:
                i = 10
        'Returns an exact domain associated with ``self``. '
        from sympy.polys.domains import QQ
        return QQ

    def gcd(self, a, b):
        if False:
            while True:
                i = 10
        'Returns GCD of ``a`` and ``b``. '
        return self.one

    def lcm(self, a, b):
        if False:
            for i in range(10):
                print('nop')
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
            i = 10
            return i + 15
        'Returns ``True`` if ``a >= 0`` and ``False`` otherwise. '
        return a >= 0

    def exsqrt(self, a):
        if False:
            while True:
                i = 10
        'Non-negative square root for ``a >= 0`` and ``None`` otherwise.\n\n        Explanation\n        ===========\n        The square root may be slightly inaccurate due to floating point\n        rounding error.\n        '
        return a ** 0.5 if a >= 0 else None
RR = RealField()