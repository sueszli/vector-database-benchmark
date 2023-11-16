"""Implementation of :class:`ExpressionDomain` class. """
from sympy.core import sympify, SympifyError
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.field import Field
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.polyutils import PicklableWithSlots
from sympy.utilities import public
eflags = {'deep': False, 'mul': True, 'power_exp': False, 'power_base': False, 'basic': False, 'multinomial': False, 'log': False}

@public
class ExpressionDomain(Field, CharacteristicZero, SimpleDomain):
    """A class for arbitrary expressions. """
    is_SymbolicDomain = is_EX = True

    class Expression(DomainElement, PicklableWithSlots):
        """An arbitrary expression. """
        __slots__ = ('ex',)

        def __init__(self, ex):
            if False:
                return 10
            if not isinstance(ex, self.__class__):
                self.ex = sympify(ex)
            else:
                self.ex = ex.ex

        def __repr__(f):
            if False:
                return 10
            return 'EX(%s)' % repr(f.ex)

        def __str__(f):
            if False:
                return 10
            return 'EX(%s)' % str(f.ex)

        def __hash__(self):
            if False:
                for i in range(10):
                    print('nop')
            return hash((self.__class__.__name__, self.ex))

        def parent(self):
            if False:
                print('Hello World!')
            return EX

        def as_expr(f):
            if False:
                while True:
                    i = 10
            return f.ex

        def numer(f):
            if False:
                return 10
            return f.__class__(f.ex.as_numer_denom()[0])

        def denom(f):
            if False:
                return 10
            return f.__class__(f.ex.as_numer_denom()[1])

        def simplify(f, ex):
            if False:
                for i in range(10):
                    print('nop')
            return f.__class__(ex.cancel().expand(**eflags))

        def __abs__(f):
            if False:
                while True:
                    i = 10
            return f.__class__(abs(f.ex))

        def __neg__(f):
            if False:
                print('Hello World!')
            return f.__class__(-f.ex)

        def _to_ex(f, g):
            if False:
                return 10
            try:
                return f.__class__(g)
            except SympifyError:
                return None

        def __lt__(f, g):
            if False:
                for i in range(10):
                    print('nop')
            return f.ex.sort_key() < g.ex.sort_key()

        def __add__(f, g):
            if False:
                i = 10
                return i + 15
            g = f._to_ex(g)
            if g is None:
                return NotImplemented
            elif g == EX.zero:
                return f
            elif f == EX.zero:
                return g
            else:
                return f.simplify(f.ex + g.ex)

        def __radd__(f, g):
            if False:
                while True:
                    i = 10
            return f.simplify(f.__class__(g).ex + f.ex)

        def __sub__(f, g):
            if False:
                print('Hello World!')
            g = f._to_ex(g)
            if g is None:
                return NotImplemented
            elif g == EX.zero:
                return f
            elif f == EX.zero:
                return -g
            else:
                return f.simplify(f.ex - g.ex)

        def __rsub__(f, g):
            if False:
                i = 10
                return i + 15
            return f.simplify(f.__class__(g).ex - f.ex)

        def __mul__(f, g):
            if False:
                while True:
                    i = 10
            g = f._to_ex(g)
            if g is None:
                return NotImplemented
            if EX.zero in (f, g):
                return EX.zero
            elif f.ex.is_Number and g.ex.is_Number:
                return f.__class__(f.ex * g.ex)
            return f.simplify(f.ex * g.ex)

        def __rmul__(f, g):
            if False:
                return 10
            return f.simplify(f.__class__(g).ex * f.ex)

        def __pow__(f, n):
            if False:
                return 10
            n = f._to_ex(n)
            if n is not None:
                return f.simplify(f.ex ** n.ex)
            else:
                return NotImplemented

        def __truediv__(f, g):
            if False:
                return 10
            g = f._to_ex(g)
            if g is not None:
                return f.simplify(f.ex / g.ex)
            else:
                return NotImplemented

        def __rtruediv__(f, g):
            if False:
                print('Hello World!')
            return f.simplify(f.__class__(g).ex / f.ex)

        def __eq__(f, g):
            if False:
                for i in range(10):
                    print('nop')
            return f.ex == f.__class__(g).ex

        def __ne__(f, g):
            if False:
                print('Hello World!')
            return not f == g

        def __bool__(f):
            if False:
                print('Hello World!')
            return not f.ex.is_zero

        def gcd(f, g):
            if False:
                for i in range(10):
                    print('nop')
            from sympy.polys import gcd
            return f.__class__(gcd(f.ex, f.__class__(g).ex))

        def lcm(f, g):
            if False:
                for i in range(10):
                    print('nop')
            from sympy.polys import lcm
            return f.__class__(lcm(f.ex, f.__class__(g).ex))
    dtype = Expression
    zero = Expression(0)
    one = Expression(1)
    rep = 'EX'
    has_assoc_Ring = False
    has_assoc_Field = True

    def __init__(self):
        if False:
            while True:
                i = 10
        pass

    def __eq__(self, other):
        if False:
            return 10
        if isinstance(other, ExpressionDomain):
            return True
        else:
            return NotImplemented

    def __hash__(self):
        if False:
            return 10
        return hash('EX')

    def to_sympy(self, a):
        if False:
            print('Hello World!')
        'Convert ``a`` to a SymPy object. '
        return a.as_expr()

    def from_sympy(self, a):
        if False:
            for i in range(10):
                print('nop')
        "Convert SymPy's expression to ``dtype``. "
        return self.dtype(a)

    def from_ZZ(K1, a, K0):
        if False:
            for i in range(10):
                print('nop')
        'Convert a Python ``int`` object to ``dtype``. '
        return K1(K0.to_sympy(a))

    def from_ZZ_python(K1, a, K0):
        if False:
            print('Hello World!')
        'Convert a Python ``int`` object to ``dtype``. '
        return K1(K0.to_sympy(a))

    def from_QQ(K1, a, K0):
        if False:
            for i in range(10):
                print('nop')
        'Convert a Python ``Fraction`` object to ``dtype``. '
        return K1(K0.to_sympy(a))

    def from_QQ_python(K1, a, K0):
        if False:
            i = 10
            return i + 15
        'Convert a Python ``Fraction`` object to ``dtype``. '
        return K1(K0.to_sympy(a))

    def from_ZZ_gmpy(K1, a, K0):
        if False:
            return 10
        'Convert a GMPY ``mpz`` object to ``dtype``. '
        return K1(K0.to_sympy(a))

    def from_QQ_gmpy(K1, a, K0):
        if False:
            print('Hello World!')
        'Convert a GMPY ``mpq`` object to ``dtype``. '
        return K1(K0.to_sympy(a))

    def from_GaussianIntegerRing(K1, a, K0):
        if False:
            print('Hello World!')
        'Convert a ``GaussianRational`` object to ``dtype``. '
        return K1(K0.to_sympy(a))

    def from_GaussianRationalField(K1, a, K0):
        if False:
            for i in range(10):
                print('nop')
        'Convert a ``GaussianRational`` object to ``dtype``. '
        return K1(K0.to_sympy(a))

    def from_AlgebraicField(K1, a, K0):
        if False:
            i = 10
            return i + 15
        'Convert an ``ANP`` object to ``dtype``. '
        return K1(K0.to_sympy(a))

    def from_RealField(K1, a, K0):
        if False:
            for i in range(10):
                print('nop')
        'Convert a mpmath ``mpf`` object to ``dtype``. '
        return K1(K0.to_sympy(a))

    def from_ComplexField(K1, a, K0):
        if False:
            for i in range(10):
                print('nop')
        'Convert a mpmath ``mpc`` object to ``dtype``. '
        return K1(K0.to_sympy(a))

    def from_PolynomialRing(K1, a, K0):
        if False:
            i = 10
            return i + 15
        'Convert a ``DMP`` object to ``dtype``. '
        return K1(K0.to_sympy(a))

    def from_FractionField(K1, a, K0):
        if False:
            for i in range(10):
                print('nop')
        'Convert a ``DMF`` object to ``dtype``. '
        return K1(K0.to_sympy(a))

    def from_ExpressionDomain(K1, a, K0):
        if False:
            while True:
                i = 10
        'Convert a ``EX`` object to ``dtype``. '
        return a

    def get_ring(self):
        if False:
            return 10
        'Returns a ring associated with ``self``. '
        return self

    def get_field(self):
        if False:
            i = 10
            return i + 15
        'Returns a field associated with ``self``. '
        return self

    def is_positive(self, a):
        if False:
            i = 10
            return i + 15
        'Returns True if ``a`` is positive. '
        return a.ex.as_coeff_mul()[0].is_positive

    def is_negative(self, a):
        if False:
            while True:
                i = 10
        'Returns True if ``a`` is negative. '
        return a.ex.could_extract_minus_sign()

    def is_nonpositive(self, a):
        if False:
            for i in range(10):
                print('nop')
        'Returns True if ``a`` is non-positive. '
        return a.ex.as_coeff_mul()[0].is_nonpositive

    def is_nonnegative(self, a):
        if False:
            while True:
                i = 10
        'Returns True if ``a`` is non-negative. '
        return a.ex.as_coeff_mul()[0].is_nonnegative

    def numer(self, a):
        if False:
            i = 10
            return i + 15
        'Returns numerator of ``a``. '
        return a.numer()

    def denom(self, a):
        if False:
            while True:
                i = 10
        'Returns denominator of ``a``. '
        return a.denom()

    def gcd(self, a, b):
        if False:
            return 10
        return self(1)

    def lcm(self, a, b):
        if False:
            while True:
                i = 10
        return a.lcm(b)
EX = ExpressionDomain()