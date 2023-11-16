from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import Function, Lambda
from sympy.core.mul import Mul
from sympy.core.numbers import Float, I, Integer, Rational, pi, oo
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.logic.boolalg import false, Or, true, Xor
from sympy.matrices.dense import Matrix
from sympy.parsing.sympy_parser import null
from sympy.polys.polytools import Poly
from sympy.printing.repr import srepr
from sympy.sets.fancysets import Range
from sympy.sets.sets import Interval
from sympy.abc import x, y
from sympy.core.sympify import sympify, _sympify, SympifyError, kernS, CantSympify, converter
from sympy.core.decorators import _sympifyit
from sympy.external import import_module
from sympy.testing.pytest import raises, XFAIL, skip
from sympy.utilities.decorator import conserve_mpmath_dps
from sympy.geometry import Point, Line
from sympy.functions.combinatorial.factorials import factorial, factorial2
from sympy.abc import _clash, _clash1, _clash2
from sympy.external.gmpy import gmpy as _gmpy, flint as _flint
from sympy.sets import FiniteSet, EmptySet
from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray
import mpmath
from collections import defaultdict, OrderedDict
numpy = import_module('numpy')

def test_issue_3538():
    if False:
        i = 10
        return i + 15
    v = sympify('exp(x)')
    assert v == exp(x)
    assert type(v) == type(exp(x))
    assert str(type(v)) == str(type(exp(x)))

def test_sympify1():
    if False:
        while True:
            i = 10
    assert sympify('x') == Symbol('x')
    assert sympify('   x') == Symbol('x')
    assert sympify('   x   ') == Symbol('x')
    assert sympify('--.5') == 0.5
    assert sympify('-1/2') == -S.Half
    assert sympify('-+--.5') == -0.5
    assert sympify('-.[3]') == Rational(-1, 3)
    assert sympify('.[3]') == Rational(1, 3)
    assert sympify('+.[3]') == Rational(1, 3)
    assert sympify('+0.[3]*10**-2') == Rational(1, 300)
    assert sympify('.[052631578947368421]') == Rational(1, 19)
    assert sympify('.0[526315789473684210]') == Rational(1, 19)
    assert sympify('.034[56]') == Rational(1711, 49500)
    assert sympify('1.22[345]', rational=True) == 1 + Rational(22, 100) + Rational(345, 99900)
    assert sympify('2/2.6', rational=True) == Rational(10, 13)
    assert sympify('2.6/2', rational=True) == Rational(13, 10)
    assert sympify('2.6e2/17', rational=True) == Rational(260, 17)
    assert sympify('2.6e+2/17', rational=True) == Rational(260, 17)
    assert sympify('2.6e-2/17', rational=True) == Rational(26, 17000)
    assert sympify('2.1+3/4', rational=True) == Rational(21, 10) + Rational(3, 4)
    assert sympify('2.234456', rational=True) == Rational(279307, 125000)
    assert sympify('2.234456e23', rational=True) == 223445600000000000000000
    assert sympify('2.234456e-23', rational=True) == Rational(279307, 12500000000000000000000000000)
    assert sympify('-2.234456e-23', rational=True) == Rational(-279307, 12500000000000000000000000000)
    assert sympify('12345678901/17', rational=True) == Rational(12345678901, 17)
    assert sympify('1/.3 + x', rational=True) == Rational(10, 3) + x
    assert sympify('222222222222/11111111111') == Rational(222222222222, 11111111111)
    assert sympify('1/.2[123456789012]') == Rational(333333333333, 70781892967)
    assert sympify('.1234567890123456', rational=True) == Rational(19290123283179, 156250000000000)

def test_sympify_Fraction():
    if False:
        print('Hello World!')
    try:
        import fractions
    except ImportError:
        pass
    else:
        value = sympify(fractions.Fraction(101, 127))
        assert value == Rational(101, 127) and type(value) is Rational

def test_sympify_gmpy():
    if False:
        for i in range(10):
            print('nop')
    if _gmpy is not None:
        import gmpy2
        value = sympify(gmpy2.mpz(1000001))
        assert value == Integer(1000001) and type(value) is Integer
        value = sympify(gmpy2.mpq(101, 127))
        assert value == Rational(101, 127) and type(value) is Rational

def test_sympify_flint():
    if False:
        while True:
            i = 10
    if _flint is not None:
        import flint
        value = sympify(flint.fmpz(1000001))
        assert value == Integer(1000001) and type(value) is Integer
        value = sympify(flint.fmpq(101, 127))
        assert value == Rational(101, 127) and type(value) is Rational

@conserve_mpmath_dps
def test_sympify_mpmath():
    if False:
        while True:
            i = 10
    value = sympify(mpmath.mpf(1.0))
    assert value == Float(1.0) and type(value) is Float
    mpmath.mp.dps = 12
    assert sympify(mpmath.pi).epsilon_eq(Float('3.14159265359'), Float('1e-12')) == True
    assert sympify(mpmath.pi).epsilon_eq(Float('3.14159265359'), Float('1e-13')) == False
    mpmath.mp.dps = 6
    assert sympify(mpmath.pi).epsilon_eq(Float('3.14159'), Float('1e-5')) == True
    assert sympify(mpmath.pi).epsilon_eq(Float('3.14159'), Float('1e-6')) == False
    mpmath.mp.dps = 15
    assert sympify(mpmath.mpc(1.0 + 2j)) == Float(1.0) + Float(2.0) * I

def test_sympify2():
    if False:
        for i in range(10):
            print('nop')

    class A:

        def _sympy_(self):
            if False:
                for i in range(10):
                    print('nop')
            return Symbol('x') ** 3
    a = A()
    assert _sympify(a) == x ** 3
    assert sympify(a) == x ** 3
    assert a == x ** 3

def test_sympify3():
    if False:
        while True:
            i = 10
    assert sympify('x**3') == x ** 3
    assert sympify('x^3') == x ** 3
    assert sympify('1/2') == Integer(1) / 2
    raises(SympifyError, lambda : _sympify('x**3'))
    raises(SympifyError, lambda : _sympify('1/2'))

def test_sympify_keywords():
    if False:
        print('Hello World!')
    raises(SympifyError, lambda : sympify('if'))
    raises(SympifyError, lambda : sympify('for'))
    raises(SympifyError, lambda : sympify('while'))
    raises(SympifyError, lambda : sympify('lambda'))

def test_sympify_float():
    if False:
        i = 10
        return i + 15
    assert sympify('1e-64') != 0
    assert sympify('1e-20000') != 0

def test_sympify_bool():
    if False:
        return 10
    assert sympify(True) is true
    assert sympify(False) is false

def test_sympyify_iterables():
    if False:
        i = 10
        return i + 15
    ans = [Rational(3, 10), Rational(1, 5)]
    assert sympify(['.3', '.2'], rational=True) == ans
    assert sympify({'x': 0, 'y': 1}) == {x: 0, y: 1}
    assert sympify(['1', '2', ['3', '4']]) == [S(1), S(2), [S(3), S(4)]]

@XFAIL
def test_issue_16772():
    if False:
        return 10
    ans = [Rational(3, 10), Rational(1, 5)]
    assert sympify(('.3', '.2'), rational=True) == Tuple(*ans)

def test_issue_16859():
    if False:
        while True:
            i = 10

    class no(float, CantSympify):
        pass
    raises(SympifyError, lambda : sympify(no(1.2)))

def test_sympify4():
    if False:
        while True:
            i = 10

    class A:

        def _sympy_(self):
            if False:
                print('Hello World!')
            return Symbol('x')
    a = A()
    assert _sympify(a) ** 3 == x ** 3
    assert sympify(a) ** 3 == x ** 3
    assert a == x

def test_sympify_text():
    if False:
        while True:
            i = 10
    assert sympify('some') == Symbol('some')
    assert sympify('core') == Symbol('core')
    assert sympify('True') is True
    assert sympify('False') is False
    assert sympify('Poly') == Poly
    assert sympify('sin') == sin

def test_sympify_function():
    if False:
        while True:
            i = 10
    assert sympify('factor(x**2-1, x)') == -(1 - x) * (x + 1)
    assert sympify('sin(pi/2)*cos(pi)') == -Integer(1)

def test_sympify_poly():
    if False:
        for i in range(10):
            print('nop')
    p = Poly(x ** 2 + x + 1, x)
    assert _sympify(p) is p
    assert sympify(p) is p

def test_sympify_factorial():
    if False:
        for i in range(10):
            print('nop')
    assert sympify('x!') == factorial(x)
    assert sympify('(x+1)!') == factorial(x + 1)
    assert sympify('(1 + y*(x + 1))!') == factorial(1 + y * (x + 1))
    assert sympify('(1 + y*(x + 1)!)^2') == (1 + y * factorial(x + 1)) ** 2
    assert sympify('y*x!') == y * factorial(x)
    assert sympify('x!!') == factorial2(x)
    assert sympify('(x+1)!!') == factorial2(x + 1)
    assert sympify('(1 + y*(x + 1))!!') == factorial2(1 + y * (x + 1))
    assert sympify('(1 + y*(x + 1)!!)^2') == (1 + y * factorial2(x + 1)) ** 2
    assert sympify('y*x!!') == y * factorial2(x)
    assert sympify('factorial2(x)!') == factorial(factorial2(x))
    raises(SympifyError, lambda : sympify('+!!'))
    raises(SympifyError, lambda : sympify(')!!'))
    raises(SympifyError, lambda : sympify('!'))
    raises(SympifyError, lambda : sympify('(!)'))
    raises(SympifyError, lambda : sympify('x!!!'))

def test_issue_3595():
    if False:
        return 10
    assert sympify('a_') == Symbol('a_')
    assert sympify('_a') == Symbol('_a')

def test_lambda():
    if False:
        i = 10
        return i + 15
    x = Symbol('x')
    assert sympify('lambda: 1') == Lambda((), 1)
    assert sympify('lambda x: x') == Lambda(x, x)
    assert sympify('lambda x: 2*x') == Lambda(x, 2 * x)
    assert sympify('lambda x, y: 2*x+y') == Lambda((x, y), 2 * x + y)

def test_lambda_raises():
    if False:
        for i in range(10):
            print('nop')
    raises(SympifyError, lambda : sympify('lambda *args: args'))
    raises(SympifyError, lambda : sympify('lambda **kwargs: kwargs[0]'))
    raises(SympifyError, lambda : sympify('lambda x = 1: x'))
    with raises(SympifyError):
        _sympify('lambda: 1')

def test_sympify_raises():
    if False:
        for i in range(10):
            print('nop')
    raises(SympifyError, lambda : sympify('fx)'))

    class A:

        def __str__(self):
            if False:
                i = 10
                return i + 15
            return 'x'
    raises(SympifyError, lambda : sympify(A()))

def test__sympify():
    if False:
        i = 10
        return i + 15
    x = Symbol('x')
    f = Function('f')
    assert _sympify(x) is x
    assert _sympify(1) == Integer(1)
    assert _sympify(0.5) == Float('0.5')
    assert _sympify(1 + 1j) == 1.0 + I * 1.0
    assert sympify(f) is f
    raises(SympifyError, lambda : _sympify(f))

    class A:

        def _sympy_(self):
            if False:
                for i in range(10):
                    print('nop')
            return Integer(5)
    a = A()
    assert _sympify(a) == Integer(5)
    raises(SympifyError, lambda : _sympify('1'))
    raises(SympifyError, lambda : _sympify([1, 2, 3]))

def test_sympifyit():
    if False:
        i = 10
        return i + 15
    x = Symbol('x')
    y = Symbol('y')

    @_sympifyit('b', NotImplemented)
    def add(a, b):
        if False:
            i = 10
            return i + 15
        return a + b
    assert add(x, 1) == x + 1
    assert add(x, 0.5) == x + Float('0.5')
    assert add(x, y) == x + y
    assert add(x, '1') == NotImplemented

    @_sympifyit('b')
    def add_raises(a, b):
        if False:
            return 10
        return a + b
    assert add_raises(x, 1) == x + 1
    assert add_raises(x, 0.5) == x + Float('0.5')
    assert add_raises(x, y) == x + y
    raises(SympifyError, lambda : add_raises(x, '1'))

def test_int_float():
    if False:
        for i in range(10):
            print('nop')

    class F1_1:

        def __float__(self):
            if False:
                print('Hello World!')
            return 1.1

    class F1_1b:
        """
        This class is still a float, even though it also implements __int__().
        """

        def __float__(self):
            if False:
                return 10
            return 1.1

        def __int__(self):
            if False:
                i = 10
                return i + 15
            return 1

    class F1_1c:
        """
        This class is still a float, because it implements _sympy_()
        """

        def __float__(self):
            if False:
                for i in range(10):
                    print('nop')
            return 1.1

        def __int__(self):
            if False:
                i = 10
                return i + 15
            return 1

        def _sympy_(self):
            if False:
                return 10
            return Float(1.1)

    class I5:

        def __int__(self):
            if False:
                for i in range(10):
                    print('nop')
            return 5

    class I5b:
        """
        This class implements both __int__() and __float__(), so it will be
        treated as Float in SymPy. One could change this behavior, by using
        float(a) == int(a), but deciding that integer-valued floats represent
        exact numbers is arbitrary and often not correct, so we do not do it.
        If, in the future, we decide to do it anyway, the tests for I5b need to
        be changed.
        """

        def __float__(self):
            if False:
                return 10
            return 5.0

        def __int__(self):
            if False:
                i = 10
                return i + 15
            return 5

    class I5c:
        """
        This class implements both __int__() and __float__(), but also
        a _sympy_() method, so it will be Integer.
        """

        def __float__(self):
            if False:
                while True:
                    i = 10
            return 5.0

        def __int__(self):
            if False:
                while True:
                    i = 10
            return 5

        def _sympy_(self):
            if False:
                return 10
            return Integer(5)
    i5 = I5()
    i5b = I5b()
    i5c = I5c()
    f1_1 = F1_1()
    f1_1b = F1_1b()
    f1_1c = F1_1c()
    assert sympify(i5) == 5
    assert isinstance(sympify(i5), Integer)
    assert sympify(i5b) == 5.0
    assert isinstance(sympify(i5b), Float)
    assert sympify(i5c) == 5
    assert isinstance(sympify(i5c), Integer)
    assert abs(sympify(f1_1) - 1.1) < 1e-05
    assert abs(sympify(f1_1b) - 1.1) < 1e-05
    assert abs(sympify(f1_1c) - 1.1) < 1e-05
    assert _sympify(i5) == 5
    assert isinstance(_sympify(i5), Integer)
    assert _sympify(i5b) == 5.0
    assert isinstance(_sympify(i5b), Float)
    assert _sympify(i5c) == 5
    assert isinstance(_sympify(i5c), Integer)
    assert abs(_sympify(f1_1) - 1.1) < 1e-05
    assert abs(_sympify(f1_1b) - 1.1) < 1e-05
    assert abs(_sympify(f1_1c) - 1.1) < 1e-05

def test_evaluate_false():
    if False:
        for i in range(10):
            print('nop')
    cases = {'2 + 3': Add(2, 3, evaluate=False), '2**2 / 3': Mul(Pow(2, 2, evaluate=False), Pow(3, -1, evaluate=False), evaluate=False), '2 + 3 * 5': Add(2, Mul(3, 5, evaluate=False), evaluate=False), '2 - 3 * 5': Add(2, Mul(-1, Mul(3, 5, evaluate=False), evaluate=False), evaluate=False), '1 / 3': Mul(1, Pow(3, -1, evaluate=False), evaluate=False), 'True | False': Or(True, False, evaluate=False), '1 + 2 + 3 + 5*3 + integrate(x)': Add(1, 2, 3, Mul(5, 3, evaluate=False), x ** 2 / 2, evaluate=False), '2 * 4 * 6 + 8': Add(Mul(2, 4, 6, evaluate=False), 8, evaluate=False), '2 - 8 / 4': Add(2, Mul(-1, Mul(8, Pow(4, -1, evaluate=False), evaluate=False), evaluate=False), evaluate=False), '2 - 2**2': Add(2, Mul(-1, Pow(2, 2, evaluate=False), evaluate=False), evaluate=False)}
    for (case, result) in cases.items():
        assert sympify(case, evaluate=False) == result

def test_issue_4133():
    if False:
        i = 10
        return i + 15
    a = sympify('Integer(4)')
    assert a == Integer(4)
    assert a.is_Integer

def test_issue_3982():
    if False:
        print('Hello World!')
    a = [3, 2.0]
    assert sympify(a) == [Integer(3), Float(2.0)]
    assert sympify(tuple(a)) == Tuple(Integer(3), Float(2.0))
    assert sympify(set(a)) == FiniteSet(Integer(3), Float(2.0))

def test_S_sympify():
    if False:
        while True:
            i = 10
    assert S(1) / 2 == sympify(1) / 2 == S.Half
    assert (-2) ** (S(1) / 2) == sqrt(2) * I

def test_issue_4788():
    if False:
        print('Hello World!')
    assert srepr(S(1.0 + 0j)) == srepr(S(1.0)) == srepr(Float(1.0))

def test_issue_4798_None():
    if False:
        while True:
            i = 10
    assert S(None) is None

def test_issue_3218():
    if False:
        while True:
            i = 10
    assert sympify('x+\ny') == x + y

def test_issue_19399():
    if False:
        return 10
    if not numpy:
        skip('numpy not installed.')
    a = numpy.array(Rational(1, 2))
    b = Rational(1, 3)
    assert (a * b, type(a * b)) == (b * a, type(b * a))

def test_issue_4988_builtins():
    if False:
        return 10
    C = Symbol('C')
    vars = {'C': C}
    exp1 = sympify('C')
    assert exp1 == C
    exp2 = sympify('C', vars)
    assert exp2 == C

def test_geometry():
    if False:
        print('Hello World!')
    p = sympify(Point(0, 1))
    assert p == Point(0, 1) and isinstance(p, Point)
    L = sympify(Line(p, (1, 0)))
    assert L == Line((0, 1), (1, 0)) and isinstance(L, Line)

def test_kernS():
    if False:
        return 10
    s = '-1 - 2*(-(-x + 1/x)/(x*(x - 1/x)**2) - 1/(x*(x - 1/x)))'
    assert -1 - 2 * (-(-x + 1 / x) / (x * (x - 1 / x) ** 2) - 1 / (x * (x - 1 / x))) == -1
    ss = kernS(s)
    assert ss != -1 and ss.simplify() == -1
    s = '-1 - 2*(-(-x + 1/x)/(x*(x - 1/x)**2) - 1/(x*(x - 1/x)))'.replace('x', '_kern')
    ss = kernS(s)
    assert ss != -1 and ss.simplify() == -1
    assert kernS('Interval(-1,-2 - 4*(-3))') == Interval(-1, Add(-2, Mul(12, 1, evaluate=False), evaluate=False))
    assert kernS('_kern') == Symbol('_kern')
    assert kernS('E**-(x)') == exp(-x)
    e = 2 * (x + y) * y
    assert kernS(['2*(x + y)*y', ('2*(x + y)*y',)]) == [e, (e,)]
    assert kernS('-(2*sin(x)**2 + 2*sin(x)*cos(x))*y/2') == -y * (2 * sin(x) ** 2 + 2 * sin(x) * cos(x)) / 2
    assert kernS('(1 - x)/(1 - x*(1-y))') == kernS('(1-x)/(1-(1-y)*x)')
    assert kernS('(1-2**-(4+1)*(1-y)*x)') == 1 - x * (1 - y) / 32
    assert kernS('(1-2**(4+1)*(1-y)*x)') == 1 - 32 * x * (1 - y)
    assert kernS('(1-2.*(1-y)*x)') == 1 - 2.0 * x * (1 - y)
    one = kernS('x - (x - 1)')
    assert one != 1 and one.expand() == 1
    assert kernS('(2*x)/(x-1)') == 2 * x / (x - 1)

def test_issue_6540_6552():
    if False:
        i = 10
        return i + 15
    assert S('[[1/3,2], (2/5,)]') == [[Rational(1, 3), 2], (Rational(2, 5),)]
    assert S('[[2/6,2], (2/4,)]') == [[Rational(1, 3), 2], (S.Half,)]
    assert S('[[[2*(1)]]]') == [[[2]]]
    assert S('Matrix([2*(1)])') == Matrix([2])

def test_issue_6046():
    if False:
        return 10
    assert str(S('Q & C', locals=_clash1)) == 'C & Q'
    assert str(S('pi(x)', locals=_clash2)) == 'pi(x)'
    locals = {}
    exec('from sympy.abc import Q, C', locals)
    assert str(S('C&Q', locals)) == 'C & Q'
    assert str(S('pi(C, Q)', locals=_clash)) == 'pi(C, Q)'
    assert len(S('pi + x', locals=_clash2).free_symbols) == 2
    raises(TypeError, lambda : S('pi + pi(x)', locals=_clash2))
    assert all((set(i.values()) == {null} for i in (_clash, _clash1, _clash2)))

def test_issue_8821_highprec_from_str():
    if False:
        return 10
    s = str(pi.evalf(128))
    p = sympify(s)
    assert Abs(sin(p)) < 1e-127

def test_issue_10295():
    if False:
        for i in range(10):
            print('nop')
    if not numpy:
        skip('numpy not installed.')
    A = numpy.array([[1, 3, -1], [0, 1, 7]])
    sA = S(A)
    assert sA.shape == (2, 3)
    for ((ri, ci), val) in numpy.ndenumerate(A):
        assert sA[ri, ci] == val
    B = numpy.array([-7, x, 3 * y ** 2])
    sB = S(B)
    assert sB.shape == (3,)
    assert B[0] == sB[0] == -7
    assert B[1] == sB[1] == x
    assert B[2] == sB[2] == 3 * y ** 2
    C = numpy.arange(0, 24)
    C.resize(2, 3, 4)
    sC = S(C)
    assert sC[0, 0, 0].is_integer
    assert sC[0, 0, 0] == 0
    a1 = numpy.array([1, 2, 3])
    a2 = numpy.array(list(range(24)))
    a2.resize(2, 4, 3)
    assert sympify(a1) == ImmutableDenseNDimArray([1, 2, 3])
    assert sympify(a2) == ImmutableDenseNDimArray(list(range(24)), (2, 4, 3))

def test_Range():
    if False:
        while True:
            i = 10
    assert sympify(range(10)) == Range(10)
    assert _sympify(range(10)) == Range(10)

def test_sympify_set():
    if False:
        i = 10
        return i + 15
    n = Symbol('n')
    assert sympify({n}) == FiniteSet(n)
    assert sympify(set()) == EmptySet

def test_sympify_numpy():
    if False:
        return 10
    if not numpy:
        skip('numpy not installed. Abort numpy tests.')
    np = numpy

    def equal(x, y):
        if False:
            print('Hello World!')
        return x == y and type(x) == type(y)
    assert sympify(np.bool_(1)) is S(True)
    try:
        assert equal(sympify(np.int_(1234567891234567891)), S(1234567891234567891))
        assert equal(sympify(np.intp(1234567891234567891)), S(1234567891234567891))
    except OverflowError:
        pass
    assert equal(sympify(np.intc(1234567891)), S(1234567891))
    assert equal(sympify(np.int8(-123)), S(-123))
    assert equal(sympify(np.int16(-12345)), S(-12345))
    assert equal(sympify(np.int32(-1234567891)), S(-1234567891))
    assert equal(sympify(np.int64(-1234567891234567891)), S(-1234567891234567891))
    assert equal(sympify(np.uint8(123)), S(123))
    assert equal(sympify(np.uint16(12345)), S(12345))
    assert equal(sympify(np.uint32(1234567891)), S(1234567891))
    assert equal(sympify(np.uint64(1234567891234567891)), S(1234567891234567891))
    assert equal(sympify(np.float32(1.123456)), Float(1.123456, precision=24))
    assert equal(sympify(np.float64(1.1234567891234)), Float(1.1234567891234, precision=53))
    ldprec = np.finfo(np.longdouble(1)).nmant + 1
    assert equal(sympify(np.longdouble(1.123456789)), Float(1.123456789, precision=ldprec))
    assert equal(sympify(np.complex64(1 + 2j)), S(1.0 + 2.0 * I))
    assert equal(sympify(np.complex128(1 + 2j)), S(1.0 + 2.0 * I))
    lcprec = np.finfo(np.longcomplex(1)).nmant + 1
    assert equal(sympify(np.longcomplex(1 + 2j)), Float(1.0, precision=lcprec) + Float(2.0, precision=lcprec) * I)
    if hasattr(np, 'float96'):
        f96prec = np.finfo(np.float96(1)).nmant + 1
        assert equal(sympify(np.float96(1.123456789)), Float(1.123456789, precision=f96prec))
    if hasattr(np, 'float128'):
        f128prec = np.finfo(np.float128(1)).nmant + 1
        assert equal(sympify(np.float128(1.123456789123)), Float(1.123456789123, precision=f128prec))

@XFAIL
def test_sympify_rational_numbers_set():
    if False:
        i = 10
        return i + 15
    ans = [Rational(3, 10), Rational(1, 5)]
    assert sympify({'.3', '.2'}, rational=True) == FiniteSet(*ans)

def test_sympify_mro():
    if False:
        print('Hello World!')
    'Tests the resolution order for classes that implement _sympy_'

    class a:

        def _sympy_(self):
            if False:
                i = 10
                return i + 15
            return Integer(1)

    class b(a):

        def _sympy_(self):
            if False:
                i = 10
                return i + 15
            return Integer(2)

    class c(a):
        pass
    assert sympify(a()) == Integer(1)
    assert sympify(b()) == Integer(2)
    assert sympify(c()) == Integer(1)

def test_sympify_converter():
    if False:
        while True:
            i = 10
    'Tests the resolution order for classes in converter'

    class a:
        pass

    class b(a):
        pass

    class c(a):
        pass
    converter[a] = lambda x: Integer(1)
    converter[b] = lambda x: Integer(2)
    assert sympify(a()) == Integer(1)
    assert sympify(b()) == Integer(2)
    assert sympify(c()) == Integer(1)

    class MyInteger(Integer):
        pass
    if int in converter:
        int_converter = converter[int]
    else:
        int_converter = None
    try:
        converter[int] = MyInteger
        assert sympify(1) == MyInteger(1)
    finally:
        if int_converter is None:
            del converter[int]
        else:
            converter[int] = int_converter

def test_issue_13924():
    if False:
        return 10
    if not numpy:
        skip('numpy not installed.')
    a = sympify(numpy.array([1]))
    assert isinstance(a, ImmutableDenseNDimArray)
    assert a[0] == 1

def test_numpy_sympify_args():
    if False:
        return 10
    if not numpy:
        skip('numpy not installed.')
    a = sympify(numpy.str_('a'))
    assert type(a) is Symbol
    assert a == Symbol('a')

    class CustomSymbol(Symbol):
        pass
    a = sympify(numpy.str_('a'), {'Symbol': CustomSymbol})
    assert isinstance(a, CustomSymbol)
    a = sympify(numpy.str_('x^y'))
    assert a == x ** y
    a = sympify(numpy.str_('x^y'), convert_xor=False)
    assert a == Xor(x, y)
    raises(SympifyError, lambda : sympify(numpy.str_('x'), strict=True))
    a = sympify(numpy.str_('1.1'))
    assert isinstance(a, Float)
    assert a == 1.1
    a = sympify(numpy.str_('1.1'), rational=True)
    assert isinstance(a, Rational)
    assert a == Rational(11, 10)
    a = sympify(numpy.str_('x + x'))
    assert isinstance(a, Mul)
    assert a == 2 * x
    a = sympify(numpy.str_('x + x'), evaluate=False)
    assert isinstance(a, Add)
    assert a == Add(x, x, evaluate=False)

def test_issue_5939():
    if False:
        return 10
    a = Symbol('a')
    b = Symbol('b')
    assert sympify('a+\nb') == a + b

def test_issue_16759():
    if False:
        return 10
    d = sympify({0.5: 1})
    assert S.Half not in d
    assert Float(0.5) in d
    assert d[0.5] is S.One
    d = sympify(OrderedDict({0.5: 1}))
    assert S.Half not in d
    assert Float(0.5) in d
    assert d[0.5] is S.One
    d = sympify(defaultdict(int, {0.5: 1}))
    assert S.Half not in d
    assert Float(0.5) in d
    assert d[0.5] is S.One

def test_issue_17811():
    if False:
        while True:
            i = 10
    a = Function('a')
    assert sympify('a(x)*5', evaluate=False) == Mul(a(x), 5, evaluate=False)

def test_issue_8439():
    if False:
        i = 10
        return i + 15
    assert sympify(float('inf')) == oo
    assert x + float('inf') == x + oo
    assert S(float('inf')) == oo

def test_issue_14706():
    if False:
        while True:
            i = 10
    if not numpy:
        skip('numpy not installed.')
    z1 = numpy.zeros((1, 1), dtype=numpy.float64)
    z2 = numpy.zeros((2, 2), dtype=numpy.float64)
    z3 = numpy.zeros((), dtype=numpy.float64)
    y1 = numpy.ones((1, 1), dtype=numpy.float64)
    y2 = numpy.ones((2, 2), dtype=numpy.float64)
    y3 = numpy.ones((), dtype=numpy.float64)
    assert numpy.all(x + z1 == numpy.full((1, 1), x))
    assert numpy.all(x + z2 == numpy.full((2, 2), x))
    assert numpy.all(z1 + x == numpy.full((1, 1), x))
    assert numpy.all(z2 + x == numpy.full((2, 2), x))
    for z in [z3, numpy.int64(0), numpy.float64(0), numpy.complex64(0)]:
        assert x + z == x
        assert z + x == x
        assert isinstance(x + z, Symbol)
        assert isinstance(z + x, Symbol)
    assert numpy.all(x + y1 == numpy.full((1, 1), x + 1.0))
    assert numpy.all(x + y2 == numpy.full((2, 2), x + 1.0))
    assert numpy.all(y1 + x == numpy.full((1, 1), x + 1.0))
    assert numpy.all(y2 + x == numpy.full((2, 2), x + 1.0))
    for y_ in [y3, numpy.int64(1), numpy.float64(1), numpy.complex64(1)]:
        assert x + y_ == y_ + x
        assert isinstance(x + y_, Add)
        assert isinstance(y_ + x, Add)
    assert x + numpy.array(x) == 2 * x
    assert x + numpy.array([x]) == numpy.array([2 * x], dtype=object)
    assert sympify(numpy.array([1])) == ImmutableDenseNDimArray([1], 1)
    assert sympify(numpy.array([[[1]]])) == ImmutableDenseNDimArray([1], (1, 1, 1))
    assert sympify(z1) == ImmutableDenseNDimArray([0.0], (1, 1))
    assert sympify(z2) == ImmutableDenseNDimArray([0.0, 0.0, 0.0, 0.0], (2, 2))
    assert sympify(z3) == ImmutableDenseNDimArray([0.0], ())
    assert sympify(z3, strict=True) == 0.0
    raises(SympifyError, lambda : sympify(numpy.array([1]), strict=True))
    raises(SympifyError, lambda : sympify(z1, strict=True))
    raises(SympifyError, lambda : sympify(z2, strict=True))

def test_issue_21536():
    if False:
        while True:
            i = 10
    u = sympify('x+3*x+2', evaluate=False)
    v = sympify('2*x+4*x+2+4', evaluate=False)
    assert u.is_Add and set(u.args) == {x, 3 * x, 2}
    assert v.is_Add and set(v.args) == {2 * x, 4 * x, 2, 4}
    assert sympify(['x+3*x+2', '2*x+4*x+2+4'], evaluate=False) == [u, v]
    u = sympify('x+3*x+2', evaluate=True)
    v = sympify('2*x+4*x+2+4', evaluate=True)
    assert u.is_Add and set(u.args) == {4 * x, 2}
    assert v.is_Add and set(v.args) == {6 * x, 6}
    assert sympify(['x+3*x+2', '2*x+4*x+2+4'], evaluate=True) == [u, v]
    u = sympify('x+3*x+2')
    v = sympify('2*x+4*x+2+4')
    assert u.is_Add and set(u.args) == {4 * x, 2}
    assert v.is_Add and set(v.args) == {6 * x, 6}
    assert sympify(['x+3*x+2', '2*x+4*x+2+4']) == [u, v]