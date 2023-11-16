from sympy.core.mod import Mod
from sympy.core.numbers import I, oo, pi
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import asin, sin
from sympy.simplify.simplify import simplify
from sympy.core import Symbol, S, Rational, Integer, Dummy, Wild, Pow
from sympy.core.assumptions import assumptions, check_assumptions, failing_assumptions, common_assumptions, _generate_assumption_rules, _load_pre_generated_assumption_rules
from sympy.core.facts import InconsistentAssumptions
from sympy.core.random import seed
from sympy.combinatorics import Permutation
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.testing.pytest import raises, XFAIL

def test_symbol_unset():
    if False:
        print('Hello World!')
    x = Symbol('x', real=True, integer=True)
    assert x.is_real is True
    assert x.is_integer is True
    assert x.is_imaginary is False
    assert x.is_noninteger is False
    assert x.is_number is False

def test_zero():
    if False:
        while True:
            i = 10
    z = Integer(0)
    assert z.is_commutative is True
    assert z.is_integer is True
    assert z.is_rational is True
    assert z.is_algebraic is True
    assert z.is_transcendental is False
    assert z.is_real is True
    assert z.is_complex is True
    assert z.is_noninteger is False
    assert z.is_irrational is False
    assert z.is_imaginary is False
    assert z.is_positive is False
    assert z.is_negative is False
    assert z.is_nonpositive is True
    assert z.is_nonnegative is True
    assert z.is_even is True
    assert z.is_odd is False
    assert z.is_finite is True
    assert z.is_infinite is False
    assert z.is_comparable is True
    assert z.is_prime is False
    assert z.is_composite is False
    assert z.is_number is True

def test_one():
    if False:
        print('Hello World!')
    z = Integer(1)
    assert z.is_commutative is True
    assert z.is_integer is True
    assert z.is_rational is True
    assert z.is_algebraic is True
    assert z.is_transcendental is False
    assert z.is_real is True
    assert z.is_complex is True
    assert z.is_noninteger is False
    assert z.is_irrational is False
    assert z.is_imaginary is False
    assert z.is_positive is True
    assert z.is_negative is False
    assert z.is_nonpositive is False
    assert z.is_nonnegative is True
    assert z.is_even is False
    assert z.is_odd is True
    assert z.is_finite is True
    assert z.is_infinite is False
    assert z.is_comparable is True
    assert z.is_prime is False
    assert z.is_number is True
    assert z.is_composite is False

def test_negativeone():
    if False:
        while True:
            i = 10
    z = Integer(-1)
    assert z.is_commutative is True
    assert z.is_integer is True
    assert z.is_rational is True
    assert z.is_algebraic is True
    assert z.is_transcendental is False
    assert z.is_real is True
    assert z.is_complex is True
    assert z.is_noninteger is False
    assert z.is_irrational is False
    assert z.is_imaginary is False
    assert z.is_positive is False
    assert z.is_negative is True
    assert z.is_nonpositive is True
    assert z.is_nonnegative is False
    assert z.is_even is False
    assert z.is_odd is True
    assert z.is_finite is True
    assert z.is_infinite is False
    assert z.is_comparable is True
    assert z.is_prime is False
    assert z.is_composite is False
    assert z.is_number is True

def test_infinity():
    if False:
        i = 10
        return i + 15
    oo = S.Infinity
    assert oo.is_commutative is True
    assert oo.is_integer is False
    assert oo.is_rational is False
    assert oo.is_algebraic is False
    assert oo.is_transcendental is False
    assert oo.is_extended_real is True
    assert oo.is_real is False
    assert oo.is_complex is False
    assert oo.is_noninteger is True
    assert oo.is_irrational is False
    assert oo.is_imaginary is False
    assert oo.is_nonzero is False
    assert oo.is_positive is False
    assert oo.is_negative is False
    assert oo.is_nonpositive is False
    assert oo.is_nonnegative is False
    assert oo.is_extended_nonzero is True
    assert oo.is_extended_positive is True
    assert oo.is_extended_negative is False
    assert oo.is_extended_nonpositive is False
    assert oo.is_extended_nonnegative is True
    assert oo.is_even is False
    assert oo.is_odd is False
    assert oo.is_finite is False
    assert oo.is_infinite is True
    assert oo.is_comparable is True
    assert oo.is_prime is False
    assert oo.is_composite is False
    assert oo.is_number is True

def test_neg_infinity():
    if False:
        print('Hello World!')
    mm = S.NegativeInfinity
    assert mm.is_commutative is True
    assert mm.is_integer is False
    assert mm.is_rational is False
    assert mm.is_algebraic is False
    assert mm.is_transcendental is False
    assert mm.is_extended_real is True
    assert mm.is_real is False
    assert mm.is_complex is False
    assert mm.is_noninteger is True
    assert mm.is_irrational is False
    assert mm.is_imaginary is False
    assert mm.is_nonzero is False
    assert mm.is_positive is False
    assert mm.is_negative is False
    assert mm.is_nonpositive is False
    assert mm.is_nonnegative is False
    assert mm.is_extended_nonzero is True
    assert mm.is_extended_positive is False
    assert mm.is_extended_negative is True
    assert mm.is_extended_nonpositive is True
    assert mm.is_extended_nonnegative is False
    assert mm.is_even is False
    assert mm.is_odd is False
    assert mm.is_finite is False
    assert mm.is_infinite is True
    assert mm.is_comparable is True
    assert mm.is_prime is False
    assert mm.is_composite is False
    assert mm.is_number is True

def test_zoo():
    if False:
        while True:
            i = 10
    zoo = S.ComplexInfinity
    assert zoo.is_complex is False
    assert zoo.is_real is False
    assert zoo.is_prime is False

def test_nan():
    if False:
        return 10
    nan = S.NaN
    assert nan.is_commutative is True
    assert nan.is_integer is None
    assert nan.is_rational is None
    assert nan.is_algebraic is None
    assert nan.is_transcendental is None
    assert nan.is_real is None
    assert nan.is_complex is None
    assert nan.is_noninteger is None
    assert nan.is_irrational is None
    assert nan.is_imaginary is None
    assert nan.is_positive is None
    assert nan.is_negative is None
    assert nan.is_nonpositive is None
    assert nan.is_nonnegative is None
    assert nan.is_even is None
    assert nan.is_odd is None
    assert nan.is_finite is None
    assert nan.is_infinite is None
    assert nan.is_comparable is False
    assert nan.is_prime is None
    assert nan.is_composite is None
    assert nan.is_number is True

def test_pos_rational():
    if False:
        print('Hello World!')
    r = Rational(3, 4)
    assert r.is_commutative is True
    assert r.is_integer is False
    assert r.is_rational is True
    assert r.is_algebraic is True
    assert r.is_transcendental is False
    assert r.is_real is True
    assert r.is_complex is True
    assert r.is_noninteger is True
    assert r.is_irrational is False
    assert r.is_imaginary is False
    assert r.is_positive is True
    assert r.is_negative is False
    assert r.is_nonpositive is False
    assert r.is_nonnegative is True
    assert r.is_even is False
    assert r.is_odd is False
    assert r.is_finite is True
    assert r.is_infinite is False
    assert r.is_comparable is True
    assert r.is_prime is False
    assert r.is_composite is False
    r = Rational(1, 4)
    assert r.is_nonpositive is False
    assert r.is_positive is True
    assert r.is_negative is False
    assert r.is_nonnegative is True
    r = Rational(5, 4)
    assert r.is_negative is False
    assert r.is_positive is True
    assert r.is_nonpositive is False
    assert r.is_nonnegative is True
    r = Rational(5, 3)
    assert r.is_nonnegative is True
    assert r.is_positive is True
    assert r.is_negative is False
    assert r.is_nonpositive is False

def test_neg_rational():
    if False:
        return 10
    r = Rational(-3, 4)
    assert r.is_positive is False
    assert r.is_nonpositive is True
    assert r.is_negative is True
    assert r.is_nonnegative is False
    r = Rational(-1, 4)
    assert r.is_nonpositive is True
    assert r.is_positive is False
    assert r.is_negative is True
    assert r.is_nonnegative is False
    r = Rational(-5, 4)
    assert r.is_negative is True
    assert r.is_positive is False
    assert r.is_nonpositive is True
    assert r.is_nonnegative is False
    r = Rational(-5, 3)
    assert r.is_nonnegative is False
    assert r.is_positive is False
    assert r.is_negative is True
    assert r.is_nonpositive is True

def test_pi():
    if False:
        i = 10
        return i + 15
    z = S.Pi
    assert z.is_commutative is True
    assert z.is_integer is False
    assert z.is_rational is False
    assert z.is_algebraic is False
    assert z.is_transcendental is True
    assert z.is_real is True
    assert z.is_complex is True
    assert z.is_noninteger is True
    assert z.is_irrational is True
    assert z.is_imaginary is False
    assert z.is_positive is True
    assert z.is_negative is False
    assert z.is_nonpositive is False
    assert z.is_nonnegative is True
    assert z.is_even is False
    assert z.is_odd is False
    assert z.is_finite is True
    assert z.is_infinite is False
    assert z.is_comparable is True
    assert z.is_prime is False
    assert z.is_composite is False

def test_E():
    if False:
        while True:
            i = 10
    z = S.Exp1
    assert z.is_commutative is True
    assert z.is_integer is False
    assert z.is_rational is False
    assert z.is_algebraic is False
    assert z.is_transcendental is True
    assert z.is_real is True
    assert z.is_complex is True
    assert z.is_noninteger is True
    assert z.is_irrational is True
    assert z.is_imaginary is False
    assert z.is_positive is True
    assert z.is_negative is False
    assert z.is_nonpositive is False
    assert z.is_nonnegative is True
    assert z.is_even is False
    assert z.is_odd is False
    assert z.is_finite is True
    assert z.is_infinite is False
    assert z.is_comparable is True
    assert z.is_prime is False
    assert z.is_composite is False

def test_I():
    if False:
        for i in range(10):
            print('nop')
    z = S.ImaginaryUnit
    assert z.is_commutative is True
    assert z.is_integer is False
    assert z.is_rational is False
    assert z.is_algebraic is True
    assert z.is_transcendental is False
    assert z.is_real is False
    assert z.is_complex is True
    assert z.is_noninteger is False
    assert z.is_irrational is False
    assert z.is_imaginary is True
    assert z.is_positive is False
    assert z.is_negative is False
    assert z.is_nonpositive is False
    assert z.is_nonnegative is False
    assert z.is_even is False
    assert z.is_odd is False
    assert z.is_finite is True
    assert z.is_infinite is False
    assert z.is_comparable is False
    assert z.is_prime is False
    assert z.is_composite is False

def test_symbol_real_false():
    if False:
        print('Hello World!')
    a = Symbol('a', real=False)
    assert a.is_real is False
    assert a.is_integer is False
    assert a.is_zero is False
    assert a.is_negative is False
    assert a.is_positive is False
    assert a.is_nonnegative is False
    assert a.is_nonpositive is False
    assert a.is_nonzero is False
    assert a.is_extended_negative is None
    assert a.is_extended_positive is None
    assert a.is_extended_nonnegative is None
    assert a.is_extended_nonpositive is None
    assert a.is_extended_nonzero is None

def test_symbol_extended_real_false():
    if False:
        return 10
    a = Symbol('a', extended_real=False)
    assert a.is_real is False
    assert a.is_integer is False
    assert a.is_zero is False
    assert a.is_negative is False
    assert a.is_positive is False
    assert a.is_nonnegative is False
    assert a.is_nonpositive is False
    assert a.is_nonzero is False
    assert a.is_extended_negative is False
    assert a.is_extended_positive is False
    assert a.is_extended_nonnegative is False
    assert a.is_extended_nonpositive is False
    assert a.is_extended_nonzero is False

def test_symbol_imaginary():
    if False:
        while True:
            i = 10
    a = Symbol('a', imaginary=True)
    assert a.is_real is False
    assert a.is_integer is False
    assert a.is_negative is False
    assert a.is_positive is False
    assert a.is_nonnegative is False
    assert a.is_nonpositive is False
    assert a.is_zero is False
    assert a.is_nonzero is False

def test_symbol_zero():
    if False:
        print('Hello World!')
    x = Symbol('x', zero=True)
    assert x.is_positive is False
    assert x.is_nonpositive
    assert x.is_negative is False
    assert x.is_nonnegative
    assert x.is_zero is True
    assert x.is_nonzero is False
    assert x.is_finite is True

def test_symbol_positive():
    if False:
        i = 10
        return i + 15
    x = Symbol('x', positive=True)
    assert x.is_positive is True
    assert x.is_nonpositive is False
    assert x.is_negative is False
    assert x.is_nonnegative is True
    assert x.is_zero is False
    assert x.is_nonzero is True

def test_neg_symbol_positive():
    if False:
        print('Hello World!')
    x = -Symbol('x', positive=True)
    assert x.is_positive is False
    assert x.is_nonpositive is True
    assert x.is_negative is True
    assert x.is_nonnegative is False
    assert x.is_zero is False
    assert x.is_nonzero is True

def test_symbol_nonpositive():
    if False:
        i = 10
        return i + 15
    x = Symbol('x', nonpositive=True)
    assert x.is_positive is False
    assert x.is_nonpositive is True
    assert x.is_negative is None
    assert x.is_nonnegative is None
    assert x.is_zero is None
    assert x.is_nonzero is None

def test_neg_symbol_nonpositive():
    if False:
        for i in range(10):
            print('nop')
    x = -Symbol('x', nonpositive=True)
    assert x.is_positive is None
    assert x.is_nonpositive is None
    assert x.is_negative is False
    assert x.is_nonnegative is True
    assert x.is_zero is None
    assert x.is_nonzero is None

def test_symbol_falsepositive():
    if False:
        for i in range(10):
            print('nop')
    x = Symbol('x', positive=False)
    assert x.is_positive is False
    assert x.is_nonpositive is None
    assert x.is_negative is None
    assert x.is_nonnegative is None
    assert x.is_zero is None
    assert x.is_nonzero is None

def test_symbol_falsepositive_mul():
    if False:
        i = 10
        return i + 15
    x = 2 * Symbol('x', positive=False)
    assert x.is_positive is False
    assert x.is_nonpositive is None
    assert x.is_negative is None
    assert x.is_nonnegative is None
    assert x.is_zero is None
    assert x.is_nonzero is None

@XFAIL
def test_symbol_infinitereal_mul():
    if False:
        for i in range(10):
            print('nop')
    ix = Symbol('ix', infinite=True, extended_real=True)
    assert (-ix).is_extended_positive is None

def test_neg_symbol_falsepositive():
    if False:
        for i in range(10):
            print('nop')
    x = -Symbol('x', positive=False)
    assert x.is_positive is None
    assert x.is_nonpositive is None
    assert x.is_negative is False
    assert x.is_nonnegative is None
    assert x.is_zero is None
    assert x.is_nonzero is None

def test_neg_symbol_falsenegative():
    if False:
        return 10
    x = -Symbol('x', negative=False)
    assert x.is_positive is False
    assert x.is_nonpositive is None
    assert x.is_negative is None
    assert x.is_nonnegative is None
    assert x.is_zero is None
    assert x.is_nonzero is None

def test_symbol_falsepositive_real():
    if False:
        print('Hello World!')
    x = Symbol('x', positive=False, real=True)
    assert x.is_positive is False
    assert x.is_nonpositive is True
    assert x.is_negative is None
    assert x.is_nonnegative is None
    assert x.is_zero is None
    assert x.is_nonzero is None

def test_neg_symbol_falsepositive_real():
    if False:
        return 10
    x = -Symbol('x', positive=False, real=True)
    assert x.is_positive is None
    assert x.is_nonpositive is None
    assert x.is_negative is False
    assert x.is_nonnegative is True
    assert x.is_zero is None
    assert x.is_nonzero is None

def test_symbol_falsenonnegative():
    if False:
        print('Hello World!')
    x = Symbol('x', nonnegative=False)
    assert x.is_positive is False
    assert x.is_nonpositive is None
    assert x.is_negative is None
    assert x.is_nonnegative is False
    assert x.is_zero is False
    assert x.is_nonzero is None

@XFAIL
def test_neg_symbol_falsenonnegative():
    if False:
        print('Hello World!')
    x = -Symbol('x', nonnegative=False)
    assert x.is_positive is None
    assert x.is_nonpositive is False
    assert x.is_negative is False
    assert x.is_nonnegative is None
    assert x.is_zero is False
    assert x.is_nonzero is True

def test_symbol_falsenonnegative_real():
    if False:
        print('Hello World!')
    x = Symbol('x', nonnegative=False, real=True)
    assert x.is_positive is False
    assert x.is_nonpositive is True
    assert x.is_negative is True
    assert x.is_nonnegative is False
    assert x.is_zero is False
    assert x.is_nonzero is True

def test_neg_symbol_falsenonnegative_real():
    if False:
        while True:
            i = 10
    x = -Symbol('x', nonnegative=False, real=True)
    assert x.is_positive is True
    assert x.is_nonpositive is False
    assert x.is_negative is False
    assert x.is_nonnegative is True
    assert x.is_zero is False
    assert x.is_nonzero is True

def test_prime():
    if False:
        i = 10
        return i + 15
    assert S.NegativeOne.is_prime is False
    assert S(-2).is_prime is False
    assert S(-4).is_prime is False
    assert S.Zero.is_prime is False
    assert S.One.is_prime is False
    assert S(2).is_prime is True
    assert S(17).is_prime is True
    assert S(4).is_prime is False

def test_composite():
    if False:
        for i in range(10):
            print('nop')
    assert S.NegativeOne.is_composite is False
    assert S(-2).is_composite is False
    assert S(-4).is_composite is False
    assert S.Zero.is_composite is False
    assert S(2).is_composite is False
    assert S(17).is_composite is False
    assert S(4).is_composite is True
    x = Dummy(integer=True, positive=True, prime=False)
    assert x.is_composite is None
    assert (x + 1).is_composite is None
    x = Dummy(positive=True, even=True, prime=False)
    assert x.is_integer is True
    assert x.is_composite is True

def test_prime_symbol():
    if False:
        i = 10
        return i + 15
    x = Symbol('x', prime=True)
    assert x.is_prime is True
    assert x.is_integer is True
    assert x.is_positive is True
    assert x.is_negative is False
    assert x.is_nonpositive is False
    assert x.is_nonnegative is True
    x = Symbol('x', prime=False)
    assert x.is_prime is False
    assert x.is_integer is None
    assert x.is_positive is None
    assert x.is_negative is None
    assert x.is_nonpositive is None
    assert x.is_nonnegative is None

def test_symbol_noncommutative():
    if False:
        for i in range(10):
            print('nop')
    x = Symbol('x', commutative=True)
    assert x.is_complex is None
    x = Symbol('x', commutative=False)
    assert x.is_integer is False
    assert x.is_rational is False
    assert x.is_algebraic is False
    assert x.is_irrational is False
    assert x.is_real is False
    assert x.is_complex is False

def test_other_symbol():
    if False:
        while True:
            i = 10
    x = Symbol('x', integer=True)
    assert x.is_integer is True
    assert x.is_real is True
    assert x.is_finite is True
    x = Symbol('x', integer=True, nonnegative=True)
    assert x.is_integer is True
    assert x.is_nonnegative is True
    assert x.is_negative is False
    assert x.is_positive is None
    assert x.is_finite is True
    x = Symbol('x', integer=True, nonpositive=True)
    assert x.is_integer is True
    assert x.is_nonpositive is True
    assert x.is_positive is False
    assert x.is_negative is None
    assert x.is_finite is True
    x = Symbol('x', odd=True)
    assert x.is_odd is True
    assert x.is_even is False
    assert x.is_integer is True
    assert x.is_finite is True
    x = Symbol('x', odd=False)
    assert x.is_odd is False
    assert x.is_even is None
    assert x.is_integer is None
    assert x.is_finite is None
    x = Symbol('x', even=True)
    assert x.is_even is True
    assert x.is_odd is False
    assert x.is_integer is True
    assert x.is_finite is True
    x = Symbol('x', even=False)
    assert x.is_even is False
    assert x.is_odd is None
    assert x.is_integer is None
    assert x.is_finite is None
    x = Symbol('x', integer=True, nonnegative=True)
    assert x.is_integer is True
    assert x.is_nonnegative is True
    assert x.is_finite is True
    x = Symbol('x', integer=True, nonpositive=True)
    assert x.is_integer is True
    assert x.is_nonpositive is True
    assert x.is_finite is True
    x = Symbol('x', rational=True)
    assert x.is_real is True
    assert x.is_finite is True
    x = Symbol('x', rational=False)
    assert x.is_real is None
    assert x.is_finite is None
    x = Symbol('x', irrational=True)
    assert x.is_real is True
    assert x.is_finite is True
    x = Symbol('x', irrational=False)
    assert x.is_real is None
    assert x.is_finite is None
    with raises(AttributeError):
        x.is_real = False
    x = Symbol('x', algebraic=True)
    assert x.is_transcendental is False
    x = Symbol('x', transcendental=True)
    assert x.is_algebraic is False
    assert x.is_rational is False
    assert x.is_integer is False

def test_evaluate_false():
    if False:
        i = 10
        return i + 15
    from sympy.core.parameters import evaluate
    from sympy.abc import x, h
    f = 2 ** x ** 7
    with evaluate(False):
        fh = f.xreplace({x: x + h})
        assert fh.exp.is_rational is None

def test_issue_3825():
    if False:
        for i in range(10):
            print('nop')
    'catch: hash instability'
    x = Symbol('x')
    y = Symbol('y')
    a1 = x + y
    a2 = y + x
    a2.is_comparable
    h1 = hash(a1)
    h2 = hash(a2)
    assert h1 == h2

def test_issue_4822():
    if False:
        print('Hello World!')
    z = (-1) ** Rational(1, 3) * (1 - I * sqrt(3))
    assert z.is_real in [True, None]

def test_hash_vs_typeinfo():
    if False:
        for i in range(10):
            print('nop')
    'seemingly different typeinfo, but in fact equal'
    x1 = Symbol('x', even=True)
    x2 = Symbol('x', integer=True, odd=False)
    assert hash(x1) == hash(x2)
    assert x1 == x2

def test_hash_vs_typeinfo_2():
    if False:
        while True:
            i = 10
    'different typeinfo should mean !eq'
    x = Symbol('x')
    x1 = Symbol('x', even=True)
    assert x != x1
    assert hash(x) != hash(x1)

def test_hash_vs_eq():
    if False:
        i = 10
        return i + 15
    'catch: different hash for equal objects'
    a = 1 + S.Pi
    ha = hash(a)
    a.is_positive
    assert a.is_positive is True
    assert ha == hash(a)
    b = a.expand(trig=True)
    hb = hash(b)
    assert a == b
    assert ha == hb

def test_Add_is_pos_neg():
    if False:
        return 10
    n = Symbol('n', extended_negative=True, infinite=True)
    nn = Symbol('n', extended_nonnegative=True, infinite=True)
    np = Symbol('n', extended_nonpositive=True, infinite=True)
    p = Symbol('p', extended_positive=True, infinite=True)
    r = Dummy(extended_real=True, finite=False)
    x = Symbol('x')
    xf = Symbol('xf', finite=True)
    assert (n + p).is_extended_positive is None
    assert (n + x).is_extended_positive is None
    assert (p + x).is_extended_positive is None
    assert (n + p).is_extended_negative is None
    assert (n + x).is_extended_negative is None
    assert (p + x).is_extended_negative is None
    assert (n + xf).is_extended_positive is False
    assert (p + xf).is_extended_positive is True
    assert (n + xf).is_extended_negative is True
    assert (p + xf).is_extended_negative is False
    assert (x - S.Infinity).is_extended_negative is None
    assert (p + nn).is_extended_positive
    assert (n + np).is_extended_negative
    assert (p + r).is_extended_positive is None

def test_Add_is_imaginary():
    if False:
        return 10
    nn = Dummy(nonnegative=True)
    assert (I * nn + I).is_imaginary

def test_Add_is_algebraic():
    if False:
        i = 10
        return i + 15
    a = Symbol('a', algebraic=True)
    b = Symbol('a', algebraic=True)
    na = Symbol('na', algebraic=False)
    nb = Symbol('nb', algebraic=False)
    x = Symbol('x')
    assert (a + b).is_algebraic
    assert (na + nb).is_algebraic is None
    assert (a + na).is_algebraic is False
    assert (a + x).is_algebraic is None
    assert (na + x).is_algebraic is None

def test_Mul_is_algebraic():
    if False:
        print('Hello World!')
    a = Symbol('a', algebraic=True)
    b = Symbol('b', algebraic=True)
    na = Symbol('na', algebraic=False)
    an = Symbol('an', algebraic=True, nonzero=True)
    nb = Symbol('nb', algebraic=False)
    x = Symbol('x')
    assert (a * b).is_algebraic is True
    assert (na * nb).is_algebraic is None
    assert (a * na).is_algebraic is None
    assert (an * na).is_algebraic is False
    assert (a * x).is_algebraic is None
    assert (na * x).is_algebraic is None

def test_Pow_is_algebraic():
    if False:
        return 10
    e = Symbol('e', algebraic=True)
    assert Pow(1, e, evaluate=False).is_algebraic
    assert Pow(0, e, evaluate=False).is_algebraic
    a = Symbol('a', algebraic=True)
    azf = Symbol('azf', algebraic=True, zero=False)
    na = Symbol('na', algebraic=False)
    ia = Symbol('ia', algebraic=True, irrational=True)
    ib = Symbol('ib', algebraic=True, irrational=True)
    r = Symbol('r', rational=True)
    x = Symbol('x')
    assert (a ** 2).is_algebraic is True
    assert (a ** r).is_algebraic is None
    assert (azf ** r).is_algebraic is True
    assert (a ** x).is_algebraic is None
    assert (na ** r).is_algebraic is None
    assert (ia ** r).is_algebraic is True
    assert (ia ** ib).is_algebraic is False
    assert (a ** e).is_algebraic is None
    assert Pow(2, sqrt(2), evaluate=False).is_algebraic is False
    assert Pow(S.GoldenRatio, sqrt(3), evaluate=False).is_algebraic is False
    t = Symbol('t', real=True, transcendental=True)
    n = Symbol('n', integer=True)
    assert (t ** n).is_algebraic is None
    assert (t ** n).is_integer is None
    assert (pi ** 3).is_algebraic is False
    r = Symbol('r', zero=True)
    assert (pi ** r).is_algebraic is True

def test_Mul_is_prime_composite():
    if False:
        i = 10
        return i + 15
    x = Symbol('x', positive=True, integer=True)
    y = Symbol('y', positive=True, integer=True)
    assert (x * y).is_prime is None
    assert ((x + 1) * (y + 1)).is_prime is False
    assert ((x + 1) * (y + 1)).is_composite is True
    x = Symbol('x', positive=True)
    assert ((x + 1) * (y + 1)).is_prime is None
    assert ((x + 1) * (y + 1)).is_composite is None

def test_Pow_is_pos_neg():
    if False:
        i = 10
        return i + 15
    z = Symbol('z', real=True)
    w = Symbol('w', nonpositive=True)
    assert (S.NegativeOne ** S(2)).is_positive is True
    assert (S.One ** z).is_positive is True
    assert (S.NegativeOne ** S(3)).is_positive is False
    assert (S.Zero ** S.Zero).is_positive is True
    assert (w ** S(3)).is_positive is False
    assert (w ** S(2)).is_positive is None
    assert (I ** 2).is_positive is False
    assert (I ** 4).is_positive is True
    p = Symbol('p', zero=True)
    q = Symbol('q', zero=False, real=True)
    j = Symbol('j', zero=False, even=True)
    x = Symbol('x', zero=True)
    y = Symbol('y', zero=True)
    assert (p ** q).is_positive is False
    assert (p ** q).is_negative is False
    assert (p ** j).is_positive is False
    assert (x ** y).is_positive is True
    assert (x ** y).is_negative is False

def test_Pow_is_prime_composite():
    if False:
        return 10
    x = Symbol('x', positive=True, integer=True)
    y = Symbol('y', positive=True, integer=True)
    assert (x ** y).is_prime is None
    assert (x ** (y + 1)).is_prime is False
    assert (x ** (y + 1)).is_composite is None
    assert ((x + 1) ** (y + 1)).is_composite is True
    assert ((-x - 1) ** (2 * y)).is_composite is True
    x = Symbol('x', positive=True)
    assert (x ** y).is_prime is None

def test_Mul_is_infinite():
    if False:
        return 10
    x = Symbol('x')
    f = Symbol('f', finite=True)
    i = Symbol('i', infinite=True)
    z = Dummy(zero=True)
    nzf = Dummy(finite=True, zero=False)
    from sympy.core.mul import Mul
    assert (x * f).is_finite is None
    assert (x * i).is_finite is None
    assert (f * i).is_finite is None
    assert (x * f * i).is_finite is None
    assert (z * i).is_finite is None
    assert (nzf * i).is_finite is False
    assert (z * f).is_finite is True
    assert Mul(0, f, evaluate=False).is_finite is True
    assert Mul(0, i, evaluate=False).is_finite is None
    assert (x * f).is_infinite is None
    assert (x * i).is_infinite is None
    assert (f * i).is_infinite is None
    assert (x * f * i).is_infinite is None
    assert (z * i).is_infinite is S.NaN.is_infinite
    assert (nzf * i).is_infinite is True
    assert (z * f).is_infinite is False
    assert Mul(0, f, evaluate=False).is_infinite is False
    assert Mul(0, i, evaluate=False).is_infinite is S.NaN.is_infinite

def test_Add_is_infinite():
    if False:
        for i in range(10):
            print('nop')
    x = Symbol('x')
    f = Symbol('f', finite=True)
    i = Symbol('i', infinite=True)
    i2 = Symbol('i2', infinite=True)
    z = Dummy(zero=True)
    nzf = Dummy(finite=True, zero=False)
    from sympy.core.add import Add
    assert (x + f).is_finite is None
    assert (x + i).is_finite is None
    assert (f + i).is_finite is False
    assert (x + f + i).is_finite is None
    assert (z + i).is_finite is False
    assert (nzf + i).is_finite is False
    assert (z + f).is_finite is True
    assert (i + i2).is_finite is None
    assert Add(0, f, evaluate=False).is_finite is True
    assert Add(0, i, evaluate=False).is_finite is False
    assert (x + f).is_infinite is None
    assert (x + i).is_infinite is None
    assert (f + i).is_infinite is True
    assert (x + f + i).is_infinite is None
    assert (z + i).is_infinite is True
    assert (nzf + i).is_infinite is True
    assert (z + f).is_infinite is False
    assert (i + i2).is_infinite is None
    assert Add(0, f, evaluate=False).is_infinite is False
    assert Add(0, i, evaluate=False).is_infinite is True

def test_special_is_rational():
    if False:
        for i in range(10):
            print('nop')
    i = Symbol('i', integer=True)
    i2 = Symbol('i2', integer=True)
    ni = Symbol('ni', integer=True, nonzero=True)
    r = Symbol('r', rational=True)
    rn = Symbol('r', rational=True, nonzero=True)
    nr = Symbol('nr', irrational=True)
    x = Symbol('x')
    assert sqrt(3).is_rational is False
    assert (3 + sqrt(3)).is_rational is False
    assert (3 * sqrt(3)).is_rational is False
    assert exp(3).is_rational is False
    assert exp(ni).is_rational is False
    assert exp(rn).is_rational is False
    assert exp(x).is_rational is None
    assert exp(log(3), evaluate=False).is_rational is True
    assert log(exp(3), evaluate=False).is_rational is True
    assert log(3).is_rational is False
    assert log(ni + 1).is_rational is False
    assert log(rn + 1).is_rational is False
    assert log(x).is_rational is None
    assert (sqrt(3) + sqrt(5)).is_rational is None
    assert (sqrt(3) + S.Pi).is_rational is False
    assert (x ** i).is_rational is None
    assert (i ** i).is_rational is True
    assert (i ** i2).is_rational is None
    assert (r ** i).is_rational is None
    assert (r ** r).is_rational is None
    assert (r ** x).is_rational is None
    assert (nr ** i).is_rational is None
    assert (nr ** Symbol('z', zero=True)).is_rational
    assert sin(1).is_rational is False
    assert sin(ni).is_rational is False
    assert sin(rn).is_rational is False
    assert sin(x).is_rational is None
    assert asin(r).is_rational is False
    assert sin(asin(3), evaluate=False).is_rational is True

@XFAIL
def test_issue_6275():
    if False:
        i = 10
        return i + 15
    x = Symbol('x')
    assert isinstance(x * 0, type(0 * S.Infinity))
    if 0 * S.Infinity is S.NaN:
        b = Symbol('b', finite=None)
        assert (b * 0).is_zero is None

def test_sanitize_assumptions():
    if False:
        return 10
    for cls in (Symbol, Dummy, Wild):
        x = cls('x', real=1, positive=0)
        assert x.is_real is True
        assert x.is_positive is False
        assert cls('', real=True, positive=None).is_positive is None
        raises(ValueError, lambda : cls('', commutative=None))
    raises(ValueError, lambda : Symbol._sanitize({'commutative': None}))

def test_special_assumptions():
    if False:
        return 10
    e = -3 - sqrt(5) + (-sqrt(10) / 2 - sqrt(2) / 2) ** 2
    assert simplify(e < 0) is S.false
    assert simplify(e > 0) is S.false
    assert (e == 0) is False
    assert e.equals(0) is True

def test_inconsistent():
    if False:
        for i in range(10):
            print('nop')
    raises(InconsistentAssumptions, lambda : Symbol('x', real=True, commutative=False))

def test_issue_6631():
    if False:
        for i in range(10):
            print('nop')
    assert ((-1) ** I).is_real is True
    assert ((-1) ** (I * 2)).is_real is True
    assert ((-1) ** (I / 2)).is_real is True
    assert ((-1) ** (I * S.Pi)).is_real is True
    assert (I ** (I + 2)).is_real is True

def test_issue_2730():
    if False:
        for i in range(10):
            print('nop')
    assert (1 / (1 + I)).is_real is False

def test_issue_4149():
    if False:
        return 10
    assert (3 + I).is_complex
    assert (3 + I).is_imaginary is False
    assert (3 * I + S.Pi * I).is_imaginary
    y = Symbol('y', real=True)
    assert (3 * I + S.Pi * I + y * I).is_imaginary is None
    p = Symbol('p', positive=True)
    assert (3 * I + S.Pi * I + p * I).is_imaginary
    n = Symbol('n', negative=True)
    assert (-3 * I - S.Pi * I + n * I).is_imaginary
    i = Symbol('i', imaginary=True)
    assert [(i ** a).is_imaginary for a in range(4)] == [False, True, False, True]
    e = S('-sqrt(3)*I/2 + 0.866025403784439*I')
    assert e.is_real is False
    assert e.is_imaginary

def test_issue_2920():
    if False:
        i = 10
        return i + 15
    n = Symbol('n', negative=True)
    assert sqrt(n).is_imaginary

def test_issue_7899():
    if False:
        for i in range(10):
            print('nop')
    x = Symbol('x', real=True)
    assert (I * x).is_real is None
    assert ((x - I) * (x - 1)).is_zero is None
    assert ((x - I) * (x - 1)).is_real is None

@XFAIL
def test_issue_7993():
    if False:
        for i in range(10):
            print('nop')
    x = Dummy(integer=True)
    y = Dummy(noninteger=True)
    assert (x - y).is_zero is False

def test_issue_8075():
    if False:
        i = 10
        return i + 15
    raises(InconsistentAssumptions, lambda : Dummy(zero=True, finite=False))
    raises(InconsistentAssumptions, lambda : Dummy(zero=True, infinite=True))

def test_issue_8642():
    if False:
        i = 10
        return i + 15
    x = Symbol('x', real=True, integer=False)
    assert (x * 2).is_integer is None, (x * 2).is_integer

def test_issues_8632_8633_8638_8675_8992():
    if False:
        print('Hello World!')
    p = Dummy(integer=True, positive=True)
    nn = Dummy(integer=True, nonnegative=True)
    assert (p - S.Half).is_positive
    assert (p - 1).is_nonnegative
    assert (nn + 1).is_positive
    assert (-p + 1).is_nonpositive
    assert (-nn - 1).is_negative
    prime = Dummy(prime=True)
    assert (prime - 2).is_nonnegative
    assert (prime - 3).is_nonnegative is None
    even = Dummy(positive=True, even=True)
    assert (even - 2).is_nonnegative
    p = Dummy(positive=True)
    assert (p / (p + 1) - 1).is_negative
    assert ((p + 2) ** 3 - S.Half).is_positive
    n = Dummy(negative=True)
    assert (n - 3).is_nonpositive

def test_issue_9115_9150():
    if False:
        for i in range(10):
            print('nop')
    n = Dummy('n', integer=True, nonnegative=True)
    assert (factorial(n) >= 1) == True
    assert (factorial(n) < 1) == False
    assert factorial(n + 1).is_even is None
    assert factorial(n + 2).is_even is True
    assert factorial(n + 2) >= 2

def test_issue_9165():
    if False:
        print('Hello World!')
    z = Symbol('z', zero=True)
    f = Symbol('f', finite=False)
    assert 0 / z is S.NaN
    assert 0 * (1 / z) is S.NaN
    assert 0 * f is S.NaN

def test_issue_10024():
    if False:
        while True:
            i = 10
    x = Dummy('x')
    assert Mod(x, 2 * pi).is_zero is None

def test_issue_10302():
    if False:
        print('Hello World!')
    x = Symbol('x')
    r = Symbol('r', real=True)
    u = -(3 * 2 ** pi) ** (1 / pi) + 2 * 3 ** (1 / pi)
    i = u + u * I
    assert i.is_real is None
    assert (u + i).is_zero is None
    assert (1 + i).is_zero is False
    a = Dummy('a', zero=True)
    assert (a + I).is_zero is False
    assert (a + r * I).is_zero is None
    assert (a + I).is_imaginary
    assert (a + x + I).is_imaginary is None
    assert (a + r * I + I).is_imaginary is None

def test_complex_reciprocal_imaginary():
    if False:
        print('Hello World!')
    assert (1 / (4 + 3 * I)).is_imaginary is False

def test_issue_16313():
    if False:
        while True:
            i = 10
    x = Symbol('x', extended_real=False)
    k = Symbol('k', real=True)
    l = Symbol('l', real=True, zero=False)
    assert (-x).is_real is False
    assert (k * x).is_real is None
    assert (l * x).is_real is False
    assert (l * x * x).is_real is None
    assert (-x).is_positive is False

def test_issue_16579():
    if False:
        return 10
    x = Symbol('x', extended_real=True, infinite=False)
    y = Symbol('y', extended_real=True, finite=False)
    assert x.is_finite is True
    assert y.is_infinite is True
    c = Symbol('c', complex=True)
    assert c.is_finite is True
    raises(InconsistentAssumptions, lambda : Dummy(complex=True, finite=False))
    nf = Symbol('nf', finite=False)
    assert nf.is_infinite is True

def test_issue_17556():
    if False:
        while True:
            i = 10
    z = I * oo
    assert z.is_imaginary is False
    assert z.is_finite is False

def test_issue_21651():
    if False:
        return 10
    k = Symbol('k', positive=True, integer=True)
    exp = 2 * 2 ** (-k)
    assert exp.is_integer is None

def test_assumptions_copy():
    if False:
        print('Hello World!')
    assert assumptions(Symbol('x'), {'commutative': True}) == {'commutative': True}
    assert assumptions(Symbol('x'), ['integer']) == {}
    assert assumptions(Symbol('x'), ['commutative']) == {'commutative': True}
    assert assumptions(Symbol('x')) == {'commutative': True}
    assert assumptions(1)['positive']
    assert assumptions(3 + I) == {'algebraic': True, 'commutative': True, 'complex': True, 'composite': False, 'even': False, 'extended_negative': False, 'extended_nonnegative': False, 'extended_nonpositive': False, 'extended_nonzero': False, 'extended_positive': False, 'extended_real': False, 'finite': True, 'imaginary': False, 'infinite': False, 'integer': False, 'irrational': False, 'negative': False, 'noninteger': False, 'nonnegative': False, 'nonpositive': False, 'nonzero': False, 'odd': False, 'positive': False, 'prime': False, 'rational': False, 'real': False, 'transcendental': False, 'zero': False}

def test_check_assumptions():
    if False:
        i = 10
        return i + 15
    assert check_assumptions(1, 0) is False
    x = Symbol('x', positive=True)
    assert check_assumptions(1, x) is True
    assert check_assumptions(1, 1) is True
    assert check_assumptions(-1, 1) is False
    i = Symbol('i', integer=True)
    assert check_assumptions(i, 1) is None
    assert check_assumptions(Dummy(integer=None), integer=True) is None
    assert check_assumptions(Dummy(integer=None), integer=False) is None
    assert check_assumptions(Dummy(integer=False), integer=True) is False
    assert check_assumptions(Dummy(integer=True), integer=False) is False
    assert check_assumptions(Dummy(integer=False), integer=None) is True
    raises(ValueError, lambda : check_assumptions(2 * x, x, positive=True))

def test_failing_assumptions():
    if False:
        print('Hello World!')
    x = Symbol('x', positive=True)
    y = Symbol('y')
    assert failing_assumptions(6 * x + y, **x.assumptions0) == {'real': None, 'imaginary': None, 'complex': None, 'hermitian': None, 'positive': None, 'nonpositive': None, 'nonnegative': None, 'nonzero': None, 'negative': None, 'zero': None, 'extended_real': None, 'finite': None, 'infinite': None, 'extended_negative': None, 'extended_nonnegative': None, 'extended_nonpositive': None, 'extended_nonzero': None, 'extended_positive': None}

def test_common_assumptions():
    if False:
        for i in range(10):
            print('nop')
    assert common_assumptions([0, 1, 2]) == {'algebraic': True, 'irrational': False, 'hermitian': True, 'extended_real': True, 'real': True, 'extended_negative': False, 'extended_nonnegative': True, 'integer': True, 'rational': True, 'imaginary': False, 'complex': True, 'commutative': True, 'noninteger': False, 'composite': False, 'infinite': False, 'nonnegative': True, 'finite': True, 'transcendental': False, 'negative': False}
    assert common_assumptions([0, 1, 2], 'positive integer'.split()) == {'integer': True}
    assert common_assumptions([0, 1, 2], []) == {}
    assert common_assumptions([], ['integer']) == {}
    assert common_assumptions([0], ['integer']) == {'integer': True}

def test_pre_generated_assumption_rules_are_valid():
    if False:
        return 10
    pre_generated_assumptions = _load_pre_generated_assumption_rules()
    generated_assumptions = _generate_assumption_rules()
    assert pre_generated_assumptions._to_python() == generated_assumptions._to_python(), 'pre-generated assumptions are invalid, see sympy.core.assumptions._generate_assumption_rules'

def test_ask_shuffle():
    if False:
        for i in range(10):
            print('nop')
    grp = PermutationGroup(Permutation(1, 0, 2), Permutation(2, 1, 3))
    seed(123)
    first = grp.random()
    seed(123)
    simplify(I)
    second = grp.random()
    seed(123)
    simplify(-I)
    third = grp.random()
    assert first == second == third