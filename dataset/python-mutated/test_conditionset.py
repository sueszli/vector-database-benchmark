from sympy.core.expr import unchanged
from sympy.sets import ConditionSet, Intersection, FiniteSet, EmptySet, Union, Contains, ImageSet
from sympy.sets.sets import SetKind
from sympy.core.function import Function, Lambda
from sympy.core.mod import Mod
from sympy.core.kind import NumberKind
from sympy.core.numbers import oo, pi
from sympy.core.relational import Eq, Ne
from sympy.core.singleton import S
from sympy.core.symbol import Symbol, symbols
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.trigonometric import asin, sin
from sympy.logic.boolalg import And
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.sets.sets import Interval
from sympy.testing.pytest import raises, warns_deprecated_sympy
w = Symbol('w')
x = Symbol('x')
y = Symbol('y')
z = Symbol('z')
f = Function('f')

def test_CondSet():
    if False:
        return 10
    sin_sols_principal = ConditionSet(x, Eq(sin(x), 0), Interval(0, 2 * pi, False, True))
    assert pi in sin_sols_principal
    assert pi / 2 not in sin_sols_principal
    assert 3 * pi not in sin_sols_principal
    assert oo not in sin_sols_principal
    assert 5 in ConditionSet(x, x ** 2 > 4, S.Reals)
    assert 1 not in ConditionSet(x, x ** 2 > 4, S.Reals)
    assert 0 not in ConditionSet(x, y > 5, Interval(1, 7))
    raises(TypeError, lambda : 6 in ConditionSet(x, y > 5, Interval(1, 7)))
    X = MatrixSymbol('X', 2, 2)
    matrix_set = ConditionSet(X, Eq(X * Matrix([[1, 1], [1, 1]]), X))
    Y = Matrix([[0, 0], [0, 0]])
    assert matrix_set.contains(Y).doit() is S.true
    Z = Matrix([[1, 2], [3, 4]])
    assert matrix_set.contains(Z).doit() is S.false
    assert isinstance(ConditionSet(x, x < 1, {x, y}).base_set, FiniteSet)
    raises(TypeError, lambda : ConditionSet(x, x + 1, {x, y}))
    raises(TypeError, lambda : ConditionSet(x, x, 1))
    I = S.Integers
    U = S.UniversalSet
    C = ConditionSet
    assert C(x, False, I) is S.EmptySet
    assert C(x, True, I) is I
    assert C(x, x < 1, C(x, x < 2, I)) == C(x, (x < 1) & (x < 2), I)
    assert C(y, y < 1, C(x, y < 2, I)) == C(x, (x < 1) & (y < 2), I), C(y, y < 1, C(x, y < 2, I))
    assert C(y, y < 1, C(x, x < 2, I)) == C(y, (y < 1) & (y < 2), I)
    assert C(y, y < 1, C(x, y < x, I)) == C(x, (x < 1) & (y < x), I)
    assert unchanged(C, y, x < 1, C(x, y < x, I))
    assert ConditionSet(x, x < 1).base_set is U
    assert ConditionSet((x,), x < 1).base_set is U
    c = ConditionSet((x, y), x < y, I ** 2)
    assert (1, 2) in c
    assert (1, pi) not in c
    raises(TypeError, lambda : C(x, x > 1, C((x, y), x > 1, I ** 2)))
    raises(TypeError, lambda : C((x, y), x + y < 2, U, U))

def test_CondSet_intersect():
    if False:
        while True:
            i = 10
    input_conditionset = ConditionSet(x, x ** 2 > 4, Interval(1, 4, False, False))
    other_domain = Interval(0, 3, False, False)
    output_conditionset = ConditionSet(x, x ** 2 > 4, Interval(1, 3, False, False))
    assert Intersection(input_conditionset, other_domain) == output_conditionset

def test_issue_9849():
    if False:
        return 10
    assert ConditionSet(x, Eq(x, x), S.Naturals) is S.Naturals
    assert ConditionSet(x, Eq(Abs(sin(x)), -1), S.Naturals) == S.EmptySet

def test_simplified_FiniteSet_in_CondSet():
    if False:
        print('Hello World!')
    assert ConditionSet(x, And(x < 1, x > -3), FiniteSet(0, 1, 2)) == FiniteSet(0)
    assert ConditionSet(x, x < 0, FiniteSet(0, 1, 2)) == EmptySet
    assert ConditionSet(x, And(x < -3), EmptySet) == EmptySet
    y = Symbol('y')
    assert ConditionSet(x, And(x > 0), FiniteSet(-1, 0, 1, y)) == Union(FiniteSet(1), ConditionSet(x, And(x > 0), FiniteSet(y)))
    assert ConditionSet(x, Eq(Mod(x, 3), 1), FiniteSet(1, 4, 2, y)) == Union(FiniteSet(1, 4), ConditionSet(x, Eq(Mod(x, 3), 1), FiniteSet(y)))

def test_free_symbols():
    if False:
        for i in range(10):
            print('nop')
    assert ConditionSet(x, Eq(y, 0), FiniteSet(z)).free_symbols == {y, z}
    assert ConditionSet(x, Eq(x, 0), FiniteSet(z)).free_symbols == {z}
    assert ConditionSet(x, Eq(x, 0), FiniteSet(x, z)).free_symbols == {x, z}
    assert ConditionSet(x, Eq(x, 0), ImageSet(Lambda(y, y ** 2), S.Integers)).free_symbols == set()

def test_bound_symbols():
    if False:
        i = 10
        return i + 15
    assert ConditionSet(x, Eq(y, 0), FiniteSet(z)).bound_symbols == [x]
    assert ConditionSet(x, Eq(x, 0), FiniteSet(x, y)).bound_symbols == [x]
    assert ConditionSet(x, x < 10, ImageSet(Lambda(y, y ** 2), S.Integers)).bound_symbols == [x]
    assert ConditionSet(x, x < 10, ConditionSet(y, y > 1, S.Integers)).bound_symbols == [x]

def test_as_dummy():
    if False:
        for i in range(10):
            print('nop')
    (_0, _1) = symbols('_0 _1')
    assert ConditionSet(x, x < 1, Interval(y, oo)).as_dummy() == ConditionSet(_0, _0 < 1, Interval(y, oo))
    assert ConditionSet(x, x < 1, Interval(x, oo)).as_dummy() == ConditionSet(_0, _0 < 1, Interval(x, oo))
    assert ConditionSet(x, x < 1, ImageSet(Lambda(y, y ** 2), S.Integers)).as_dummy() == ConditionSet(_0, _0 < 1, ImageSet(Lambda(_0, _0 ** 2), S.Integers))
    e = ConditionSet((x, y), x <= y, S.Reals ** 2)
    assert e.bound_symbols == [x, y]
    assert e.as_dummy() == ConditionSet((_0, _1), _0 <= _1, S.Reals ** 2)
    assert e.as_dummy() == ConditionSet((y, x), y <= x, S.Reals ** 2).as_dummy()

def test_subs_CondSet():
    if False:
        print('Hello World!')
    s = FiniteSet(z, y)
    c = ConditionSet(x, x < 2, s)
    assert c.subs(x, y) == c
    assert c.subs(z, y) == ConditionSet(x, x < 2, FiniteSet(y))
    assert c.xreplace({x: y}) == ConditionSet(y, y < 2, s)
    assert ConditionSet(x, x < y, s).subs(y, w) == ConditionSet(x, x < w, s.subs(y, w))
    n = Symbol('n', negative=True)
    assert ConditionSet(n, 0 < n, S.Integers) is S.EmptySet
    p = Symbol('p', positive=True)
    assert ConditionSet(n, n < y, S.Integers).subs(n, x) == ConditionSet(n, n < y, S.Integers)
    raises(ValueError, lambda : ConditionSet(x + 1, x < 1, S.Integers))
    assert ConditionSet(p, n < x, Interval(-5, 5)).subs(x, p) == Interval(-5, 5), ConditionSet(p, n < x, Interval(-5, 5)).subs(x, p)
    assert ConditionSet(n, n < x, Interval(-oo, 0)).subs(x, p) == Interval(-oo, 0)
    assert ConditionSet(f(x), f(x) < 1, {w, z}).subs(f(x), y) == ConditionSet(f(x), f(x) < 1, {w, z})
    k = Symbol('k')
    img1 = ImageSet(Lambda(k, 2 * k * pi + asin(y)), S.Integers)
    img2 = ImageSet(Lambda(k, 2 * k * pi + asin(S.One / 3)), S.Integers)
    assert ConditionSet(x, Contains(y, Interval(-1, 1)), img1).subs(y, S.One / 3).dummy_eq(img2)
    assert (0, 1) in ConditionSet((x, y), x + y < 3, S.Integers ** 2)
    raises(TypeError, lambda : ConditionSet(n, n < -10, Interval(0, 10)))

def test_subs_CondSet_tebr():
    if False:
        return 10
    with warns_deprecated_sympy():
        assert ConditionSet((x, y), {x + 1, x + y}, S.Reals ** 2) == ConditionSet((x, y), Eq(x + 1, 0) & Eq(x + y, 0), S.Reals ** 2)

def test_dummy_eq():
    if False:
        while True:
            i = 10
    C = ConditionSet
    I = S.Integers
    c = C(x, x < 1, I)
    assert c.dummy_eq(C(y, y < 1, I))
    assert c.dummy_eq(1) == False
    assert c.dummy_eq(C(x, x < 1, S.Reals)) == False
    c1 = ConditionSet((x, y), Eq(x + 1, 0) & Eq(x + y, 0), S.Reals ** 2)
    c2 = ConditionSet((x, y), Eq(x + 1, 0) & Eq(x + y, 0), S.Reals ** 2)
    c3 = ConditionSet((x, y), Eq(x + 1, 0) & Eq(x + y, 0), S.Complexes ** 2)
    assert c1.dummy_eq(c2)
    assert c1.dummy_eq(c3) is False
    assert c.dummy_eq(c1) is False
    assert c1.dummy_eq(c) is False
    m = Symbol('m')
    n = Symbol('n')
    a = Symbol('a')
    d1 = ImageSet(Lambda(m, m * pi), S.Integers)
    d2 = ImageSet(Lambda(n, n * pi), S.Integers)
    c1 = ConditionSet(x, Ne(a, 0), d1)
    c2 = ConditionSet(x, Ne(a, 0), d2)
    assert c1.dummy_eq(c2)

def test_contains():
    if False:
        print('Hello World!')
    assert 6 in ConditionSet(x, x > 5, Interval(1, 7))
    assert (8 in ConditionSet(x, y > 5, Interval(1, 7))) is False
    raises(TypeError, lambda : 6 in ConditionSet(x, y > 5, Interval(1, 7)))
    raises(TypeError, lambda : 0 in ConditionSet(x, 1 / x >= 0, S.Reals))
    assert ConditionSet(x, y > 5, Interval(1, 7)).contains(6) == (y > 5)
    assert ConditionSet(x, y > 5, Interval(1, 7)).contains(8) is S.false
    assert ConditionSet(x, y > 5, Interval(1, 7)).contains(w) == And(Contains(w, Interval(1, 7)), y > 5)
    assert ConditionSet(x, 1 / x >= 0, S.Reals).contains(0) == Contains(0, ConditionSet(x, 1 / x >= 0, S.Reals), evaluate=False)
    c = ConditionSet((x, y), x + y > 1, S.Integers ** 2)
    assert not c.contains(1)
    assert c.contains((2, 1))
    assert not c.contains((0, 1))
    c = ConditionSet((w, (x, y)), w + x + y > 1, S.Integers * S.Integers ** 2)
    assert not c.contains(1)
    assert not c.contains((1, 2))
    assert not c.contains(((1, 2), 3))
    assert not c.contains(((1, 2), (3, 4)))
    assert c.contains((1, (3, 4)))

def test_as_relational():
    if False:
        for i in range(10):
            print('nop')
    assert ConditionSet((x, y), x > 1, S.Integers ** 2).as_relational((x, y)) == (x > 1) & Contains(x, S.Integers) & Contains(y, S.Integers)
    assert ConditionSet(x, x > 1, S.Integers).as_relational(x) == Contains(x, S.Integers) & (x > 1)

def test_flatten():
    if False:
        return 10
    'Tests whether there is basic denesting functionality'
    inner = ConditionSet(x, sin(x) + x > 0)
    outer = ConditionSet(x, Contains(x, inner), S.Reals)
    assert outer == ConditionSet(x, sin(x) + x > 0, S.Reals)
    inner = ConditionSet(y, sin(y) + y > 0)
    outer = ConditionSet(x, Contains(y, inner), S.Reals)
    assert outer != ConditionSet(x, sin(x) + x > 0, S.Reals)
    inner = ConditionSet(x, sin(x) + x > 0).intersect(Interval(-1, 1))
    outer = ConditionSet(x, Contains(x, inner), S.Reals)
    assert outer == ConditionSet(x, sin(x) + x > 0, Interval(-1, 1))

def test_duplicate():
    if False:
        return 10
    from sympy.core.function import BadSignatureError
    dup = symbols('a,a')
    raises(BadSignatureError, lambda : ConditionSet(dup, x < 0))

def test_SetKind_ConditionSet():
    if False:
        while True:
            i = 10
    assert ConditionSet(x, Eq(sin(x), 0), Interval(0, 2 * pi)).kind is SetKind(NumberKind)
    assert ConditionSet(x, x < 0).kind is SetKind(NumberKind)