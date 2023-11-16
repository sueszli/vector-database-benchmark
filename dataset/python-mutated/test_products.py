from sympy.concrete.products import Product, product
from sympy.concrete.summations import Sum
from sympy.core.function import Derivative, Function, diff
from sympy.core.numbers import Rational, oo, pi
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol, symbols
from sympy.functions.combinatorial.factorials import rf, factorial
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.simplify.combsimp import combsimp
from sympy.simplify.simplify import simplify
from sympy.testing.pytest import raises
(a, k, n, m, x) = symbols('a,k,n,m,x', integer=True)
f = Function('f')

def test_karr_convention():
    if False:
        for i in range(10):
            print('nop')
    i = Symbol('i', integer=True)
    k = Symbol('k', integer=True)
    j = Symbol('j', integer=True, positive=True)
    m = k
    n = k + j
    a = m
    b = n - 1
    S1 = Product(i ** 2, (i, a, b)).doit()
    m = k + j
    n = k
    a = m
    b = n - 1
    S2 = Product(i ** 2, (i, a, b)).doit()
    assert S1 * S2 == 1
    m = k
    n = k
    a = m
    b = n - 1
    Sz = Product(i ** 2, (i, a, b)).doit()
    assert Sz == 1
    f = Function('f')
    m = 2
    n = 11
    a = m
    b = n - 1
    S1 = Product(f(i), (i, a, b)).doit()
    m = 11
    n = 2
    a = m
    b = n - 1
    S2 = Product(f(i), (i, a, b)).doit()
    assert simplify(S1 * S2) == 1
    m = 5
    n = 5
    a = m
    b = n - 1
    Sz = Product(f(i), (i, a, b)).doit()
    assert Sz == 1

def test_karr_proposition_2a():
    if False:
        for i in range(10):
            print('nop')
    (i, u, v) = symbols('i u v', integer=True)

    def test_the_product(m, n):
        if False:
            while True:
                i = 10
        g = i ** 3 + 2 * i ** 2 - 3 * i
        f = simplify(g.subs(i, i + 1) / g)
        a = m
        b = n - 1
        P = Product(f, (i, a, b)).doit()
        assert combsimp(P / (g.subs(i, n) / g.subs(i, m))) == 1
    test_the_product(u, u + v)
    test_the_product(u, u)
    test_the_product(u + v, u)

def test_karr_proposition_2b():
    if False:
        print('Hello World!')
    (i, u, v, w) = symbols('i u v w', integer=True)

    def test_the_product(l, n, m):
        if False:
            print('Hello World!')
        s = i ** 3
        a = l
        b = n - 1
        S1 = Product(s, (i, a, b)).doit()
        a = l
        b = m - 1
        S2 = Product(s, (i, a, b)).doit()
        a = m
        b = n - 1
        S3 = Product(s, (i, a, b)).doit()
        assert combsimp(S1 / (S2 * S3)) == 1
    test_the_product(u, u + v, u + v + w)
    test_the_product(u, u + v, u + v)
    test_the_product(u, u + v + w, v)
    test_the_product(u, u, u + v)
    test_the_product(u, u, u)
    test_the_product(u + v, u + v, u)
    test_the_product(u + v, u, u + w)
    test_the_product(u + v, u, u)
    test_the_product(u + v + w, u + v, u)

def test_simple_products():
    if False:
        i = 10
        return i + 15
    assert product(2, (k, a, n)) == 2 ** (n - a + 1)
    assert product(k, (k, 1, n)) == factorial(n)
    assert product(k ** 3, (k, 1, n)) == factorial(n) ** 3
    assert product(k + 1, (k, 0, n - 1)) == factorial(n)
    assert product(k + 1, (k, a, n - 1)) == rf(1 + a, n - a)
    assert product(cos(k), (k, 0, 5)) == cos(1) * cos(2) * cos(3) * cos(4) * cos(5)
    assert product(cos(k), (k, 3, 5)) == cos(3) * cos(4) * cos(5)
    assert product(cos(k), (k, 1, Rational(5, 2))) != cos(1) * cos(2)
    assert isinstance(product(k ** k, (k, 1, n)), Product)
    assert Product(x ** k, (k, 1, n)).variables == [k]
    raises(ValueError, lambda : Product(n))
    raises(ValueError, lambda : Product(n, k))
    raises(ValueError, lambda : Product(n, k, 1))
    raises(ValueError, lambda : Product(n, k, 1, 10))
    raises(ValueError, lambda : Product(n, (k, 1)))
    assert product(1, (n, 1, oo)) == 1
    assert product(2, (n, 1, oo)) is oo
    assert product(-1, (n, 1, oo)).func is Product

def test_multiple_products():
    if False:
        i = 10
        return i + 15
    assert product(x, (n, 1, k), (k, 1, m)) == x ** (m ** 2 / 2 + m / 2)
    assert product(f(n), (n, 1, m), (m, 1, k)) == Product(f(n), (n, 1, m), (m, 1, k)).doit()
    assert Product(f(n), (m, 1, k), (n, 1, k)).doit() == Product(Product(f(n), (m, 1, k)), (n, 1, k)).doit() == product(f(n), (m, 1, k), (n, 1, k)) == product(product(f(n), (m, 1, k)), (n, 1, k)) == Product(f(n) ** k, (n, 1, k))
    assert Product(x, (x, 1, k), (k, 1, n)).doit() == Product(factorial(k), (k, 1, n))
    assert Product(x ** k, (n, 1, k), (k, 1, m)).variables == [n, k]

def test_rational_products():
    if False:
        return 10
    assert product(1 + 1 / k, (k, 1, n)) == rf(2, n) / factorial(n)

def test_special_products():
    if False:
        for i in range(10):
            print('nop')
    assert product((4 * k) ** 2 / (4 * k ** 2 - 1), (k, 1, n)) == 4 ** n * factorial(n) ** 2 / rf(S.Half, n) / rf(Rational(3, 2), n)
    assert product(1 + a / k ** 2, (k, 1, n)) == rf(1 - sqrt(-a), n) * rf(1 + sqrt(-a), n) / factorial(n) ** 2

def test__eval_product():
    if False:
        i = 10
        return i + 15
    from sympy.abc import i, n
    a = Function('a')
    assert product(2 * a(i), (i, 1, n)) == 2 ** n * Product(a(i), (i, 1, n))
    assert product(2 ** i, (i, 1, n)) == 2 ** (n * (n + 1) / 2)
    (k, m) = symbols('k m', integer=True)
    assert product(2 ** i, (i, k, m)) == 2 ** (-k ** 2 / 2 + k / 2 + m ** 2 / 2 + m / 2)
    n = Symbol('n', negative=True, integer=True)
    p = Symbol('p', positive=True, integer=True)
    assert product(2 ** i, (i, n, p)) == 2 ** (-n ** 2 / 2 + n / 2 + p ** 2 / 2 + p / 2)
    assert product(2 ** i, (i, p, n)) == 2 ** (n ** 2 / 2 + n / 2 - p ** 2 / 2 + p / 2)

def test_product_pow():
    if False:
        for i in range(10):
            print('nop')
    assert product(2 ** f(k), (k, 1, n)) == 2 ** Sum(f(k), (k, 1, n))
    assert product(2 ** (2 * f(k)), (k, 1, n)) == 2 ** Sum(2 * f(k), (k, 1, n))

def test_infinite_product():
    if False:
        while True:
            i = 10
    assert isinstance(Product(2 ** (1 / factorial(n)), (n, 0, oo)), Product)

def test_conjugate_transpose():
    if False:
        print('Hello World!')
    p = Product(x ** k, (k, 1, 3))
    assert p.adjoint().doit() == p.doit().adjoint()
    assert p.conjugate().doit() == p.doit().conjugate()
    assert p.transpose().doit() == p.doit().transpose()
    (A, B) = symbols('A B', commutative=False)
    p = Product(A * B ** k, (k, 1, 3))
    assert p.adjoint().doit() == p.doit().adjoint()
    assert p.conjugate().doit() == p.doit().conjugate()
    assert p.transpose().doit() == p.doit().transpose()
    p = Product(B ** k * A, (k, 1, 3))
    assert p.adjoint().doit() == p.doit().adjoint()
    assert p.conjugate().doit() == p.doit().conjugate()
    assert p.transpose().doit() == p.doit().transpose()

def test_simplify_prod():
    if False:
        for i in range(10):
            print('nop')
    (y, t, b, c, v, d) = symbols('y, t, b, c, v, d', integer=True)
    _simplify = lambda e: simplify(e, doit=False)
    assert _simplify(Product(x * y, (x, n, m), (y, a, k)) * Product(y, (x, n, m), (y, a, k))) == Product(x * y ** 2, (x, n, m), (y, a, k))
    assert _simplify(3 * y * Product(x, (x, n, m)) * Product(x, (x, m + 1, a))) == 3 * y * Product(x, (x, n, a))
    assert _simplify(Product(x, (x, k + 1, a)) * Product(x, (x, n, k))) == Product(x, (x, n, a))
    assert _simplify(Product(x, (x, k + 1, a)) * Product(x + 1, (x, n, k))) == Product(x, (x, k + 1, a)) * Product(x + 1, (x, n, k))
    assert _simplify(Product(x, (t, a, b)) * Product(y, (t, a, b)) * Product(x, (t, b + 1, c))) == Product(x * y, (t, a, b)) * Product(x, (t, b + 1, c))
    assert _simplify(Product(x, (t, a, b)) * Product(x, (t, b + 1, c)) * Product(y, (t, a, b))) == Product(x * y, (t, a, b)) * Product(x, (t, b + 1, c))
    assert _simplify(Product(sin(t) ** 2 + cos(t) ** 2 + 1, (t, a, b))) == Product(2, (t, a, b))
    assert _simplify(Product(sin(t) ** 2 + cos(t) ** 2 - 1, (t, a, b))) == Product(0, (t, a, b))
    assert _simplify(Product(v * Product(sin(t) ** 2 + cos(t) ** 2, (t, a, b)), (v, c, d))) == Product(v * Product(1, (t, a, b)), (v, c, d))

def test_change_index():
    if False:
        for i in range(10):
            print('nop')
    (b, y, c, d, z) = symbols('b, y, c, d, z', integer=True)
    assert Product(x, (x, a, b)).change_index(x, x + 1, y) == Product(y - 1, (y, a + 1, b + 1))
    assert Product(x ** 2, (x, a, b)).change_index(x, x - 1) == Product((x + 1) ** 2, (x, a - 1, b - 1))
    assert Product(x ** 2, (x, a, b)).change_index(x, -x, y) == Product((-y) ** 2, (y, -b, -a))
    assert Product(x, (x, a, b)).change_index(x, -x - 1) == Product(-x - 1, (x, -b - 1, -a - 1))
    assert Product(x * y, (x, a, b), (y, c, d)).change_index(x, x - 1, z) == Product((z + 1) * y, (z, a - 1, b - 1), (y, c, d))

def test_reorder():
    if False:
        for i in range(10):
            print('nop')
    (b, y, c, d, z) = symbols('b, y, c, d, z', integer=True)
    assert Product(x * y, (x, a, b), (y, c, d)).reorder((0, 1)) == Product(x * y, (y, c, d), (x, a, b))
    assert Product(x, (x, a, b), (x, c, d)).reorder((0, 1)) == Product(x, (x, c, d), (x, a, b))
    assert Product(x * y + z, (x, a, b), (z, m, n), (y, c, d)).reorder((2, 0), (0, 1)) == Product(x * y + z, (z, m, n), (y, c, d), (x, a, b))
    assert Product(x * y * z, (x, a, b), (y, c, d), (z, m, n)).reorder((0, 1), (1, 2), (0, 2)) == Product(x * y * z, (x, a, b), (z, m, n), (y, c, d))
    assert Product(x * y * z, (x, a, b), (y, c, d), (z, m, n)).reorder((x, y), (y, z), (x, z)) == Product(x * y * z, (x, a, b), (z, m, n), (y, c, d))
    assert Product(x * y, (x, a, b), (y, c, d)).reorder((x, 1)) == Product(x * y, (y, c, d), (x, a, b))
    assert Product(x * y, (x, a, b), (y, c, d)).reorder((y, x)) == Product(x * y, (y, c, d), (x, a, b))

def test_Product_is_convergent():
    if False:
        while True:
            i = 10
    assert Product(1 / n ** 2, (n, 1, oo)).is_convergent() is S.false
    assert Product(exp(1 / n ** 2), (n, 1, oo)).is_convergent() is S.true
    assert Product(1 / n, (n, 1, oo)).is_convergent() is S.false
    assert Product(1 + 1 / n, (n, 1, oo)).is_convergent() is S.false
    assert Product(1 + 1 / n ** 2, (n, 1, oo)).is_convergent() is S.true

def test_reverse_order():
    if False:
        i = 10
        return i + 15
    (x, y, a, b, c, d) = symbols('x, y, a, b, c, d', integer=True)
    assert Product(x, (x, 0, 3)).reverse_order(0) == Product(1 / x, (x, 4, -1))
    assert Product(x * y, (x, 1, 5), (y, 0, 6)).reverse_order(0, 1) == Product(x * y, (x, 6, 0), (y, 7, -1))
    assert Product(x, (x, 1, 2)).reverse_order(0) == Product(1 / x, (x, 3, 0))
    assert Product(x, (x, 1, 3)).reverse_order(0) == Product(1 / x, (x, 4, 0))
    assert Product(x, (x, 1, a)).reverse_order(0) == Product(1 / x, (x, a + 1, 0))
    assert Product(x, (x, a, 5)).reverse_order(0) == Product(1 / x, (x, 6, a - 1))
    assert Product(x, (x, a + 1, a + 5)).reverse_order(0) == Product(1 / x, (x, a + 6, a))
    assert Product(x, (x, a + 1, a + 2)).reverse_order(0) == Product(1 / x, (x, a + 3, a))
    assert Product(x, (x, a + 1, a + 1)).reverse_order(0) == Product(1 / x, (x, a + 2, a))
    assert Product(x, (x, a, b)).reverse_order(0) == Product(1 / x, (x, b + 1, a - 1))
    assert Product(x, (x, a, b)).reverse_order(x) == Product(1 / x, (x, b + 1, a - 1))
    assert Product(x * y, (x, a, b), (y, 2, 5)).reverse_order(x, 1) == Product(x * y, (x, b + 1, a - 1), (y, 6, 1))
    assert Product(x * y, (x, a, b), (y, 2, 5)).reverse_order(y, x) == Product(x * y, (x, b + 1, a - 1), (y, 6, 1))

def test_issue_9983():
    if False:
        i = 10
        return i + 15
    n = Symbol('n', integer=True, positive=True)
    p = Product(1 + 1 / n ** Rational(2, 3), (n, 1, oo))
    assert p.is_convergent() is S.false
    assert product(1 + 1 / n ** Rational(2, 3), (n, 1, oo)) == p.doit()

def test_issue_13546():
    if False:
        i = 10
        return i + 15
    n = Symbol('n')
    k = Symbol('k')
    p = Product(n + 1 / 2 ** k, (k, 0, n - 1)).doit()
    assert p.subs(n, 2).doit() == Rational(15, 2)

def test_issue_14036():
    if False:
        print('Hello World!')
    (a, n) = symbols('a n')
    assert product(1 - a ** 2 / (n * pi) ** 2, [n, 1, oo]) != 0

def test_rewrite_Sum():
    if False:
        for i in range(10):
            print('nop')
    assert Product(1 - S.Half ** 2 / k ** 2, (k, 1, oo)).rewrite(Sum) == exp(Sum(log(1 - 1 / (4 * k ** 2)), (k, 1, oo)))

def test_KroneckerDelta_Product():
    if False:
        i = 10
        return i + 15
    y = Symbol('y')
    assert Product(x * KroneckerDelta(x, y), (x, 0, 1)).doit() == 0

def test_issue_20848():
    if False:
        i = 10
        return i + 15
    _i = Dummy('i')
    (t, y, z) = symbols('t y z')
    assert diff(Product(x, (y, 1, z)), x).as_dummy() == Sum(Product(x, (y, 1, _i - 1)) * Product(x, (y, _i + 1, z)), (_i, 1, z)).as_dummy()
    assert diff(Product(x, (y, 1, z)), x).doit() == x ** (z - 1) * z
    assert diff(Product(x, (y, x, z)), x) == Derivative(Product(x, (y, x, z)), x)
    assert diff(Product(t, (x, 1, z)), x) == S(0)
    assert Product(sin(n * x), (n, -1, 1)).diff(x).doit() == S(0)