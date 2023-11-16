from sympy.testing.pytest import XFAIL
from sympy.parsing.latex.lark import parse_latex_lark
from sympy.external import import_module
from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.function import Derivative, Function
from sympy.core.numbers import E, oo, Rational
from sympy.core.power import Pow
from sympy.core.parameters import evaluate
from sympy.core.relational import GreaterThan, LessThan, StrictGreaterThan, StrictLessThan, Unequality
from sympy.core.symbol import Symbol
from sympy.functions.combinatorial.factorials import binomial, factorial
from sympy.functions.elementary.complexes import Abs, conjugate
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.miscellaneous import root, sqrt, Min, Max
from sympy.functions.elementary.trigonometric import asin, cos, csc, sec, sin, tan
from sympy.integrals.integrals import Integral
from sympy.series.limits import Limit
from sympy.core.relational import Eq, Ne, Lt, Le, Gt, Ge
from sympy.physics.quantum import Bra, Ket, InnerProduct
from sympy.abc import x, y, z, a, b, c, d, t, k, n
from .test_latex import theta, f, _Add, _Mul, _Pow, _Sqrt, _Conjugate, _Abs, _factorial, _exp, _binomial
lark = import_module('lark')
disabled = lark is None

def _Min(*args):
    if False:
        while True:
            i = 10
    return Min(*args, evaluate=False)

def _Max(*args):
    if False:
        while True:
            i = 10
    return Max(*args, evaluate=False)

def _log(a, b=E):
    if False:
        for i in range(10):
            print('nop')
    if b == E:
        return log(a, evaluate=False)
    else:
        return log(a, b, evaluate=False)
SYMBOL_EXPRESSION_PAIRS = [('x_0', Symbol('x_{0}')), ('x_{1}', Symbol('x_{1}')), ('x_a', Symbol('x_{a}')), ('x_{b}', Symbol('x_{b}')), ('h_\\theta', Symbol('h_{theta}')), ('h_{\\theta}', Symbol('h_{theta}')), ("y''_1", Symbol("y_{1}''")), ("y_1''", Symbol("y_{1}''")), ('\\mathit{x}', Symbol('x')), ('\\mathit{test}', Symbol('test')), ('\\mathit{TEST}', Symbol('TEST')), ('\\mathit{HELLO world}', Symbol('HELLO world'))]
UNEVALUATED_SIMPLE_EXPRESSION_PAIRS = [('0', 0), ('1', 1), ('-3.14', -3.14), ('(-7.13)(1.5)', _Mul(-7.13, 1.5)), ('1+1', _Add(1, 1)), ('0+1', _Add(0, 1)), ('1*2', _Mul(1, 2)), ('0*1', _Mul(0, 1)), ('x', x), ('2x', 2 * x), ('3x - 1', _Add(_Mul(3, x), -1)), ('-c', -c), ('\\infty', oo), ('a \\cdot b', a * b), ('1 \\times 2 ', _Mul(1, 2)), ('a / b', a / b), ('a \\div b', a / b), ('a + b', a + b), ('a + b - a', _Add(a + b, -a)), ('(x + y) z', _Mul(_Add(x, y), z)), ("a'b+ab'", _Add(_Mul(Symbol("a'"), b), _Mul(a, Symbol("b'"))))]
EVALUATED_SIMPLE_EXPRESSION_PAIRS = [('(-7.13)(1.5)', -10.695), ('1+1', 2), ('0+1', 1), ('1*2', 2), ('0*1', 0), ('2x', 2 * x), ('3x - 1', 3 * x - 1), ('-c', -c), ('a \\cdot b', a * b), ('1 \\times 2 ', 2), ('a / b', a / b), ('a \\div b', a / b), ('a + b', a + b), ('a + b - a', b), ('(x + y) z', (x + y) * z)]
UNEVALUATED_FRACTION_EXPRESSION_PAIRS = [('\\frac{a}{b}', a / b), ('\\dfrac{a}{b}', a / b), ('\\tfrac{a}{b}', a / b), ('\\frac12', _Mul(1, _Pow(2, -1))), ('\\frac12y', _Mul(_Mul(1, _Pow(2, -1)), y)), ('\\frac1234', _Mul(_Mul(1, _Pow(2, -1)), 34)), ('\\frac2{3}', _Mul(2, _Pow(3, -1))), ('\\frac{a + b}{c}', _Mul(a + b, _Pow(c, -1))), ('\\frac{7}{3}', _Mul(7, _Pow(3, -1)))]
EVALUATED_FRACTION_EXPRESSION_PAIRS = [('\\frac{a}{b}', a / b), ('\\dfrac{a}{b}', a / b), ('\\tfrac{a}{b}', a / b), ('\\frac12', Rational(1, 2)), ('\\frac12y', y / 2), ('\\frac1234', 17), ('\\frac2{3}', Rational(2, 3)), ('\\frac{a + b}{c}', (a + b) / c), ('\\frac{7}{3}', Rational(7, 3))]
RELATION_EXPRESSION_PAIRS = [('x = y', Eq(x, y)), ('x \\neq y', Ne(x, y)), ('x < y', Lt(x, y)), ('x > y', Gt(x, y)), ('x \\leq y', Le(x, y)), ('x \\geq y', Ge(x, y)), ('x \\le y', Le(x, y)), ('x \\ge y', Ge(x, y)), ('x < y', StrictLessThan(x, y)), ('x \\leq y', LessThan(x, y)), ('x > y', StrictGreaterThan(x, y)), ('x \\geq y', GreaterThan(x, y)), ('x \\neq y', Unequality(x, y)), ('a^2 + b^2 = c^2', Eq(a ** 2 + b ** 2, c ** 2))]
UNEVALUATED_POWER_EXPRESSION_PAIRS = [('x^2', x ** 2), ('x^\\frac{1}{2}', _Pow(x, _Mul(1, _Pow(2, -1)))), ('x^{3 + 1}', x ** _Add(3, 1)), ('\\pi^{|xy|}', Symbol('pi') ** _Abs(x * y)), ('5^0 - 4^0', _Add(_Pow(5, 0), _Mul(-1, _Pow(4, 0))))]
EVALUATED_POWER_EXPRESSION_PAIRS = [('x^2', x ** 2), ('x^\\frac{1}{2}', sqrt(x)), ('x^{3 + 1}', x ** 4), ('\\pi^{|xy|}', Symbol('pi') ** _Abs(x * y)), ('5^0 - 4^0', 0)]
UNEVALUATED_INTEGRAL_EXPRESSION_PAIRS = [('\\int x dx', Integral(_Mul(1, x), x)), ('\\int x \\, dx', Integral(_Mul(1, x), x)), ('\\int x d\\theta', Integral(_Mul(1, x), theta)), ('\\int (x^2 - y)dx', Integral(_Mul(1, x ** 2 - y), x)), ('\\int x + a dx', Integral(_Mul(1, _Add(x, a)), x)), ('\\int da', Integral(_Mul(1, 1), a)), ('\\int_0^7 dx', Integral(_Mul(1, 1), (x, 0, 7))), ('\\int\\limits_{0}^{1} x dx', Integral(_Mul(1, x), (x, 0, 1))), ('\\int_a^b x dx', Integral(_Mul(1, x), (x, a, b))), ('\\int^b_a x dx', Integral(_Mul(1, x), (x, a, b))), ('\\int_{a}^b x dx', Integral(_Mul(1, x), (x, a, b))), ('\\int^{b}_a x dx', Integral(_Mul(1, x), (x, a, b))), ('\\int_{a}^{b} x dx', Integral(_Mul(1, x), (x, a, b))), ('\\int^{b}_{a} x dx', Integral(_Mul(1, x), (x, a, b))), ('\\int_{f(a)}^{f(b)} f(z) dz', Integral(f(z), (z, f(a), f(b)))), ('\\int a + b + c dx', Integral(_Mul(1, _Add(_Add(a, b), c)), x)), ('\\int \\frac{dz}{z}', Integral(_Mul(1, _Mul(1, Pow(z, -1))), z)), ('\\int \\frac{3 dz}{z}', Integral(_Mul(1, _Mul(3, _Pow(z, -1))), z)), ('\\int \\frac{1}{x} dx', Integral(_Mul(1, _Mul(1, Pow(x, -1))), x)), ('\\int \\frac{1}{a} + \\frac{1}{b} dx', Integral(_Mul(1, _Add(_Mul(1, _Pow(a, -1)), _Mul(1, Pow(b, -1)))), x)), ('\\int \\frac{1}{x} + 1 dx', Integral(_Mul(1, _Add(_Mul(1, _Pow(x, -1)), 1)), x))]
EVALUATED_INTEGRAL_EXPRESSION_PAIRS = [('\\int x dx', Integral(x, x)), ('\\int x \\, dx', Integral(x, x)), ('\\int x d\\theta', Integral(x, theta)), ('\\int (x^2 - y)dx', Integral(x ** 2 - y, x)), ('\\int x + a dx', Integral(x + a, x)), ('\\int da', Integral(1, a)), ('\\int_0^7 dx', Integral(1, (x, 0, 7))), ('\\int\\limits_{0}^{1} x dx', Integral(x, (x, 0, 1))), ('\\int_a^b x dx', Integral(x, (x, a, b))), ('\\int^b_a x dx', Integral(x, (x, a, b))), ('\\int_{a}^b x dx', Integral(x, (x, a, b))), ('\\int^{b}_a x dx', Integral(x, (x, a, b))), ('\\int_{a}^{b} x dx', Integral(x, (x, a, b))), ('\\int^{b}_{a} x dx', Integral(x, (x, a, b))), ('\\int_{f(a)}^{f(b)} f(z) dz', Integral(f(z), (z, f(a), f(b)))), ('\\int a + b + c dx', Integral(a + b + c, x)), ('\\int \\frac{dz}{z}', Integral(Pow(z, -1), z)), ('\\int \\frac{3 dz}{z}', Integral(3 * Pow(z, -1), z)), ('\\int \\frac{1}{x} dx', Integral(1 / x, x)), ('\\int \\frac{1}{a} + \\frac{1}{b} dx', Integral(1 / a + 1 / b, x)), ('\\int \\frac{1}{x} + 1 dx', Integral(1 / x + 1, x))]
DERIVATIVE_EXPRESSION_PAIRS = [('\\frac{d}{dx} x', Derivative(x, x)), ('\\frac{d}{dt} x', Derivative(x, t)), ('\\frac{d}{dx} ( \\tan x )', Derivative(tan(x), x)), ('\\frac{d f(x)}{dx}', Derivative(f(x), x)), ('\\frac{d\\theta(x)}{dx}', Derivative(Function('theta')(x), x))]
TRIGONOMETRIC_EXPRESSION_PAIRS = [('\\sin \\theta', sin(theta)), ('\\sin(\\theta)', sin(theta)), ('\\sin^{-1} a', asin(a)), ('\\sin a \\cos b', _Mul(sin(a), cos(b))), ('\\sin \\cos \\theta', sin(cos(theta))), ('\\sin(\\cos \\theta)', sin(cos(theta))), ('(\\csc x)(\\sec y)', csc(x) * sec(y)), ('\\frac{\\sin{x}}2', _Mul(sin(x), _Pow(2, -1)))]
UNEVALUATED_LIMIT_EXPRESSION_PAIRS = [('\\lim_{x \\to 3} a', Limit(a, x, 3, dir='+-')), ('\\lim_{x \\rightarrow 3} a', Limit(a, x, 3, dir='+-')), ('\\lim_{x \\Rightarrow 3} a', Limit(a, x, 3, dir='+-')), ('\\lim_{x \\longrightarrow 3} a', Limit(a, x, 3, dir='+-')), ('\\lim_{x \\Longrightarrow 3} a', Limit(a, x, 3, dir='+-')), ('\\lim_{x \\to 3^{+}} a', Limit(a, x, 3, dir='+')), ('\\lim_{x \\to 3^{-}} a', Limit(a, x, 3, dir='-')), ('\\lim_{x \\to 3^+} a', Limit(a, x, 3, dir='+')), ('\\lim_{x \\to 3^-} a', Limit(a, x, 3, dir='-')), ('\\lim_{x \\to \\infty} \\frac{1}{x}', Limit(_Mul(1, _Pow(x, -1)), x, oo))]
EVALUATED_LIMIT_EXPRESSION_PAIRS = [('\\lim_{x \\to \\infty} \\frac{1}{x}', Limit(1 / x, x, oo))]
UNEVALUATED_SQRT_EXPRESSION_PAIRS = [('\\sqrt{x}', sqrt(x)), ('\\sqrt{x + b}', sqrt(_Add(x, b))), ('\\sqrt[3]{\\sin x}', _Pow(sin(x), _Pow(3, -1))), ('\\sqrt[y]{\\sin x}', root(sin(x), y)), ('\\sqrt[\\theta]{\\sin x}', root(sin(x), theta)), ('\\sqrt{\\frac{12}{6}}', _Sqrt(_Mul(12, _Pow(6, -1))))]
EVALUATED_SQRT_EXPRESSION_PAIRS = [('\\sqrt{x}', sqrt(x)), ('\\sqrt{x + b}', sqrt(x + b)), ('\\sqrt[3]{\\sin x}', root(sin(x), 3)), ('\\sqrt[y]{\\sin x}', root(sin(x), y)), ('\\sqrt[\\theta]{\\sin x}', root(sin(x), theta)), ('\\sqrt{\\frac{12}{6}}', sqrt(2))]
UNEVALUATED_FACTORIAL_EXPRESSION_PAIRS = [('x!', _factorial(x)), ('100!', _factorial(100)), ('\\theta!', _factorial(theta)), ('(x + 1)!', _factorial(_Add(x, 1))), ('(x!)!', _factorial(_factorial(x))), ('x!!!', _factorial(_factorial(_factorial(x)))), ('5!7!', _Mul(_factorial(5), _factorial(7)))]
EVALUATED_FACTORIAL_EXPRESSION_PAIRS = [('x!', factorial(x)), ('100!', factorial(100)), ('\\theta!', factorial(theta)), ('(x + 1)!', factorial(x + 1)), ('(x!)!', factorial(factorial(x))), ('x!!!', factorial(factorial(factorial(x)))), ('5!7!', factorial(5) * factorial(7))]
UNEVALUATED_SUM_EXPRESSION_PAIRS = [('\\sum_{k = 1}^{3} c', Sum(_Mul(1, c), (k, 1, 3))), ('\\sum_{k = 1}^3 c', Sum(_Mul(1, c), (k, 1, 3))), ('\\sum^{3}_{k = 1} c', Sum(_Mul(1, c), (k, 1, 3))), ('\\sum^3_{k = 1} c', Sum(_Mul(1, c), (k, 1, 3))), ('\\sum_{k = 1}^{10} k^2', Sum(_Mul(1, k ** 2), (k, 1, 10))), ('\\sum_{n = 0}^{\\infty} \\frac{1}{n!}', Sum(_Mul(1, _Mul(1, _Pow(_factorial(n), -1))), (n, 0, oo)))]
EVALUATED_SUM_EXPRESSION_PAIRS = [('\\sum_{k = 1}^{3} c', Sum(c, (k, 1, 3))), ('\\sum_{k = 1}^3 c', Sum(c, (k, 1, 3))), ('\\sum^{3}_{k = 1} c', Sum(c, (k, 1, 3))), ('\\sum^3_{k = 1} c', Sum(c, (k, 1, 3))), ('\\sum_{k = 1}^{10} k^2', Sum(k ** 2, (k, 1, 10))), ('\\sum_{n = 0}^{\\infty} \\frac{1}{n!}', Sum(1 / factorial(n), (n, 0, oo)))]
UNEVALUATED_PRODUCT_EXPRESSION_PAIRS = [('\\prod_{a = b}^{c} x', Product(x, (a, b, c))), ('\\prod_{a = b}^c x', Product(x, (a, b, c))), ('\\prod^{c}_{a = b} x', Product(x, (a, b, c))), ('\\prod^c_{a = b} x', Product(x, (a, b, c)))]
APPLIED_FUNCTION_EXPRESSION_PAIRS = [('f(x)', f(x)), ('f(x, y)', f(x, y)), ('f(x, y, z)', f(x, y, z)), ("f'_1(x)", Function("f_{1}'")(x)), ("f_{1}''(x+y)", Function("f_{1}''")(x + y)), ('h_{\\theta}(x_0, x_1)', Function('h_{theta}')(Symbol('x_{0}'), Symbol('x_{1}')))]
UNEVALUATED_COMMON_FUNCTION_EXPRESSION_PAIRS = [('|x|', _Abs(x)), ('||x||', _Abs(Abs(x))), ('|x||y|', _Abs(x) * _Abs(y)), ('||x||y||', _Abs(_Abs(x) * _Abs(y))), ('\\lfloor x \\rfloor', floor(x)), ('\\lceil x \\rceil', ceiling(x)), ('\\exp x', _exp(x)), ('\\exp(x)', _exp(x)), ('\\lg x', _log(x, 10)), ('\\ln x', _log(x)), ('\\ln xy', _log(x * y)), ('\\log x', _log(x)), ('\\log xy', _log(x * y)), ('\\log_{2} x', _log(x, 2)), ('\\log_{a} x', _log(x, a)), ('\\log_{11} x', _log(x, 11)), ('\\log_{a^2} x', _log(x, _Pow(a, 2))), ('\\log_2 x', _log(x, 2)), ('\\log_a x', _log(x, a)), ('\\overline{z}', _Conjugate(z)), ('\\overline{\\overline{z}}', _Conjugate(_Conjugate(z))), ('\\overline{x + y}', _Conjugate(_Add(x, y))), ('\\overline{x} + \\overline{y}', _Conjugate(x) + _Conjugate(y)), ('\\min(a, b)', _Min(a, b)), ('\\min(a, b, c - d, xy)', _Min(a, b, c - d, x * y)), ('\\max(a, b)', _Max(a, b)), ('\\max(a, b, c - d, xy)', _Max(a, b, c - d, x * y)), ('\\langle x |', Bra('x')), ('| x \\rangle', Ket('x')), ('\\langle x | y \\rangle', InnerProduct(Bra('x'), Ket('y')))]
EVALUATED_COMMON_FUNCTION_EXPRESSION_PAIRS = [('|x|', Abs(x)), ('||x||', Abs(Abs(x))), ('|x||y|', Abs(x) * Abs(y)), ('||x||y||', Abs(Abs(x) * Abs(y))), ('\\lfloor x \\rfloor', floor(x)), ('\\lceil x \\rceil', ceiling(x)), ('\\exp x', exp(x)), ('\\exp(x)', exp(x)), ('\\lg x', log(x, 10)), ('\\ln x', log(x)), ('\\ln xy', log(x * y)), ('\\log x', log(x)), ('\\log xy', log(x * y)), ('\\log_{2} x', log(x, 2)), ('\\log_{a} x', log(x, a)), ('\\log_{11} x', log(x, 11)), ('\\log_{a^2} x', log(x, _Pow(a, 2))), ('\\log_2 x', log(x, 2)), ('\\log_a x', log(x, a)), ('\\overline{z}', conjugate(z)), ('\\overline{\\overline{z}}', conjugate(conjugate(z))), ('\\overline{x + y}', conjugate(x + y)), ('\\overline{x} + \\overline{y}', conjugate(x) + conjugate(y)), ('\\min(a, b)', Min(a, b)), ('\\min(a, b, c - d, xy)', Min(a, b, c - d, x * y)), ('\\max(a, b)', Max(a, b)), ('\\max(a, b, c - d, xy)', Max(a, b, c - d, x * y)), ('\\langle x |', Bra('x')), ('| x \\rangle', Ket('x')), ('\\langle x | y \\rangle', InnerProduct(Bra('x'), Ket('y')))]
SPACING_RELATED_EXPRESSION_PAIRS = [('a \\, b', _Mul(a, b)), ('a \\thinspace b', _Mul(a, b)), ('a \\: b', _Mul(a, b)), ('a \\medspace b', _Mul(a, b)), ('a \\; b', _Mul(a, b)), ('a \\thickspace b', _Mul(a, b)), ('a \\quad b', _Mul(a, b)), ('a \\qquad b', _Mul(a, b)), ('a \\! b', _Mul(a, b)), ('a \\negthinspace b', _Mul(a, b)), ('a \\negmedspace b', _Mul(a, b)), ('a \\negthickspace b', _Mul(a, b))]
UNEVALUATED_BINOMIAL_EXPRESSION_PAIRS = [('\\binom{n}{k}', _binomial(n, k)), ('\\tbinom{n}{k}', _binomial(n, k)), ('\\dbinom{n}{k}', _binomial(n, k)), ('\\binom{n}{0}', _binomial(n, 0)), ('x^\\binom{n}{k}', _Pow(x, _binomial(n, k)))]
EVALUATED_BINOMIAL_EXPRESSION_PAIRS = [('\\binom{n}{k}', binomial(n, k)), ('\\tbinom{n}{k}', binomial(n, k)), ('\\dbinom{n}{k}', binomial(n, k)), ('\\binom{n}{0}', binomial(n, 0)), ('x^\\binom{n}{k}', x ** binomial(n, k))]
MISCELLANEOUS_EXPRESSION_PAIRS = [('\\left(x + y\\right) z', _Mul(_Add(x, y), z)), ('\\left( x + y\\right ) z', _Mul(_Add(x, y), z)), ('\\left(  x + y\\right ) z', _Mul(_Add(x, y), z))]

def test_symbol_expressions():
    if False:
        i = 10
        return i + 15
    expected_failures = {6, 7}
    for (i, (latex_str, sympy_expr)) in enumerate(SYMBOL_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str

def test_simple_expressions():
    if False:
        i = 10
        return i + 15
    expected_failures = {20}
    for (i, (latex_str, sympy_expr)) in enumerate(UNEVALUATED_SIMPLE_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str
    for (i, (latex_str, sympy_expr)) in enumerate(EVALUATED_SIMPLE_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str

def test_fraction_expressions():
    if False:
        while True:
            i = 10
    for (latex_str, sympy_expr) in UNEVALUATED_FRACTION_EXPRESSION_PAIRS:
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str
    for (latex_str, sympy_expr) in EVALUATED_FRACTION_EXPRESSION_PAIRS:
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str

def test_relation_expressions():
    if False:
        return 10
    for (latex_str, sympy_expr) in RELATION_EXPRESSION_PAIRS:
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str

def test_power_expressions():
    if False:
        i = 10
        return i + 15
    expected_failures = {3}
    for (i, (latex_str, sympy_expr)) in enumerate(UNEVALUATED_POWER_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str
    for (i, (latex_str, sympy_expr)) in enumerate(EVALUATED_POWER_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str

def test_integral_expressions():
    if False:
        return 10
    expected_failures = {14}
    for (i, (latex_str, sympy_expr)) in enumerate(UNEVALUATED_INTEGRAL_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, i
    for (i, (latex_str, sympy_expr)) in enumerate(EVALUATED_INTEGRAL_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str

def test_derivative_expressions():
    if False:
        print('Hello World!')
    expected_failures = {3, 4}
    for (i, (latex_str, sympy_expr)) in enumerate(DERIVATIVE_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str
    for (i, (latex_str, sympy_expr)) in enumerate(DERIVATIVE_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str

def test_trigonometric_expressions():
    if False:
        while True:
            i = 10
    expected_failures = {3}
    for (i, (latex_str, sympy_expr)) in enumerate(TRIGONOMETRIC_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str

def test_limit_expressions():
    if False:
        while True:
            i = 10
    for (latex_str, sympy_expr) in UNEVALUATED_LIMIT_EXPRESSION_PAIRS:
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str

def test_square_root_expressions():
    if False:
        print('Hello World!')
    for (latex_str, sympy_expr) in UNEVALUATED_SQRT_EXPRESSION_PAIRS:
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str
    for (latex_str, sympy_expr) in EVALUATED_SQRT_EXPRESSION_PAIRS:
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str

def test_factorial_expressions():
    if False:
        while True:
            i = 10
    for (latex_str, sympy_expr) in UNEVALUATED_FACTORIAL_EXPRESSION_PAIRS:
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str
    for (latex_str, sympy_expr) in EVALUATED_FACTORIAL_EXPRESSION_PAIRS:
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str

def test_sum_expressions():
    if False:
        while True:
            i = 10
    for (latex_str, sympy_expr) in UNEVALUATED_SUM_EXPRESSION_PAIRS:
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str
    for (latex_str, sympy_expr) in EVALUATED_SUM_EXPRESSION_PAIRS:
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str

def test_product_expressions():
    if False:
        i = 10
        return i + 15
    for (latex_str, sympy_expr) in UNEVALUATED_PRODUCT_EXPRESSION_PAIRS:
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str

@XFAIL
def test_applied_function_expressions():
    if False:
        print('Hello World!')
    expected_failures = {0, 3, 4}
    for (i, (latex_str, sympy_expr)) in enumerate(APPLIED_FUNCTION_EXPRESSION_PAIRS):
        if i in expected_failures:
            continue
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str

def test_common_function_expressions():
    if False:
        i = 10
        return i + 15
    for (latex_str, sympy_expr) in UNEVALUATED_COMMON_FUNCTION_EXPRESSION_PAIRS:
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str
    for (latex_str, sympy_expr) in EVALUATED_COMMON_FUNCTION_EXPRESSION_PAIRS:
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str

@XFAIL
def test_spacing():
    if False:
        print('Hello World!')
    for (latex_str, sympy_expr) in SPACING_RELATED_EXPRESSION_PAIRS:
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str

def test_binomial_expressions():
    if False:
        for i in range(10):
            print('nop')
    for (latex_str, sympy_expr) in UNEVALUATED_BINOMIAL_EXPRESSION_PAIRS:
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str
    for (latex_str, sympy_expr) in EVALUATED_BINOMIAL_EXPRESSION_PAIRS:
        assert parse_latex_lark(latex_str) == sympy_expr, latex_str

def test_miscellaneous_expressions():
    if False:
        for i in range(10):
            print('nop')
    for (latex_str, sympy_expr) in MISCELLANEOUS_EXPRESSION_PAIRS:
        with evaluate(False):
            assert parse_latex_lark(latex_str) == sympy_expr, latex_str