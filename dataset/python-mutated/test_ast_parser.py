from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.parsing.ast_parser import parse_expr
from sympy.testing.pytest import raises
from sympy.core.sympify import SympifyError
import warnings

def test_parse_expr():
    if False:
        print('Hello World!')
    (a, b) = symbols('a, b')
    assert parse_expr('a + b', {}) == a + b
    raises(SympifyError, lambda : parse_expr('a + ', {}))
    assert parse_expr('1 + 2', {}) == S(3)
    assert parse_expr('1 + 2.0', {}) == S(3.0)
    assert parse_expr('Rational(1, 2)', {}) == S(1) / 2
    assert parse_expr('a', {'a': a}) == a
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert parse_expr('6 * 7', {}) == S(42)