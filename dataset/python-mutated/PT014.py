import pytest

@pytest.mark.parametrize('x', [1, 1, 2])
def test_error_literal(x):
    if False:
        while True:
            i = 10
    ...
a = 1
b = 2
c = 3

@pytest.mark.parametrize('x', [a, a, b, b, b, c])
def test_error_expr_simple(x):
    if False:
        i = 10
        return i + 15
    ...

@pytest.mark.parametrize('x', [(a, b), (a, b), (b, c)])
def test_error_expr_complex(x):
    if False:
        while True:
            i = 10
    ...

@pytest.mark.parametrize('x', [a, b, a, c, a])
def test_error_parentheses(x):
    if False:
        print('Hello World!')
    ...

@pytest.mark.parametrize('x', [a, b, a, c, a])
def test_error_parentheses_trailing_comma(x):
    if False:
        while True:
            i = 10
    ...

@pytest.mark.parametrize('x', [1, 2])
def test_ok(x):
    if False:
        while True:
            i = 10
    ...