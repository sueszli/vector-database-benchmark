import pytest

@pytest.mark.parametrize('param', [1, 2, 3])
def test_always_ok(param):
    if False:
        while True:
            i = 10
    ...

@pytest.mark.parametrize('param1,param2', [(1, 2), (3, 4)])
def test_csv(param1, param2):
    if False:
        while True:
            i = 10
    ...

@pytest.mark.parametrize('   param1,   ,    param2   , ', [(1, 2), (3, 4)])
def test_csv_with_whitespace(param1, param2):
    if False:
        i = 10
        return i + 15
    ...

@pytest.mark.parametrize('param1,param2', [(1, 2), (3, 4)])
def test_csv_bad_quotes(param1, param2):
    if False:
        print('Hello World!')
    ...

@pytest.mark.parametrize(('param1', 'param2'), [(1, 2), (3, 4)])
def test_tuple(param1, param2):
    if False:
        while True:
            i = 10
    ...

@pytest.mark.parametrize(('param1',), [1, 2, 3])
def test_tuple_one_elem(param1, param2):
    if False:
        for i in range(10):
            print('nop')
    ...

@pytest.mark.parametrize(['param1', 'param2'], [(1, 2), (3, 4)])
def test_list(param1, param2):
    if False:
        while True:
            i = 10
    ...

@pytest.mark.parametrize(['param1'], [1, 2, 3])
def test_list_one_elem(param1, param2):
    if False:
        return 10
    ...

@pytest.mark.parametrize([some_expr, another_expr], [1, 2, 3])
def test_list_expressions(param1, param2):
    if False:
        return 10
    ...

@pytest.mark.parametrize([some_expr, 'param2'], [1, 2, 3])
def test_list_mixed_expr_literal(param1, param2):
    if False:
        return 10
    ...

@pytest.mark.parametrize('param1, param2, param3', [(1, 2, 3), (4, 5, 6)])
def test_implicit_str_concat_with_parens(param1, param2, param3):
    if False:
        i = 10
        return i + 15
    ...

@pytest.mark.parametrize('param1, param2, param3', [(1, 2, 3), (4, 5, 6)])
def test_implicit_str_concat_no_parens(param1, param2, param3):
    if False:
        return 10
    ...

@pytest.mark.parametrize('param1, param2, param3', [(1, 2, 3), (4, 5, 6)])
def test_implicit_str_concat_with_multi_parens(param1, param2, param3):
    if False:
        for i in range(10):
            print('nop')
    ...

@pytest.mark.parametrize('param1,param2', [(1, 2), (3, 4)])
def test_csv_with_parens(param1, param2):
    if False:
        for i in range(10):
            print('nop')
    ...