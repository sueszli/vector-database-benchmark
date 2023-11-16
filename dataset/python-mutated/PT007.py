import pytest

@pytest.mark.parametrize('param', (1, 2))
def test_tuple(param):
    if False:
        for i in range(10):
            print('nop')
    ...

@pytest.mark.parametrize(('param1', 'param2'), ((1, 2), (3, 4)))
def test_tuple_of_tuples(param1, param2):
    if False:
        i = 10
        return i + 15
    ...

@pytest.mark.parametrize(('param1', 'param2'), ([1, 2], [3, 4]))
def test_tuple_of_lists(param1, param2):
    if False:
        while True:
            i = 10
    ...

@pytest.mark.parametrize('param', [1, 2])
def test_list(param):
    if False:
        print('Hello World!')
    ...

@pytest.mark.parametrize(('param1', 'param2'), [(1, 2), (3, 4)])
def test_list_of_tuples(param1, param2):
    if False:
        return 10
    ...

@pytest.mark.parametrize(('param1', 'param2'), [[1, 2], [3, 4]])
def test_list_of_lists(param1, param2):
    if False:
        print('Hello World!')
    ...

@pytest.mark.parametrize('param1,param2', [[1, 2], [3, 4]])
def test_csv_name_list_of_lists(param1, param2):
    if False:
        print('Hello World!')
    ...

@pytest.mark.parametrize('param', [[1, 2], [3, 4]])
def test_single_list_of_lists(param):
    if False:
        return 10
    ...

@pytest.mark.parametrize('a', [1, 2])
@pytest.mark.parametrize(('b', 'c'), ((3, 4), (5, 6)))
def test_multiple_decorators(a, b, c):
    if False:
        return 10
    pass