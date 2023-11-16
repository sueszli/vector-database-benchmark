from returns.result import Failure, Result, Success

def test_bind():
    if False:
        return 10
    'Ensures that bind works.'

    def factory(inner_value: int) -> Result[int, str]:
        if False:
            return 10
        if inner_value > 0:
            return Success(inner_value * 2)
        return Failure(str(inner_value))
    input_value = 5
    bound: Result[int, str] = Success(input_value)
    assert bound.bind(factory) == factory(input_value)
    assert Success(input_value).bind(factory) == factory(input_value)
    assert str(bound.bind(factory)) == '<Success: 10>'
    input_value = 0
    bound2: Result[int, str] = Success(input_value)
    assert bound2.bind(factory) == factory(input_value)
    assert str(bound2.bind(factory)) == '<Failure: 0>'

def test_left_identity_success():
    if False:
        for i in range(10):
            print('nop')
    'Ensures that left identity works for Success container.'

    def factory(inner_value: int) -> Result[int, str]:
        if False:
            i = 10
            return i + 15
        return Success(inner_value * 2)
    input_value = 5
    bound: Result[int, str] = Success(input_value)
    assert bound.bind(factory) == factory(input_value)

def test_left_identity_failure():
    if False:
        while True:
            i = 10
    'Ensures that left identity works for Failure container.'

    def factory(inner_value: int) -> Result[int, int]:
        if False:
            print('Hello World!')
        return Failure(6)
    input_value = 5
    bound: Result[int, int] = Failure(input_value)
    assert bound.bind(factory) == Failure(input_value)
    assert Failure(input_value).bind(factory) == Failure(5)
    assert str(bound) == '<Failure: 5>'

def test_lash_success():
    if False:
        while True:
            i = 10
    'Ensures that lash works for Success container.'

    def factory(inner_value) -> Result[int, str]:
        if False:
            i = 10
            return i + 15
        return Success(inner_value * 2)
    bound = Success(5).lash(factory)
    assert bound == Success(5)
    assert Success(5).lash(factory) == Success(5)
    assert str(bound) == '<Success: 5>'

def test_lash_failure():
    if False:
        return 10
    'Ensures that lash works for Failure container.'

    def factory(inner_value: int) -> Result[str, int]:
        if False:
            i = 10
            return i + 15
        return Failure(inner_value + 1)
    expected = 6
    bound: Result[str, int] = Failure(5)
    assert bound.lash(factory) == Failure(expected)
    assert Failure(5).lash(factory) == Failure(expected)
    assert str(bound.lash(factory)) == '<Failure: 6>'