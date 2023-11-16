from returns.result import Failure, Success

def test_map_success():
    if False:
        for i in range(10):
            print('nop')
    'Ensures that Success is mappable.'
    assert Success(5).map(str) == Success('5')

def test_alt_failure():
    if False:
        i = 10
        return i + 15
    'Ensures that Failure is mappable.'
    assert Failure(5).map(str) == Failure(5)
    assert Failure(5).alt(str) == Failure('5')

def test_alt_success():
    if False:
        print('Hello World!')
    'Ensures that Success.alt is NoOp.'
    assert Success(5).alt(str) == Success(5)