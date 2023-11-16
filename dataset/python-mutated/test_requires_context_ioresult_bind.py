from returns.context import RequiresContext
from returns.context import RequiresContextIOResult as RCR
from returns.context import RequiresContextResult
from returns.io import IOFailure, IOResult, IOSuccess
from returns.result import Failure, Result, Success

def test_bind():
    if False:
        while True:
            i = 10
    'Ensures that bind works.'

    def factory(inner_value: int) -> RCR[float, str, int]:
        if False:
            i = 10
            return i + 15
        if inner_value > 0:
            return RCR(lambda deps: IOSuccess(inner_value / deps))
        return RCR.from_failure(str(inner_value))
    input_value = 5
    bound: RCR[int, str, int] = RCR.from_value(input_value)
    assert bound.bind(factory)(2) == factory(input_value)(2)
    assert bound.bind(factory)(2) == IOSuccess(2.5)
    assert RCR.from_value(0).bind(factory)(2) == factory(0)(2) == IOFailure('0')

def test_bind_regular_result():
    if False:
        i = 10
        return i + 15
    'Ensures that regular ``Result`` can be bound.'

    def factory(inner_value: int) -> Result[int, str]:
        if False:
            for i in range(10):
                print('nop')
        if inner_value > 0:
            return Success(inner_value + 1)
        return Failure('nope')
    first: RCR[int, str, int] = RCR.from_value(1)
    third: RCR[int, str, int] = RCR.from_failure('a')
    assert first.bind_result(factory)(RCR.no_args) == IOSuccess(2)
    assert RCR.from_value(0).bind_result(factory)(RCR.no_args) == IOFailure('nope')
    assert third.bind_result(factory)(RCR.no_args) == IOFailure('a')

def test_bind_ioresult():
    if False:
        for i in range(10):
            print('nop')
    'Ensures that io ``Result`` can be bound.'

    def factory(inner_value: int) -> IOResult[int, str]:
        if False:
            for i in range(10):
                print('nop')
        if inner_value > 0:
            return IOSuccess(inner_value + 1)
        return IOFailure('nope')
    first: RCR[int, str, int] = RCR.from_value(1)
    third: RCR[int, str, int] = RCR.from_failure('a')
    assert first.bind_ioresult(factory)(RCR.no_args) == IOSuccess(2)
    assert RCR.from_value(0).bind_ioresult(factory)(RCR.no_args) == IOFailure('nope')
    assert third.bind_ioresult(factory)(RCR.no_args) == IOFailure('a')

def test_bind_regular_context():
    if False:
        print('Hello World!')
    'Ensures that regular ``RequiresContext`` can be bound.'

    def factory(inner_value: int) -> RequiresContext[float, int]:
        if False:
            for i in range(10):
                print('nop')
        return RequiresContext(lambda deps: inner_value / deps)
    first: RCR[int, str, int] = RCR.from_value(1)
    third: RCR[int, str, int] = RCR.from_failure('a')
    assert first.bind_context(factory)(2) == IOSuccess(0.5)
    assert RCR.from_value(2).bind_context(factory)(1) == IOSuccess(2.0)
    assert third.bind_context(factory)(1) == IOFailure('a')

def test_bind_result_context():
    if False:
        print('Hello World!')
    'Ensures that ``RequiresContextResult`` can be bound.'

    def factory(inner_value: int) -> RequiresContextResult[float, str, int]:
        if False:
            while True:
                i = 10
        return RequiresContextResult(lambda deps: Success(inner_value / deps))
    first: RCR[int, str, int] = RCR.from_value(1)
    third: RCR[int, str, int] = RCR.from_failure('a')
    assert first.bind_context_result(factory)(2) == IOSuccess(0.5)
    assert RCR.from_value(2).bind_context_result(factory)(1) == IOSuccess(2.0)
    assert third.bind_context_result(factory)(1) == IOFailure('a')

def test_lash_success():
    if False:
        i = 10
        return i + 15
    'Ensures that lash works for Success container.'

    def factory(inner_value) -> RCR[int, str, int]:
        if False:
            while True:
                i = 10
        return RCR.from_value(inner_value * 2)
    assert RCR.from_value(5).lash(factory)(0) == RCR.from_value(5)(0)
    assert RCR.from_failure(5).lash(factory)(0) == RCR.from_value(10)(0)

def test_lash_failure():
    if False:
        return 10
    'Ensures that lash works for Failure container.'

    def factory(inner_value) -> RCR[int, str, int]:
        if False:
            for i in range(10):
                print('nop')
        return RCR.from_failure(inner_value * 2)
    assert RCR.from_value(5).lash(factory)(0) == RCR.from_value(5)(0)
    assert RCR.from_failure(5).lash(factory)(0) == RCR.from_failure(10)(0)