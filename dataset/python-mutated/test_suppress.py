from __future__ import annotations
import pytest
from airflow.providers.amazon.aws.utils.suppress import return_on_error

@pytest.mark.db_test
def test_suppress_function(caplog):
    if False:
        while True:
            i = 10

    @return_on_error('error')
    def fn(value: str, exc: Exception | None=None) -> str:
        if False:
            while True:
                i = 10
        if exc:
            raise exc
        return value
    caplog.set_level('DEBUG', 'airflow.providers.amazon.aws.utils.suppress')
    caplog.clear()
    assert fn('no-error') == 'no-error'
    assert not caplog.messages
    assert fn('foo', ValueError('boooo')) == 'error'
    assert "Encountered error during execution function/method 'fn'" in caplog.messages
    caplog.clear()
    with pytest.raises(SystemExit, match='42'):
        fn('bar', SystemExit(42))
    assert not caplog.messages
    assert fn() == 'error'
    assert "Encountered error during execution function/method 'fn'" in caplog.messages

def test_suppress_methods():
    if False:
        print('Hello World!')

    class FakeClass:

        @return_on_error('Oops!… I Did It Again')
        def some_method(self, value, exc: Exception | None=None) -> str:
            if False:
                i = 10
                return i + 15
            if exc:
                raise exc
            return value

        @staticmethod
        @return_on_error(0)
        def some_staticmethod(value, exc: Exception | None=None) -> int:
            if False:
                i = 10
                return i + 15
            if exc:
                raise exc
            return value

        @classmethod
        @return_on_error("It's fine")
        def some_classmethod(cls, value, exc: Exception | None=None) -> str:
            if False:
                print('Hello World!')
            if exc:
                raise exc
            return value
    assert FakeClass().some_method('no-error') == 'no-error'
    assert FakeClass.some_staticmethod(42) == 42
    assert FakeClass.some_classmethod('really-no-error-here') == 'really-no-error-here'
    assert FakeClass().some_method('foo', KeyError('foo')) == 'Oops!… I Did It Again'
    assert FakeClass.some_staticmethod(42, RuntimeError('bar')) == 0
    assert FakeClass.some_classmethod('bar', OSError('Windows detected!')) == "It's fine"