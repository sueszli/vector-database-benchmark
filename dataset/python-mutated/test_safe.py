from typing import Union
import pytest
from returns.result import Success, safe

@safe
def _function(number: int) -> float:
    if False:
        i = 10
        return i + 15
    return number / number

@safe(exceptions=(ZeroDivisionError,))
def _function_two(number: Union[int, str]) -> float:
    if False:
        i = 10
        return i + 15
    assert isinstance(number, int)
    return number / number

@safe((ZeroDivisionError,))
def _function_three(number: Union[int, str]) -> float:
    if False:
        return 10
    assert isinstance(number, int)
    return number / number

def test_safe_success():
    if False:
        while True:
            i = 10
    'Ensures that safe decorator works correctly for Success case.'
    assert _function(1) == Success(1.0)

def test_safe_failure():
    if False:
        while True:
            i = 10
    'Ensures that safe decorator works correctly for Failure case.'
    failed = _function(0)
    assert isinstance(failed.failure(), ZeroDivisionError)

def test_safe_failure_with_expected_error():
    if False:
        while True:
            i = 10
    'Ensures that safe decorator works correctly for Failure case.'
    failed = _function_two(0)
    assert isinstance(failed.failure(), ZeroDivisionError)
    failed2 = _function_three(0)
    assert isinstance(failed2.failure(), ZeroDivisionError)

def test_safe_failure_with_non_expected_error():
    if False:
        return 10
    'Ensures that safe decorator works correctly for Failure case.'
    with pytest.raises(AssertionError):
        _function_two('0')