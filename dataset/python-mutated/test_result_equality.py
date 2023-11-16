from copy import copy, deepcopy
import pytest
from returns.primitives.exceptions import ImmutableStateError
from returns.result import Failure, Success

def test_equals():
    if False:
        return 10
    'Ensures that ``.equals`` method works correctly.'
    inner_value = 1
    assert Success(inner_value).equals(Success(inner_value))
    assert Failure(inner_value).equals(Failure(inner_value))

def test_not_equals():
    if False:
        return 10
    'Ensures that ``.equals`` method works correctly.'
    inner_value = 1
    assert not Success(inner_value).equals(Failure(inner_value))
    assert not Success(inner_value).equals(Success(0))
    assert not Failure(inner_value).equals(Success(inner_value))
    assert not Failure(inner_value).equals(Failure(0))

def test_non_equality():
    if False:
        i = 10
        return i + 15
    'Ensures that containers are not compared to regular values.'
    input_value = 5
    assert Failure(input_value) != input_value
    assert Success(input_value) != input_value
    assert Failure(input_value) != Success(input_value)
    assert hash(Failure(1))
    assert hash(Success(1))

def test_is_compare():
    if False:
        while True:
            i = 10
    'Ensures that `is` operator works correctly.'
    left = Failure(1)
    right = Success(1)
    assert left.bind(lambda state: state) is left
    assert right.lash(lambda state: state) is right
    assert right is not Success(1)

def test_immutability_failure():
    if False:
        i = 10
        return i + 15
    'Ensures that Failure container is immutable.'
    with pytest.raises(ImmutableStateError):
        Failure(0)._inner_state = 1
    with pytest.raises(ImmutableStateError):
        Failure(1).missing = 2
    with pytest.raises(ImmutableStateError):
        del Failure(0)._inner_state
    with pytest.raises(AttributeError):
        Failure(1).missing

def test_immutability_success():
    if False:
        i = 10
        return i + 15
    'Ensures that Success container is immutable.'
    with pytest.raises(ImmutableStateError):
        Success(0)._inner_state = 1
    with pytest.raises(ImmutableStateError):
        Success(1).missing = 2
    with pytest.raises(ImmutableStateError):
        del Success(0)._inner_state
    with pytest.raises(AttributeError):
        Success(1).missing

def test_success_immutable_copy():
    if False:
        while True:
            i = 10
    'Ensures that Success returns it self when passed to copy function.'
    success = Success(1)
    assert success is copy(success)

def test_success_immutable_deepcopy():
    if False:
        while True:
            i = 10
    'Ensures that Success returns it self when passed to deepcopy function.'
    success = Success(1)
    assert success is deepcopy(success)

def test_failure_immutable_copy():
    if False:
        for i in range(10):
            print('nop')
    'Ensures that Failure returns it self when passed to copy function.'
    failure = Failure(0)
    assert failure is copy(failure)

def test_failure_immutable_deepcopy():
    if False:
        while True:
            i = 10
    'Ensures that Failure returns it self when passed to deepcopy function.'
    failure = Failure(0)
    assert failure is deepcopy(failure)