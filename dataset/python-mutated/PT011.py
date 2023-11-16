from pickle import PicklingError, UnpicklingError
import socket
import pytest

def test_ok():
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError, match="Can't divide by 0"):
        raise ValueError("Can't divide by 0")

def test_ok_different_error_from_config():
    if False:
        print('Hello World!')
    with pytest.raises(ZeroDivisionError):
        raise ZeroDivisionError("Can't divide by 0")

def test_error_no_argument_given():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError):
        raise ValueError("Can't divide 1 by 0")
    with pytest.raises(socket.error):
        raise ValueError("Can't divide 1 by 0")
    with pytest.raises(PicklingError):
        raise PicklingError("Can't pickle")
    with pytest.raises(UnpicklingError):
        raise UnpicklingError("Can't unpickle")

def test_error_match_is_empty():
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError, match=None):
        raise ValueError("Can't divide 1 by 0")
    with pytest.raises(ValueError, match=''):
        raise ValueError("Can't divide 1 by 0")
    with pytest.raises(ValueError, match=f''):
        raise ValueError("Can't divide 1 by 0")