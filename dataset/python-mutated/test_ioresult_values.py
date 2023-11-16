import pytest
from returns.io import IO, IOFailure, IOSuccess
from returns.primitives.exceptions import UnwrapFailedError

def test_ioresult_value_or():
    if False:
        print('Hello World!')
    'Ensures that ``value_or`` works correctly.'
    assert IOSuccess(1).value_or(0) == IO(1)
    assert IOFailure(1).value_or(0) == IO(0)

def test_unwrap_iosuccess():
    if False:
        return 10
    'Ensures that unwrap works for IOSuccess container.'
    assert IOSuccess(5).unwrap() == IO(5)

def test_unwrap_iofailure():
    if False:
        i = 10
        return i + 15
    'Ensures that unwrap works for IOFailure container.'
    with pytest.raises(UnwrapFailedError):
        IOFailure(5).unwrap()

def test_unwrap_iofailure_with_exception():
    if False:
        print('Hello World!')
    'Ensures that unwrap raises from the original exception.'
    expected_exception = ValueError('error')
    with pytest.raises(UnwrapFailedError) as excinfo:
        IOFailure(expected_exception).unwrap()
    assert 'ValueError: error' in str(excinfo.getrepr())

def test_failure_iosuccess():
    if False:
        return 10
    'Ensures that failure works for IOSuccess container.'
    with pytest.raises(UnwrapFailedError):
        IOSuccess(5).failure()

def test_failure_iofailure():
    if False:
        for i in range(10):
            print('nop')
    'Ensures that failure works for IOFailure container.'
    assert IOFailure(5).failure() == IO(5)