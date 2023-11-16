import pytest

def test_ok():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises():
        pass

def test_error():
    if False:
        i = 10
        return i + 15
    with pytest.raises(UnicodeError):
        pass