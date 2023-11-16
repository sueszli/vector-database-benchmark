import pytest
from hypothesis import given, strategies as st
given_booleans = given(st.booleans())

@given_booleans
def test_has_an_arg_named_x(x):
    if False:
        while True:
            i = 10
    pass

@given_booleans
def test_has_an_arg_named_y(y):
    if False:
        i = 10
        return i + 15
    pass
given_named_booleans = given(z=st.text())

def test_fail_independently():
    if False:
        for i in range(10):
            print('nop')

    @given_named_booleans
    def test_z1(z):
        if False:
            while True:
                i = 10
        raise AssertionError

    @given_named_booleans
    def test_z2(z):
        if False:
            print('Hello World!')
        pass
    with pytest.raises(AssertionError):
        test_z1()
    test_z2()