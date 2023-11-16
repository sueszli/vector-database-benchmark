from hypothesis import given, assume
from hypothesis import strategies as st
from pyo3_pytests import othermod
INTEGER32_ST = st.integers(min_value=-2 ** 31, max_value=2 ** 31 - 1)
USIZE_ST = st.integers(min_value=othermod.USIZE_MIN, max_value=othermod.USIZE_MAX)

@given(x=INTEGER32_ST)
def test_double(x):
    if False:
        i = 10
        return i + 15
    expected = x * 2
    assume(-2 ** 31 <= expected <= 2 ** 31 - 1)
    assert othermod.double(x) == expected

def test_modclass():
    if False:
        while True:
            i = 10
    repr(othermod.ModClass)
    assert isinstance(othermod.ModClass, type)

def test_modclass_instance():
    if False:
        print('Hello World!')
    mi = othermod.ModClass()
    repr(mi)
    repr(mi.__class__)
    assert isinstance(mi, othermod.ModClass)
    assert isinstance(mi, object)

@given(x=USIZE_ST)
def test_modclas_noop(x):
    if False:
        for i in range(10):
            print('nop')
    mi = othermod.ModClass()
    assert mi.noop(x) == x