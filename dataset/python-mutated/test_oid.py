import copy
import pytest
from cryptography.hazmat._oid import ObjectIdentifier

def test_basic_oid():
    if False:
        return 10
    assert ObjectIdentifier('1.2.3.4').dotted_string == '1.2.3.4'

def test_oid_equal():
    if False:
        print('Hello World!')
    assert ObjectIdentifier('1.2.3.4') == ObjectIdentifier('1.2.3.4')

def test_oid_deepcopy():
    if False:
        i = 10
        return i + 15
    oid = ObjectIdentifier('1.2.3.4')
    assert oid == copy.deepcopy(oid)

def test_oid_constraint():
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError):
        ObjectIdentifier('1')
    with pytest.raises(ValueError):
        ObjectIdentifier('3.2.1')
    with pytest.raises(ValueError):
        ObjectIdentifier('1.40')
    with pytest.raises(ValueError):
        ObjectIdentifier('0.42')
    with pytest.raises(ValueError):
        ObjectIdentifier('1.2.foo.bar')
    with pytest.raises(ValueError):
        ObjectIdentifier('1.2.0xf00.0xba4')
    with pytest.raises(ValueError):
        ObjectIdentifier('1.2.-3.-4')