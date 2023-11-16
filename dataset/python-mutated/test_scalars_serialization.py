from graphql import Undefined
from ..scalars import Boolean, Float, Int, String

def test_serializes_output_int():
    if False:
        print('Hello World!')
    assert Int.serialize(1) == 1
    assert Int.serialize(0) == 0
    assert Int.serialize(-1) == -1
    assert Int.serialize(0.1) == 0
    assert Int.serialize(1.1) == 1
    assert Int.serialize(-1.1) == -1
    assert Int.serialize(100000.0) == 100000
    assert Int.serialize(9876504321) is Undefined
    assert Int.serialize(-9876504321) is Undefined
    assert Int.serialize(1e+100) is Undefined
    assert Int.serialize(-1e+100) is Undefined
    assert Int.serialize('-1.1') == -1
    assert Int.serialize('one') is Undefined
    assert Int.serialize(False) == 0
    assert Int.serialize(True) == 1

def test_serializes_output_float():
    if False:
        print('Hello World!')
    assert Float.serialize(1) == 1.0
    assert Float.serialize(0) == 0.0
    assert Float.serialize(-1) == -1.0
    assert Float.serialize(0.1) == 0.1
    assert Float.serialize(1.1) == 1.1
    assert Float.serialize(-1.1) == -1.1
    assert Float.serialize('-1.1') == -1.1
    assert Float.serialize('one') is Undefined
    assert Float.serialize(False) == 0
    assert Float.serialize(True) == 1

def test_serializes_output_string():
    if False:
        return 10
    assert String.serialize('string') == 'string'
    assert String.serialize(1) == '1'
    assert String.serialize(-1.1) == '-1.1'
    assert String.serialize(True) == 'true'
    assert String.serialize(False) == 'false'
    assert String.serialize('😁') == '😁'

def test_serializes_output_boolean():
    if False:
        print('Hello World!')
    assert Boolean.serialize('string') is True
    assert Boolean.serialize('') is False
    assert Boolean.serialize(1) is True
    assert Boolean.serialize(0) is False
    assert Boolean.serialize(True) is True
    assert Boolean.serialize(False) is False