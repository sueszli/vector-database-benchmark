from errbot.backends.test import TestOccupant, TestPerson

def test_identifier_eq():
    if False:
        i = 10
        return i + 15
    a = TestPerson('foo')
    b = TestPerson('foo')
    assert a == b

def test_identifier_ineq():
    if False:
        print('Hello World!')
    a = TestPerson('foo')
    b = TestPerson('bar')
    assert not a == b
    assert a != b

def test_mucidentifier_eq():
    if False:
        i = 10
        return i + 15
    a = TestOccupant('foo', 'room')
    b = TestOccupant('foo', 'room')
    assert a == b

def test_mucidentifier_ineq1():
    if False:
        print('Hello World!')
    a = TestOccupant('foo', 'room')
    b = TestOccupant('bar', 'room')
    assert not a == b
    assert a != b

def test_mucidentifier_ineq2():
    if False:
        i = 10
        return i + 15
    a = TestOccupant('foo', 'room1')
    b = TestOccupant('foo', 'room2')
    assert not a == b
    assert a != b