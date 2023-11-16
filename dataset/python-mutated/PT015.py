import pytest

def test_ok():
    if False:
        for i in range(10):
            print('nop')
    assert [0]

def test_error():
    if False:
        while True:
            i = 10
    assert None
    assert False
    assert 0
    assert 0.0
    assert ''
    assert f''
    assert []
    assert ()
    assert {}
    assert list()
    assert set()
    assert tuple()
    assert dict()
    assert frozenset()
    assert list([])
    assert set(set())
    assert tuple('')