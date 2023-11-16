from __future__ import annotations
import pytest
pytest
from numpy.testing import assert_allclose
from tests.support.util.api import verify_all
import bokeh.driving as bd
ALL = ('bounce', 'cosine', 'count', 'force', 'linear', 'repeat', 'sine')

def _collector(results):
    if False:
        while True:
            i = 10

    def foo(val):
        if False:
            for i in range(10):
                print('nop')
        results.append(val)
    return foo
w = 0.3
A = 3
phi = 0.1
offset = 2
Test___all__ = verify_all(bd, ALL)

def test_bounce() -> None:
    if False:
        for i in range(10):
            print('nop')
    results = []
    func = bd.bounce([0, 1, 5, -1])(_collector(results))
    for i in range(8):
        func()
    assert results == [0, 1, 5, -1, -1, 5, 1, 0]

def test_cosine() -> None:
    if False:
        i = 10
        return i + 15
    results = []
    func = bd.cosine(w, A, phi, offset)(_collector(results))
    for i in range(4):
        func()
    assert_allclose(results, [4.985012495834077, 4.763182982008655, 4.294526561853465, 3.6209069176044197])

def test_count() -> None:
    if False:
        while True:
            i = 10
    results = []
    func = bd.count()(_collector(results))
    for i in range(8):
        func()
    assert results == list(range(8))

def test_force() -> None:
    if False:
        print('Hello World!')
    results = []
    seq = (x for x in ['foo', 'bar', 'baz'])
    w = bd.force(_collector(results), seq)
    w()
    assert results == ['foo']
    w()
    assert results == ['foo', 'bar']
    w()
    assert results == ['foo', 'bar', 'baz']

def test_linear() -> None:
    if False:
        return 10
    results = []
    func = bd.linear(m=2.5, b=3.7)(_collector(results))
    for i in range(4):
        func()
    assert_allclose(results, [3.7, 6.2, 8.7, 11.2])

def test_repeat() -> None:
    if False:
        i = 10
        return i + 15
    results = []
    func = bd.repeat([0, 1, 5, -1])(_collector(results))
    for i in range(8):
        func()
    assert results == [0, 1, 5, -1, 0, 1, 5, -1]

def test_sine() -> None:
    if False:
        return 10
    results = []
    func = bd.sine(w, A, phi, offset)(_collector(results))
    for i in range(4):
        func()
    assert_allclose(results, [2.2995002499404844, 3.1682550269259515, 3.932653061713073, 4.524412954423689])

def test__advance() -> None:
    if False:
        i = 10
        return i + 15
    results = []
    testf = _collector(results)
    s = bd._advance(testf)
    next(s)
    assert results == [0]
    next(s)
    assert results == [0, 1]
    next(s)
    assert results == [0, 1, 2]
    next(s)
    assert results == [0, 1, 2, 3]