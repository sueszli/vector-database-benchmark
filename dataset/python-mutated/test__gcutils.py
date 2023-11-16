""" Test for assert_deallocated context manager and gc utilities
"""
import gc
from scipy._lib._gcutils import set_gc_state, gc_state, assert_deallocated, ReferenceError, IS_PYPY
from numpy.testing import assert_equal
import pytest

def test_set_gc_state():
    if False:
        i = 10
        return i + 15
    gc_status = gc.isenabled()
    try:
        for state in (True, False):
            gc.enable()
            set_gc_state(state)
            assert_equal(gc.isenabled(), state)
            gc.disable()
            set_gc_state(state)
            assert_equal(gc.isenabled(), state)
    finally:
        if gc_status:
            gc.enable()

def test_gc_state():
    if False:
        i = 10
        return i + 15
    gc_status = gc.isenabled()
    try:
        for pre_state in (True, False):
            set_gc_state(pre_state)
            for with_state in (True, False):
                with gc_state(with_state):
                    assert_equal(gc.isenabled(), with_state)
                assert_equal(gc.isenabled(), pre_state)
                with gc_state(with_state):
                    assert_equal(gc.isenabled(), with_state)
                    set_gc_state(not with_state)
                assert_equal(gc.isenabled(), pre_state)
    finally:
        if gc_status:
            gc.enable()

@pytest.mark.skipif(IS_PYPY, reason='Test not meaningful on PyPy')
def test_assert_deallocated():
    if False:
        print('Hello World!')

    class C:

        def __init__(self, arg0, arg1, name='myname'):
            if False:
                return 10
            self.name = name
    for gc_current in (True, False):
        with gc_state(gc_current):
            with assert_deallocated(C, 0, 2, 'another name') as c:
                assert_equal(c.name, 'another name')
                del c
            with assert_deallocated(C, 0, 2, name='third name'):
                pass
            assert_equal(gc.isenabled(), gc_current)

@pytest.mark.skipif(IS_PYPY, reason='Test not meaningful on PyPy')
def test_assert_deallocated_nodel():
    if False:
        print('Hello World!')

    class C:
        pass
    with pytest.raises(ReferenceError):
        with assert_deallocated(C) as _:
            pass

@pytest.mark.skipif(IS_PYPY, reason='Test not meaningful on PyPy')
def test_assert_deallocated_circular():
    if False:
        while True:
            i = 10

    class C:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self._circular = self
    with pytest.raises(ReferenceError):
        with assert_deallocated(C) as c:
            del c

@pytest.mark.skipif(IS_PYPY, reason='Test not meaningful on PyPy')
def test_assert_deallocated_circular2():
    if False:
        print('Hello World!')

    class C:

        def __init__(self):
            if False:
                while True:
                    i = 10
            self._circular = self
    with pytest.raises(ReferenceError):
        with assert_deallocated(C):
            pass