import pytest

def test_keep_alive_argument(capture):
    if False:
        while True:
            i = 10
    from pybind11_tests import Parent, Child
    with capture:
        p = Parent()
    assert capture == 'Allocating parent.'
    with capture:
        p.addChild(Child())
        pytest.gc_collect()
    assert capture == '\n        Allocating child.\n        Releasing child.\n    '
    with capture:
        del p
        pytest.gc_collect()
    assert capture == 'Releasing parent.'
    with capture:
        p = Parent()
    assert capture == 'Allocating parent.'
    with capture:
        p.addChildKeepAlive(Child())
        pytest.gc_collect()
    assert capture == 'Allocating child.'
    with capture:
        del p
        pytest.gc_collect()
    assert capture == '\n        Releasing parent.\n        Releasing child.\n    '

def test_keep_alive_return_value(capture):
    if False:
        return 10
    from pybind11_tests import Parent
    with capture:
        p = Parent()
    assert capture == 'Allocating parent.'
    with capture:
        p.returnChild()
        pytest.gc_collect()
    assert capture == '\n        Allocating child.\n        Releasing child.\n    '
    with capture:
        del p
        pytest.gc_collect()
    assert capture == 'Releasing parent.'
    with capture:
        p = Parent()
    assert capture == 'Allocating parent.'
    with capture:
        p.returnChildKeepAlive()
        pytest.gc_collect()
    assert capture == 'Allocating child.'
    with capture:
        del p
        pytest.gc_collect()
    assert capture == '\n        Releasing parent.\n        Releasing child.\n    '

def test_return_none(capture):
    if False:
        for i in range(10):
            print('nop')
    from pybind11_tests import Parent
    with capture:
        p = Parent()
    assert capture == 'Allocating parent.'
    with capture:
        p.returnNullChildKeepAliveChild()
        pytest.gc_collect()
    assert capture == ''
    with capture:
        del p
        pytest.gc_collect()
    assert capture == 'Releasing parent.'
    with capture:
        p = Parent()
    assert capture == 'Allocating parent.'
    with capture:
        p.returnNullChildKeepAliveParent()
        pytest.gc_collect()
    assert capture == ''
    with capture:
        del p
        pytest.gc_collect()
    assert capture == 'Releasing parent.'