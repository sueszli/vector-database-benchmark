import pytest
import env
from pybind11_tests import ConstructorStats
from pybind11_tests import call_policies as m

@pytest.mark.xfail('env.PYPY', reason='sometimes comes out 1 off on PyPy', strict=False)
def test_keep_alive_argument(capture):
    if False:
        return 10
    n_inst = ConstructorStats.detail_reg_inst()
    with capture:
        p = m.Parent()
    assert capture == 'Allocating parent.'
    with capture:
        p.addChild(m.Child())
        assert ConstructorStats.detail_reg_inst() == n_inst + 1
    assert capture == '\n        Allocating child.\n        Releasing child.\n    '
    with capture:
        del p
        assert ConstructorStats.detail_reg_inst() == n_inst
    assert capture == 'Releasing parent.'
    with capture:
        p = m.Parent()
    assert capture == 'Allocating parent.'
    with capture:
        p.addChildKeepAlive(m.Child())
        assert ConstructorStats.detail_reg_inst() == n_inst + 2
    assert capture == 'Allocating child.'
    with capture:
        del p
        assert ConstructorStats.detail_reg_inst() == n_inst
    assert capture == '\n        Releasing parent.\n        Releasing child.\n    '
    p = m.Parent()
    c = m.Child()
    assert ConstructorStats.detail_reg_inst() == n_inst + 2
    m.free_function(p, c)
    del c
    assert ConstructorStats.detail_reg_inst() == n_inst + 2
    del p
    assert ConstructorStats.detail_reg_inst() == n_inst
    with pytest.raises(RuntimeError) as excinfo:
        m.invalid_arg_index()
    assert str(excinfo.value) == 'Could not activate keep_alive!'

def test_keep_alive_return_value(capture):
    if False:
        return 10
    n_inst = ConstructorStats.detail_reg_inst()
    with capture:
        p = m.Parent()
    assert capture == 'Allocating parent.'
    with capture:
        p.returnChild()
        assert ConstructorStats.detail_reg_inst() == n_inst + 1
    assert capture == '\n        Allocating child.\n        Releasing child.\n    '
    with capture:
        del p
        assert ConstructorStats.detail_reg_inst() == n_inst
    assert capture == 'Releasing parent.'
    with capture:
        p = m.Parent()
    assert capture == 'Allocating parent.'
    with capture:
        p.returnChildKeepAlive()
        assert ConstructorStats.detail_reg_inst() == n_inst + 2
    assert capture == 'Allocating child.'
    with capture:
        del p
        assert ConstructorStats.detail_reg_inst() == n_inst
    assert capture == '\n        Releasing parent.\n        Releasing child.\n    '
    p = m.Parent()
    assert ConstructorStats.detail_reg_inst() == n_inst + 1
    with capture:
        m.Parent.staticFunction(p)
        assert ConstructorStats.detail_reg_inst() == n_inst + 2
    assert capture == 'Allocating child.'
    with capture:
        del p
        assert ConstructorStats.detail_reg_inst() == n_inst
    assert capture == '\n        Releasing parent.\n        Releasing child.\n    '

@pytest.mark.xfail('env.PYPY', reason='_PyObject_GetDictPtr is unimplemented')
def test_alive_gc(capture):
    if False:
        return 10
    n_inst = ConstructorStats.detail_reg_inst()
    p = m.ParentGC()
    p.addChildKeepAlive(m.Child())
    assert ConstructorStats.detail_reg_inst() == n_inst + 2
    lst = [p]
    lst.append(lst)
    with capture:
        del p, lst
        assert ConstructorStats.detail_reg_inst() == n_inst
    assert capture == '\n        Releasing parent.\n        Releasing child.\n    '

def test_alive_gc_derived(capture):
    if False:
        while True:
            i = 10

    class Derived(m.Parent):
        pass
    n_inst = ConstructorStats.detail_reg_inst()
    p = Derived()
    p.addChildKeepAlive(m.Child())
    assert ConstructorStats.detail_reg_inst() == n_inst + 2
    lst = [p]
    lst.append(lst)
    with capture:
        del p, lst
        assert ConstructorStats.detail_reg_inst() == n_inst
    assert capture == '\n        Releasing parent.\n        Releasing child.\n    '

def test_alive_gc_multi_derived(capture):
    if False:
        return 10

    class Derived(m.Parent, m.Child):

        def __init__(self):
            if False:
                i = 10
                return i + 15
            m.Parent.__init__(self)
            m.Child.__init__(self)
    n_inst = ConstructorStats.detail_reg_inst()
    p = Derived()
    p.addChildKeepAlive(m.Child())
    assert ConstructorStats.detail_reg_inst() == n_inst + 3
    lst = [p]
    lst.append(lst)
    with capture:
        del p, lst
        assert ConstructorStats.detail_reg_inst() == n_inst
    assert capture == '\n        Releasing parent.\n        Releasing child.\n        Releasing child.\n    '

def test_return_none(capture):
    if False:
        while True:
            i = 10
    n_inst = ConstructorStats.detail_reg_inst()
    with capture:
        p = m.Parent()
    assert capture == 'Allocating parent.'
    with capture:
        p.returnNullChildKeepAliveChild()
        assert ConstructorStats.detail_reg_inst() == n_inst + 1
    assert capture == ''
    with capture:
        del p
        assert ConstructorStats.detail_reg_inst() == n_inst
    assert capture == 'Releasing parent.'
    with capture:
        p = m.Parent()
    assert capture == 'Allocating parent.'
    with capture:
        p.returnNullChildKeepAliveParent()
        assert ConstructorStats.detail_reg_inst() == n_inst + 1
    assert capture == ''
    with capture:
        del p
        assert ConstructorStats.detail_reg_inst() == n_inst
    assert capture == 'Releasing parent.'

def test_keep_alive_constructor(capture):
    if False:
        print('Hello World!')
    n_inst = ConstructorStats.detail_reg_inst()
    with capture:
        p = m.Parent(m.Child())
        assert ConstructorStats.detail_reg_inst() == n_inst + 2
    assert capture == '\n        Allocating child.\n        Allocating parent.\n    '
    with capture:
        del p
        assert ConstructorStats.detail_reg_inst() == n_inst
    assert capture == '\n        Releasing parent.\n        Releasing child.\n    '

def test_call_guard():
    if False:
        while True:
            i = 10
    assert m.unguarded_call() == 'unguarded'
    assert m.guarded_call() == 'guarded'
    assert m.multiple_guards_correct_order() == 'guarded & guarded'
    assert m.multiple_guards_wrong_order() == 'unguarded & guarded'
    if hasattr(m, 'with_gil'):
        assert m.with_gil() == 'GIL held'
        assert m.without_gil() == 'GIL released'