import gc
import weakref
import pytest
import env
from pybind11_tests import custom_type_setup as m

@pytest.fixture()
def gc_tester():
    if False:
        for i in range(10):
            print('nop')
    'Tests that an object is garbage collected.\n\n    Assumes that any unreferenced objects are fully collected after calling\n    `gc.collect()`.  That is true on CPython, but does not appear to reliably\n    hold on PyPy.\n    '
    weak_refs = []

    def add_ref(obj):
        if False:
            while True:
                i = 10
        if hasattr(gc, 'is_tracked'):
            assert gc.is_tracked(obj)
        weak_refs.append(weakref.ref(obj))
    yield add_ref
    gc.collect()
    for ref in weak_refs:
        assert ref() is None

@pytest.mark.skipif('env.PYPY')
def test_self_cycle(gc_tester):
    if False:
        print('Hello World!')
    obj = m.OwnsPythonObjects()
    obj.value = obj
    gc_tester(obj)

@pytest.mark.skipif('env.PYPY')
def test_indirect_cycle(gc_tester):
    if False:
        print('Hello World!')
    obj = m.OwnsPythonObjects()
    obj_list = [obj]
    obj.value = obj_list
    gc_tester(obj)