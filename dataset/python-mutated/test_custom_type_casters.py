import pytest
from pybind11_tests import custom_type_casters as m

def test_noconvert_args(msg):
    if False:
        print('Hello World!')
    a = m.ArgInspector()
    assert msg(a.f('hi')) == '\n        loading ArgInspector1 argument WITH conversion allowed.  Argument value = hi\n    '
    assert msg(a.g('this is a', 'this is b')) == '\n        loading ArgInspector1 argument WITHOUT conversion allowed.  Argument value = this is a\n        loading ArgInspector1 argument WITH conversion allowed.  Argument value = this is b\n        13\n        loading ArgInspector2 argument WITH conversion allowed.  Argument value = (default arg inspector 2)\n    '
    assert msg(a.g('this is a', 'this is b', 42)) == '\n        loading ArgInspector1 argument WITHOUT conversion allowed.  Argument value = this is a\n        loading ArgInspector1 argument WITH conversion allowed.  Argument value = this is b\n        42\n        loading ArgInspector2 argument WITH conversion allowed.  Argument value = (default arg inspector 2)\n    '
    assert msg(a.g('this is a', 'this is b', 42, 'this is d')) == '\n        loading ArgInspector1 argument WITHOUT conversion allowed.  Argument value = this is a\n        loading ArgInspector1 argument WITH conversion allowed.  Argument value = this is b\n        42\n        loading ArgInspector2 argument WITH conversion allowed.  Argument value = this is d\n    '
    assert a.h('arg 1') == 'loading ArgInspector2 argument WITHOUT conversion allowed.  Argument value = arg 1'
    assert msg(m.arg_inspect_func('A1', 'A2')) == '\n        loading ArgInspector2 argument WITH conversion allowed.  Argument value = A1\n        loading ArgInspector1 argument WITHOUT conversion allowed.  Argument value = A2\n    '
    assert m.floats_preferred(4) == 2.0
    assert m.floats_only(4.0) == 2.0
    with pytest.raises(TypeError) as excinfo:
        m.floats_only(4)
    assert msg(excinfo.value) == '\n        floats_only(): incompatible function arguments. The following argument types are supported:\n            1. (f: float) -> float\n\n        Invoked with: 4\n    '
    assert m.ints_preferred(4) == 2
    assert m.ints_preferred(True) == 0
    with pytest.raises(TypeError) as excinfo:
        m.ints_preferred(4.0)
    assert msg(excinfo.value) == '\n        ints_preferred(): incompatible function arguments. The following argument types are supported:\n            1. (i: int) -> int\n\n        Invoked with: 4.0\n    '
    assert m.ints_only(4) == 2
    with pytest.raises(TypeError) as excinfo:
        m.ints_only(4.0)
    assert msg(excinfo.value) == '\n        ints_only(): incompatible function arguments. The following argument types are supported:\n            1. (i: int) -> int\n\n        Invoked with: 4.0\n    '

def test_custom_caster_destruction():
    if False:
        for i in range(10):
            print('nop')
    'Tests that returning a pointer to a type that gets converted with a custom type caster gets\n    destroyed when the function has py::return_value_policy::take_ownership policy applied.\n    '
    cstats = m.destruction_tester_cstats()
    z = m.custom_caster_no_destroy()
    assert cstats.alive() == 1
    assert cstats.default_constructions == 1
    assert z
    z = m.custom_caster_destroy()
    assert z
    assert cstats.default_constructions == 2
    z = m.custom_caster_destroy_const()
    assert z
    assert cstats.default_constructions == 3
    assert cstats.alive() == 1

def test_custom_caster_other_lib():
    if False:
        print('Hello World!')
    assert m.other_lib_type(True)