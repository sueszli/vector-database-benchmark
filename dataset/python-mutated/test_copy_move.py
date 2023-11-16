import pytest
from pybind11_tests import copy_move_policies as m

def test_lacking_copy_ctor():
    if False:
        print('Hello World!')
    with pytest.raises(RuntimeError) as excinfo:
        m.lacking_copy_ctor.get_one()
    assert 'is non-copyable!' in str(excinfo.value)

def test_lacking_move_ctor():
    if False:
        print('Hello World!')
    with pytest.raises(RuntimeError) as excinfo:
        m.lacking_move_ctor.get_one()
    assert 'is neither movable nor copyable!' in str(excinfo.value)

def test_move_and_copy_casts():
    if False:
        while True:
            i = 10
    'Cast some values in C++ via custom type casters and count the number of moves/copies.'
    cstats = m.move_and_copy_cstats()
    (c_m, c_mc, c_c) = (cstats['MoveOnlyInt'], cstats['MoveOrCopyInt'], cstats['CopyOnlyInt'])
    assert m.move_and_copy_casts(3) == 18
    assert c_m.copy_assignments + c_m.copy_constructions == 0
    assert c_m.move_assignments == 2
    assert c_m.move_constructions >= 2
    assert c_mc.alive() == 0
    assert c_mc.copy_assignments + c_mc.copy_constructions == 0
    assert c_mc.move_assignments == 2
    assert c_mc.move_constructions >= 2
    assert c_c.alive() == 0
    assert c_c.copy_assignments == 2
    assert c_c.copy_constructions >= 2
    assert c_m.alive() + c_mc.alive() + c_c.alive() == 0

def test_move_and_copy_loads():
    if False:
        print('Hello World!')
    'Call some functions that load arguments via custom type casters and count the number of\n    moves/copies.'
    cstats = m.move_and_copy_cstats()
    (c_m, c_mc, c_c) = (cstats['MoveOnlyInt'], cstats['MoveOrCopyInt'], cstats['CopyOnlyInt'])
    assert m.move_only(10) == 10
    assert m.move_or_copy(11) == 11
    assert m.copy_only(12) == 12
    assert m.move_pair((13, 14)) == 27
    assert m.move_tuple((15, 16, 17)) == 48
    assert m.copy_tuple((18, 19)) == 37
    assert m.move_copy_nested((1, ((2, 3, (4,)), 5))) == 15
    assert c_m.copy_assignments + c_m.copy_constructions == 0
    assert c_m.move_assignments == 6
    assert c_m.move_constructions == 9
    assert c_mc.copy_assignments + c_mc.copy_constructions == 0
    assert c_mc.move_assignments == 5
    assert c_mc.move_constructions == 8
    assert c_c.copy_assignments == 4
    assert c_c.copy_constructions == 6
    assert c_m.alive() + c_mc.alive() + c_c.alive() == 0

@pytest.mark.skipif(not m.has_optional, reason='no <optional>')
def test_move_and_copy_load_optional():
    if False:
        print('Hello World!')
    'Tests move/copy loads of std::optional arguments'
    cstats = m.move_and_copy_cstats()
    (c_m, c_mc, c_c) = (cstats['MoveOnlyInt'], cstats['MoveOrCopyInt'], cstats['CopyOnlyInt'])
    assert m.move_optional(10) == 10
    assert m.move_or_copy_optional(11) == 11
    assert m.copy_optional(12) == 12
    assert m.move_optional_tuple((3, 4, 5)) == 12
    assert c_m.copy_assignments + c_m.copy_constructions == 0
    assert c_m.move_assignments == 2
    assert c_m.move_constructions == 5
    assert c_mc.copy_assignments + c_mc.copy_constructions == 0
    assert c_mc.move_assignments == 2
    assert c_mc.move_constructions == 5
    assert c_c.copy_assignments == 2
    assert c_c.copy_constructions == 5
    assert c_m.alive() + c_mc.alive() + c_c.alive() == 0

def test_private_op_new():
    if False:
        return 10
    'An object with a private `operator new` cannot be returned by value'
    with pytest.raises(RuntimeError) as excinfo:
        m.private_op_new_value()
    assert 'is neither movable nor copyable' in str(excinfo.value)
    assert m.private_op_new_reference().value == 1

def test_move_fallback():
    if False:
        while True:
            i = 10
    '#389: rvp::move should fall-through to copy on non-movable objects'
    m1 = m.get_moveissue1(1)
    assert m1.value == 1
    m2 = m.get_moveissue2(2)
    assert m2.value == 2

def test_pytype_rvalue_cast():
    if False:
        while True:
            i = 10
    'Make sure that cast from pytype rvalue to other pytype works'
    value = m.get_pytype_rvalue_castissue(1.0)
    assert value == 1