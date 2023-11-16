from pytest_pyodide import run_in_pyodide

@run_in_pyodide(packages=['frozenlist'])
def test_subclass(selenium):
    if False:
        i = 10
        return i + 15
    from collections.abc import MutableSequence
    from frozenlist import FrozenList
    assert issubclass(FrozenList, MutableSequence)

@run_in_pyodide(packages=['frozenlist'])
def test_iface(selenium):
    if False:
        return 10
    from collections.abc import MutableSequence
    from frozenlist import FrozenList
    SKIP_METHODS = {'__abstractmethods__', '__slots__'}
    for name in set(dir(MutableSequence)) - SKIP_METHODS:
        if name.startswith('_') and (not name.endswith('_')):
            continue
        assert hasattr(FrozenList, name)

@run_in_pyodide(packages=['frozenlist'])
def test_ctor_default(selenium):
    if False:
        i = 10
        return i + 15
    from frozenlist import FrozenList
    _list = FrozenList([])
    assert not _list.frozen

@run_in_pyodide(packages=['frozenlist'])
def test_ctor(selenium):
    if False:
        i = 10
        return i + 15
    from frozenlist import FrozenList
    _list = FrozenList([1])
    assert not _list.frozen

@run_in_pyodide(packages=['frozenlist'])
def test_ctor_copy_list(selenium):
    if False:
        return 10
    from frozenlist import FrozenList
    orig = [1]
    _list = FrozenList(orig)
    del _list[0]
    assert _list != orig

@run_in_pyodide(packages=['frozenlist'])
def test_freeze(selenium):
    if False:
        print('Hello World!')
    from frozenlist import FrozenList
    _list = FrozenList()
    _list.freeze()
    assert _list.frozen

@run_in_pyodide(packages=['frozenlist'])
def test_repr(selenium):
    if False:
        for i in range(10):
            print('nop')
    from frozenlist import FrozenList
    _list = FrozenList([1])
    assert repr(_list) == '<FrozenList(frozen=False, [1])>'
    _list.freeze()
    assert repr(_list) == '<FrozenList(frozen=True, [1])>'

@run_in_pyodide(packages=['frozenlist'])
def test_getitem(selenium):
    if False:
        while True:
            i = 10
    from frozenlist import FrozenList
    _list = FrozenList([1, 2])
    assert _list[1] == 2

@run_in_pyodide(packages=['frozenlist'])
def test_setitem(selenium):
    if False:
        i = 10
        return i + 15
    from frozenlist import FrozenList
    _list = FrozenList([1, 2])
    _list[1] = 3
    assert _list[1] == 3

@run_in_pyodide(packages=['frozenlist'])
def test_delitem(selenium):
    if False:
        return 10
    from frozenlist import FrozenList
    _list = FrozenList([1, 2])
    del _list[0]
    assert len(_list) == 1
    assert _list[0] == 2

@run_in_pyodide(packages=['frozenlist'])
def test_len(selenium):
    if False:
        print('Hello World!')
    from frozenlist import FrozenList
    _list = FrozenList([1])
    assert len(_list) == 1

@run_in_pyodide(packages=['frozenlist'])
def test_iter(selenium):
    if False:
        for i in range(10):
            print('nop')
    from frozenlist import FrozenList
    _list = FrozenList([1, 2])
    assert list(iter(_list)) == [1, 2]

@run_in_pyodide(packages=['frozenlist'])
def test_reversed(selenium):
    if False:
        return 10
    from frozenlist import FrozenList
    _list = FrozenList([1, 2])
    assert list(reversed(_list)) == [2, 1]

@run_in_pyodide(packages=['frozenlist'])
def test_eq(selenium):
    if False:
        for i in range(10):
            print('nop')
    from frozenlist import FrozenList
    _list = FrozenList([1])
    assert _list == [1]

@run_in_pyodide(packages=['frozenlist'])
def test_ne(selenium):
    if False:
        print('Hello World!')
    from frozenlist import FrozenList
    _list = FrozenList([1])
    assert _list != [2]

@run_in_pyodide(packages=['frozenlist'])
def test_le(selenium):
    if False:
        return 10
    from frozenlist import FrozenList
    _list = FrozenList([1])
    assert _list <= [1]

@run_in_pyodide(packages=['frozenlist'])
def test_lt(selenium):
    if False:
        i = 10
        return i + 15
    from frozenlist import FrozenList
    _list = FrozenList([1])
    assert _list <= [3]

@run_in_pyodide(packages=['frozenlist'])
def test_ge(selenium):
    if False:
        for i in range(10):
            print('nop')
    from frozenlist import FrozenList
    _list = FrozenList([1])
    assert _list >= [1]

@run_in_pyodide(packages=['frozenlist'])
def test_gt(selenium):
    if False:
        for i in range(10):
            print('nop')
    from frozenlist import FrozenList
    _list = FrozenList([2])
    assert _list > [1]

@run_in_pyodide(packages=['frozenlist'])
def test_insert(selenium):
    if False:
        while True:
            i = 10
    from frozenlist import FrozenList
    _list = FrozenList([2])
    _list.insert(0, 1)
    assert _list == [1, 2]

@run_in_pyodide(packages=['frozenlist'])
def test_frozen_setitem(selenium):
    if False:
        i = 10
        return i + 15
    import pytest
    from frozenlist import FrozenList
    _list = FrozenList([1])
    _list.freeze()
    with pytest.raises(RuntimeError):
        _list[0] = 2

@run_in_pyodide(packages=['frozenlist'])
def test_frozen_delitem(selenium):
    if False:
        print('Hello World!')
    import pytest
    from frozenlist import FrozenList
    _list = FrozenList([1])
    _list.freeze()
    with pytest.raises(RuntimeError):
        del _list[0]

@run_in_pyodide(packages=['frozenlist'])
def test_frozen_insert(selenium):
    if False:
        print('Hello World!')
    import pytest
    from frozenlist import FrozenList
    _list = FrozenList([1])
    _list.freeze()
    with pytest.raises(RuntimeError):
        _list.insert(0, 2)

@run_in_pyodide(packages=['frozenlist'])
def test_contains(selenium):
    if False:
        for i in range(10):
            print('nop')
    from frozenlist import FrozenList
    _list = FrozenList([2])
    assert 2 in _list

@run_in_pyodide(packages=['frozenlist'])
def test_iadd(selenium):
    if False:
        i = 10
        return i + 15
    from frozenlist import FrozenList
    _list = FrozenList([1])
    _list += [2]
    assert _list == [1, 2]

@run_in_pyodide(packages=['frozenlist'])
def test_iadd_frozen(selenium):
    if False:
        print('Hello World!')
    import pytest
    from frozenlist import FrozenList
    _list = FrozenList([1])
    _list.freeze()
    with pytest.raises(RuntimeError):
        _list += [2]
    assert _list == [1]

@run_in_pyodide(packages=['frozenlist'])
def test_index(selenium):
    if False:
        for i in range(10):
            print('nop')
    from frozenlist import FrozenList
    _list = FrozenList([1])
    assert _list.index(1) == 0

@run_in_pyodide(packages=['frozenlist'])
def test_remove(selenium):
    if False:
        print('Hello World!')
    from frozenlist import FrozenList
    _list = FrozenList([1])
    _list.remove(1)
    assert len(_list) == 0

@run_in_pyodide(packages=['frozenlist'])
def test_remove_frozen(selenium):
    if False:
        return 10
    import pytest
    from frozenlist import FrozenList
    _list = FrozenList([1])
    _list.freeze()
    with pytest.raises(RuntimeError):
        _list.remove(1)
    assert _list == [1]

@run_in_pyodide(packages=['frozenlist'])
def test_clear(selenium):
    if False:
        while True:
            i = 10
    from frozenlist import FrozenList
    _list = FrozenList([1])
    _list.clear()
    assert len(_list) == 0

@run_in_pyodide(packages=['frozenlist'])
def test_clear_frozen(selenium):
    if False:
        print('Hello World!')
    import pytest
    from frozenlist import FrozenList
    _list = FrozenList([1])
    _list.freeze()
    with pytest.raises(RuntimeError):
        _list.clear()
    assert _list == [1]

@run_in_pyodide(packages=['frozenlist'])
def test_extend(selenium):
    if False:
        return 10
    from frozenlist import FrozenList
    _list = FrozenList([1])
    _list.extend([2])
    assert _list == [1, 2]

@run_in_pyodide(packages=['frozenlist'])
def test_extend_frozen(selenium):
    if False:
        i = 10
        return i + 15
    import pytest
    from frozenlist import FrozenList
    _list = FrozenList([1])
    _list.freeze()
    with pytest.raises(RuntimeError):
        _list.extend([2])
    assert _list == [1]

@run_in_pyodide(packages=['frozenlist'])
def test_reverse(selenium):
    if False:
        print('Hello World!')
    from frozenlist import FrozenList
    _list = FrozenList([1, 2])
    _list.reverse()
    assert _list == [2, 1]

@run_in_pyodide(packages=['frozenlist'])
def test_reverse_frozen(selenium):
    if False:
        while True:
            i = 10
    import pytest
    from frozenlist import FrozenList
    _list = FrozenList([1, 2])
    _list.freeze()
    with pytest.raises(RuntimeError):
        _list.reverse()
    assert _list == [1, 2]

@run_in_pyodide(packages=['frozenlist'])
def test_pop(selenium):
    if False:
        return 10
    from frozenlist import FrozenList
    _list = FrozenList([1, 2])
    assert _list.pop(0) == 1
    assert _list == [2]

@run_in_pyodide(packages=['frozenlist'])
def test_pop_default(selenium):
    if False:
        print('Hello World!')
    from frozenlist import FrozenList
    _list = FrozenList([1, 2])
    assert _list.pop() == 2
    assert _list == [1]

@run_in_pyodide(packages=['frozenlist'])
def test_pop_frozen(selenium):
    if False:
        i = 10
        return i + 15
    import pytest
    from frozenlist import FrozenList
    _list = FrozenList([1, 2])
    _list.freeze()
    with pytest.raises(RuntimeError):
        _list.pop()
    assert _list == [1, 2]

@run_in_pyodide(packages=['frozenlist'])
def test_append(selenium):
    if False:
        i = 10
        return i + 15
    from frozenlist import FrozenList
    _list = FrozenList([1, 2])
    _list.append(3)
    assert _list == [1, 2, 3]

@run_in_pyodide(packages=['frozenlist'])
def test_append_frozen(selenium):
    if False:
        i = 10
        return i + 15
    import pytest
    from frozenlist import FrozenList
    _list = FrozenList([1, 2])
    _list.freeze()
    with pytest.raises(RuntimeError):
        _list.append(3)
    assert _list == [1, 2]

@run_in_pyodide(packages=['frozenlist'])
def test_hash(selenium):
    if False:
        for i in range(10):
            print('nop')
    import pytest
    from frozenlist import FrozenList
    _list = FrozenList([1, 2])
    with pytest.raises(RuntimeError):
        hash(_list)

@run_in_pyodide(packages=['frozenlist'])
def test_hash_frozen(selenium):
    if False:
        print('Hello World!')
    from frozenlist import FrozenList
    _list = FrozenList([1, 2])
    _list.freeze()
    h = hash(_list)
    assert h == hash((1, 2))

@run_in_pyodide(packages=['frozenlist'])
def test_dict_key(selenium):
    if False:
        return 10
    import pytest
    from frozenlist import FrozenList
    _list = FrozenList([1, 2])
    with pytest.raises(RuntimeError):
        {_list: 'hello'}
    _list.freeze()
    {_list: 'hello'}

@run_in_pyodide(packages=['frozenlist'])
def test_count(selenium):
    if False:
        i = 10
        return i + 15
    from frozenlist import FrozenList
    _list = FrozenList([1, 2])
    assert _list.count(1) == 1