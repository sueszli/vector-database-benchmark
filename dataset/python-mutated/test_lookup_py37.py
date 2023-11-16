from __future__ import annotations
import collections
import collections.abc
import contextlib
import re
import sys
import pytest
from hypothesis import assume, given
if sys.version_info < (3, 9):
    pytestmark = pytest.mark.xfail(raises=Exception, reason='Requires Python 3.9 (PEP 585) or later.')

class Elem:
    pass

class Value:
    pass

def check(t, ex):
    if False:
        while True:
            i = 10
    assert isinstance(ex, t)
    assert all((isinstance(e, Elem) for e in ex))
    assume(ex)

@given(...)
def test_resolving_standard_tuple1_as_generic(x: tuple[Elem]):
    if False:
        i = 10
        return i + 15
    check(tuple, x)

@given(...)
def test_resolving_standard_tuple2_as_generic(x: tuple[Elem, Elem]):
    if False:
        return 10
    check(tuple, x)

@given(...)
def test_resolving_standard_tuple_variadic_as_generic(x: tuple[Elem, ...]):
    if False:
        for i in range(10):
            print('nop')
    check(tuple, x)

@given(...)
def test_resolving_standard_list_as_generic(x: list[Elem]):
    if False:
        for i in range(10):
            print('nop')
    check(list, x)

@given(...)
def test_resolving_standard_dict_as_generic(x: dict[Elem, Value]):
    if False:
        while True:
            i = 10
    check(dict, x)
    assert all((isinstance(e, Value) for e in x.values()))

@given(...)
def test_resolving_standard_set_as_generic(x: set[Elem]):
    if False:
        print('Hello World!')
    check(set, x)

@given(...)
def test_resolving_standard_frozenset_as_generic(x: frozenset[Elem]):
    if False:
        i = 10
        return i + 15
    check(frozenset, x)

@given(...)
def test_resolving_standard_deque_as_generic(x: collections.deque[Elem]):
    if False:
        while True:
            i = 10
    check(collections.deque, x)

@given(...)
def test_resolving_standard_defaultdict_as_generic(x: collections.defaultdict[Elem, Value]):
    if False:
        while True:
            i = 10
    check(collections.defaultdict, x)
    assert all((isinstance(e, Value) for e in x.values()))

@given(...)
def test_resolving_standard_ordered_dict_as_generic(x: collections.OrderedDict[Elem, Value]):
    if False:
        print('Hello World!')
    check(collections.OrderedDict, x)
    assert all((isinstance(e, Value) for e in x.values()))

@given(...)
def test_resolving_standard_counter_as_generic(x: collections.Counter[Elem]):
    if False:
        print('Hello World!')
    check(collections.Counter, x)
    assume(any(x.values()))

@given(...)
def test_resolving_standard_chainmap_as_generic(x: collections.ChainMap[Elem, Value]):
    if False:
        for i in range(10):
            print('nop')
    check(collections.ChainMap, x)
    assert all((isinstance(e, Value) for e in x.values()))

@given(...)
def test_resolving_standard_iterable_as_generic(x: collections.abc.Iterable[Elem]):
    if False:
        for i in range(10):
            print('nop')
    check(collections.abc.Iterable, x)

@given(...)
def test_resolving_standard_iterator_as_generic(x: collections.abc.Iterator[Elem]):
    if False:
        while True:
            i = 10
    check(collections.abc.Iterator, x)

@given(...)
def test_resolving_standard_generator_as_generic(x: collections.abc.Generator[Elem, None, Value]):
    if False:
        return 10
    assert isinstance(x, collections.abc.Generator)
    try:
        while True:
            e = next(x)
            assert isinstance(e, Elem)
            x.send(None)
    except StopIteration as stop:
        assert isinstance(stop.value, Value)

@given(...)
def test_resolving_standard_reversible_as_generic(x: collections.abc.Reversible[Elem]):
    if False:
        print('Hello World!')
    check(collections.abc.Reversible, x)

@given(...)
def test_resolving_standard_container_as_generic(x: collections.abc.Container[Elem]):
    if False:
        print('Hello World!')
    check(collections.abc.Container, x)

@given(...)
def test_resolving_standard_collection_as_generic(x: collections.abc.Collection[Elem]):
    if False:
        for i in range(10):
            print('nop')
    check(collections.abc.Collection, x)

@given(...)
def test_resolving_standard_callable_ellipsis(x: collections.abc.Callable[..., Elem]):
    if False:
        return 10
    assert isinstance(x, collections.abc.Callable)
    assert callable(x)
    assert isinstance(x(), Elem)
    assert isinstance(x(1, 2, 3, a=4, b=5, c=6), Elem)

@given(...)
def test_resolving_standard_callable_no_args(x: collections.abc.Callable[[], Elem]):
    if False:
        while True:
            i = 10
    assert isinstance(x, collections.abc.Callable)
    assert callable(x)
    assert isinstance(x(), Elem)
    with pytest.raises(TypeError):
        x(1)
    with pytest.raises(TypeError):
        x(a=1)

@given(...)
def test_resolving_standard_collections_set_as_generic(x: collections.abc.Set[Elem]):
    if False:
        print('Hello World!')
    check(collections.abc.Set, x)

@given(...)
def test_resolving_standard_collections_mutableset_as_generic(x: collections.abc.MutableSet[Elem]):
    if False:
        return 10
    check(collections.abc.MutableSet, x)

@given(...)
def test_resolving_standard_mapping_as_generic(x: collections.abc.Mapping[Elem, Value]):
    if False:
        while True:
            i = 10
    check(collections.abc.Mapping, x)
    assert all((isinstance(e, Value) for e in x.values()))

@given(...)
def test_resolving_standard_mutable_mapping_as_generic(x: collections.abc.MutableMapping[Elem, Value]):
    if False:
        i = 10
        return i + 15
    check(collections.abc.MutableMapping, x)
    assert all((isinstance(e, Value) for e in x.values()))

@given(...)
def test_resolving_standard_sequence_as_generic(x: collections.abc.Sequence[Elem]):
    if False:
        while True:
            i = 10
    check(collections.abc.Sequence, x)

@given(...)
def test_resolving_standard_mutable_sequence_as_generic(x: collections.abc.MutableSequence[Elem]):
    if False:
        i = 10
        return i + 15
    check(collections.abc.MutableSequence, x)

@given(...)
def test_resolving_standard_keysview_as_generic(x: collections.abc.KeysView[Elem]):
    if False:
        return 10
    check(collections.abc.KeysView, x)

@given(...)
def test_resolving_standard_itemsview_as_generic(x: collections.abc.ItemsView[Elem, Value]):
    if False:
        i = 10
        return i + 15
    assert isinstance(x, collections.abc.ItemsView)
    assert all((isinstance(e, Elem) and isinstance(v, Value) for (e, v) in x))
    assume(x)

@given(...)
def test_resolving_standard_valuesview_as_generic(x: collections.abc.ValuesView[Elem]):
    if False:
        i = 10
        return i + 15
    check(collections.abc.ValuesView, x)

@pytest.mark.xfail
@given(...)
def test_resolving_standard_contextmanager_as_generic(x: contextlib.AbstractContextManager[Elem]):
    if False:
        i = 10
        return i + 15
    assert isinstance(x, contextlib.AbstractContextManager)

@given(...)
def test_resolving_standard_re_match_bytes_as_generic(x: re.Match[bytes]):
    if False:
        for i in range(10):
            print('nop')
    assert isinstance(x, re.Match)
    assert isinstance(x[0], bytes)

@given(...)
def test_resolving_standard_re_match_str_as_generic(x: re.Match[str]):
    if False:
        while True:
            i = 10
    assert isinstance(x, re.Match)
    assert isinstance(x[0], str)

@given(...)
def test_resolving_standard_re_pattern_bytes_as_generic(x: re.Pattern[bytes]):
    if False:
        i = 10
        return i + 15
    assert isinstance(x, re.Pattern)
    assert isinstance(x.pattern, bytes)

@given(...)
def test_resolving_standard_re_pattern_str_as_generic(x: re.Pattern[str]):
    if False:
        i = 10
        return i + 15
    assert isinstance(x, re.Pattern)
    assert isinstance(x.pattern, str)