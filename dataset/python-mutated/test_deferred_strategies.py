from hypothesis import strategies as st
from tests.common.debug import minimal

def test_large_branching_tree():
    if False:
        while True:
            i = 10
    tree = st.deferred(lambda : st.integers() | st.tuples(tree, tree, tree, tree, tree))
    assert minimal(tree) == 0
    assert minimal(tree, lambda x: isinstance(x, tuple)) == (0,) * 5

def test_non_trivial_json():
    if False:
        return 10
    json = st.deferred(lambda : st.none() | st.floats() | st.text() | lists | objects)
    lists = st.lists(json)
    objects = st.dictionaries(st.text(), json)
    assert minimal(json) is None
    small_list = minimal(json, lambda x: isinstance(x, list) and x)
    assert small_list == [None]
    x = minimal(json, lambda x: isinstance(x, dict) and isinstance(x.get(''), list))
    assert x == {'': []}

def test_self_recursive_lists():
    if False:
        while True:
            i = 10
    x = st.deferred(lambda : st.lists(x))
    assert minimal(x) == []
    assert minimal(x, bool) == [[]]
    assert minimal(x, lambda x: len(x) > 1) == [[], []]