from featuretools.utils import Trie

def test_get_node():
    if False:
        while True:
            i = 10
    t = Trie(default=lambda : 'default')
    t.get_node([1, 2, 3]).value = '123'
    t.get_node([1, 2, 4]).value = '124'
    sub = t.get_node([1, 2])
    assert sub.get_node([3]).value == '123'
    assert sub.get_node([4]).value == '124'
    sub.get_node([4, 5]).value = '1245'
    assert t.get_node([1, 2, 4, 5]).value == '1245'

def test_setting_and_getting():
    if False:
        print('Hello World!')
    t = Trie(default=lambda : 'default')
    assert t.get_node([1, 2, 3]).value == 'default'
    t.get_node([1, 2, 3]).value = '123'
    t.get_node([1, 2, 4]).value = '124'
    assert t.get_node([1, 2, 3]).value == '123'
    assert t.get_node([1, 2, 4]).value == '124'
    assert t.get_node([1]).value == 'default'
    t.get_node([1]).value = '1'
    assert t.get_node([1]).value == '1'
    t.get_node([1, 2, 3]).value = 'updated'
    assert t.get_node([1, 2, 3]).value == 'updated'

def test_iteration():
    if False:
        i = 10
        return i + 15
    t = Trie(default=lambda : 'default', path_constructor=tuple)
    t.get_node((1, 2, 3)).value = '123'
    t.get_node((1, 2, 4)).value = '124'
    expected = [((), 'default'), ((1,), 'default'), ((1, 2), 'default'), ((1, 2, 3), '123'), ((1, 2, 4), '124')]
    for (i, value) in enumerate(t):
        assert value == expected[i]