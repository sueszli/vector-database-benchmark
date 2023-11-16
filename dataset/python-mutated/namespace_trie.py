import pytest
import spack.util.naming

@pytest.fixture()
def trie():
    if False:
        i = 10
        return i + 15
    return spack.util.naming.NamespaceTrie()

def test_add_single(trie):
    if False:
        print('Hello World!')
    trie['foo'] = 'bar'
    assert trie.is_prefix('foo')
    assert trie.has_value('foo')
    assert trie['foo'] == 'bar'

def test_add_multiple(trie):
    if False:
        while True:
            i = 10
    trie['foo.bar'] = 'baz'
    assert not trie.has_value('foo')
    assert trie.is_prefix('foo')
    assert trie.is_prefix('foo.bar')
    assert trie.has_value('foo.bar')
    assert trie['foo.bar'] == 'baz'
    assert not trie.is_prefix('foo.bar.baz')
    assert not trie.has_value('foo.bar.baz')

def test_add_three(trie):
    if False:
        return 10
    trie['foo.bar.baz'] = 'quux'
    assert trie.is_prefix('foo')
    assert not trie.has_value('foo')
    assert trie.is_prefix('foo.bar')
    assert not trie.has_value('foo.bar')
    assert trie.is_prefix('foo.bar.baz')
    assert trie.has_value('foo.bar.baz')
    assert trie['foo.bar.baz'] == 'quux'
    assert not trie.is_prefix('foo.bar.baz.quux')
    assert not trie.has_value('foo.bar.baz.quux')
    trie['foo.bar'] = 'blah'
    assert trie.is_prefix('foo')
    assert not trie.has_value('foo')
    assert trie.is_prefix('foo.bar')
    assert trie.has_value('foo.bar')
    assert trie['foo.bar'] == 'blah'
    assert trie.is_prefix('foo.bar.baz')
    assert trie.has_value('foo.bar.baz')
    assert trie['foo.bar.baz'] == 'quux'
    assert not trie.is_prefix('foo.bar.baz.quux')
    assert not trie.has_value('foo.bar.baz.quux')

def test_add_none_single(trie):
    if False:
        return 10
    trie['foo'] = None
    assert trie.is_prefix('foo')
    assert trie.has_value('foo')
    assert trie['foo'] is None
    assert not trie.is_prefix('foo.bar')
    assert not trie.has_value('foo.bar')

def test_add_none_multiple(trie):
    if False:
        while True:
            i = 10
    trie['foo.bar'] = None
    assert trie.is_prefix('foo')
    assert not trie.has_value('foo')
    assert trie.is_prefix('foo.bar')
    assert trie.has_value('foo.bar')
    assert trie['foo.bar'] is None
    assert not trie.is_prefix('foo.bar.baz')
    assert not trie.has_value('foo.bar.baz')