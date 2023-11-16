import pytest
from dagster._check import CheckError
from dagster._core.definitions.dependency import NodeHandle
from dagster._seven import json

def test_handle_path():
    if False:
        for i in range(10):
            print('nop')
    handle = NodeHandle('baz', NodeHandle('bar', NodeHandle('foo', None)))
    assert handle.path == ['foo', 'bar', 'baz']
    assert NodeHandle.from_path(handle.path) == handle
    handle = NodeHandle('foo', None)
    assert handle.path == ['foo']
    assert NodeHandle.from_path(handle.path) == handle

def test_handle_to_from_string():
    if False:
        for i in range(10):
            print('nop')
    handle = NodeHandle('baz', NodeHandle('bar', NodeHandle('foo', None)))
    assert handle.to_string() == 'foo.bar.baz'
    assert NodeHandle.from_string(handle.to_string()) == handle
    handle = NodeHandle('foo', None)
    assert handle.to_string() == 'foo'
    assert NodeHandle.from_string(handle.to_string()) == handle

def test_is_or_descends_from():
    if False:
        i = 10
        return i + 15
    ancestor = NodeHandle('baz', NodeHandle('bar', NodeHandle('foo', NodeHandle('quux', None))))
    descendant = NodeHandle('baz', NodeHandle('bar', NodeHandle('foo', None)))
    assert not descendant.is_or_descends_from(ancestor)
    ancestor = NodeHandle('baz', NodeHandle('bar', NodeHandle('foo', None)))
    descendant = NodeHandle('baz', NodeHandle('bar', NodeHandle('foo', None)))
    assert descendant.is_or_descends_from(ancestor)
    ancestor = NodeHandle('bar', NodeHandle('foo', None))
    descendant = NodeHandle('baz', NodeHandle('bar', NodeHandle('foo', None)))
    assert descendant.is_or_descends_from(ancestor)
    ancestor = NodeHandle('foo', None)
    descendant = NodeHandle('baz', NodeHandle('bar', NodeHandle('foo', None)))
    assert descendant.is_or_descends_from(ancestor)
    ancestor = NodeHandle('foo', None)
    descendant = NodeHandle('foo', None)
    assert descendant.is_or_descends_from(ancestor)
    ancestor = NodeHandle('foo', None)
    descendant = NodeHandle('bar', None)
    assert not descendant.is_or_descends_from(ancestor)
    ancestor = NodeHandle('baz', NodeHandle('bar', NodeHandle('foo', None)))
    descendant = NodeHandle('baz', None)
    assert not descendant.is_or_descends_from(ancestor)
    ancestor = NodeHandle('baz', NodeHandle('bar', NodeHandle('foo', None)))
    descendant = NodeHandle('baz', NodeHandle('bar', None))
    assert not descendant.is_or_descends_from(ancestor)

def test_pop():
    if False:
        print('Hello World!')
    handle = NodeHandle('baz', NodeHandle('bar', NodeHandle('foo', None)))
    assert handle.pop(NodeHandle('foo', None)) == NodeHandle('baz', NodeHandle('bar', None))
    assert handle.pop(NodeHandle('bar', NodeHandle('foo', None))) == NodeHandle('baz', None)
    with pytest.raises(CheckError, match='does not descend from'):
        handle = NodeHandle('baz', NodeHandle('bar', NodeHandle('foo', None)))
        handle.pop(NodeHandle('quux', None))

def test_with_ancestor():
    if False:
        return 10
    handle = NodeHandle('baz', NodeHandle('bar', NodeHandle('foo', None)))
    assert handle.with_ancestor(None) == handle
    assert handle.with_ancestor(NodeHandle('quux', None)) == NodeHandle('baz', NodeHandle('bar', NodeHandle('foo', NodeHandle('quux', None))))

def test_dict_roundtrip():
    if False:
        while True:
            i = 10
    handle = NodeHandle('baz', NodeHandle('bar', NodeHandle('foo', None)))
    assert NodeHandle.from_dict(json.loads(json.dumps(handle._asdict()))) == handle
    handle = NodeHandle('foo', None)
    assert NodeHandle.from_dict(json.loads(json.dumps(handle._asdict()))) == handle