from __future__ import annotations
from typing import Any
from unittest import mock
import pytest
from xarray.backends.lru_cache import LRUCache

def test_simple() -> None:
    if False:
        return 10
    cache: LRUCache[Any, Any] = LRUCache(maxsize=2)
    cache['x'] = 1
    cache['y'] = 2
    assert cache['x'] == 1
    assert cache['y'] == 2
    assert len(cache) == 2
    assert dict(cache) == {'x': 1, 'y': 2}
    assert list(cache.keys()) == ['x', 'y']
    assert list(cache.items()) == [('x', 1), ('y', 2)]
    cache['z'] = 3
    assert len(cache) == 2
    assert list(cache.items()) == [('y', 2), ('z', 3)]

def test_trivial() -> None:
    if False:
        while True:
            i = 10
    cache: LRUCache[Any, Any] = LRUCache(maxsize=0)
    cache['x'] = 1
    assert len(cache) == 0

def test_invalid() -> None:
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(TypeError):
        LRUCache(maxsize=None)
    with pytest.raises(ValueError):
        LRUCache(maxsize=-1)

def test_update_priority() -> None:
    if False:
        for i in range(10):
            print('nop')
    cache: LRUCache[Any, Any] = LRUCache(maxsize=2)
    cache['x'] = 1
    cache['y'] = 2
    assert list(cache) == ['x', 'y']
    assert 'x' in cache
    assert list(cache) == ['y', 'x']
    assert cache['y'] == 2
    assert list(cache) == ['x', 'y']
    cache['x'] = 3
    assert list(cache.items()) == [('y', 2), ('x', 3)]

def test_del() -> None:
    if False:
        for i in range(10):
            print('nop')
    cache: LRUCache[Any, Any] = LRUCache(maxsize=2)
    cache['x'] = 1
    cache['y'] = 2
    del cache['x']
    assert dict(cache) == {'y': 2}

def test_on_evict() -> None:
    if False:
        return 10
    on_evict = mock.Mock()
    cache = LRUCache(maxsize=1, on_evict=on_evict)
    cache['x'] = 1
    cache['y'] = 2
    on_evict.assert_called_once_with('x', 1)

def test_on_evict_trivial() -> None:
    if False:
        while True:
            i = 10
    on_evict = mock.Mock()
    cache = LRUCache(maxsize=0, on_evict=on_evict)
    cache['x'] = 1
    on_evict.assert_called_once_with('x', 1)

def test_resize() -> None:
    if False:
        for i in range(10):
            print('nop')
    cache: LRUCache[Any, Any] = LRUCache(maxsize=2)
    assert cache.maxsize == 2
    cache['w'] = 0
    cache['x'] = 1
    cache['y'] = 2
    assert list(cache.items()) == [('x', 1), ('y', 2)]
    cache.maxsize = 10
    cache['z'] = 3
    assert list(cache.items()) == [('x', 1), ('y', 2), ('z', 3)]
    cache.maxsize = 1
    assert list(cache.items()) == [('z', 3)]
    with pytest.raises(ValueError):
        cache.maxsize = -1