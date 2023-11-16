import unittest
from unittest.mock import MagicMock
from cachetools import TTLCache
from streamlit.runtime.memory_session_storage import MemorySessionStorage

class MemorySessionStorageTest(unittest.TestCase):
    """Test MemorySessionStorage.

    These tests are intentionally extremely simple to ensure that we don't just end up
    testing cachetools.TTLCache. We try to just verify that we've wrapped TTLCache
    correctly, and in particular we avoid testing cache expiry functionality.
    """

    def test_uses_ttl_cache(self):
        if False:
            print('Hello World!')
        "Verify that the backing cache of a MemorySessionStorage is a TTLCache.\n\n        We do this because we're intentionally avoiding writing tests around cache\n        expiry because the cachetools library should do this for us. In the case\n        that the backing cache for a MemorySessionStorage ever changes, we'll likely be\n        responsible for adding our own tests.\n        "
        store = MemorySessionStorage()
        self.assertIsInstance(store._cache, TTLCache)

    def test_get(self):
        if False:
            while True:
                i = 10
        store = MemorySessionStorage()
        store._cache['foo'] = 'bar'
        self.assertEqual(store.get('foo'), 'bar')
        self.assertEqual(store.get('baz'), None)

    def test_save(self):
        if False:
            while True:
                i = 10
        store = MemorySessionStorage()
        session_info = MagicMock()
        session_info.session.id = 'foo'
        store.save(session_info)
        self.assertEqual(store.get('foo'), session_info)

    def test_delete(self):
        if False:
            for i in range(10):
                print('nop')
        store = MemorySessionStorage()
        store._cache['foo'] = 'bar'
        store.delete('foo')
        self.assertEqual(store.get('foo'), None)

    def test_list(self):
        if False:
            while True:
                i = 10
        store = MemorySessionStorage()
        store._cache['foo'] = 'bar'
        store._cache['baz'] = 'qux'
        self.assertEqual(store.list(), ['bar', 'qux'])