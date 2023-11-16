"""Unit tests for DummyCacheStorage and MemoryCacheStorageManager"""
import unittest
from streamlit.runtime.caching.storage import CacheStorageContext, CacheStorageKeyNotFoundError
from streamlit.runtime.caching.storage.dummy_cache_storage import DummyCacheStorage, MemoryCacheStorageManager

class DummyCacheStorageManagerTest(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        super().setUp()
        self.context = CacheStorageContext(function_key='func-key', function_display_name='func-display-name', persist='disk')
        self.dummy_cache_storage = DummyCacheStorage()
        self.storage_manager = MemoryCacheStorageManager()
        self.storage = self.storage_manager.create(self.context)

    def test_in_memory_wrapped_dummy_cache_storage_get_not_found(self):
        if False:
            while True:
                i = 10
        '\n        Test that storage.get() returns CacheStorageKeyNotFoundError when key is not\n        present.\n        '
        with self.assertRaises(CacheStorageKeyNotFoundError):
            self.storage.get('some-key')

    def test_in_memory_wrapped_dummy_cache_storage_get_found(self):
        if False:
            return 10
        '\n        Test that storage.get() returns the value when key is present.\n        '
        self.storage.set('some-key', b'some-value')
        self.assertEqual(self.storage.get('some-key'), b'some-value')

    def test_in_memory_wrapped_dummy_cache_storage_storage_set(self):
        if False:
            return 10
        '\n        Test that storage.set() sets the value correctly.\n        '
        self.storage.set('new-key', b'new-value')
        self.assertEqual(self.storage.get('new-key'), b'new-value')

    def test_in_memory_wrapped_dummy_cache_storage_storage_set_override(self):
        if False:
            return 10
        '\n        Test that storage.set() overrides the value.\n        '
        self.storage.set('another_key', b'another_value')
        self.storage.set('another_key', b'new_value')
        self.assertEqual(self.storage.get('another_key'), b'new_value')

    def test_in_memory_wrapped_dummy_cache_storage_storage_delete(self):
        if False:
            while True:
                i = 10
        '\n        Test that storage.delete() deletes the value correctly.\n        '
        self.storage.set('new-key', b'new-value')
        self.storage.delete('new-key')
        with self.assertRaises(CacheStorageKeyNotFoundError):
            self.storage.get('new-key')

class DummyCacheStorageTest(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.storage = DummyCacheStorage()

    def test_dummy_storage_get_always_not_found(self):
        if False:
            i = 10
            return i + 15
        'Test that storage.get() always returns CacheStorageKeyNotFoundError.'
        with self.assertRaises(CacheStorageKeyNotFoundError):
            self.storage.get('some-key')
        self.storage.set('some-key', b'some-value')
        with self.assertRaises(CacheStorageKeyNotFoundError):
            self.storage.get('some-key')

    def test_storage_set(self):
        if False:
            print('Hello World!')
        '\n        Test that storage.set() works correctly, at always do nothing without\n        raising exception.'
        self.storage.set('new-key', b'new-value')
        with self.assertRaises(CacheStorageKeyNotFoundError):
            self.storage.get('new-key')

    def test_storage_delete(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that storage.delete() works correctly, at always do nothing without\n        raising exception.\n        '
        self.storage.delete('another-key')
        self.storage.delete('another-key')
        self.storage.delete('another-key')

    def test_storage_clear(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that storage.clear() works correctly, at always do nothing without\n        raising exception.\n        '
        self.storage.clear()

    def test_storage_close(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that storage.close() works correctly, at always do nothing without\n        raising exception.\n        '
        self.storage.close()