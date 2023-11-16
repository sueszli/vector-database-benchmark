"""Unit tests for InMemoryCacheStorageWrapper"""
import unittest
from unittest.mock import patch
from testfixtures import TempDirectory
from streamlit.runtime.caching.storage import CacheStorageContext, CacheStorageKeyNotFoundError
from streamlit.runtime.caching.storage.dummy_cache_storage import DummyCacheStorage
from streamlit.runtime.caching.storage.in_memory_cache_storage_wrapper import InMemoryCacheStorageWrapper
from streamlit.runtime.caching.storage.local_disk_cache_storage import LocalDiskCacheStorage

class InMemoryCacheStorageWrapperTest(unittest.TestCase):
    """Unit tests for InMemoryCacheStorageWrapper"""

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        self.tempdir = TempDirectory(create=True)
        self.patch_get_cache_folder_path = patch('streamlit.runtime.caching.storage.local_disk_cache_storage.get_cache_folder_path', return_value=self.tempdir.path)
        self.patch_get_cache_folder_path.start()

    def tearDown(self):
        if False:
            print('Hello World!')
        super().tearDown()
        self.patch_get_cache_folder_path.stop()
        self.tempdir.cleanup()

    def get_storage_context(self):
        if False:
            print('Hello World!')
        return CacheStorageContext(function_key='func-key', function_display_name='func-display-name', persist='disk')

    def test_in_memory_cache_storage_wrapper_works_with_local_disk_storage(self):
        if False:
            while True:
                i = 10
        '\n        InMemoryCacheStorageWrapper should work with local disk storage without raising\n        an exception\n        '
        context = self.get_storage_context()
        InMemoryCacheStorageWrapper(persist_storage=LocalDiskCacheStorage(context), context=context)

    def test_in_memory_cache_storage_wrapper_works_with_dummy_storage(self):
        if False:
            i = 10
            return i + 15
        '\n        InMemoryCacheStorageWrapper should work with dummy storage without raising\n        an exception\n        '
        context = self.get_storage_context()
        InMemoryCacheStorageWrapper(persist_storage=DummyCacheStorage(), context=context)

    def test_in_memory_cache_storage_wrapper_get_key_in_persist_storage(self):
        if False:
            print('Hello World!')
        "\n        Test that storage.get() returns the value from persist storage\n        if value doesn't exist in memory.\n        "
        context = self.get_storage_context()
        persist_storage = LocalDiskCacheStorage(context)
        wrapped_storage = InMemoryCacheStorageWrapper(persist_storage=persist_storage, context=context)
        persist_storage.set('some-key', b'some-value')
        with patch.object(persist_storage, 'get', wraps=persist_storage.get) as mock_persist_get:
            self.assertEqual(wrapped_storage.get('some-key'), b'some-value')
            mock_persist_get.assert_called_once_with('some-key')
            self.assertEqual(wrapped_storage.get('some-key'), b'some-value')
            mock_persist_get.assert_called_once()

    def test_in_memory_cache_storage_wrapper_get_key_in_memory_storage(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that storage.get() returns the value from in_memory storage\n        if value exists in memory.\n        '
        context = self.get_storage_context()
        persist_storage = LocalDiskCacheStorage(context)
        wrapped_storage = InMemoryCacheStorageWrapper(persist_storage=persist_storage, context=context)
        wrapped_storage.set('some-key', b'some-value')
        with patch.object(persist_storage, 'get', wraps=persist_storage.get) as mock_persist_get:
            self.assertEqual(wrapped_storage.get('some-key'), b'some-value')
            mock_persist_get.assert_not_called()

    def test_in_memory_cache_storage_wrapper_set(self):
        if False:
            return 10
        '\n        Test that storage.set() sets value both in in-memory cache and\n        in persist storage\n        '
        context = self.get_storage_context()
        persist_storage = LocalDiskCacheStorage(context)
        wrapped_storage = InMemoryCacheStorageWrapper(persist_storage=persist_storage, context=context)
        persist_storage.set('some-key', b'some-value')
        with patch.object(persist_storage, 'set', wraps=persist_storage.set) as mock_persist_set:
            wrapped_storage.set('some-key', b'some-value')
            mock_persist_set.assert_called_once_with('some-key', b'some-value')
        self.assertEqual(wrapped_storage.get('some-key'), b'some-value')

    def test_in_memory_cache_storage_wrapper_delete(self):
        if False:
            while True:
                i = 10
        '\n        Test that storage.delete() deletes value both in in-memory cache\n        and in persist storage\n        '
        context = self.get_storage_context()
        persist_storage = LocalDiskCacheStorage(context)
        wrapped_storage = InMemoryCacheStorageWrapper(persist_storage=persist_storage, context=context)
        wrapped_storage.set('some-key', b'some-value')
        with patch.object(persist_storage, 'delete', wraps=persist_storage.delete) as mock_persist_delete:
            wrapped_storage.delete('some-key')
            mock_persist_delete.assert_called_once_with('some-key')
        with self.assertRaises(CacheStorageKeyNotFoundError):
            wrapped_storage.get('some-key')

    def test_in_memory_cache_storage_wrapper_close(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that storage.close() closes the underlying persist storage\n        '
        context = self.get_storage_context()
        persist_storage = LocalDiskCacheStorage(context)
        wrapped_storage = InMemoryCacheStorageWrapper(persist_storage=persist_storage, context=context)
        with patch.object(persist_storage, 'close', wraps=persist_storage.close) as mock_persist_close:
            wrapped_storage.close()
            mock_persist_close.assert_called_once()