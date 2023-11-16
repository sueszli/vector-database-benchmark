"""Unit tests for LocalDiskCacheStorage and LocalDiskCacheStorageManager"""
import logging
import math
import os.path
import shutil
import unittest
from unittest.mock import MagicMock, patch
from testfixtures import TempDirectory
from streamlit import util
from streamlit.logger import get_logger
from streamlit.runtime.caching.storage import CacheStorageContext, CacheStorageError, CacheStorageKeyNotFoundError
from streamlit.runtime.caching.storage.in_memory_cache_storage_wrapper import InMemoryCacheStorageWrapper
from streamlit.runtime.caching.storage.local_disk_cache_storage import LocalDiskCacheStorage, LocalDiskCacheStorageManager

class LocalDiskCacheStorageManagerTest(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            return 10
        super().setUp()
        self.tempdir = TempDirectory(create=True)
        self.patch_get_cache_folder_path = patch('streamlit.runtime.caching.storage.local_disk_cache_storage.get_cache_folder_path', return_value=self.tempdir.path)
        self.patch_get_cache_folder_path.start()

    def tearDown(self) -> None:
        if False:
            print('Hello World!')
        super().tearDown()
        self.patch_get_cache_folder_path.stop()
        self.tempdir.cleanup()

    def test_create_persist_context(self):
        if False:
            i = 10
            return i + 15
        'Tests that LocalDiskCacheStorageManager.create()\n        returns a LocalDiskCacheStorage with correct parameters from context, if\n        persist="disk"\n        '
        context = CacheStorageContext(function_key='func-key', function_display_name='func-display-name', persist='disk', ttl_seconds=60, max_entries=100)
        manager = LocalDiskCacheStorageManager()
        storage = manager.create(context)
        self.assertIsInstance(storage, InMemoryCacheStorageWrapper)
        self.assertEqual(storage.ttl_seconds, 60)
        self.assertEqual(storage.max_entries, 100)

    def test_create_not_persist_context(self):
        if False:
            i = 10
            return i + 15
        'Tests that LocalDiskCacheStorageManager.create()\n        returns a LocalDiskCacheStorage with correct parameters from context, if\n        persist is None\n        '
        context = CacheStorageContext(function_key='func-key', function_display_name='func-display-name', persist=None, ttl_seconds=None, max_entries=None)
        manager = LocalDiskCacheStorageManager()
        storage = manager.create(context)
        self.assertIsInstance(storage, InMemoryCacheStorageWrapper)
        self.assertEqual(storage.ttl_seconds, math.inf)
        self.assertEqual(storage.max_entries, math.inf)

    def test_check_context_with_persist_and_ttl(self):
        if False:
            print('Hello World!')
        'Tests that LocalDiskCacheStorageManager.check_context() writes a warning\n        in logs when persist="disk" and ttl_seconds is not None\n        '
        context = CacheStorageContext(function_key='func-key', function_display_name='func-display-name', persist='disk', ttl_seconds=60, max_entries=100)
        with self.assertLogs('streamlit.runtime.caching.storage.local_disk_cache_storage', level=logging.WARNING) as logs:
            manager = LocalDiskCacheStorageManager()
            manager.check_context(context)
            output = ''.join(logs.output)
            self.assertIn("The cached function 'func-display-name' has a TTL that will be ignored. Persistent cached functions currently don't support TTL.", output)

    def test_check_context_without_persist(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests that LocalDiskCacheStorageManager.check_context() does not\n        write a warning in logs when persist is None and ttl_seconds is NOT None.\n        '
        context = CacheStorageContext(function_key='func-key', function_display_name='func-display-name', persist=None, ttl_seconds=60, max_entries=100)
        with self.assertLogs('streamlit.runtime.caching.storage.local_disk_cache_storage', level=logging.WARNING) as logs:
            manager = LocalDiskCacheStorageManager()
            manager.check_context(context)
            get_logger('streamlit.runtime.caching.storage.local_disk_cache_storage').warning('irrelevant warning so assertLogs passes')
            output = ''.join(logs.output)
            self.assertNotIn("The cached function 'func-display-name' has a TTL that will be ignored. Persistent cached functions currently don't support TTL.", output)

    @patch('shutil.rmtree', wraps=shutil.rmtree)
    def test_clear_all(self, mock_rmtree):
        if False:
            for i in range(10):
                print('nop')
        'Tests that LocalDiskCacheStorageManager.clear_all() calls shutil.rmtree\n        to remove the cache folder\n        '
        manager = LocalDiskCacheStorageManager()
        manager.clear_all()
        mock_rmtree.assert_called_once()

class LocalDiskPersistCacheStorageTest(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.context = CacheStorageContext(function_key='func-key', function_display_name='func-display-name', persist='disk')
        self.storage = LocalDiskCacheStorage(self.context)
        self.tempdir = TempDirectory(create=True)
        self.patch_get_cache_folder_path = patch('streamlit.runtime.caching.storage.local_disk_cache_storage.get_cache_folder_path', return_value=self.tempdir.path)
        self.patch_get_cache_folder_path.start()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        super().tearDown()
        self.storage.clear()
        self.patch_get_cache_folder_path.stop()
        self.tempdir.cleanup()

    def test_storage_get_not_found(self):
        if False:
            i = 10
            return i + 15
        'Test that storage.get() returns the correct value.'
        with self.assertRaises(CacheStorageKeyNotFoundError):
            self.storage.get('some-key')

    def test_storage_get_found(self):
        if False:
            return 10
        'Test that storage.get() returns the correct value.'
        self.storage.set('some-key', b'some-value')
        self.assertEqual(self.storage.get('some-key'), b'some-value')

    def test_storage_set(self):
        if False:
            return 10
        'Test that storage.set() writes the correct value to disk.'
        self.storage.set('new-key', b'new-value')
        self.assertTrue(os.path.exists(self.tempdir.path + '/func-key-new-key.memo'))
        with open(self.tempdir.path + '/func-key-new-key.memo', 'rb') as f:
            self.assertEqual(f.read(), b'new-value')

    @patch('streamlit.runtime.caching.storage.local_disk_cache_storage.streamlit_write', MagicMock(side_effect=util.Error('mock exception')))
    def test_storage_set_error(self):
        if False:
            print('Hello World!')
        'Test that storage.set() raises an exception when it fails to write to disk.'
        with self.assertRaises(CacheStorageError) as e:
            self.storage.set('uniqueKey', b'new-value')
        self.assertEqual(str(e.exception), 'Unable to write to cache')

    def test_storage_set_override(self):
        if False:
            i = 10
            return i + 15
        'Test that storage.set() overrides the value of an existing key.'
        self.storage.set('another_key', b'another_value')
        self.storage.set('another_key', b'new_value')
        self.assertEqual(self.storage.get('another_key'), b'new_value')

    def test_storage_delete(self):
        if False:
            return 10
        'Test that storage.delete() removes the correct file from disk.'
        self.storage.set('new-key', b'new-value')
        self.assertTrue(os.path.exists(self.tempdir.path + '/func-key-new-key.memo'))
        self.storage.delete('new-key')
        self.assertFalse(os.path.exists(self.tempdir.path + '/func-key-new-key.memo'))
        with self.assertRaises(CacheStorageKeyNotFoundError):
            self.storage.get('new-key')

    def test_storage_clear(self):
        if False:
            while True:
                i = 10
        'Test that storage.clear() removes all storage files from disk.'
        self.storage.set('some-key', b'some-value')
        self.storage.set('another-key', b'another-value')
        self.assertTrue(os.path.exists(self.tempdir.path + '/func-key-some-key.memo'))
        self.assertTrue(os.path.exists(self.tempdir.path + '/func-key-another-key.memo'))
        self.storage.clear()
        self.assertFalse(os.path.exists(self.tempdir.path + '/func-key-some-key.memo'))
        self.assertFalse(os.path.exists(self.tempdir.path + '/func-key-another-key.memo'))
        with self.assertRaises(CacheStorageKeyNotFoundError):
            self.storage.get('some-key')
        with self.assertRaises(CacheStorageKeyNotFoundError):
            self.storage.get('another-key')
        self.assertEqual(os.listdir(self.tempdir.path), [])

    def test_storage_clear_not_existing_cache_directory(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that clear() is not crashing if the cache directory does not exist.'
        self.tempdir.cleanup()
        self.storage.clear()

    def test_storage_clear_call_listdir_existing_cache_directory(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that clear() call os.listdir if cache folder does not exist.'
        with patch('os.listdir') as mock_listdir:
            self.storage.clear()
        mock_listdir.assert_called_once()

    def test_storage_clear_not_call_listdir_not_existing_cache_directory(self):
        if False:
            while True:
                i = 10
        "Test that clear() doesn't call os.listdir if cache folder does not exist."
        self.tempdir.cleanup()
        with patch('os.listdir') as mock_listdir:
            self.storage.clear()
        mock_listdir.assert_not_called()

    def test_storage_close(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that storage.close() does not raise any exception.'
        self.storage.close()