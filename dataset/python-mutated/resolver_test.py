"""Tests for tensorflow_hub.resolver."""
import os
import re
import socket
import tempfile
import threading
import time
import unittest
from unittest import mock
import uuid
from absl import flags
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_hub import compressed_module_resolver
from tensorflow_hub import config
from tensorflow_hub import registry
from tensorflow_hub import resolver
from tensorflow_hub import test_utils
from tensorflow_hub import tf_utils
from tensorflow_hub import uncompressed_module_resolver
FLAGS = flags.FLAGS

class PathResolverTest(tf.test.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.resolver = resolver.PathResolver()

    def testAlwaysSupported(self):
        if False:
            for i in range(10):
                print('nop')
        os.chdir(os.path.join(self.get_temp_dir()))
        self.assertTrue(self.resolver.is_supported('/tmp'))
        tf.compat.v1.gfile.MkDir('foo/')
        self.assertTrue(self.resolver.is_supported('./foo/'))
        self.assertTrue(self.resolver.is_supported('foo/'))
        self.assertTrue(self.resolver.is_supported('bar/'))
        self.assertTrue(self.resolver.is_supported('foo/bar'))
        self.assertTrue(self.resolver.is_supported('nope://throw-OpError'))

    def testCallWithValidHandle(self):
        if False:
            while True:
                i = 10
        tmp_path = os.path.join(self.get_temp_dir(), '1234')
        tf.compat.v1.gfile.MkDir(tmp_path)
        path = self.resolver(tmp_path)
        self.assertEqual(path, tmp_path)

    def testCallWhenHandleDirectoryDoesNotExist(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(IOError, '/foo/ does not exist.'):
            self.resolver('/foo/')

class FakeResolver(resolver.Resolver):
    """Fake Resolver used to test composite Resolvers."""

    def __init__(self, prefix):
        if False:
            for i in range(10):
                print('nop')
        self.prefix = prefix

    def is_supported(self, handle):
        if False:
            print('Hello World!')
        return handle.startswith(self.prefix)

    def __call__(self, handle):
        if False:
            i = 10
            return i + 15
        if handle.endswith('error'):
            raise ValueError('error for: ' + handle)
        return handle + '-resolved_by_' + self.prefix

class CompressedResolverTest(tf.test.TestCase):

    def testCacheDir(self):
        if False:
            for i in range(10):
                print('nop')
        cache_dir = resolver.tfhub_cache_dir()
        self.assertEqual(cache_dir, None)
        cache_dir = resolver.tfhub_cache_dir(use_temp=True)
        self.assertEquals(cache_dir, os.path.join(tempfile.gettempdir(), 'tfhub_modules'))
        cache_dir = resolver.tfhub_cache_dir(default_cache_dir='/d', use_temp=True)
        self.assertEqual('/d', cache_dir)
        FLAGS.tfhub_cache_dir = '/e'
        cache_dir = resolver.tfhub_cache_dir(default_cache_dir='/d', use_temp=True)
        self.assertEqual('/e', cache_dir)
        FLAGS.tfhub_cache_dir = ''
        os.environ[resolver._TFHUB_CACHE_DIR] = '/f'
        cache_dir = resolver.tfhub_cache_dir(default_cache_dir='/d', use_temp=True)
        self.assertEqual('/f', cache_dir)
        FLAGS.tfhub_cache_dir = '/e'
        cache_dir = resolver.tfhub_cache_dir(default_cache_dir='/d', use_temp=True)
        self.assertEqual('/f', cache_dir)
        FLAGS.tfhub_cache_dir = ''
        os.unsetenv(resolver._TFHUB_CACHE_DIR)

    def testDirSize(self):
        if False:
            i = 10
            return i + 15
        fake_task_uid = 1234
        test_dir = resolver._temp_download_dir(self.get_temp_dir(), fake_task_uid)
        tf.compat.v1.gfile.MakeDirs(test_dir)
        tf_utils.atomic_write_string_to_file(os.path.join(test_dir, 'file1'), 'content1', False)
        tf_utils.atomic_write_string_to_file(os.path.join(test_dir, 'file2'), 'content2', False)
        test_sub_dir = os.path.join(test_dir, 'sub_dir')
        tf.compat.v1.gfile.MakeDirs(test_sub_dir)
        tf_utils.atomic_write_string_to_file(os.path.join(test_sub_dir, 'file3'), 'content3', False)
        self.assertEqual(3 * 8, resolver._dir_size(test_dir))
        self.assertEqual(8, resolver._dir_size(test_sub_dir))
        fake_lock_filename = resolver._lock_filename(self.get_temp_dir())
        tf_utils.atomic_write_string_to_file(fake_lock_filename, resolver._lock_file_contents(fake_task_uid), False)
        self.assertEqual(3 * 8, resolver._locked_tmp_dir_size(fake_lock_filename))
        tf.compat.v1.gfile.DeleteRecursively(test_dir)
        self.assertEqual(0, resolver._locked_tmp_dir_size(fake_lock_filename))

    def testLockFileName(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEquals('/a/b/c.lock', resolver._lock_filename('/a/b/c/'))

    def testTempDownloadDir(self):
        if False:
            i = 10
            return i + 15
        self.assertEquals('/a/b.t.tmp', resolver._temp_download_dir('/a/b/', 't'))

    def testReadTaskUidFromLockFile(self):
        if False:
            i = 10
            return i + 15
        module_dir = os.path.join(self.get_temp_dir(), 'module')
        task_uid = uuid.uuid4().hex
        lock_filename = resolver._lock_filename(module_dir)
        tf_utils.atomic_write_string_to_file(lock_filename, resolver._lock_file_contents(task_uid), overwrite=False)
        self.assertEqual(task_uid, resolver._task_uid_from_lock_file(lock_filename))

    def testWaitForLockToDisappear_DownloadCompletes(self):
        if False:
            while True:
                i = 10
        module_dir = os.path.join(self.get_temp_dir(), 'module')
        task_uid = uuid.uuid4().hex
        lock_filename = resolver._lock_filename(module_dir)
        tf_utils.atomic_write_string_to_file(lock_filename, resolver._lock_file_contents(task_uid), overwrite=False)
        thread = threading.Thread(target=resolver._wait_for_lock_to_disappear, args=('module', lock_filename, 600))
        thread.start()
        tf.compat.v1.gfile.Remove(lock_filename)
        thread.join(10)

    def testWaitForLockToDisappear_DownloadOngoing(self):
        if False:
            print('Hello World!')
        module_dir = os.path.join(self.get_temp_dir(), 'module')
        task_uid = uuid.uuid4().hex
        lock_filename = resolver._lock_filename(module_dir)
        lock_file_content = resolver._lock_file_contents(task_uid)
        tf_utils.atomic_write_string_to_file(lock_filename, lock_file_content, overwrite=False)
        lock_expiration_wait_time_secs = 10
        thread = threading.Thread(target=resolver._wait_for_lock_to_disappear, args=('module', lock_filename, lock_expiration_wait_time_secs))
        thread.start()
        tmp_dir = resolver._temp_download_dir(self.get_temp_dir(), task_uid)
        tf.compat.v1.gfile.MakeDirs(tmp_dir)
        for x in range(2 * lock_expiration_wait_time_secs):
            tf_utils.atomic_write_string_to_file(os.path.join(tmp_dir, 'file_%d' % x), 'test', overwrite=False)
            self.assertEqual(lock_file_content, tf_utils.read_file_to_string(lock_filename))
            time.sleep(1)
        thread.join(lock_expiration_wait_time_secs)

    def testWaitForLockToDisappear_DownloadAborted(self):
        if False:
            while True:
                i = 10
        module_dir = os.path.join(self.get_temp_dir(), 'module')
        task_uid = uuid.uuid4().hex
        lock_filename = resolver._lock_filename(module_dir)
        lock_file_content = resolver._lock_file_contents(task_uid)
        tf_utils.atomic_write_string_to_file(lock_filename, lock_file_content, overwrite=False)
        tmp_dir = resolver._temp_download_dir(self.get_temp_dir(), task_uid)
        tf.compat.v1.gfile.MakeDirs(tmp_dir)
        thread = threading.Thread(target=resolver._wait_for_lock_to_disappear, args=('module', lock_filename, 10))
        thread.start()
        thread.join(30)
        self.assertFalse(tf.compat.v1.gfile.Exists(lock_filename))

    def testModuleAlreadyDownloaded(self):
        if False:
            return 10
        module_dir = os.path.join(self.get_temp_dir(), 'module')

        def fake_download_fn_with_rogue_behavior(handle, tmp_dir):
            if False:
                return 10
            del handle, tmp_dir
            tf.compat.v1.gfile.MakeDirs(module_dir)
            tf_utils.atomic_write_string_to_file(os.path.join(module_dir, 'file'), 'content', False)
        self.assertEqual(module_dir, resolver.atomic_download('module', fake_download_fn_with_rogue_behavior, module_dir))
        self.assertEqual(tf.compat.v1.gfile.ListDirectory(module_dir), ['file'])
        self.assertFalse(tf.compat.v1.gfile.Exists(resolver._lock_filename(module_dir)))
        parent_dir = os.path.abspath(os.path.join(module_dir, '..'))
        self.assertEqual(sorted(tf.compat.v1.gfile.ListDirectory(parent_dir)), ['module', 'module.descriptor.txt'])
        self.assertRegexpMatches(tf_utils.read_file_to_string(resolver._module_descriptor_file(module_dir)), 'Module: module\nDownload Time: .*\nDownloader Hostname: %s .PID:%d.' % (re.escape(socket.gethostname()), os.getpid()))
        with mock.patch.object(tf_utils, 'atomic_write_string_to_file', side_effect=ValueError('This error should never be raised!')):
            self.assertEqual(module_dir, resolver.atomic_download('module', fake_download_fn_with_rogue_behavior, module_dir))
            self.assertEqual(tf.compat.v1.gfile.ListDirectory(module_dir), ['file'])
            self.assertFalse(tf.compat.v1.gfile.Exists(resolver._lock_filename(module_dir)))

    def testModuleDownloadedWhenEmptyFolderExists(self):
        if False:
            while True:
                i = 10
        module_dir = os.path.join(self.get_temp_dir(), 'module')

        def fake_download_fn(handle, tmp_dir):
            if False:
                return 10
            del handle, tmp_dir
            tf.compat.v1.gfile.MakeDirs(module_dir)
            tf_utils.atomic_write_string_to_file(os.path.join(module_dir, 'file'), 'content', False)
        self.assertFalse(tf.compat.v1.gfile.Exists(module_dir))
        tf.compat.v1.gfile.MakeDirs(module_dir)
        self.assertEqual(module_dir, resolver.atomic_download('module', fake_download_fn, module_dir))
        self.assertEqual(tf.compat.v1.gfile.ListDirectory(module_dir), ['file'])
        self.assertFalse(tf.compat.v1.gfile.Exists(resolver._lock_filename(module_dir)))
        parent_dir = os.path.abspath(os.path.join(module_dir, '..'))
        self.assertEqual(sorted(tf.compat.v1.gfile.ListDirectory(parent_dir)), ['module', 'module.descriptor.txt'])
        self.assertRegexpMatches(tf_utils.read_file_to_string(resolver._module_descriptor_file(module_dir)), 'Module: module\nDownload Time: .*\nDownloader Hostname: %s .PID:%d.' % (re.escape(socket.gethostname()), os.getpid()))

    def testModuleConcurrentDownload(self):
        if False:
            i = 10
            return i + 15
        module_dir = os.path.join(self.get_temp_dir(), 'module')

        def second_download_fn(handle, tmp_dir):
            if False:
                while True:
                    i = 10
            del handle, tmp_dir
            self.fail('This should not be called. The module should have been downloaded already.')
        second_download_thread = threading.Thread(target=resolver.atomic_download, args=('module', second_download_fn, module_dir))

        def first_download_fn(handle, tmp_dir):
            if False:
                while True:
                    i = 10
            del handle, tmp_dir
            tf.compat.v1.gfile.MakeDirs(module_dir)
            tf_utils.atomic_write_string_to_file(os.path.join(module_dir, 'file'), 'content', False)
            second_download_thread.start()
        self.assertEqual(module_dir, resolver.atomic_download('module', first_download_fn, module_dir))
        second_download_thread.join(30)

    def testModuleLockLostDownloadKilled(self):
        if False:
            return 10
        module_dir = os.path.join(self.get_temp_dir(), 'module')
        download_aborted_msg = 'Download aborted.'

        def kill_download(handle, tmp_dir):
            if False:
                while True:
                    i = 10
            del handle, tmp_dir
            tf.compat.v1.gfile.Remove(resolver._lock_filename(module_dir))
            raise OSError(download_aborted_msg)
        try:
            resolver.atomic_download('module', kill_download, module_dir)
            self.fail('atomic_download() should have thrown an exception.')
        except OSError as _:
            pass
        parent_dir = os.path.abspath(os.path.join(module_dir, '..'))
        self.assertEqual(tf.compat.v1.gfile.ListDirectory(parent_dir), [])

    def testNotFoundGCSBucket(self):
        if False:
            while True:
                i = 10
        module_dir = ''

        def dummy_download_fn(handle, tmp_dir):
            if False:
                while True:
                    i = 10
            del handle, tmp_dir
            return
        with unittest.mock.patch('tensorflow_hub.tf_utils.atomic_write_string_to_file') as mock_:
            mock_.side_effect = tf.errors.NotFoundError(None, None, 'Test')
            try:
                resolver.atomic_download('module', dummy_download_fn, module_dir)
                assert False
            except tf.errors.NotFoundError as e:
                self.assertEqual('Test', e.message)

class UncompressedResolverTest(tf.test.TestCase):

    def testModuleRunningWithUncompressedContext(self):
        if False:
            for i in range(10):
                print('nop')
        module_export_path = os.path.join(self.get_temp_dir(), 'module')
        with tf.Graph().as_default():
            test_utils.export_module(module_export_path)
            with mock.patch.object(uncompressed_module_resolver.HttpUncompressedFileResolver, '_request_gcs_location', return_value=module_export_path) as mocked_urlopen:
                with test_utils.UncompressedLoadFormatContext():
                    m = hub.Module('https://tfhub.dev/google/model/1')
                mocked_urlopen.assert_called_once_with('https://tfhub.dev/google/model/1?tf-hub-format=uncompressed')
            out = m(11)
            with tf.compat.v1.Session() as sess:
                self.assertAllClose(sess.run(out), 121)

class LoadFormatResolverBehaviorTest(tf.test.TestCase):
    """Test that the right resolvers are called depending on the load format."""

    def _assert_resolver_is_called(self, http_resolver):
        if False:
            return 10
        module_url = 'https://tfhub.dev/google/model/1'
        with mock.patch.object(http_resolver, '__call__', side_effect=ValueError) as mocked_call:
            try:
                hub.Module(module_url)
                self.fail('Failure expected since mock raises it as side effect.')
            except ValueError:
                pass
        mocked_call.assert_called_once_with(module_url)

    def _assert_compressed_resolver_called(self):
        if False:
            return 10
        self._assert_resolver_is_called(compressed_module_resolver.HttpCompressedFileResolver)

    def _assert_uncompressed_resolver_called(self):
        if False:
            for i in range(10):
                print('nop')
        self._assert_resolver_is_called(uncompressed_module_resolver.HttpUncompressedFileResolver)

    def test_load_format_auto(self):
        if False:
            while True:
                i = 10
        self._assert_compressed_resolver_called()

    def test_load_format_compressed(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.CompressedLoadFormatContext():
            self._assert_compressed_resolver_called()

    def test_load_format_uncompressed(self):
        if False:
            i = 10
            return i + 15
        with test_utils.UncompressedLoadFormatContext():
            self._assert_uncompressed_resolver_called()
if __name__ == '__main__':
    registry._clear()
    config._run()
    tf.test.main()