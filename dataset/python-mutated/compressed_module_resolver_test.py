"""Tests for tensorflow_hub.compressed_module_resolver."""
import os
import re
import socket
import ssl
import tarfile
import tempfile
import unittest
from unittest import mock
import urllib.request
import uuid
from absl import flags
from absl.testing import parameterized
import tensorflow as tf
from tensorflow_hub import compressed_module_resolver
from tensorflow_hub import resolver
from tensorflow_hub import test_utils
from tensorflow_hub import tf_utils
FLAGS = flags.FLAGS

class HttpCompressedFileResolverTest(tf.test.TestCase, parameterized.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        os.chdir(self.get_temp_dir())
        self.files = ['file1', 'file2', 'file3']
        for cur_file in self.files:
            with tf.compat.v1.gfile.GFile(cur_file, mode='w') as f:
                f.write(cur_file)
        with tf.compat.v1.gfile.GFile('mock_module', mode='w') as f:
            f.write('module')
        tar = tarfile.open('mock_module.tar', 'w')
        for name in self.files:
            tar.add(name)
        tar.close()
        tar = tarfile.open('mock_module.tar.gz', 'w:gz')
        for name in self.files:
            tar.add(name)
        tar.close()
        self.server_port = test_utils.start_http_server()
        self.module_handle = 'http://localhost:%d/mock_module.tar.gz' % self.server_port
        self.redirect_server_port = test_utils.start_http_server(redirect='http://localhost:%d' % self.server_port)
        self.smart_server_port = test_utils.start_smart_module_server(self.module_handle)
        self.smart_handle = 'http://localhost:%d/mock_module' % self.smart_server_port

    def testGetModulePathTar(self):
        if False:
            for i in range(10):
                print('nop')
        FLAGS.tfhub_cache_dir = os.path.join(self.get_temp_dir(), 'cache_dir')
        http_resolver = compressed_module_resolver.HttpCompressedFileResolver()
        path = http_resolver('http://localhost:%d/mock_module.tar' % self.server_port)
        files = os.listdir(path)
        self.assertListEqual(sorted(files), ['file1', 'file2', 'file3'])

    def testGetModulePathTarGz(self):
        if False:
            print('Hello World!')
        FLAGS.tfhub_cache_dir = os.path.join(self.get_temp_dir(), 'cache_dir')
        http_resolver = compressed_module_resolver.HttpCompressedFileResolver()
        path = http_resolver(self.module_handle)
        files = os.listdir(path)
        self.assertListEqual(sorted(files), ['file1', 'file2', 'file3'])

    def testGetModuleFromSmartLocation(self):
        if False:
            while True:
                i = 10
        FLAGS.tfhub_cache_dir = os.path.join(self.get_temp_dir(), 'cache_dir')
        http_resolver = compressed_module_resolver.HttpCompressedFileResolver()
        path = http_resolver(self.smart_handle)
        files = os.listdir(path)
        self.assertListEqual(sorted(files), ['file1', 'file2', 'file3'])

    def testModuleDescriptor(self):
        if False:
            return 10
        FLAGS.tfhub_cache_dir = os.path.join(self.get_temp_dir(), 'cache_dir')
        http_resolver = compressed_module_resolver.HttpCompressedFileResolver()
        path = http_resolver(self.module_handle)
        desc = tf_utils.read_file_to_string(resolver._module_descriptor_file(path))
        self.assertRegexpMatches(desc, 'Module: %s\nDownload Time: .*\nDownloader Hostname: %s .PID:%d.' % (re.escape(self.module_handle), re.escape(socket.gethostname()), os.getpid()))

    def testNoCacheDirSet(self):
        if False:
            print('Hello World!')
        FLAGS.tfhub_cache_dir = ''
        http_resolver = compressed_module_resolver.HttpCompressedFileResolver()
        handle = 'http://localhost:%d/mock_module.tar.gz' % self.server_port
        path = http_resolver(handle)
        files = os.listdir(path)
        self.assertListEqual(sorted(files), ['file1', 'file2', 'file3'])
        self.assertStartsWith(path, tempfile.gettempdir())

    def testIsTarFile(self):
        if False:
            while True:
                i = 10
        self.assertTrue(compressed_module_resolver._is_tarfile('foo.tar'))
        self.assertTrue(compressed_module_resolver._is_tarfile('foo.tar.gz'))
        self.assertTrue(compressed_module_resolver._is_tarfile('foo.tgz'))
        self.assertFalse(compressed_module_resolver._is_tarfile('foo'))
        self.assertFalse(compressed_module_resolver._is_tarfile('footar'))

    def testAppendFormatQuery(self):
        if False:
            return 10
        tests = [('https://example.com/module.tar.gz', 'https://example.com/module.tar.gz?tf-hub-format=compressed'), ('https://example.com/module', 'https://example.com/module?tf-hub-format=compressed'), ('https://example.com/module?extra=abc', 'https://example.com/module?extra=abc&tf-hub-format=compressed'), ('https://example.com/module?extra=abc', 'https://example.com/module?extra=abc&tf-hub-format=compressed'), ('https://example.com/module?extra=abc&tf-hub-format=test', 'https://example.com/module?extra=abc&tf-hub-format=test&tf-hub-format=compressed')]
        http_resolver = compressed_module_resolver.HttpCompressedFileResolver()
        for (handle, expected) in tests:
            self.assertTrue(http_resolver._append_compressed_format_query(handle), expected)

    @parameterized.parameters(('', ssl.CERT_REQUIRED), ('TRUE', ssl.CERT_REQUIRED), ('true', ssl.CERT_NONE))
    def testGetModulePathTarGz_withEnvVariable(self, env_value, expected_mode):
        if False:
            while True:
                i = 10
        FLAGS.tfhub_cache_dir = os.path.join(self.get_temp_dir(), 'cache_dir')
        with unittest.mock.patch.dict(os.environ, {resolver._TFHUB_DISABLE_CERT_VALIDATION: env_value}):
            http_resolver = compressed_module_resolver.HttpCompressedFileResolver()
            path = http_resolver(self.module_handle)
        self.assertEqual(http_resolver._context.verify_mode, expected_mode)
        self.assertCountEqual(os.listdir(path), ['file1', 'file2', 'file3'])

    def testAbandondedLockFile(self):
        if False:
            i = 10
            return i + 15
        FLAGS.tfhub_cache_dir = os.path.join(self.get_temp_dir(), 'cache_dir')
        module_dir = compressed_module_resolver._module_dir(self.module_handle)
        task_uid = uuid.uuid4().hex
        lock_filename = resolver._lock_filename(module_dir)
        tf_utils.atomic_write_string_to_file(lock_filename, resolver._lock_file_contents(task_uid), overwrite=False)
        with unittest.mock.patch.object(compressed_module_resolver.HttpCompressedFileResolver, '_lock_file_timeout_sec', return_value=10):
            http_resolver = compressed_module_resolver.HttpCompressedFileResolver()
            handle = 'http://localhost:%d/mock_module.tar.gz' % self.server_port
            path = http_resolver(handle)
        files = os.listdir(path)
        self.assertListEqual(sorted(files), ['file1', 'file2', 'file3'])
        self.assertFalse(tf.compat.v1.gfile.Exists(lock_filename))

    def testModuleAlreadyDownloaded(self):
        if False:
            while True:
                i = 10
        FLAGS.tfhub_cache_dir = os.path.join(self.get_temp_dir(), 'cache_dir')
        http_resolver = compressed_module_resolver.HttpCompressedFileResolver()
        path = http_resolver(self.module_handle)
        files = sorted(os.listdir(path))
        self.assertListEqual(files, ['file1', 'file2', 'file3'])
        creation_times = [tf.compat.v1.gfile.Stat(os.path.join(path, f)).mtime_nsec for f in files]
        path = http_resolver(self.module_handle)
        files = sorted(os.listdir(path))
        self.assertListEqual(files, ['file1', 'file2', 'file3'])
        self.assertListEqual(creation_times, [tf.compat.v1.gfile.Stat(os.path.join(path, f)).mtime_nsec for f in files])

    def testCorruptedArchive(self):
        if False:
            return 10
        with tf.compat.v1.gfile.GFile('bad_archive.tar.gz', mode='w') as f:
            f.write('bad_archive')
        http_resolver = compressed_module_resolver.HttpCompressedFileResolver()
        try:
            http_resolver('http://localhost:%d/bad_archive.tar.gz' % self.server_port)
            self.fail('Corrupted archive should have failed to resolve.')
        except IOError as e:
            self.assertEqual('http://localhost:%d/bad_archive.tar.gz does not appear to be a valid module.' % self.server_port, str(e))
        try:
            http_resolver('http://localhost:%d/bad_archive.tar.gz' % self.redirect_server_port)
            self.fail('Corrupted archive should have failed to resolve.')
        except IOError as e:
            self.assertEqual('http://localhost:%d/bad_archive.tar.gz does not appear to be a valid module.' % self.redirect_server_port, str(e))

    def testLoadFromCn(self):
        if False:
            for i in range(10):
                print('nop')
        http_resolver = compressed_module_resolver.HttpCompressedFileResolver()
        with mock.patch.object(urllib.request, 'urlopen', autospec=True, return_value=urllib.request.urlopen('http://localhost:%d/mock_module.tar.gz' % self.server_port)) as mock_urlopen:
            path = http_resolver('https://hub.tensorflow.google.cn/google/bit/s-r50x1/1')
        mock_urlopen.assert_called_once_with('https://gcs.tensorflow.google.cn/tfhub-modules/google/bit/s-r50x1/1.tar.gz', context=mock.ANY)
        self.assertCountEqual(os.listdir(path), ['file1', 'file2', 'file3'])
if __name__ == '__main__':
    tf.test.main()