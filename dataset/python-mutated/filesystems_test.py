"""Unit tests for LocalFileSystem."""
import filecmp
import logging
import os
import shutil
import tempfile
import unittest
import mock
from apache_beam.io import localfilesystem
from apache_beam.io.filesystem import BeamIOError
from apache_beam.io.filesystems import FileSystems

def _gen_fake_join(separator):
    if False:
        print('Hello World!')
    'Returns a callable that joins paths with the given separator.'

    def _join(first_path, *paths):
        if False:
            print('Hello World!')
        return separator.join((first_path.rstrip(separator),) + paths)
    return _join

class FileSystemsTest(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        if False:
            print('Hello World!')
        shutil.rmtree(self.tmpdir)

    def test_get_scheme(self):
        if False:
            i = 10
            return i + 15
        self.assertIsNone(FileSystems.get_scheme('/abc/cdf'))
        self.assertIsNone(FileSystems.get_scheme('c:\\abc\\cdf'))
        self.assertEqual(FileSystems.get_scheme('gs://abc/cdf'), 'gs')

    def test_get_filesystem(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(isinstance(FileSystems.get_filesystem('/tmp'), localfilesystem.LocalFileSystem))
        self.assertTrue(isinstance(FileSystems.get_filesystem('c:\\abc\\def'), localfilesystem.LocalFileSystem))
        with self.assertRaises(ValueError):
            FileSystems.get_filesystem('error://abc/def')

    @mock.patch('apache_beam.io.localfilesystem.os')
    def test_unix_path_join(self, *unused_mocks):
        if False:
            for i in range(10):
                print('nop')
        localfilesystem.os.path.join.side_effect = _gen_fake_join('/')
        self.assertEqual('/tmp/path/to/file', FileSystems.join('/tmp/path', 'to', 'file'))
        self.assertEqual('/tmp/path/to/file', FileSystems.join('/tmp/path', 'to/file'))
        self.assertEqual('/tmp/path/to/file', FileSystems.join('/', 'tmp/path', 'to/file'))
        self.assertEqual('/tmp/path/to/file', FileSystems.join('/tmp/', 'path', 'to/file'))

    @mock.patch('apache_beam.io.localfilesystem.os')
    def test_windows_path_join(self, *unused_mocks):
        if False:
            print('Hello World!')
        localfilesystem.os.path.join.side_effect = _gen_fake_join('\\')
        self.assertEqual('C:\\tmp\\path\\to\\file', FileSystems.join('C:\\tmp\\path', 'to', 'file'))
        self.assertEqual('C:\\tmp\\path\\to\\file', FileSystems.join('C:\\tmp\\path', 'to\\file'))
        self.assertEqual('C:\\tmp\\path\\to\\file', FileSystems.join('C:\\tmp\\path\\\\', 'to', 'file'))

    def test_mkdirs(self):
        if False:
            while True:
                i = 10
        path = os.path.join(self.tmpdir, 't1/t2')
        FileSystems.mkdirs(path)
        self.assertTrue(os.path.isdir(path))

    def test_mkdirs_failed(self):
        if False:
            i = 10
            return i + 15
        path = os.path.join(self.tmpdir, 't1/t2')
        FileSystems.mkdirs(path)
        with self.assertRaises(IOError):
            FileSystems.mkdirs(path)
        with self.assertRaises(IOError):
            FileSystems.mkdirs(os.path.join(self.tmpdir, 't1'))

    def test_match_file(self):
        if False:
            while True:
                i = 10
        path = os.path.join(self.tmpdir, 'f1')
        open(path, 'a').close()
        result = FileSystems.match([path])[0]
        files = [f.path for f in result.metadata_list]
        self.assertEqual(files, [path])

    def test_match_file_empty(self):
        if False:
            return 10
        path = os.path.join(self.tmpdir, 'f2')
        result = FileSystems.match([path])[0]
        files = [f.path for f in result.metadata_list]
        self.assertEqual(files, [])

    def test_match_file_exception(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(BeamIOError, '^Unable to get the Filesystem') as error:
            FileSystems.match([None])
        self.assertEqual(list(error.exception.exception_details), [None])

    def test_match_directory_with_files(self):
        if False:
            for i in range(10):
                print('nop')
        path1 = os.path.join(self.tmpdir, 'f1')
        path2 = os.path.join(self.tmpdir, 'f2')
        open(path1, 'a').close()
        open(path2, 'a').close()
        path = os.path.join(self.tmpdir, '*')
        result = FileSystems.match([path])[0]
        files = [f.path for f in result.metadata_list]
        self.assertCountEqual(files, [path1, path2])

    def test_match_directory(self):
        if False:
            return 10
        result = FileSystems.match([self.tmpdir])[0]
        files = [f.path for f in result.metadata_list]
        self.assertEqual(files, [self.tmpdir])

    def test_copy(self):
        if False:
            i = 10
            return i + 15
        path1 = os.path.join(self.tmpdir, 'f1')
        path2 = os.path.join(self.tmpdir, 'f2')
        with open(path1, 'a') as f:
            f.write('Hello')
        FileSystems.copy([path1], [path2])
        self.assertTrue(filecmp.cmp(path1, path2))

    def test_copy_error(self):
        if False:
            print('Hello World!')
        path1 = os.path.join(self.tmpdir, 'f1')
        path2 = os.path.join(self.tmpdir, 'f2')
        with self.assertRaisesRegex(BeamIOError, '^Copy operation failed') as error:
            FileSystems.copy([path1], [path2])
        self.assertEqual(list(error.exception.exception_details.keys()), [(path1, path2)])

    def test_copy_directory(self):
        if False:
            print('Hello World!')
        path_t1 = os.path.join(self.tmpdir, 't1')
        path_t2 = os.path.join(self.tmpdir, 't2')
        FileSystems.mkdirs(path_t1)
        FileSystems.mkdirs(path_t2)
        path1 = os.path.join(path_t1, 'f1')
        path2 = os.path.join(path_t2, 'f1')
        with open(path1, 'a') as f:
            f.write('Hello')
        FileSystems.copy([path_t1], [path_t2])
        self.assertTrue(filecmp.cmp(path1, path2))

    def test_rename(self):
        if False:
            while True:
                i = 10
        path1 = os.path.join(self.tmpdir, 'f1')
        path2 = os.path.join(self.tmpdir, 'f2')
        with open(path1, 'a') as f:
            f.write('Hello')
        FileSystems.rename([path1], [path2])
        self.assertTrue(FileSystems.exists(path2))
        self.assertFalse(FileSystems.exists(path1))

    def test_rename_error(self):
        if False:
            i = 10
            return i + 15
        path1 = os.path.join(self.tmpdir, 'f1')
        path2 = os.path.join(self.tmpdir, 'f2')
        with self.assertRaisesRegex(BeamIOError, '^Rename operation failed') as error:
            FileSystems.rename([path1], [path2])
        self.assertEqual(list(error.exception.exception_details.keys()), [(path1, path2)])

    def test_rename_directory(self):
        if False:
            print('Hello World!')
        path_t1 = os.path.join(self.tmpdir, 't1')
        path_t2 = os.path.join(self.tmpdir, 't2')
        FileSystems.mkdirs(path_t1)
        path1 = os.path.join(path_t1, 'f1')
        path2 = os.path.join(path_t2, 'f1')
        with open(path1, 'a') as f:
            f.write('Hello')
        FileSystems.rename([path_t1], [path_t2])
        self.assertTrue(FileSystems.exists(path_t2))
        self.assertFalse(FileSystems.exists(path_t1))
        self.assertTrue(FileSystems.exists(path2))
        self.assertFalse(FileSystems.exists(path1))

    def test_exists(self):
        if False:
            for i in range(10):
                print('nop')
        path1 = os.path.join(self.tmpdir, 'f1')
        path2 = os.path.join(self.tmpdir, 'f2')
        with open(path1, 'a') as f:
            f.write('Hello')
        self.assertTrue(FileSystems.exists(path1))
        self.assertFalse(FileSystems.exists(path2))

    def test_delete(self):
        if False:
            for i in range(10):
                print('nop')
        path1 = os.path.join(self.tmpdir, 'f1')
        with open(path1, 'a') as f:
            f.write('Hello')
        self.assertTrue(FileSystems.exists(path1))
        FileSystems.delete([path1])
        self.assertFalse(FileSystems.exists(path1))

    def test_delete_error(self):
        if False:
            i = 10
            return i + 15
        path1 = os.path.join(self.tmpdir, 'f1')
        with self.assertRaisesRegex(BeamIOError, '^Delete operation failed') as error:
            FileSystems.delete([path1])
        self.assertEqual(list(error.exception.exception_details.keys()), [path1])
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()