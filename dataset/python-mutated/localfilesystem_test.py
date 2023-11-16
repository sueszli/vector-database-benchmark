"""Unit tests for LocalFileSystem."""
import filecmp
import logging
import os
import shutil
import tempfile
import unittest
import mock
from parameterized import param
from parameterized import parameterized
from apache_beam.io import localfilesystem
from apache_beam.io.filesystem import BeamIOError
from apache_beam.options.pipeline_options import PipelineOptions

def _gen_fake_join(separator):
    if False:
        while True:
            i = 10
    'Returns a callable that joins paths with the given separator.'

    def _join(first_path, *paths):
        if False:
            return 10
        return separator.join((first_path.rstrip(separator),) + paths)
    return _join

def _gen_fake_split(separator):
    if False:
        return 10
    'Returns a callable that splits a with the given separator.'

    def _split(path):
        if False:
            for i in range(10):
                print('nop')
        sep_index = path.rfind(separator)
        if sep_index >= 0:
            return (path[:sep_index], path[sep_index + 1:])
        else:
            return (path, '')
    return _split

class LocalFileSystemTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.tmpdir = tempfile.mkdtemp()
        pipeline_options = PipelineOptions()
        self.fs = localfilesystem.LocalFileSystem(pipeline_options)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        shutil.rmtree(self.tmpdir)

    def test_scheme(self):
        if False:
            i = 10
            return i + 15
        self.assertIsNone(self.fs.scheme())
        self.assertIsNone(localfilesystem.LocalFileSystem.scheme())

    @mock.patch('apache_beam.io.localfilesystem.os')
    def test_unix_path_join(self, *unused_mocks):
        if False:
            for i in range(10):
                print('nop')
        localfilesystem.os.path.join.side_effect = _gen_fake_join('/')
        self.assertEqual('/tmp/path/to/file', self.fs.join('/tmp/path', 'to', 'file'))
        self.assertEqual('/tmp/path/to/file', self.fs.join('/tmp/path', 'to/file'))

    @mock.patch('apache_beam.io.localfilesystem.os')
    def test_windows_path_join(self, *unused_mocks):
        if False:
            while True:
                i = 10
        localfilesystem.os.path.join.side_effect = _gen_fake_join('\\')
        self.assertEqual('C:\\tmp\\path\\to\\file', self.fs.join('C:\\tmp\\path', 'to', 'file'))
        self.assertEqual('C:\\tmp\\path\\to\\file', self.fs.join('C:\\tmp\\path', 'to\\file'))

    @mock.patch('apache_beam.io.localfilesystem.os')
    def test_unix_path_split(self, os_mock):
        if False:
            while True:
                i = 10
        os_mock.path.abspath.side_effect = lambda a: a
        os_mock.path.split.side_effect = _gen_fake_split('/')
        self.assertEqual(('/tmp/path/to', 'file'), self.fs.split('/tmp/path/to/file'))
        self.assertEqual(('', 'tmp'), self.fs.split('/tmp'))

    @mock.patch('apache_beam.io.localfilesystem.os')
    def test_windows_path_split(self, os_mock):
        if False:
            i = 10
            return i + 15
        os_mock.path.abspath = lambda a: a
        os_mock.path.split.side_effect = _gen_fake_split('\\')
        self.assertEqual(('C:\\tmp\\path\\to', 'file'), self.fs.split('C:\\tmp\\path\\to\\file'))
        self.assertEqual(('C:', 'tmp'), self.fs.split('C:\\tmp'))

    def test_mkdirs(self):
        if False:
            print('Hello World!')
        path = os.path.join(self.tmpdir, 't1/t2')
        self.fs.mkdirs(path)
        self.assertTrue(os.path.isdir(path))

    def test_mkdirs_failed(self):
        if False:
            return 10
        path = os.path.join(self.tmpdir, 't1/t2')
        self.fs.mkdirs(path)
        with self.assertRaises(IOError):
            self.fs.mkdirs(path)
        with self.assertRaises(IOError):
            self.fs.mkdirs(os.path.join(self.tmpdir, 't1'))

    def test_match_file(self):
        if False:
            for i in range(10):
                print('nop')
        path = os.path.join(self.tmpdir, 'f1')
        open(path, 'a').close()
        result = self.fs.match([path])[0]
        files = [f.path for f in result.metadata_list]
        self.assertEqual(files, [path])

    def test_match_file_empty(self):
        if False:
            for i in range(10):
                print('nop')
        path = os.path.join(self.tmpdir, 'f2')
        result = self.fs.match([path])[0]
        files = [f.path for f in result.metadata_list]
        self.assertEqual(files, [])

    def test_match_file_exception(self):
        if False:
            return 10
        with self.assertRaisesRegex(BeamIOError, '^Match operation failed') as error:
            self.fs.match([None])
        self.assertEqual(list(error.exception.exception_details.keys()), [None])

    @parameterized.expand([param('*', files=['a', 'b', os.path.join('c', 'x')], expected=['a', 'b']), param('**', files=['a', os.path.join('b', 'x'), os.path.join('c', 'x')], expected=['a', os.path.join('b', 'x'), os.path.join('c', 'x')]), param(os.path.join('*', '*'), files=['a', os.path.join('b', 'x'), os.path.join('c', 'x'), os.path.join('d', 'x', 'y')], expected=[os.path.join('b', 'x'), os.path.join('c', 'x')]), param(os.path.join('**', '*'), files=['a', os.path.join('b', 'x'), os.path.join('c', 'x'), os.path.join('d', 'x', 'y')], expected=[os.path.join('b', 'x'), os.path.join('c', 'x'), os.path.join('d', 'x', 'y')])])
    def test_match_glob(self, pattern, files, expected):
        if False:
            i = 10
            return i + 15
        for filename in files:
            full_path = os.path.join(self.tmpdir, filename)
            dirname = os.path.dirname(full_path)
            if not dirname == full_path:
                assert os.path.commonprefix([self.tmpdir, full_path]) == self.tmpdir
                try:
                    self.fs.mkdirs(dirname)
                except IOError:
                    pass
            open(full_path, 'a').close()
        full_pattern = os.path.join(self.tmpdir, pattern)
        result = self.fs.match([full_pattern])[0]
        files = [os.path.relpath(f.path, self.tmpdir) for f in result.metadata_list]
        self.assertCountEqual(files, expected)

    def test_match_directory(self):
        if False:
            for i in range(10):
                print('nop')
        result = self.fs.match([self.tmpdir])[0]
        files = [f.path for f in result.metadata_list]
        self.assertEqual(files, [self.tmpdir])

    def test_match_directory_contents(self):
        if False:
            return 10
        path1 = os.path.join(self.tmpdir, 'f1')
        path2 = os.path.join(self.tmpdir, 'f2')
        open(path1, 'a').close()
        open(path2, 'a').close()
        result = self.fs.match([os.path.join(self.tmpdir, '*')])[0]
        files = [f.path for f in result.metadata_list]
        self.assertCountEqual(files, [path1, path2])

    def test_copy(self):
        if False:
            while True:
                i = 10
        path1 = os.path.join(self.tmpdir, 'f1')
        path2 = os.path.join(self.tmpdir, 'f2')
        with open(path1, 'a') as f:
            f.write('Hello')
        self.fs.copy([path1], [path2])
        self.assertTrue(filecmp.cmp(path1, path2))

    def test_copy_error(self):
        if False:
            while True:
                i = 10
        path1 = os.path.join(self.tmpdir, 'f1')
        path2 = os.path.join(self.tmpdir, 'f2')
        with self.assertRaisesRegex(BeamIOError, '^Copy operation failed') as error:
            self.fs.copy([path1], [path2])
        self.assertEqual(list(error.exception.exception_details.keys()), [(path1, path2)])

    def test_copy_directory(self):
        if False:
            while True:
                i = 10
        path_t1 = os.path.join(self.tmpdir, 't1')
        path_t2 = os.path.join(self.tmpdir, 't2')
        self.fs.mkdirs(path_t1)
        self.fs.mkdirs(path_t2)
        path1 = os.path.join(path_t1, 'f1')
        path2 = os.path.join(path_t2, 'f1')
        with open(path1, 'a') as f:
            f.write('Hello')
        self.fs.copy([path_t1], [path_t2])
        self.assertTrue(filecmp.cmp(path1, path2))

    def test_rename(self):
        if False:
            while True:
                i = 10
        path1 = os.path.join(self.tmpdir, 'f1')
        path2 = os.path.join(self.tmpdir, 'f2')
        with open(path1, 'a') as f:
            f.write('Hello')
        self.fs.rename([path1], [path2])
        self.assertTrue(self.fs.exists(path2))
        self.assertFalse(self.fs.exists(path1))

    def test_rename_error(self):
        if False:
            i = 10
            return i + 15
        path1 = os.path.join(self.tmpdir, 'f1')
        path2 = os.path.join(self.tmpdir, 'f2')
        with self.assertRaisesRegex(BeamIOError, '^Rename operation failed') as error:
            self.fs.rename([path1], [path2])
        self.assertEqual(list(error.exception.exception_details.keys()), [(path1, path2)])

    def test_rename_directory(self):
        if False:
            i = 10
            return i + 15
        path_t1 = os.path.join(self.tmpdir, 't1')
        path_t2 = os.path.join(self.tmpdir, 't2')
        self.fs.mkdirs(path_t1)
        path1 = os.path.join(path_t1, 'f1')
        path2 = os.path.join(path_t2, 'f1')
        with open(path1, 'a') as f:
            f.write('Hello')
        self.fs.rename([path_t1], [path_t2])
        self.assertTrue(self.fs.exists(path_t2))
        self.assertFalse(self.fs.exists(path_t1))
        self.assertTrue(self.fs.exists(path2))
        self.assertFalse(self.fs.exists(path1))

    def test_exists(self):
        if False:
            print('Hello World!')
        path1 = os.path.join(self.tmpdir, 'f1')
        path2 = os.path.join(self.tmpdir, 'f2')
        with open(path1, 'a') as f:
            f.write('Hello')
        self.assertTrue(self.fs.exists(path1))
        self.assertFalse(self.fs.exists(path2))

    def test_checksum(self):
        if False:
            return 10
        path1 = os.path.join(self.tmpdir, 'f1')
        path2 = os.path.join(self.tmpdir, 'f2')
        with open(path1, 'a') as f:
            f.write('Hello')
        with open(path2, 'a') as f:
            f.write('foo')
        checksum1 = self.fs.checksum(path1)
        checksum2 = self.fs.checksum(path2)
        self.assertEqual(checksum1, str(5))
        self.assertEqual(checksum2, str(3))
        self.assertEqual(checksum1, str(self.fs.size(path1)))
        self.assertEqual(checksum2, str(self.fs.size(path2)))

    def make_tree(self, path, value, expected_leaf_count=None):
        if False:
            i = 10
            return i + 15
        'Create a file+directory structure from a simple dict-based DSL\n\n    :param path: root path to create directories+files under\n    :param value: a specification of what ``path`` should contain: ``None`` to\n     make it an empty directory, a string literal to make it a file with those\n      contents, and a ``dict`` to make it a non-empty directory and recurse\n    :param expected_leaf_count: only be set at the top of a recursive call\n     stack; after the whole tree has been created, verify the presence and\n     number of all files+directories, as a sanity check\n    '
        if value is None:
            os.makedirs(path)
        elif isinstance(value, str):
            dir = os.path.dirname(path)
            if not os.path.exists(dir):
                os.makedirs(dir)
            with open(path, 'a') as f:
                f.write(value)
        elif isinstance(value, dict):
            for (basename, v) in value.items():
                self.make_tree(os.path.join(path, basename), v)
        else:
            raise Exception('Unexpected value in tempdir tree: %s' % value)
        if expected_leaf_count is not None:
            self.assertEqual(self.check_tree(path, value), expected_leaf_count)

    def check_tree(self, path, value, expected_leaf_count=None):
        if False:
            while True:
                i = 10
        'Verify a directory+file structure according to the rules described in\n    ``make_tree``\n\n    :param path: path to check under\n    :param value: DSL-representation of expected files+directories under\n    ``path``\n    :return: number of leaf files/directories that were verified\n    '
        actual_leaf_count = None
        if value is None:
            self.assertTrue(os.path.exists(path), msg=path)
            self.assertEqual(os.listdir(path), [])
            actual_leaf_count = 1
        elif isinstance(value, str):
            with open(path, 'r') as f:
                self.assertEqual(f.read(), value, msg=path)
            actual_leaf_count = 1
        elif isinstance(value, dict):
            actual_leaf_count = sum([self.check_tree(os.path.join(path, basename), v) for (basename, v) in value.items()])
        else:
            raise Exception('Unexpected value in tempdir tree: %s' % value)
        if expected_leaf_count is not None:
            self.assertEqual(actual_leaf_count, expected_leaf_count)
        return actual_leaf_count
    _test_tree = {'path1': '111', 'path2': {'2': '222', 'emptydir': None}, 'aaa': {'b1': 'b1', 'b2': None, 'bbb': {'ccc': {'ddd': 'DDD'}}, 'c': None}}

    def test_delete_globs(self):
        if False:
            while True:
                i = 10
        dir = os.path.join(self.tmpdir, 'dir')
        self.make_tree(dir, self._test_tree, expected_leaf_count=7)
        self.fs.delete([os.path.join(dir, 'path*'), os.path.join(dir, 'aaa', 'b*')])
        self.check_tree(dir, {'aaa': {'c': None}}, expected_leaf_count=1)

    def test_recursive_delete(self):
        if False:
            i = 10
            return i + 15
        dir = os.path.join(self.tmpdir, 'dir')
        self.make_tree(dir, self._test_tree, expected_leaf_count=7)
        self.fs.delete([dir])
        self.check_tree(self.tmpdir, {'': None}, expected_leaf_count=1)

    def test_delete_glob_errors(self):
        if False:
            while True:
                i = 10
        dir = os.path.join(self.tmpdir, 'dir')
        self.make_tree(dir, self._test_tree, expected_leaf_count=7)
        with self.assertRaisesRegex(BeamIOError, '^Delete operation failed') as error:
            self.fs.delete([os.path.join(dir, 'path*'), os.path.join(dir, 'aaa', 'b*'), os.path.join(dir, 'aaa', 'd*')])
        self.check_tree(dir, {'aaa': {'c': None}}, expected_leaf_count=1)
        self.assertEqual(list(error.exception.exception_details.keys()), [os.path.join(dir, 'aaa', 'd*')])
        with self.assertRaisesRegex(BeamIOError, '^Delete operation failed') as error:
            self.fs.delete([os.path.join(dir, 'path*')])
        self.check_tree(dir, {'aaa': {'c': None}}, expected_leaf_count=1)
        self.assertEqual(list(error.exception.exception_details.keys()), [os.path.join(dir, 'path*')])

    def test_delete(self):
        if False:
            return 10
        path1 = os.path.join(self.tmpdir, 'f1')
        with open(path1, 'a') as f:
            f.write('Hello')
        self.assertTrue(self.fs.exists(path1))
        self.fs.delete([path1])
        self.assertFalse(self.fs.exists(path1))

    def test_delete_error(self):
        if False:
            return 10
        path1 = os.path.join(self.tmpdir, 'f1')
        with self.assertRaisesRegex(BeamIOError, '^Delete operation failed') as error:
            self.fs.delete([path1])
        self.assertEqual(list(error.exception.exception_details.keys()), [path1])
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()