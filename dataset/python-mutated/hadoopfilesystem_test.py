"""Unit tests for :class:`HadoopFileSystem`."""
import io
import logging
import posixpath
import time
import unittest
from parameterized import parameterized_class
from apache_beam.io import hadoopfilesystem as hdfs
from apache_beam.io.filesystem import BeamIOError
from apache_beam.options.pipeline_options import HadoopFileSystemOptions
from apache_beam.options.pipeline_options import PipelineOptions

class FakeFile(io.BytesIO):
    """File object for FakeHdfs"""
    __hash__ = None

    def __init__(self, path, mode='', type='FILE', time_ms=None):
        if False:
            print('Hello World!')
        io.BytesIO.__init__(self)
        if time_ms is None:
            time_ms = int(time.time() * 1000)
        self.time_ms = time_ms
        self.stat = {'path': path, 'mode': mode, 'type': type}
        self.saved_data = None

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        'Equality of two files. Timestamp not included in comparison'
        return self.stat == other.stat and self.getvalue() == self.getvalue()

    def close(self):
        if False:
            while True:
                i = 10
        self.saved_data = self.getvalue()
        io.BytesIO.close(self)

    def __enter__(self):
        if False:
            while True:
                i = 10
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            i = 10
            return i + 15
        self.close()

    @property
    def size(self):
        if False:
            i = 10
            return i + 15
        if self.closed:
            if self.saved_data is None:
                return 0
            return len(self.saved_data)
        return len(self.getvalue())

    def get_file_status(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns a partial WebHDFS FileStatus object.'
        return {hdfs._FILE_STATUS_PATH_SUFFIX: posixpath.basename(self.stat['path']), hdfs._FILE_STATUS_LENGTH: self.size, hdfs._FILE_STATUS_TYPE: self.stat['type'], hdfs._FILE_STATUS_UPDATED: self.time_ms}

    def get_file_checksum(self):
        if False:
            while True:
                i = 10
        'Returns a WebHDFS FileChecksum object.'
        return {hdfs._FILE_CHECKSUM_ALGORITHM: 'fake_algo', hdfs._FILE_CHECKSUM_BYTES: 'checksum_byte_sequence', hdfs._FILE_CHECKSUM_LENGTH: 5}

class FakeHdfsError(Exception):
    """Generic error for FakeHdfs methods."""

class FakeHdfs(object):
    """Fake implementation of ``hdfs.Client``."""

    def __init__(self):
        if False:
            return 10
        self.files = {}

    def write(self, path):
        if False:
            while True:
                i = 10
        if self.status(path, strict=False) is not None:
            raise FakeHdfsError('Path already exists: %s' % path)
        new_file = FakeFile(path, 'wb')
        self.files[path] = new_file
        return new_file

    def read(self, path, offset=0, length=None):
        if False:
            while True:
                i = 10
        old_file = self.files.get(path, None)
        if old_file is None:
            raise FakeHdfsError('Path not found: %s' % path)
        if old_file.stat['type'] == 'DIRECTORY':
            raise FakeHdfsError('Cannot open a directory: %s' % path)
        if not old_file.closed:
            raise FakeHdfsError('File already opened: %s' % path)
        new_file = FakeFile(path, 'rb')
        if old_file.saved_data:
            if length is None:
                new_file.write(old_file.saved_data)
            else:
                new_file.write(old_file.saved_data[:offset + length])
            new_file.seek(offset)
        return new_file

    def list(self, path, status=False):
        if False:
            for i in range(10):
                print('nop')
        if not status:
            raise ValueError('status must be True')
        fs = self.status(path, strict=False)
        if fs is not None and fs[hdfs._FILE_STATUS_TYPE] == hdfs._FILE_STATUS_TYPE_FILE:
            raise ValueError('list must be called on a directory, got file: %s' % path)
        result = []
        for file in self.files.values():
            if file.stat['path'].startswith(path):
                fs = file.get_file_status()
                result.append((fs[hdfs._FILE_STATUS_PATH_SUFFIX], fs))
        return result

    def makedirs(self, path):
        if False:
            for i in range(10):
                print('nop')
        self.files[path] = FakeFile(path, type='DIRECTORY')

    def status(self, path, strict=True):
        if False:
            return 10
        f = self.files.get(path)
        if f is None:
            if strict:
                raise FakeHdfsError('Path not found: %s' % path)
            else:
                return f
        return f.get_file_status()

    def delete(self, path, recursive=True):
        if False:
            i = 10
            return i + 15
        if not recursive:
            raise FakeHdfsError('Non-recursive mode not implemented')
        _ = self.status(path)
        for filepath in list(self.files):
            if filepath.startswith(path):
                del self.files[filepath]

    def walk(self, path):
        if False:
            print('Hello World!')
        paths = [path]
        while paths:
            path = paths.pop()
            files = []
            dirs = []
            for full_path in self.files:
                if not full_path.startswith(path):
                    continue
                short_path = posixpath.relpath(full_path, path)
                if '/' not in short_path:
                    if self.status(full_path)[hdfs._FILE_STATUS_TYPE] == 'DIRECTORY':
                        if short_path != '.':
                            dirs.append(short_path)
                    else:
                        files.append(short_path)
            yield (path, dirs, files)
            paths = [posixpath.join(path, dir) for dir in dirs]

    def rename(self, path1, path2):
        if False:
            for i in range(10):
                print('nop')
        if self.status(path1, strict=False) is None:
            raise FakeHdfsError('Path1 not found: %s' % path1)
        files_to_rename = [path for path in self.files if path == path1 or path.startswith(path1 + '/')]
        for fullpath in files_to_rename:
            f = self.files.pop(fullpath)
            newpath = path2 + fullpath[len(path1):]
            f.stat['path'] = newpath
            self.files[newpath] = f

    def checksum(self, path):
        if False:
            for i in range(10):
                print('nop')
        f = self.files.get(path, None)
        if f is None:
            raise FakeHdfsError('Path not found: %s' % path)
        return f.get_file_checksum()

@parameterized_class(('full_urls',), [(False,), (True,)])
class HadoopFileSystemTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self._fake_hdfs = FakeHdfs()
        hdfs.hdfs.InsecureClient = lambda *args, **kwargs: self._fake_hdfs
        pipeline_options = PipelineOptions()
        hdfs_options = pipeline_options.view_as(HadoopFileSystemOptions)
        hdfs_options.hdfs_host = ''
        hdfs_options.hdfs_port = 0
        hdfs_options.hdfs_user = ''
        self.fs = hdfs.HadoopFileSystem(pipeline_options)
        self.fs._full_urls = self.full_urls
        if self.full_urls:
            self.tmpdir = 'hdfs://test_dir'
        else:
            self.tmpdir = 'hdfs://server/test_dir'
        for filename in ['old_file1', 'old_file2']:
            url = self.fs.join(self.tmpdir, filename)
            self.fs.create(url).close()

    def test_scheme(self):
        if False:
            return 10
        self.assertEqual(self.fs.scheme(), 'hdfs')
        self.assertEqual(hdfs.HadoopFileSystem.scheme(), 'hdfs')

    def test_parse_url(self):
        if False:
            i = 10
            return i + 15
        cases = [('hdfs://', ('', '/'), False), ('hdfs://', None, True), ('hdfs://a', ('', '/a'), False), ('hdfs://a', ('a', '/'), True), ('hdfs://a/', ('', '/a/'), False), ('hdfs://a/', ('a', '/'), True), ('hdfs://a/b', ('', '/a/b'), False), ('hdfs://a/b', ('a', '/b'), True), ('hdfs://a/b/', ('', '/a/b/'), False), ('hdfs://a/b/', ('a', '/b/'), True), ('hdfs:/a/b', None, False), ('hdfs:/a/b', None, True), ('invalid', None, False), ('invalid', None, True)]
        for (url, expected, full_urls) in cases:
            if self.full_urls != full_urls:
                continue
            try:
                result = self.fs._parse_url(url)
            except ValueError:
                self.assertIsNone(expected, msg=(url, expected, full_urls))
                continue
            self.assertEqual(expected, result, msg=(url, expected, full_urls))

    def test_url_join(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual('hdfs://tmp/path/to/file', self.fs.join('hdfs://tmp/path', 'to', 'file'))
        self.assertEqual('hdfs://tmp/path/to/file', self.fs.join('hdfs://tmp/path', 'to/file'))
        self.assertEqual('hdfs://tmp/path/', self.fs.join('hdfs://tmp/path/', ''))
        if not self.full_urls:
            self.assertEqual('hdfs://bar', self.fs.join('hdfs://foo', '/bar'))
            self.assertEqual('hdfs://bar', self.fs.join('hdfs://foo/', '/bar'))
            with self.assertRaises(ValueError):
                self.fs.join('/no/scheme', 'file')
        else:
            self.assertEqual('hdfs://foo/bar', self.fs.join('hdfs://foo', '/bar'))
            self.assertEqual('hdfs://foo/bar', self.fs.join('hdfs://foo/', '/bar'))

    def test_url_split(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(('hdfs://tmp/path/to', 'file'), self.fs.split('hdfs://tmp/path/to/file'))
        if not self.full_urls:
            self.assertEqual(('hdfs://', 'tmp'), self.fs.split('hdfs://tmp'))
            self.assertEqual(('hdfs://tmp', ''), self.fs.split('hdfs://tmp/'))
            self.assertEqual(('hdfs://tmp', 'a'), self.fs.split('hdfs://tmp/a'))
        else:
            self.assertEqual(('hdfs://tmp/', ''), self.fs.split('hdfs://tmp'))
            self.assertEqual(('hdfs://tmp/', ''), self.fs.split('hdfs://tmp/'))
            self.assertEqual(('hdfs://tmp/', 'a'), self.fs.split('hdfs://tmp/a'))
        self.assertEqual(('hdfs://tmp/a', ''), self.fs.split('hdfs://tmp/a/'))
        with self.assertRaisesRegex(ValueError, 'parse'):
            self.fs.split('tmp')

    def test_mkdirs(self):
        if False:
            for i in range(10):
                print('nop')
        url = self.fs.join(self.tmpdir, 't1/t2')
        self.fs.mkdirs(url)
        self.assertTrue(self.fs.exists(url))

    def test_mkdirs_failed(self):
        if False:
            return 10
        url = self.fs.join(self.tmpdir, 't1/t2')
        self.fs.mkdirs(url)
        with self.assertRaises(IOError):
            self.fs.mkdirs(url)

    def test_match_file(self):
        if False:
            return 10
        expected_files = [self.fs.join(self.tmpdir, filename) for filename in ['old_file1', 'old_file2']]
        match_patterns = expected_files
        result = self.fs.match(match_patterns)
        returned_files = [f.path for match_result in result for f in match_result.metadata_list]
        self.assertCountEqual(expected_files, returned_files)

    def test_match_file_with_limits(self):
        if False:
            i = 10
            return i + 15
        expected_files = [self.fs.join(self.tmpdir, filename) for filename in ['old_file1', 'old_file2']]
        result = self.fs.match([self.tmpdir + '/'], [1])[0]
        files = [f.path for f in result.metadata_list]
        self.assertEqual(len(files), 1)
        self.assertIn(files[0], expected_files)

    def test_match_file_with_zero_limit(self):
        if False:
            print('Hello World!')
        result = self.fs.match([self.tmpdir + '/'], [0])[0]
        self.assertEqual(len(result.metadata_list), 0)

    def test_match_file_empty(self):
        if False:
            return 10
        url = self.fs.join(self.tmpdir, 'nonexistent_file')
        result = self.fs.match([url])[0]
        files = [f.path for f in result.metadata_list]
        self.assertEqual(files, [])

    def test_match_file_error(self):
        if False:
            return 10
        url = self.fs.join(self.tmpdir, 'old_file1')
        bad_url = 'bad_url'
        with self.assertRaisesRegex(BeamIOError, '^Match operation failed .* %s' % bad_url):
            result = self.fs.match([bad_url, url])[0]
            files = [f.path for f in result.metadata_list]
            self.assertEqual(files, [self.fs._parse_url(url)])

    def test_match_directory(self):
        if False:
            i = 10
            return i + 15
        expected_files = [self.fs.join(self.tmpdir, filename) for filename in ['old_file1', 'old_file2']]
        result = self.fs.match([self.tmpdir + '/'])[0]
        files = [f.path for f in result.metadata_list]
        self.assertCountEqual(files, expected_files)

    def test_match_directory_trailing_slash(self):
        if False:
            i = 10
            return i + 15
        expected_files = [self.fs.join(self.tmpdir, filename) for filename in ['old_file1', 'old_file2']]
        result = self.fs.match([self.tmpdir + '/'])[0]
        files = [f.path for f in result.metadata_list]
        self.assertCountEqual(files, expected_files)

    def test_create_success(self):
        if False:
            while True:
                i = 10
        url = self.fs.join(self.tmpdir, 'new_file')
        handle = self.fs.create(url)
        self.assertIsNotNone(handle)
        (_, url) = self.fs._parse_url(url)
        expected_file = FakeFile(url, 'wb')
        self.assertEqual(self._fake_hdfs.files[url], expected_file)

    def test_create_write_read_compressed(self):
        if False:
            i = 10
            return i + 15
        url = self.fs.join(self.tmpdir, 'new_file.gz')
        handle = self.fs.create(url)
        self.assertIsNotNone(handle)
        (_, path) = self.fs._parse_url(url)
        expected_file = FakeFile(path, 'wb')
        self.assertEqual(self._fake_hdfs.files[path], expected_file)
        data = b'abc' * 10
        handle.write(data)
        self.assertNotEqual(data, self._fake_hdfs.files[path].getvalue())
        handle.close()
        handle = self.fs.open(url)
        read_data = handle.read(len(data))
        self.assertEqual(data, read_data)
        handle.close()

    def test_random_read_large_file(self):
        if False:
            return 10
        url = self.fs.join(self.tmpdir, 'read_length')
        handle = self.fs.create(url)
        data = b'test' * 10000000
        handle.write(data)
        handle.close()
        handle = self.fs.open(url)
        handle.seek(100)
        read_data = handle.read(3)
        self.assertEqual(data[100:103], read_data)
        read_data = handle.read(4)
        self.assertEqual(data[103:107], read_data)

    def test_open(self):
        if False:
            while True:
                i = 10
        url = self.fs.join(self.tmpdir, 'old_file1')
        handle = self.fs.open(url)
        expected_data = b''
        data = handle.read()
        self.assertEqual(data, expected_data)

    def test_open_bad_path(self):
        if False:
            return 10
        with self.assertRaises(FakeHdfsError):
            self.fs.open(self.fs.join(self.tmpdir, 'nonexistent/path'))

    def _cmpfiles(self, url1, url2):
        if False:
            i = 10
            return i + 15
        with self.fs.open(url1) as f1:
            with self.fs.open(url2) as f2:
                data1 = f1.read()
                data2 = f2.read()
                return data1 == data2

    def test_copy_file(self):
        if False:
            i = 10
            return i + 15
        url1 = self.fs.join(self.tmpdir, 'new_file1')
        url2 = self.fs.join(self.tmpdir, 'new_file2')
        url3 = self.fs.join(self.tmpdir, 'new_file3')
        with self.fs.create(url1) as f1:
            f1.write(b'Hello')
        self.fs.copy([url1, url1], [url2, url3])
        self.assertTrue(self._cmpfiles(url1, url2))
        self.assertTrue(self._cmpfiles(url1, url3))

    def test_copy_file_overwrite_error(self):
        if False:
            for i in range(10):
                print('nop')
        url1 = self.fs.join(self.tmpdir, 'new_file1')
        url2 = self.fs.join(self.tmpdir, 'new_file2')
        with self.fs.create(url1) as f1:
            f1.write(b'Hello')
        with self.fs.create(url2) as f2:
            f2.write(b'nope')
        with self.assertRaisesRegex(BeamIOError, 'already exists.*%s' % posixpath.basename(url2)):
            self.fs.copy([url1], [url2])

    def test_copy_file_error(self):
        if False:
            return 10
        url1 = self.fs.join(self.tmpdir, 'new_file1')
        url2 = self.fs.join(self.tmpdir, 'new_file2')
        url3 = self.fs.join(self.tmpdir, 'new_file3')
        url4 = self.fs.join(self.tmpdir, 'new_file4')
        with self.fs.create(url3) as f:
            f.write(b'Hello')
        with self.assertRaisesRegex(BeamIOError, '^Copy operation failed .*%s.*%s.* not found' % (url1, url2)):
            self.fs.copy([url1, url3], [url2, url4])
        self.assertTrue(self._cmpfiles(url3, url4))

    def test_copy_directory(self):
        if False:
            i = 10
            return i + 15
        url_t1 = self.fs.join(self.tmpdir, 't1')
        url_t1_inner = self.fs.join(self.tmpdir, 't1/inner')
        url_t2 = self.fs.join(self.tmpdir, 't2')
        url_t2_inner = self.fs.join(self.tmpdir, 't2/inner')
        self.fs.mkdirs(url_t1)
        self.fs.mkdirs(url_t1_inner)
        self.fs.mkdirs(url_t2)
        url1 = self.fs.join(url_t1_inner, 'f1')
        url2 = self.fs.join(url_t2_inner, 'f1')
        with self.fs.create(url1) as f:
            f.write(b'Hello')
        self.fs.copy([url_t1], [url_t2])
        self.assertTrue(self._cmpfiles(url1, url2))

    def test_copy_directory_overwrite_error(self):
        if False:
            while True:
                i = 10
        url_t1 = self.fs.join(self.tmpdir, 't1')
        url_t1_inner = self.fs.join(self.tmpdir, 't1/inner')
        url_t2 = self.fs.join(self.tmpdir, 't2')
        url_t2_inner = self.fs.join(self.tmpdir, 't2/inner')
        self.fs.mkdirs(url_t1)
        self.fs.mkdirs(url_t1_inner)
        self.fs.mkdirs(url_t2)
        self.fs.mkdirs(url_t2_inner)
        url1 = self.fs.join(url_t1, 'f1')
        url1_inner = self.fs.join(url_t1_inner, 'f2')
        url2 = self.fs.join(url_t2, 'f1')
        unused_url2_inner = self.fs.join(url_t2_inner, 'f2')
        url3_inner = self.fs.join(url_t2_inner, 'f3')
        for url in [url1, url1_inner, url3_inner]:
            with self.fs.create(url) as f:
                f.write(b'Hello')
        with self.fs.create(url2) as f:
            f.write(b'nope')
        with self.assertRaisesRegex(BeamIOError, 'already exists'):
            self.fs.copy([url_t1], [url_t2])

    def test_rename_file(self):
        if False:
            for i in range(10):
                print('nop')
        url1 = self.fs.join(self.tmpdir, 'f1')
        url2 = self.fs.join(self.tmpdir, 'f2')
        with self.fs.create(url1) as f:
            f.write(b'Hello')
        self.fs.rename([url1], [url2])
        self.assertFalse(self.fs.exists(url1))
        self.assertTrue(self.fs.exists(url2))

    def test_rename_file_error(self):
        if False:
            i = 10
            return i + 15
        url1 = self.fs.join(self.tmpdir, 'f1')
        url2 = self.fs.join(self.tmpdir, 'f2')
        url3 = self.fs.join(self.tmpdir, 'f3')
        url4 = self.fs.join(self.tmpdir, 'f4')
        with self.fs.create(url3) as f:
            f.write(b'Hello')
        with self.assertRaisesRegex(BeamIOError, '^Rename operation failed .*%s.*%s' % (url1, url2)):
            self.fs.rename([url1, url3], [url2, url4])
        self.assertFalse(self.fs.exists(url3))
        self.assertTrue(self.fs.exists(url4))

    def test_rename_directory(self):
        if False:
            return 10
        url_t1 = self.fs.join(self.tmpdir, 't1')
        url_t2 = self.fs.join(self.tmpdir, 't2')
        self.fs.mkdirs(url_t1)
        url1 = self.fs.join(url_t1, 'f1')
        url2 = self.fs.join(url_t2, 'f1')
        with self.fs.create(url1) as f:
            f.write(b'Hello')
        self.fs.rename([url_t1], [url_t2])
        self.assertFalse(self.fs.exists(url_t1))
        self.assertTrue(self.fs.exists(url_t2))
        self.assertFalse(self.fs.exists(url1))
        self.assertTrue(self.fs.exists(url2))

    def test_exists(self):
        if False:
            return 10
        url1 = self.fs.join(self.tmpdir, 'old_file1')
        url2 = self.fs.join(self.tmpdir, 'nonexistent')
        self.assertTrue(self.fs.exists(url1))
        self.assertFalse(self.fs.exists(url2))

    def test_size(self):
        if False:
            i = 10
            return i + 15
        url = self.fs.join(self.tmpdir, 'f1')
        with self.fs.create(url) as f:
            f.write(b'Hello')
        self.assertEqual(5, self.fs.size(url))

    def test_checksum(self):
        if False:
            print('Hello World!')
        url = self.fs.join(self.tmpdir, 'f1')
        with self.fs.create(url) as f:
            f.write(b'Hello')
        self.assertEqual('fake_algo-5-checksum_byte_sequence', self.fs.checksum(url))

    def test_last_updated(self):
        if False:
            print('Hello World!')
        url = self.fs.join(self.tmpdir, 'f1')
        with self.fs.create(url) as f:
            f.write(b'Hello')
        tolerance = 5 * 60
        result = self.fs.last_updated(url)
        self.assertAlmostEqual(result, time.time(), delta=tolerance)

    def test_delete_file(self):
        if False:
            i = 10
            return i + 15
        url = self.fs.join(self.tmpdir, 'old_file1')
        self.assertTrue(self.fs.exists(url))
        self.fs.delete([url])
        self.assertFalse(self.fs.exists(url))

    def test_delete_dir(self):
        if False:
            i = 10
            return i + 15
        url_t1 = self.fs.join(self.tmpdir, 'new_dir1')
        url_t2 = self.fs.join(url_t1, 'new_dir2')
        url1 = self.fs.join(url_t2, 'new_file1')
        url2 = self.fs.join(url_t2, 'new_file2')
        self.fs.mkdirs(url_t1)
        self.fs.mkdirs(url_t2)
        self.fs.create(url1).close()
        self.fs.create(url2).close()
        self.assertTrue(self.fs.exists(url1))
        self.assertTrue(self.fs.exists(url2))
        self.fs.delete([url_t1])
        self.assertFalse(self.fs.exists(url_t1))
        self.assertFalse(self.fs.exists(url_t2))
        self.assertFalse(self.fs.exists(url2))
        self.assertFalse(self.fs.exists(url1))

    def test_delete_error(self):
        if False:
            while True:
                i = 10
        url1 = self.fs.join(self.tmpdir, 'nonexistent')
        url2 = self.fs.join(self.tmpdir, 'old_file1')
        self.assertTrue(self.fs.exists(url2))
        (_, path1) = self.fs._parse_url(url1)
        with self.assertRaisesRegex(BeamIOError, '^Delete operation failed .* %s' % path1):
            self.fs.delete([url1, url2])
        self.assertFalse(self.fs.exists(url2))

class HadoopFileSystemRuntimeValueProviderTest(unittest.TestCase):
    """Tests pipeline_options, in the form of a
  RuntimeValueProvider.runtime_options object."""

    def setUp(self):
        if False:
            print('Hello World!')
        self._fake_hdfs = FakeHdfs()
        hdfs.hdfs.InsecureClient = lambda *args, **kwargs: self._fake_hdfs

    def test_dict_options(self):
        if False:
            i = 10
            return i + 15
        pipeline_options = {'hdfs_host': '', 'hdfs_port': 0, 'hdfs_user': ''}
        self.fs = hdfs.HadoopFileSystem(pipeline_options=pipeline_options)
        self.assertFalse(self.fs._full_urls)

    def test_dict_options_missing(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(ValueError, 'hdfs_host'):
            self.fs = hdfs.HadoopFileSystem(pipeline_options={'hdfs_port': 0, 'hdfs_user': ''})
        with self.assertRaisesRegex(ValueError, 'hdfs_port'):
            self.fs = hdfs.HadoopFileSystem(pipeline_options={'hdfs_host': '', 'hdfs_user': ''})
        with self.assertRaisesRegex(ValueError, 'hdfs_user'):
            self.fs = hdfs.HadoopFileSystem(pipeline_options={'hdfs_host': '', 'hdfs_port': 0})

    def test_dict_options_full_urls(self):
        if False:
            while True:
                i = 10
        pipeline_options = {'hdfs_host': '', 'hdfs_port': 0, 'hdfs_user': '', 'hdfs_full_urls': 'invalid'}
        with self.assertRaisesRegex(ValueError, 'hdfs_full_urls'):
            self.fs = hdfs.HadoopFileSystem(pipeline_options=pipeline_options)
        pipeline_options['hdfs_full_urls'] = True
        self.fs = hdfs.HadoopFileSystem(pipeline_options=pipeline_options)
        self.assertTrue(self.fs._full_urls)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()