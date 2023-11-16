"""Unit tests for the S3 File System"""
import logging
import unittest
import mock
from apache_beam.io.aws.clients.s3 import messages
from apache_beam.io.filesystem import BeamIOError
from apache_beam.io.filesystem import FileMetadata
from apache_beam.options.pipeline_options import PipelineOptions
try:
    from apache_beam.io.aws import s3filesystem
except ImportError:
    s3filesystem = None

@unittest.skipIf(s3filesystem is None, 'AWS dependencies are not installed')
class S3FileSystemTest(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        pipeline_options = PipelineOptions()
        self.fs = s3filesystem.S3FileSystem(pipeline_options=pipeline_options)

    def test_scheme(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.fs.scheme(), 's3')
        self.assertEqual(s3filesystem.S3FileSystem.scheme(), 's3')

    def test_join(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual('s3://bucket/path/to/file', self.fs.join('s3://bucket/path', 'to', 'file'))
        self.assertEqual('s3://bucket/path/to/file', self.fs.join('s3://bucket/path', 'to/file'))
        self.assertEqual('s3://bucket/path/to/file', self.fs.join('s3://bucket/path', '/to/file'))
        self.assertEqual('s3://bucket/path/to/file', self.fs.join('s3://bucket/path/', 'to', 'file'))
        self.assertEqual('s3://bucket/path/to/file', self.fs.join('s3://bucket/path/', 'to/file'))
        self.assertEqual('s3://bucket/path/to/file', self.fs.join('s3://bucket/path/', '/to/file'))
        with self.assertRaises(ValueError):
            self.fs.join('/bucket/path/', '/to/file')

    def test_split(self):
        if False:
            print('Hello World!')
        self.assertEqual(('s3://foo/bar', 'baz'), self.fs.split('s3://foo/bar/baz'))
        self.assertEqual(('s3://foo', ''), self.fs.split('s3://foo/'))
        self.assertEqual(('s3://foo', ''), self.fs.split('s3://foo'))
        with self.assertRaises(ValueError):
            self.fs.split('/no/s3/prefix')

    @mock.patch('apache_beam.io.aws.s3filesystem.s3io')
    def test_match_single(self, unused_mock_arg):
        if False:
            while True:
                i = 10
        s3io_mock = mock.MagicMock()
        s3filesystem.s3io.S3IO = lambda options: s3io_mock
        s3io_mock._status.return_value = {'size': 1, 'last_updated': 9999999.0}
        expected_results = [FileMetadata('s3://bucket/file1', 1, 9999999.0)]
        match_result = self.fs.match(['s3://bucket/file1'])[0]
        self.assertEqual(match_result.metadata_list, expected_results)
        s3io_mock._status.assert_called_once_with('s3://bucket/file1')

    @mock.patch('apache_beam.io.aws.s3filesystem.s3io')
    def test_match_multiples(self, unused_mock_arg):
        if False:
            i = 10
            return i + 15
        s3io_mock = mock.MagicMock()
        s3filesystem.s3io.S3IO = lambda options: s3io_mock
        s3io_mock.list_files.return_value = iter([('s3://bucket/file1', (1, 9999999.0)), ('s3://bucket/file2', (2, 8888888.0))])
        expected_results = set([FileMetadata('s3://bucket/file1', 1, 9999999.0), FileMetadata('s3://bucket/file2', 2, 8888888.0)])
        match_result = self.fs.match(['s3://bucket/'])[0]
        self.assertEqual(set(match_result.metadata_list), expected_results)
        s3io_mock.list_files.assert_called_once_with('s3://bucket/', with_metadata=True)

    @mock.patch('apache_beam.io.aws.s3filesystem.s3io')
    def test_match_multiples_limit(self, unused_mock_arg):
        if False:
            print('Hello World!')
        s3io_mock = mock.MagicMock()
        limit = 1
        s3filesystem.s3io.S3IO = lambda options: s3io_mock
        s3io_mock.list_files.return_value = iter([('s3://bucket/file1', (1, 99999.0))])
        expected_results = set([FileMetadata('s3://bucket/file1', 1, 99999.0)])
        match_result = self.fs.match(['s3://bucket/'], [limit])[0]
        self.assertEqual(set(match_result.metadata_list), expected_results)
        self.assertEqual(len(match_result.metadata_list), limit)
        s3io_mock.list_files.assert_called_once_with('s3://bucket/', with_metadata=True)

    @mock.patch('apache_beam.io.aws.s3filesystem.s3io')
    def test_match_multiples_error(self, unused_mock_arg):
        if False:
            i = 10
            return i + 15
        s3io_mock = mock.MagicMock()
        s3filesystem.s3io.S3IO = lambda options: s3io_mock
        exception = IOError('Failed')
        s3io_mock.list_files.side_effect = exception
        with self.assertRaises(BeamIOError) as error:
            self.fs.match(['s3://bucket/'])
        self.assertIn('Match operation failed', str(error.exception))
        s3io_mock.list_files.assert_called_once_with('s3://bucket/', with_metadata=True)

    @mock.patch('apache_beam.io.aws.s3filesystem.s3io')
    def test_match_multiple_patterns(self, unused_mock_arg):
        if False:
            while True:
                i = 10
        s3io_mock = mock.MagicMock()
        s3filesystem.s3io.S3IO = lambda options: s3io_mock
        s3io_mock.list_files.side_effect = [iter([('s3://bucket/file1', (1, 99999.0))]), iter([('s3://bucket/file2', (2, 88888.0))])]
        expected_results = [[FileMetadata('s3://bucket/file1', 1, 99999.0)], [FileMetadata('s3://bucket/file2', 2, 88888.0)]]
        result = self.fs.match(['s3://bucket/file1*', 's3://bucket/file2*'])
        self.assertEqual([mr.metadata_list for mr in result], expected_results)

    @mock.patch('apache_beam.io.aws.s3filesystem.s3io')
    def test_create(self, unused_mock_arg):
        if False:
            while True:
                i = 10
        s3io_mock = mock.MagicMock()
        s3filesystem.s3io.S3IO = lambda options: s3io_mock
        _ = self.fs.create('s3://bucket/from1', 'application/octet-stream')
        s3io_mock.open.assert_called_once_with('s3://bucket/from1', 'wb', mime_type='application/octet-stream')

    @mock.patch('apache_beam.io.aws.s3filesystem.s3io')
    def test_open(self, unused_mock_arg):
        if False:
            return 10
        s3io_mock = mock.MagicMock()
        s3filesystem.s3io.S3IO = lambda options: s3io_mock
        _ = self.fs.open('s3://bucket/from1', 'application/octet-stream')
        s3io_mock.open.assert_called_once_with('s3://bucket/from1', 'rb', mime_type='application/octet-stream')

    @mock.patch('apache_beam.io.aws.s3filesystem.s3io')
    def test_copy_file(self, unused_mock_arg):
        if False:
            return 10
        s3io_mock = mock.MagicMock()
        s3filesystem.s3io.S3IO = lambda options: s3io_mock
        sources = ['s3://bucket/from1', 's3://bucket/from2']
        destinations = ['s3://bucket/to1', 's3://bucket/to2']
        self.fs.copy(sources, destinations)
        src_dest_pairs = list(zip(sources, destinations))
        s3io_mock.copy_paths.assert_called_once_with(src_dest_pairs)

    @mock.patch('apache_beam.io.aws.s3filesystem.s3io')
    def test_copy_file_error(self, unused_mock_arg):
        if False:
            for i in range(10):
                print('nop')
        s3io_mock = mock.MagicMock()
        s3filesystem.s3io.S3IO = lambda options: s3io_mock
        sources = ['s3://bucket/from1', 's3://bucket/from2', 's3://bucket/from3']
        destinations = ['s3://bucket/to1', 's3://bucket/to2']
        with self.assertRaises(BeamIOError):
            self.fs.copy(sources, destinations)

    @mock.patch('apache_beam.io.aws.s3filesystem.s3io')
    def test_delete(self, unused_mock_arg):
        if False:
            for i in range(10):
                print('nop')
        s3io_mock = mock.MagicMock()
        s3filesystem.s3io.S3IO = lambda options: s3io_mock
        s3io_mock.size.return_value = 0
        files = ['s3://bucket/from1', 's3://bucket/from2', 's3://bucket/from3']
        self.fs.delete(files)
        s3io_mock.delete_paths.assert_called_once_with(files)

    @mock.patch('apache_beam.io.aws.s3filesystem.s3io')
    def test_delete_error(self, unused_mock_arg):
        if False:
            i = 10
            return i + 15
        s3io_mock = mock.MagicMock()
        s3filesystem.s3io.S3IO = lambda options: s3io_mock
        problematic_directory = 's3://nonexistent-bucket/tree/'
        exception = messages.S3ClientError('Not found', 404)
        s3io_mock.delete_paths.return_value = {problematic_directory: exception, 's3://bucket/object1': None, 's3://bucket/object2': None}
        s3io_mock.size.return_value = 0
        files = [problematic_directory, 's3://bucket/object1', 's3://bucket/object2']
        expected_results = {problematic_directory: exception}
        with self.assertRaises(BeamIOError) as error:
            self.fs.delete(files)
        self.assertIn('Delete operation failed', str(error.exception))
        self.assertEqual(error.exception.exception_details, expected_results)
        s3io_mock.delete_paths.assert_called()

    @mock.patch('apache_beam.io.aws.s3filesystem.s3io')
    def test_rename(self, unused_mock_arg):
        if False:
            for i in range(10):
                print('nop')
        s3io_mock = mock.MagicMock()
        s3filesystem.s3io.S3IO = lambda options: s3io_mock
        sources = ['s3://bucket/from1', 's3://bucket/from2']
        destinations = ['s3://bucket/to1', 's3://bucket/to2']
        self.fs.rename(sources, destinations)
        src_dest_pairs = list(zip(sources, destinations))
        s3io_mock.rename_files.assert_called_once_with(src_dest_pairs)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()