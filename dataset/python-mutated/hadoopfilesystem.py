""":class:`~apache_beam.io.filesystem.FileSystem` implementation for accessing
Hadoop Distributed File System files."""
import io
import logging
import posixpath
import re
from typing import BinaryIO
import hdfs
from apache_beam.io import filesystemio
from apache_beam.io.filesystem import BeamIOError
from apache_beam.io.filesystem import CompressedFile
from apache_beam.io.filesystem import CompressionTypes
from apache_beam.io.filesystem import FileMetadata
from apache_beam.io.filesystem import FileSystem
from apache_beam.options.pipeline_options import HadoopFileSystemOptions
from apache_beam.options.pipeline_options import PipelineOptions
__all__ = ['HadoopFileSystem']
_HDFS_PREFIX = 'hdfs:/'
_URL_RE = re.compile('^' + _HDFS_PREFIX + '(/.*)')
_FULL_URL_RE = re.compile('^' + _HDFS_PREFIX + '/([^/]+)(/.*)*')
_COPY_BUFFER_SIZE = 2 ** 16
_DEFAULT_BUFFER_SIZE = 20 * 1024 * 1024
_FILE_CHECKSUM_ALGORITHM = 'algorithm'
_FILE_CHECKSUM_BYTES = 'bytes'
_FILE_CHECKSUM_LENGTH = 'length'
_FILE_STATUS_LENGTH = 'length'
_FILE_STATUS_UPDATED = 'modificationTime'
_FILE_STATUS_PATH_SUFFIX = 'pathSuffix'
_FILE_STATUS_TYPE = 'type'
_FILE_STATUS_TYPE_DIRECTORY = 'DIRECTORY'
_FILE_STATUS_TYPE_FILE = 'FILE'
_LOGGER = logging.getLogger(__name__)

class HdfsDownloader(filesystemio.Downloader):

    def __init__(self, hdfs_client, path):
        if False:
            i = 10
            return i + 15
        self._hdfs_client = hdfs_client
        self._path = path
        self._size = self._hdfs_client.status(path)[_FILE_STATUS_LENGTH]

    @property
    def size(self):
        if False:
            while True:
                i = 10
        return self._size

    def get_range(self, start, end):
        if False:
            print('Hello World!')
        with self._hdfs_client.read(self._path, offset=start, length=end - start) as reader:
            return reader.read()

class HdfsUploader(filesystemio.Uploader):

    def __init__(self, hdfs_client, path):
        if False:
            i = 10
            return i + 15
        self._hdfs_client = hdfs_client
        if self._hdfs_client.status(path, strict=False) is not None:
            raise BeamIOError('Path already exists: %s' % path)
        self._handle_context = self._hdfs_client.write(path)
        self._handle = self._handle_context.__enter__()

    def put(self, data):
        if False:
            for i in range(10):
                print('nop')
        self._handle.write(bytes(data))

    def finish(self):
        if False:
            print('Hello World!')
        self._handle.__exit__(None, None, None)
        self._handle = None
        self._handle_context = None

class HadoopFileSystem(FileSystem):
    """``FileSystem`` implementation that supports HDFS.

  URL arguments to methods expect strings starting with ``hdfs://``.
  """

    def __init__(self, pipeline_options):
        if False:
            i = 10
            return i + 15
        'Initializes a connection to HDFS.\n\n    Connection configuration is done by passing pipeline options.\n    See :class:`~apache_beam.options.pipeline_options.HadoopFileSystemOptions`.\n    '
        super().__init__(pipeline_options)
        logging.getLogger('hdfs.client').setLevel(logging.WARN)
        if pipeline_options is None:
            raise ValueError('pipeline_options is not set')
        if isinstance(pipeline_options, PipelineOptions):
            hdfs_options = pipeline_options.view_as(HadoopFileSystemOptions)
            hdfs_host = hdfs_options.hdfs_host
            hdfs_port = hdfs_options.hdfs_port
            hdfs_user = hdfs_options.hdfs_user
            self._full_urls = hdfs_options.hdfs_full_urls
        else:
            hdfs_host = pipeline_options.get('hdfs_host')
            hdfs_port = pipeline_options.get('hdfs_port')
            hdfs_user = pipeline_options.get('hdfs_user')
            self._full_urls = pipeline_options.get('hdfs_full_urls', False)
        if hdfs_host is None:
            raise ValueError('hdfs_host is not set')
        if hdfs_port is None:
            raise ValueError('hdfs_port is not set')
        if hdfs_user is None:
            raise ValueError('hdfs_user is not set')
        if not isinstance(self._full_urls, bool):
            raise ValueError('hdfs_full_urls should be bool, got: %s', self._full_urls)
        self._hdfs_client = hdfs.InsecureClient('http://%s:%s' % (hdfs_host, str(hdfs_port)), user=hdfs_user)

    @classmethod
    def scheme(cls):
        if False:
            while True:
                i = 10
        return 'hdfs'

    def _parse_url(self, url):
        if False:
            for i in range(10):
                print('nop')
        "Verifies that url begins with hdfs:// prefix, strips it and adds a\n    leading /.\n\n    Parsing behavior is determined by HadoopFileSystemOptions.hdfs_full_urls.\n\n    Args:\n      url: (str) A URL in the form hdfs://path/...\n        or in the form hdfs://server/path/...\n\n    Raises:\n      ValueError if the URL doesn't match the expect format.\n\n    Returns:\n      (str, str) If using hdfs_full_urls, for an input of\n      'hdfs://server/path/...' will return (server, '/path/...').\n      Otherwise, for an input of 'hdfs://path/...', will return\n      ('', '/path/...').\n    "
        if not self._full_urls:
            m = _URL_RE.match(url)
            if m is None:
                raise ValueError('Could not parse url: %s' % url)
            return ('', m.group(1))
        else:
            m = _FULL_URL_RE.match(url)
            if m is None:
                raise ValueError('Could not parse url: %s' % url)
            return (m.group(1), m.group(2) or '/')

    def join(self, base_url, *paths):
        if False:
            for i in range(10):
                print('nop')
        'Join two or more pathname components.\n\n    Args:\n      base_url: string path of the first component of the path.\n        Must start with hdfs://.\n      paths: path components to be added\n\n    Returns:\n      Full url after combining all the passed components.\n    '
        (server, basepath) = self._parse_url(base_url)
        return _HDFS_PREFIX + self._join(server, basepath, *paths)

    def _join(self, server, basepath, *paths):
        if False:
            return 10
        res = posixpath.join(basepath, *paths)
        if server:
            server = '/' + server
        return server + res

    def split(self, url):
        if False:
            return 10
        (server, rel_path) = self._parse_url(url)
        if server:
            server = '/' + server
        (head, tail) = posixpath.split(rel_path)
        return (_HDFS_PREFIX + server + head, tail)

    def mkdirs(self, url):
        if False:
            for i in range(10):
                print('nop')
        (_, path) = self._parse_url(url)
        if self._exists(path):
            raise BeamIOError('Path already exists: %s' % path)
        return self._mkdirs(path)

    def _mkdirs(self, path):
        if False:
            i = 10
            return i + 15
        self._hdfs_client.makedirs(path)

    def has_dirs(self):
        if False:
            for i in range(10):
                print('nop')
        return True

    def _list(self, url):
        if False:
            while True:
                i = 10
        try:
            (server, path) = self._parse_url(url)
            for res in self._hdfs_client.list(path, status=True):
                yield FileMetadata(_HDFS_PREFIX + self._join(server, path, res[0]), res[1][_FILE_STATUS_LENGTH], res[1][_FILE_STATUS_UPDATED] / 1000.0)
        except Exception as e:
            raise BeamIOError('List operation failed', {url: e})

    @staticmethod
    def _add_compression(stream, path, mime_type, compression_type):
        if False:
            for i in range(10):
                print('nop')
        if mime_type != 'application/octet-stream':
            _LOGGER.warning('Mime types are not supported. Got non-default mime_type: %s', mime_type)
        if compression_type == CompressionTypes.AUTO:
            compression_type = CompressionTypes.detect_compression_type(path)
        if compression_type != CompressionTypes.UNCOMPRESSED:
            return CompressedFile(stream)
        return stream

    def create(self, url, mime_type='application/octet-stream', compression_type=CompressionTypes.AUTO):
        if False:
            for i in range(10):
                print('nop')
        '\n    Returns:\n      A Python File-like object.\n    '
        (_, path) = self._parse_url(url)
        return self._create(path, mime_type, compression_type)

    def _create(self, path, mime_type='application/octet-stream', compression_type=CompressionTypes.AUTO):
        if False:
            i = 10
            return i + 15
        stream = io.BufferedWriter(filesystemio.UploaderStream(HdfsUploader(self._hdfs_client, path)), buffer_size=_DEFAULT_BUFFER_SIZE)
        return self._add_compression(stream, path, mime_type, compression_type)

    def open(self, url, mime_type='application/octet-stream', compression_type=CompressionTypes.AUTO):
        if False:
            for i in range(10):
                print('nop')
        '\n    Returns:\n      A Python File-like object.\n    '
        (_, path) = self._parse_url(url)
        return self._open(path, mime_type, compression_type)

    def _open(self, path, mime_type='application/octet-stream', compression_type=CompressionTypes.AUTO):
        if False:
            i = 10
            return i + 15
        stream = io.BufferedReader(filesystemio.DownloaderStream(HdfsDownloader(self._hdfs_client, path)), buffer_size=_DEFAULT_BUFFER_SIZE)
        return self._add_compression(stream, path, mime_type, compression_type)

    def copy(self, source_file_names, destination_file_names):
        if False:
            print('Hello World!')
        '\n    It is an error if any file to copy already exists at the destination.\n\n    Raises ``BeamIOError`` if any error occurred.\n\n    Args:\n      source_file_names: iterable of URLs.\n      destination_file_names: iterable of URLs.\n    '
        if len(source_file_names) != len(destination_file_names):
            raise BeamIOError('source_file_names and destination_file_names should be equal in length: %d != %d' % (len(source_file_names), len(destination_file_names)))

        def _copy_file(source, destination):
            if False:
                for i in range(10):
                    print('nop')
            with self._open(source) as f1:
                with self._create(destination) as f2:
                    while True:
                        buf = f1.read(_COPY_BUFFER_SIZE)
                        if not buf:
                            break
                        f2.write(buf)

        def _copy_path(source, destination):
            if False:
                while True:
                    i = 10
            'Recursively copy the file tree from the source to the destination.'
            if self._hdfs_client.status(source)[_FILE_STATUS_TYPE] != _FILE_STATUS_TYPE_DIRECTORY:
                _copy_file(source, destination)
                return
            for (path, dirs, files) in self._hdfs_client.walk(source):
                for dir in dirs:
                    new_dir = self._join('', destination, dir)
                    if not self._exists(new_dir):
                        self._mkdirs(new_dir)
                rel_path = posixpath.relpath(path, source)
                if rel_path == '.':
                    rel_path = ''
                for file in files:
                    _copy_file(self._join('', path, file), self._join('', destination, rel_path, file))
        exceptions = {}
        for (source, destination) in zip(source_file_names, destination_file_names):
            try:
                (_, rel_source) = self._parse_url(source)
                (_, rel_destination) = self._parse_url(destination)
                _copy_path(rel_source, rel_destination)
            except Exception as e:
                exceptions[source, destination] = e
        if exceptions:
            raise BeamIOError('Copy operation failed', exceptions)

    def rename(self, source_file_names, destination_file_names):
        if False:
            print('Hello World!')
        exceptions = {}
        for (source, destination) in zip(source_file_names, destination_file_names):
            try:
                (_, rel_source) = self._parse_url(source)
                (_, rel_destination) = self._parse_url(destination)
                try:
                    self._hdfs_client.rename(rel_source, rel_destination)
                except hdfs.HdfsError as e:
                    raise BeamIOError('libhdfs error in renaming %s to %s' % (source, destination), e)
            except Exception as e:
                exceptions[source, destination] = e
        if exceptions:
            raise BeamIOError('Rename operation failed', exceptions)

    def exists(self, url):
        if False:
            return 10
        'Checks existence of url in HDFS.\n\n    Args:\n      url: String in the form hdfs://...\n\n    Returns:\n      True if url exists as a file or directory in HDFS.\n    '
        (_, path) = self._parse_url(url)
        return self._exists(path)

    def _exists(self, path):
        if False:
            for i in range(10):
                print('nop')
        'Returns True if path exists as a file or directory in HDFS.\n\n    Args:\n      path: String in the form /...\n    '
        return self._hdfs_client.status(path, strict=False) is not None

    def size(self, url):
        if False:
            return 10
        "Fetches file size for a URL.\n\n    Returns:\n      int size of path according to the FileSystem.\n\n    Raises:\n      ``BeamIOError``: if url doesn't exist.\n    "
        return self.metadata(url).size_in_bytes

    def last_updated(self, url):
        if False:
            for i in range(10):
                print('nop')
        "Fetches last updated time for a URL.\n\n    Args:\n      url: string url of file.\n\n    Returns: float UNIX Epoch time\n\n    Raises:\n      ``BeamIOError``: if path doesn't exist.\n    "
        return self.metadata(url).last_updated_in_seconds

    def checksum(self, url):
        if False:
            return 10
        "Fetches a checksum description for a URL.\n\n    Returns:\n      String describing the checksum.\n\n    Raises:\n      ``BeamIOError``: if url doesn't exist.\n    "
        (_, path) = self._parse_url(url)
        file_checksum = self._hdfs_client.checksum(path)
        return '%s-%d-%s' % (file_checksum[_FILE_CHECKSUM_ALGORITHM], file_checksum[_FILE_CHECKSUM_LENGTH], file_checksum[_FILE_CHECKSUM_BYTES])

    def metadata(self, url):
        if False:
            i = 10
            return i + 15
        "Fetch metadata fields of a file on the FileSystem.\n\n    Args:\n      url: string url of a file.\n\n    Returns:\n      :class:`~apache_beam.io.filesystem.FileMetadata`.\n\n    Raises:\n      ``BeamIOError``: if url doesn't exist.\n    "
        (_, path) = self._parse_url(url)
        status = self._hdfs_client.status(path, strict=False)
        if status is None:
            raise BeamIOError('File not found: %s' % url)
        return FileMetadata(url, status[_FILE_STATUS_LENGTH], status[_FILE_STATUS_UPDATED] / 1000.0)

    def delete(self, urls):
        if False:
            for i in range(10):
                print('nop')
        exceptions = {}
        for url in urls:
            try:
                (_, path) = self._parse_url(url)
                self._hdfs_client.delete(path, recursive=True)
            except Exception as e:
                exceptions[url] = e
        if exceptions:
            raise BeamIOError('Delete operation failed', exceptions)