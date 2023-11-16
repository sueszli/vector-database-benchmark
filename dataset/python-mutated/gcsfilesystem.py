"""GCS file system implementation for accessing files on GCS.

**Updates to the I/O connector code**

For any significant updates to this I/O connector, please consider involving
corresponding code reviewers mentioned in
https://github.com/apache/beam/blob/master/sdks/python/OWNERS
"""
from typing import BinaryIO
from apache_beam.io.filesystem import BeamIOError
from apache_beam.io.filesystem import CompressedFile
from apache_beam.io.filesystem import CompressionTypes
from apache_beam.io.filesystem import FileMetadata
from apache_beam.io.filesystem import FileSystem
from apache_beam.io.gcp import gcsio
__all__ = ['GCSFileSystem']

class GCSFileSystem(FileSystem):
    """A GCS ``FileSystem`` implementation for accessing files on GCS.
  """
    CHUNK_SIZE = gcsio.MAX_BATCH_OPERATION_SIZE
    GCS_PREFIX = 'gs://'

    def __init__(self, pipeline_options):
        if False:
            print('Hello World!')
        super().__init__(pipeline_options)
        self._pipeline_options = pipeline_options

    @classmethod
    def scheme(cls):
        if False:
            return 10
        'URI scheme for the FileSystem\n    '
        return 'gs'

    def join(self, basepath, *paths):
        if False:
            for i in range(10):
                print('nop')
        'Join two or more pathname components for the filesystem\n\n    Args:\n      basepath: string path of the first component of the path\n      paths: path components to be added\n\n    Returns: full path after combining all the passed components\n    '
        if not basepath.startswith(GCSFileSystem.GCS_PREFIX):
            raise ValueError('Basepath %r must be GCS path.' % basepath)
        path = basepath
        for p in paths:
            path = path.rstrip('/') + '/' + p.lstrip('/')
        return path

    def split(self, path):
        if False:
            i = 10
            return i + 15
        "Splits the given path into two parts.\n\n    Splits the path into a pair (head, tail) such that tail contains the last\n    component of the path and head contains everything up to that.\n\n    Head will include the GCS prefix ('gs://').\n\n    Args:\n      path: path as a string\n    Returns:\n      a pair of path components as strings.\n    "
        path = path.strip()
        if not path.startswith(GCSFileSystem.GCS_PREFIX):
            raise ValueError('Path %r must be GCS path.' % path)
        prefix_len = len(GCSFileSystem.GCS_PREFIX)
        last_sep = path[prefix_len:].rfind('/')
        if last_sep >= 0:
            last_sep += prefix_len
        if last_sep > 0:
            return (path[:last_sep], path[last_sep + 1:])
        elif last_sep < 0:
            return (path, '')
        else:
            raise ValueError('Invalid path: %s' % path)

    def mkdirs(self, path):
        if False:
            while True:
                i = 10
        'Recursively create directories for the provided path.\n\n    Args:\n      path: string path of the directory structure that should be created\n\n    Raises:\n      IOError: if leaf directory already exists.\n    '
        pass

    def has_dirs(self):
        if False:
            for i in range(10):
                print('nop')
        'Whether this FileSystem supports directories.'
        return False

    def _list(self, dir_or_prefix):
        if False:
            for i in range(10):
                print('nop')
        "List files in a location.\n\n    Listing is non-recursive, for filesystems that support directories.\n\n    Args:\n      dir_or_prefix: (string) A directory or location prefix (for filesystems\n        that don't have directories).\n\n    Returns:\n      Generator of ``FileMetadata`` objects.\n\n    Raises:\n      ``BeamIOError``: if listing fails, but not if no files were found.\n    "
        try:
            for (path, (size, updated)) in self._gcsIO().list_files(dir_or_prefix, with_metadata=True):
                yield FileMetadata(path, size, updated)
        except Exception as e:
            raise BeamIOError('List operation failed', {dir_or_prefix: e})

    def _gcsIO(self):
        if False:
            i = 10
            return i + 15
        return gcsio.GcsIO(pipeline_options=self._pipeline_options)

    def _path_open(self, path, mode, mime_type='application/octet-stream', compression_type=CompressionTypes.AUTO):
        if False:
            while True:
                i = 10
        'Helper functions to open a file in the provided mode.\n    '
        compression_type = FileSystem._get_compression_type(path, compression_type)
        mime_type = CompressionTypes.mime_type(compression_type, mime_type)
        raw_file = self._gcsIO().open(path, mode, mime_type=mime_type)
        if compression_type == CompressionTypes.UNCOMPRESSED:
            return raw_file
        return CompressedFile(raw_file, compression_type=compression_type)

    def create(self, path, mime_type='application/octet-stream', compression_type=CompressionTypes.AUTO):
        if False:
            return 10
        'Returns a write channel for the given file path.\n\n    Args:\n      path: string path of the file object to be written to the system\n      mime_type: MIME type to specify the type of content in the file object\n      compression_type: Type of compression to be used for this object\n\n    Returns: file handle with a close function for the user to use\n    '
        return self._path_open(path, 'wb', mime_type, compression_type)

    def open(self, path, mime_type='application/octet-stream', compression_type=CompressionTypes.AUTO):
        if False:
            for i in range(10):
                print('nop')
        'Returns a read channel for the given file path.\n\n    Args:\n      path: string path of the file object to be written to the system\n      mime_type: MIME type to specify the type of content in the file object\n      compression_type: Type of compression to be used for this object\n\n    Returns: file handle with a close function for the user to use\n    '
        return self._path_open(path, 'rb', mime_type, compression_type)

    def copy(self, source_file_names, destination_file_names):
        if False:
            for i in range(10):
                print('nop')
        'Recursively copy the file tree from the source to the destination\n\n    Args:\n      source_file_names: list of source file objects that needs to be copied\n      destination_file_names: list of destination of the new object\n\n    Raises:\n      ``BeamIOError``: if any of the copy operations fail\n    '
        err_msg = 'source_file_names and destination_file_names should be equal in length'
        assert len(source_file_names) == len(destination_file_names), err_msg

        def _copy_path(source, destination):
            if False:
                i = 10
                return i + 15
            'Recursively copy the file tree from the source to the destination\n      '
            if not destination.startswith(GCSFileSystem.GCS_PREFIX):
                raise ValueError('Destination %r must be GCS path.' % destination)
            if source.endswith('/'):
                self._gcsIO().copytree(source, destination)
            else:
                self._gcsIO().copy(source, destination)
        exceptions = {}
        for (source, destination) in zip(source_file_names, destination_file_names):
            try:
                _copy_path(source, destination)
            except Exception as e:
                exceptions[source, destination] = e
        if exceptions:
            raise BeamIOError('Copy operation failed', exceptions)

    def rename(self, source_file_names, destination_file_names):
        if False:
            i = 10
            return i + 15
        'Rename the files at the source list to the destination list.\n    Source and destination lists should be of the same size.\n\n    Args:\n      source_file_names: List of file paths that need to be moved\n      destination_file_names: List of destination_file_names for the files\n\n    Raises:\n      ``BeamIOError``: if any of the rename operations fail\n    '
        err_msg = 'source_file_names and destination_file_names should be equal in length'
        assert len(source_file_names) == len(destination_file_names), err_msg
        gcs_batches = []
        gcs_current_batch = []
        for (src, dest) in zip(source_file_names, destination_file_names):
            gcs_current_batch.append((src, dest))
            if len(gcs_current_batch) == self.CHUNK_SIZE:
                gcs_batches.append(gcs_current_batch)
                gcs_current_batch = []
        if gcs_current_batch:
            gcs_batches.append(gcs_current_batch)
        exceptions = {}
        for batch in gcs_batches:
            copy_statuses = self._gcsIO().copy_batch(batch)
            copy_succeeded = []
            for (src, dest, exception) in copy_statuses:
                if exception:
                    exceptions[src, dest] = exception
                else:
                    copy_succeeded.append((src, dest))
            delete_batch = [src for (src, dest) in copy_succeeded]
            delete_statuses = self._gcsIO().delete_batch(delete_batch)
            for (i, (src, exception)) in enumerate(delete_statuses):
                dest = copy_succeeded[i][1]
                if exception:
                    exceptions[src, dest] = exception
        if exceptions:
            raise BeamIOError('Rename operation failed', exceptions)

    def exists(self, path):
        if False:
            i = 10
            return i + 15
        'Check if the provided path exists on the FileSystem.\n\n    Args:\n      path: string path that needs to be checked.\n\n    Returns: boolean flag indicating if path exists\n    '
        return self._gcsIO().exists(path)

    def size(self, path):
        if False:
            for i in range(10):
                print('nop')
        "Get size of path on the FileSystem.\n\n    Args:\n      path: string path in question.\n\n    Returns: int size of path according to the FileSystem.\n\n    Raises:\n      ``BeamIOError``: if path doesn't exist.\n    "
        return self._gcsIO().size(path)

    def last_updated(self, path):
        if False:
            print('Hello World!')
        "Get UNIX Epoch time in seconds on the FileSystem.\n\n    Args:\n      path: string path of file.\n\n    Returns: float UNIX Epoch time\n\n    Raises:\n      ``BeamIOError``: if path doesn't exist.\n    "
        return self._gcsIO().last_updated(path)

    def checksum(self, path):
        if False:
            print('Hello World!')
        "Fetch checksum metadata of a file on the\n    :class:`~apache_beam.io.filesystem.FileSystem`.\n\n    Args:\n      path: string path of a file.\n\n    Returns: string containing checksum\n\n    Raises:\n      ``BeamIOError``: if path isn't a file or doesn't exist.\n    "
        try:
            return self._gcsIO().checksum(path)
        except Exception as e:
            raise BeamIOError('Checksum operation failed', {path: e})

    def metadata(self, path):
        if False:
            while True:
                i = 10
        "Fetch metadata fields of a file on the FileSystem.\n\n    Args:\n      path: string path of a file.\n\n    Returns:\n      :class:`~apache_beam.io.filesystem.FileMetadata`.\n\n    Raises:\n      ``BeamIOError``: if path isn't a file or doesn't exist.\n    "
        try:
            file_metadata = self._gcsIO()._status(path)
            return FileMetadata(path, file_metadata['size'], file_metadata['last_updated'])
        except Exception as e:
            raise BeamIOError('Metadata operation failed', {path: e})

    def delete(self, paths):
        if False:
            i = 10
            return i + 15
        'Deletes files or directories at the provided paths.\n    Directories will be deleted recursively.\n\n    Args:\n      paths: list of paths that give the file objects to be deleted\n    '

        def _delete_path(path):
            if False:
                return 10
            'Recursively delete the file or directory at the provided path.\n      '
            if path.endswith('/'):
                path_to_use = path + '*'
            else:
                path_to_use = path
            match_result = self.match([path_to_use])[0]
            statuses = self._gcsIO().delete_batch([m.path for m in match_result.metadata_list])
            failures = [e for (_, e) in statuses if e is not None]
            if failures:
                raise failures[0]
        exceptions = {}
        for path in paths:
            try:
                _delete_path(path)
            except Exception as e:
                exceptions[path] = e
        if exceptions:
            raise BeamIOError('Delete operation failed', exceptions)