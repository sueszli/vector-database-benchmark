"""Azure Blob Storage Implementation for accesing files on
Azure Blob Storage.
"""
from apache_beam.io.azure import blobstorageio
from apache_beam.io.filesystem import BeamIOError
from apache_beam.io.filesystem import CompressedFile
from apache_beam.io.filesystem import CompressionTypes
from apache_beam.io.filesystem import FileMetadata
from apache_beam.io.filesystem import FileSystem
__all__ = ['BlobStorageFileSystem']

class BlobStorageFileSystem(FileSystem):
    """An Azure Blob Storage ``FileSystem`` implementation for accesing files on
  Azure Blob Storage.
  """
    CHUNK_SIZE = blobstorageio.MAX_BATCH_OPERATION_SIZE
    AZURE_FILE_SYSTEM_PREFIX = 'azfs://'

    def __init__(self, pipeline_options):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(pipeline_options)
        self._pipeline_options = pipeline_options

    @classmethod
    def scheme(cls):
        if False:
            i = 10
            return i + 15
        'URI scheme for the FileSystem\n    '
        return 'azfs'

    def join(self, basepath, *paths):
        if False:
            for i in range(10):
                print('nop')
        'Join two or more pathname components for the filesystem\n\n    Args:\n      basepath: string path of the first component of the path\n      paths: path components to be added\n\n    Returns: full path after combining all the passed components\n    '
        if not basepath.startswith(BlobStorageFileSystem.AZURE_FILE_SYSTEM_PREFIX):
            raise ValueError('Basepath %r must be an Azure Blob Storage path.' % basepath)
        path = basepath
        for p in paths:
            path = path.rstrip('/') + '/' + p.lstrip('/')
        return path

    def split(self, path):
        if False:
            for i in range(10):
                print('nop')
        'Splits the given path into two parts.\n\n    Splits the path into a pair (head, tail) such that tail contains the last\n    component of the path and head contains everything up to that.\n    For file-systems other than the local file-system, head should include the\n    prefix.\n\n    Args:\n      path: path as a string\n\n    Returns:\n      a pair of path components as strings.\n    '
        path = path.strip()
        if not path.startswith(BlobStorageFileSystem.AZURE_FILE_SYSTEM_PREFIX):
            raise ValueError('Path %r must be Azure Blob Storage path.' % path)
        prefix_len = len(BlobStorageFileSystem.AZURE_FILE_SYSTEM_PREFIX)
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
            return 10
        'Recursively create directories for the provided path.\n\n    Args:\n      path: string path of the directory structure that should be created\n\n    Raises:\n      IOError: if leaf directory already exists.\n    '
        pass

    def has_dirs(self):
        if False:
            return 10
        'Whether this FileSystem supports directories.'
        return False

    def _list(self, dir_or_prefix):
        if False:
            for i in range(10):
                print('nop')
        "List files in a location.\n    Listing is non-recursive (for filesystems that support directories).\n    Args:\n      dir_or_prefix: (string) A directory or location prefix (for filesystems\n        that don't have directories).\n    Returns:\n      Generator of ``FileMetadata`` objects.\n    Raises:\n      ``BeamIOError``: if listing fails, but not if no files were found.\n    "
        try:
            for (path, (size, updated)) in self._blobstorageIO().list_files(dir_or_prefix, with_metadata=True):
                yield FileMetadata(path, size, updated)
        except Exception as e:
            raise BeamIOError('List operation failed', {dir_or_prefix: e})

    def _blobstorageIO(self):
        if False:
            for i in range(10):
                print('nop')
        return blobstorageio.BlobStorageIO(pipeline_options=self._pipeline_options)

    def _path_open(self, path, mode, mime_type='application/octet-stream', compression_type=CompressionTypes.AUTO):
        if False:
            for i in range(10):
                print('nop')
        'Helper functions to open a file in the provided mode.\n    '
        compression_type = FileSystem._get_compression_type(path, compression_type)
        mime_type = CompressionTypes.mime_type(compression_type, mime_type)
        raw_file = self._blobstorageIO().open(path, mode, mime_type=mime_type)
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
            while True:
                i = 10
        'Returns a read channel for the given file path.\n\n    Args:\n      path: string path of the file object to be read\n      mime_type: MIME type to specify the type of content in the file object\n      compression_type: Type of compression to be used for this object\n\n    Returns: file handle with a close function for the user to use\n    '
        return self._path_open(path, 'rb', mime_type, compression_type)

    def copy(self, source_file_names, destination_file_names):
        if False:
            for i in range(10):
                print('nop')
        'Recursively copy the file tree from the source to the destination\n\n    Args:\n      source_file_names: list of source file objects that needs to be copied\n      destination_file_names: list of destination of the new object\n\n    Raises:\n      ``BeamIOError``: if any of the copy operations fail\n    '
        if not len(source_file_names) == len(destination_file_names):
            message = 'Unable to copy unequal number of sources and destinations.'
            raise BeamIOError(message)
        src_dest_pairs = list(zip(source_file_names, destination_file_names))
        return self._blobstorageIO().copy_paths(src_dest_pairs)

    def rename(self, source_file_names, destination_file_names):
        if False:
            return 10
        'Rename the files at the source list to the destination list.\n    Source and destination lists should be of the same size.\n\n    Args:\n      source_file_names: List of file paths that need to be moved\n      destination_file_names: List of destination_file_names for the files\n\n    Raises:\n      ``BeamIOError``: if any of the rename operations fail\n    '
        if not len(source_file_names) == len(destination_file_names):
            message = 'Unable to rename unequal number of sources and destinations.'
            raise BeamIOError(message)
        src_dest_pairs = list(zip(source_file_names, destination_file_names))
        results = self._blobstorageIO().rename_files(src_dest_pairs)
        exceptions = {(src, dest): error for (src, dest, error) in results if error is not None}
        if exceptions:
            raise BeamIOError('Rename operation failed.', exceptions)

    def exists(self, path):
        if False:
            return 10
        'Check if the provided path exists on the FileSystem.\n\n    Args:\n      path: string path that needs to be checked.\n\n    Returns: boolean flag indicating if path exists\n    '
        try:
            return self._blobstorageIO().exists(path)
        except Exception as e:
            raise BeamIOError('Exists operation failed', {path: e})

    def size(self, path):
        if False:
            for i in range(10):
                print('nop')
        "Get size in bytes of a file on the FileSystem.\n\n    Args:\n      path: string filepath of file.\n\n    Returns: int size of file according to the FileSystem.\n\n    Raises:\n      ``BeamIOError``: if path doesn't exist.\n    "
        try:
            return self._blobstorageIO().size(path)
        except Exception as e:
            raise BeamIOError('Size operation failed', {path: e})

    def last_updated(self, path):
        if False:
            while True:
                i = 10
        "Get UNIX Epoch time in seconds on the FileSystem.\n\n    Args:\n      path: string path of file.\n\n    Returns: float UNIX Epoch time\n\n    Raises:\n      ``BeamIOError``: if path doesn't exist.\n    "
        try:
            return self._blobstorageIO().last_updated(path)
        except Exception as e:
            raise BeamIOError('Last updated operation failed', {path: e})

    def checksum(self, path):
        if False:
            for i in range(10):
                print('nop')
        "Fetch checksum metadata of a file on the\n    :class:`~apache_beam.io.filesystem.FileSystem`.\n\n    Args:\n      path: string path of a file.\n\n    Returns: string containing checksum\n\n    Raises:\n      ``BeamIOError``: if path isn't a file or doesn't exist.\n    "
        try:
            return self._blobstorageIO().checksum(path)
        except Exception as e:
            raise BeamIOError('Checksum operation failed', {path, e})

    def metadata(self, path):
        if False:
            print('Hello World!')
        "Fetch metadata fields of a file on the FileSystem.\n\n    Args:\n      path: string path of a file.\n\n    Returns:\n      :class:`~apache_beam.io.filesystem.FileMetadata`.\n\n    Raises:\n      ``BeamIOError``: if path isn't a file or doesn't exist.\n    "
        try:
            file_metadata = self._blobstorageIO()._status(path)
            return FileMetadata(path, file_metadata['size'], file_metadata['last_updated'])
        except Exception as e:
            raise BeamIOError('Metadata operation failed', {path: e})

    def delete(self, paths):
        if False:
            return 10
        'Deletes files or directories at the provided paths.\n    Directories will be deleted recursively.\n\n    Args:\n      paths: list of paths that give the file objects to be deleted\n\n    Raises:\n      ``BeamIOError``: if any of the delete operations fail\n    '
        results = self._blobstorageIO().delete_paths(paths)
        exceptions = {path: error for (path, error) in results.items() if error is not None}
        if exceptions:
            raise BeamIOError('Delete operation failed', exceptions)