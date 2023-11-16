"""Local File system implementation for accessing files on disk."""
import io
import os
import shutil
from typing import BinaryIO
from apache_beam.io.filesystem import BeamIOError
from apache_beam.io.filesystem import CompressedFile
from apache_beam.io.filesystem import CompressionTypes
from apache_beam.io.filesystem import FileMetadata
from apache_beam.io.filesystem import FileSystem
__all__ = ['LocalFileSystem']

class LocalFileSystem(FileSystem):
    """A Local ``FileSystem`` implementation for accessing files on disk.
  """

    @classmethod
    def scheme(cls):
        if False:
            while True:
                i = 10
        'URI scheme for the FileSystem\n    '
        return None

    def join(self, basepath, *paths):
        if False:
            i = 10
            return i + 15
        'Join two or more pathname components for the filesystem\n\n    Args:\n      basepath: string path of the first component of the path\n      paths: path components to be added\n\n    Returns: full path after combining all the passed components\n    '
        return os.path.join(basepath, *paths)

    def split(self, path):
        if False:
            while True:
                i = 10
        'Splits the given path into two parts.\n\n    Splits the path into a pair (head, tail) such that tail contains the last\n    component of the path and head contains everything up to that.\n\n    Args:\n      path: path as a string\n    Returns:\n      a pair of path components as strings.\n    '
        return os.path.split(os.path.abspath(path))

    def mkdirs(self, path):
        if False:
            i = 10
            return i + 15
        'Recursively create directories for the provided path.\n\n    Args:\n      path: string path of the directory structure that should be created\n\n    Raises:\n      IOError: if leaf directory already exists.\n    '
        try:
            os.makedirs(path)
        except OSError as err:
            raise IOError(err)

    def has_dirs(self):
        if False:
            i = 10
            return i + 15
        'Whether this FileSystem supports directories.'
        return True

    def _url_dirname(self, url_or_path):
        if False:
            for i in range(10):
                print('nop')
        'Pass through to os.path.dirname.\n\n    This version uses os.path instead of posixpath to be compatible with the\n    host OS.\n\n    Args:\n      url_or_path: A string in the form of /some/path.\n    '
        return os.path.dirname(url_or_path)

    def _list(self, dir_or_prefix):
        if False:
            print('Hello World!')
        "List files in a location.\n\n    Listing is non-recursive, for filesystems that support directories.\n\n    Args:\n      dir_or_prefix: (string) A directory or location prefix (for filesystems\n        that don't have directories).\n\n    Returns:\n      Generator of ``FileMetadata`` objects.\n\n    Raises:\n      ``BeamIOError``: if listing fails, but not if no files were found.\n    "
        if not self.exists(dir_or_prefix):
            return

        def list_files(root):
            if False:
                for i in range(10):
                    print('nop')
            for (dirpath, _, files) in os.walk(root):
                for filename in files:
                    yield self.join(dirpath, filename)
        try:
            for f in list_files(dir_or_prefix):
                try:
                    yield FileMetadata(f, os.path.getsize(f), os.path.getmtime(f))
                except OSError:
                    pass
        except Exception as e:
            raise BeamIOError('List operation failed', {dir_or_prefix: e})

    def _path_open(self, path, mode, mime_type='application/octet-stream', compression_type=CompressionTypes.AUTO):
        if False:
            return 10
        'Helper functions to open a file in the provided mode.\n    '
        compression_type = FileSystem._get_compression_type(path, compression_type)
        raw_file = io.open(path, mode)
        if compression_type == CompressionTypes.UNCOMPRESSED:
            return raw_file
        else:
            return CompressedFile(raw_file, compression_type=compression_type)

    def create(self, path, mime_type='application/octet-stream', compression_type=CompressionTypes.AUTO):
        if False:
            i = 10
            return i + 15
        'Returns a write channel for the given file path.\n\n    Args:\n      path: string path of the file object to be written to the system\n      mime_type: MIME type to specify the type of content in the file object\n      compression_type: Type of compression to be used for this object\n\n    Returns: file handle with a close function for the user to use\n    '
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return self._path_open(path, 'wb', mime_type, compression_type)

    def open(self, path, mime_type='application/octet-stream', compression_type=CompressionTypes.AUTO):
        if False:
            return 10
        'Returns a read channel for the given file path.\n\n    Args:\n      path: string path of the file object to be written to the system\n      mime_type: MIME type to specify the type of content in the file object\n      compression_type: Type of compression to be used for this object\n\n    Returns: file handle with a close function for the user to use\n    '
        return self._path_open(path, 'rb', mime_type, compression_type)

    def copy(self, source_file_names, destination_file_names):
        if False:
            i = 10
            return i + 15
        'Recursively copy the file tree from the source to the destination\n\n    Args:\n      source_file_names: list of source file objects that needs to be copied\n      destination_file_names: list of destination of the new object\n\n    Raises:\n      ``BeamIOError``: if any of the copy operations fail\n    '
        err_msg = 'source_file_names and destination_file_names should be equal in length'
        assert len(source_file_names) == len(destination_file_names), err_msg

        def _copy_path(source, destination):
            if False:
                i = 10
                return i + 15
            'Recursively copy the file tree from the source to the destination\n      '
            try:
                if os.path.exists(destination):
                    if os.path.isdir(destination):
                        shutil.rmtree(destination)
                    else:
                        os.remove(destination)
                if os.path.isdir(source):
                    shutil.copytree(source, destination)
                else:
                    shutil.copy2(source, destination)
            except OSError as err:
                raise IOError(err)
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
            return 10
        'Rename the files at the source list to the destination list.\n    Source and destination lists should be of the same size.\n\n    Args:\n      source_file_names: List of file paths that need to be moved\n      destination_file_names: List of destination_file_names for the files\n\n    Raises:\n      ``BeamIOError``: if any of the rename operations fail\n    '
        err_msg = 'source_file_names and destination_file_names should be equal in length'
        assert len(source_file_names) == len(destination_file_names), err_msg

        def _rename_file(source, destination):
            if False:
                while True:
                    i = 10
            'Rename a single file object'
            try:
                os.rename(source, destination)
            except OSError as err:
                raise IOError(err)
        exceptions = {}
        for (source, destination) in zip(source_file_names, destination_file_names):
            try:
                _rename_file(source, destination)
            except Exception as e:
                exceptions[source, destination] = e
        if exceptions:
            raise BeamIOError('Rename operation failed', exceptions)

    def exists(self, path):
        if False:
            print('Hello World!')
        'Check if the provided path exists on the FileSystem.\n\n    Args:\n      path: string path that needs to be checked.\n\n    Returns: boolean flag indicating if path exists\n    '
        return os.path.exists(path)

    def size(self, path):
        if False:
            return 10
        "Get size of path on the FileSystem.\n\n    Args:\n      path: string path in question.\n\n    Returns: int size of path according to the FileSystem.\n\n    Raises:\n      ``BeamIOError``: if path doesn't exist.\n    "
        try:
            return os.path.getsize(path)
        except Exception as e:
            raise BeamIOError('Size operation failed', {path: e})

    def last_updated(self, path):
        if False:
            print('Hello World!')
        "Get UNIX Epoch time in seconds on the FileSystem.\n\n    Args:\n      path: string path of file.\n\n    Returns: float UNIX Epoch time\n\n    Raises:\n      ``BeamIOError``: if path doesn't exist.\n    "
        if not self.exists(path):
            raise BeamIOError('Path does not exist: %s' % path)
        return os.path.getmtime(path)

    def checksum(self, path):
        if False:
            print('Hello World!')
        "Fetch checksum metadata of a file on the\n    :class:`~apache_beam.io.filesystem.FileSystem`.\n\n    Args:\n      path: string path of a file.\n\n    Returns: string containing file size.\n\n    Raises:\n      ``BeamIOError``: if path isn't a file or doesn't exist.\n    "
        if not self.exists(path):
            raise BeamIOError('Path does not exist: %s' % path)
        return str(os.path.getsize(path))

    def metadata(self, path):
        if False:
            i = 10
            return i + 15
        "Fetch metadata fields of a file on the FileSystem.\n\n    Args:\n      path: string path of a file.\n\n    Returns:\n      :class:`~apache_beam.io.filesystem.FileMetadata`.\n\n    Raises:\n      ``BeamIOError``: if path isn't a file or doesn't exist.\n    "
        if not self.exists(path):
            raise BeamIOError('Path does not exist: %s' % path)
        return FileMetadata(path, os.path.getsize(path), os.path.getmtime(path))

    def delete(self, paths):
        if False:
            while True:
                i = 10
        'Deletes files or directories at the provided paths.\n    Directories will be deleted recursively.\n\n    Args:\n      paths: list of paths that give the file objects to be deleted\n\n    Raises:\n      ``BeamIOError``: if any of the delete operations fail\n    '

        def _delete_path(path):
            if False:
                return 10
            'Recursively delete the file or directory at the provided path.\n      '
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            except OSError as err:
                raise IOError(err)
        exceptions = {}

        def try_delete(path):
            if False:
                for i in range(10):
                    print('nop')
            try:
                _delete_path(path)
            except Exception as e:
                exceptions[path] = e
        for match_result in self.match(paths):
            metadata_list = match_result.metadata_list
            if not metadata_list:
                exceptions[match_result.pattern] = IOError('No files found to delete under: %s' % match_result.pattern)
            for metadata in match_result.metadata_list:
                try_delete(metadata.path)
        if exceptions:
            raise BeamIOError('Delete operation failed', exceptions)