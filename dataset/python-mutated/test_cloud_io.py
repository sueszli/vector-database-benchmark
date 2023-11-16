import os
import fsspec
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem
from lightning.fabric.utilities.cloud_io import _is_dir, get_filesystem

def test_get_filesystem_custom_filesystem():
    if False:
        return 10
    _DUMMY_PRFEIX = 'dummy'

    class DummyFileSystem(LocalFileSystem):
        ...
    fsspec.register_implementation(_DUMMY_PRFEIX, DummyFileSystem, clobber=True)
    output_file = os.path.join(f'{_DUMMY_PRFEIX}://', 'tmpdir/tmp_file')
    assert isinstance(get_filesystem(output_file), DummyFileSystem)

def test_get_filesystem_local_filesystem():
    if False:
        while True:
            i = 10
    assert isinstance(get_filesystem('tmpdir/tmp_file'), LocalFileSystem)

def test_is_dir_with_local_filesystem(tmp_path):
    if False:
        return 10
    fs = LocalFileSystem()
    tmp_existing_directory = tmp_path
    tmp_non_existing_directory = tmp_path / 'non_existing'
    assert _is_dir(fs, tmp_existing_directory)
    assert not _is_dir(fs, tmp_non_existing_directory)

def test_is_dir_with_object_storage_filesystem():
    if False:
        i = 10
        return i + 15

    class MockAzureBlobFileSystem(AbstractFileSystem):

        def isdir(self, path):
            if False:
                for i in range(10):
                    print('nop')
            return path.startswith('azure://') and (not path.endswith('.txt'))

        def isfile(self, path):
            if False:
                i = 10
                return i + 15
            return path.startswith('azure://') and path.endswith('.txt')

    class MockGCSFileSystem(AbstractFileSystem):

        def isdir(self, path):
            if False:
                while True:
                    i = 10
            return path.startswith('gcs://') and (not path.endswith('.txt'))

        def isfile(self, path):
            if False:
                return 10
            return path.startswith('gcs://') and path.endswith('.txt')

    class MockS3FileSystem(AbstractFileSystem):

        def isdir(self, path):
            if False:
                return 10
            return path.startswith('s3://') and (not path.endswith('.txt'))

        def isfile(self, path):
            if False:
                for i in range(10):
                    print('nop')
            return path.startswith('s3://') and path.endswith('.txt')
    fsspec.register_implementation('azure', MockAzureBlobFileSystem, clobber=True)
    fsspec.register_implementation('gcs', MockGCSFileSystem, clobber=True)
    fsspec.register_implementation('s3', MockS3FileSystem, clobber=True)
    azure_directory = 'azure://container/directory/'
    azure_file = 'azure://container/file.txt'
    gcs_directory = 'gcs://bucket/directory/'
    gcs_file = 'gcs://bucket/file.txt'
    s3_directory = 's3://bucket/directory/'
    s3_file = 's3://bucket/file.txt'
    assert _is_dir(get_filesystem(azure_directory), azure_directory)
    assert _is_dir(get_filesystem(azure_directory), azure_directory, strict=True)
    assert not _is_dir(get_filesystem(azure_directory), azure_file)
    assert not _is_dir(get_filesystem(azure_directory), azure_file, strict=True)
    assert _is_dir(get_filesystem(gcs_directory), gcs_directory)
    assert _is_dir(get_filesystem(gcs_directory), gcs_directory, strict=True)
    assert not _is_dir(get_filesystem(gcs_directory), gcs_file)
    assert not _is_dir(get_filesystem(gcs_directory), gcs_file, strict=True)
    assert _is_dir(get_filesystem(s3_directory), s3_directory)
    assert _is_dir(get_filesystem(s3_directory), s3_directory, strict=True)
    assert not _is_dir(get_filesystem(s3_directory), s3_file)
    assert not _is_dir(get_filesystem(s3_directory), s3_file, strict=True)