"""Tests for cloud_storage_services."""
from __future__ import annotations
from core.platform.storage import cloud_storage_services
from core.tests import test_utils
from google.cloud import storage
from typing import Dict, List, Optional

class MockClient:

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self.buckets: Dict[str, MockBucket] = {}

    def get_bucket(self, bucket_name: str) -> MockBucket:
        if False:
            i = 10
            return i + 15
        'Gets mocked Cloud Storage bucket.\n\n        Args:\n            bucket_name: str. The name of the storage bucket to return.\n\n        Returns:\n            MockBucket. Cloud Storage bucket.\n        '
        return self.buckets[bucket_name]

    def list_blobs(self, bucket: MockBucket, prefix: Optional[str]=None) -> List[MockBlob]:
        if False:
            return 10
        'Lists all blobs with some prefix.\n\n        Args:\n            bucket: MockBucket. The mock GCS bucket.\n            prefix: str|None. The prefix which the blobs should have.\n\n        Returns:\n            list(MockBlob). A list of blobs.\n        '
        return [blob for (name, blob) in bucket.blobs.items() if prefix is None or name.startswith(prefix)]

class MockBucket:

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        self.blobs: Dict[str, MockBlob] = {}

    def get_blob(self, filepath: str) -> Optional[MockBlob]:
        if False:
            i = 10
            return i + 15
        "Gets a blob object by filepath. This will return None if the\n        blob doesn't exist.\n\n        Args:\n            filepath: str. Filepath of the blob.\n\n        Returns:\n            MockBlob. The blob.\n        "
        return self.blobs.get(filepath)

    def copy_blob(self, src_blob: MockBlob, bucket: MockBucket, new_name: Optional[str]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Copies the given blob to the given bucket, optionally\n        with a new name.\n\n        Args:\n            src_blob: MockBlob. Source blob which should be copied.\n            bucket: MockBucket. The target bucket into which the blob\n                should be copied.\n            new_name: str|None. The new name of the blob. When None the name\n                of src_blob will be used.\n        '
        blob = bucket.blob(new_name if new_name else src_blob.filepath)
        blob.upload_from_string(src_blob.download_as_bytes(), content_type=src_blob.content_type)

    def blob(self, filepath: str) -> MockBlob:
        if False:
            return 10
        'Creates new blob in this bucket.\n\n        Args:\n            filepath: str. Filepath of the blob.\n\n        Returns:\n            MockBlob. The newly created blob.\n        '
        blob = MockBlob(filepath)
        self.blobs[filepath] = blob
        return blob

class MockBlob:
    __slots__ = ('filepath', 'raw_bytes', 'content_type', 'deleted')

    def __init__(self, filepath: str) -> None:
        if False:
            while True:
                i = 10
        self.filepath = filepath
        self.deleted = False
        self.raw_bytes = b''
        self.content_type: Optional[str] = None

    def upload_from_string(self, raw_bytes: bytes, content_type: Optional[str]=None) -> None:
        if False:
            while True:
                i = 10
        'Sets the blob data.\n\n        Args:\n            raw_bytes: bytes. The blob data.\n            content_type: str. The content type of the blob.\n        '
        self.raw_bytes = raw_bytes
        self.content_type = content_type

    def download_as_bytes(self) -> bytes:
        if False:
            while True:
                i = 10
        'Gets the blob data as bytes.\n\n        Returns:\n            bytes. The blob data.\n        '
        return self.raw_bytes

    def delete(self) -> None:
        if False:
            while True:
                i = 10
        'Marks the blob as deleted.'
        self.deleted = True

class CloudStorageServicesTests(test_utils.TestBase):

    def setUp(self) -> None:
        if False:
            return 10
        super().setUp()
        self.client = MockClient()
        self.bucket_1 = MockBucket()
        self.bucket_2 = MockBucket()
        self.client.buckets['bucket_1'] = self.bucket_1
        self.client.buckets['bucket_2'] = self.bucket_2
        self.get_client_swap = self.swap(storage, 'Client', lambda : self.client)
        self.get_bucket_swap = self.swap(cloud_storage_services, '_get_bucket', self.client.get_bucket)

    def test_isfile_when_file_exists_returns_true(self) -> None:
        if False:
            print('Hello World!')
        self.bucket_1.blobs['path/to/file.txt'] = MockBlob('path/to/file.txt')
        with self.get_client_swap:
            self.assertTrue(cloud_storage_services.isfile('bucket_1', 'path/to/file.txt'))

    def test_isfile_when_file_does_not_exist_returns_false(self) -> None:
        if False:
            i = 10
            return i + 15
        with self.get_bucket_swap:
            self.assertFalse(cloud_storage_services.isfile('bucket_1', 'path/to/file.txt'))

    def test_get_when_file_exists_returns_file_contents(self) -> None:
        if False:
            return 10
        self.bucket_1.blobs['path/to/file.txt'] = MockBlob('path/to/file.txt')
        self.bucket_1.blobs['path/to/file.txt'].upload_from_string(b'abc')
        self.bucket_2.blobs['path/file.txt'] = MockBlob('path/file.txt')
        self.bucket_2.blobs['path/file.txt'].upload_from_string(b'xyz')
        with self.get_bucket_swap:
            self.assertEqual(cloud_storage_services.get('bucket_1', 'path/to/file.txt'), b'abc')
            self.assertEqual(cloud_storage_services.get('bucket_2', 'path/file.txt'), b'xyz')

    def test_commit_saves_file_into_bucket(self) -> None:
        if False:
            i = 10
            return i + 15
        with self.get_bucket_swap:
            cloud_storage_services.commit('bucket_1', 'path/to/file.txt', b'abc', 'audio/mpeg')
            cloud_storage_services.commit('bucket_2', 'path/file.txt', b'xyz', 'image/png')
        self.assertEqual(self.bucket_1.blobs['path/to/file.txt'].raw_bytes, b'abc')
        self.assertEqual(self.bucket_1.blobs['path/to/file.txt'].content_type, 'audio/mpeg')
        self.assertEqual(self.bucket_2.blobs['path/file.txt'].raw_bytes, b'xyz')
        self.assertEqual(self.bucket_2.blobs['path/file.txt'].content_type, 'image/png')

    def test_delete_removes_file_from_bucket(self) -> None:
        if False:
            i = 10
            return i + 15
        self.bucket_1.blobs['path/to/file.txt'] = MockBlob('path/to/file.txt')
        self.bucket_1.blobs['path/to/file.txt'].upload_from_string(b'abc')
        self.bucket_2.blobs['path/file.txt'] = MockBlob('path/file.txt')
        self.bucket_2.blobs['path/file.txt'].upload_from_string(b'xyz')
        self.assertFalse(self.bucket_1.blobs['path/to/file.txt'].deleted)
        self.assertFalse(self.bucket_2.blobs['path/file.txt'].deleted)
        with self.get_bucket_swap:
            cloud_storage_services.delete('bucket_1', 'path/to/file.txt')
            cloud_storage_services.delete('bucket_2', 'path/file.txt')
        self.assertTrue(self.bucket_1.blobs['path/to/file.txt'].deleted)
        self.assertTrue(self.bucket_2.blobs['path/file.txt'].deleted)

    def test_copy_creates_copy_in_the_bucket(self) -> None:
        if False:
            return 10
        self.bucket_1.blobs['path/to/file.txt'] = MockBlob('path/to/file.txt')
        self.bucket_1.blobs['path/to/file.txt'].upload_from_string(b'abc', content_type='audio/mpeg')
        self.bucket_2.blobs['path/file.txt'] = MockBlob('path/file.txt')
        self.bucket_2.blobs['path/file.txt'].upload_from_string(b'xyz', content_type='image/png')
        with self.get_bucket_swap:
            cloud_storage_services.copy('bucket_1', 'path/to/file.txt', 'other/path/to/file.txt')
            cloud_storage_services.copy('bucket_2', 'path/file.txt', 'other/path/file.txt')
        self.assertEqual(self.bucket_1.blobs['other/path/to/file.txt'].raw_bytes, b'abc')
        self.assertEqual(self.bucket_1.blobs['other/path/to/file.txt'].content_type, 'audio/mpeg')
        self.assertEqual(self.bucket_2.blobs['other/path/file.txt'].raw_bytes, b'xyz')
        self.assertEqual(self.bucket_2.blobs['other/path/file.txt'].content_type, 'image/png')

    def test_listdir_lists_files_with_provided_prefix(self) -> None:
        if False:
            print('Hello World!')
        self.bucket_1.blobs['path/to/file.txt'] = MockBlob('path/to/file.txt')
        self.bucket_1.blobs['path/to/file.txt'].upload_from_string(b'abc')
        self.bucket_1.blobs['pathto/file.txt'] = MockBlob('pathto/file.txt')
        self.bucket_1.blobs['pathto/file.txt'].upload_from_string(b'def')
        self.bucket_1.blobs['path/to/file2.txt'] = MockBlob('path/to/file2.txt')
        self.bucket_1.blobs['path/to/file2.txt'].upload_from_string(b'ghi')
        with self.get_client_swap, self.get_bucket_swap:
            path_blobs = cloud_storage_services.listdir('bucket_1', 'path')
            path_slash_blobs = cloud_storage_services.listdir('bucket_1', 'path/')
        self.assertItemsEqual(path_blobs, [self.bucket_1.blobs['path/to/file.txt'], self.bucket_1.blobs['pathto/file.txt'], self.bucket_1.blobs['path/to/file2.txt']])
        self.assertItemsEqual(path_slash_blobs, [self.bucket_1.blobs['path/to/file.txt'], self.bucket_1.blobs['path/to/file2.txt']])