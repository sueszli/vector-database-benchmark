"""Tests for cloud_storage_emulator."""
from __future__ import annotations
from core.platform.storage import cloud_storage_emulator
from core.tests import test_utils

class BlobUnitTests(test_utils.TestBase):
    """Tests for EmulatorBlob."""

    def test_init_blob_with_str_creates_blob(self) -> None:
        if False:
            return 10
        blob = cloud_storage_emulator.EmulatorBlob('name', 'string', 'image/png')
        self.assertEqual(blob.name, 'name')
        self.assertEqual(blob.download_as_bytes(), b'string')
        self.assertEqual(blob.content_type, 'image/png')

    def test_init_blob_with_bytes_creates_blob(self) -> None:
        if False:
            print('Hello World!')
        blob = cloud_storage_emulator.EmulatorBlob('name', b'string', 'image/png')
        self.assertEqual(blob.name, 'name')
        self.assertEqual(blob.download_as_bytes(), b'string')
        self.assertEqual(blob.content_type, 'image/png')

    def test_init_blob_with_none_content_type_creates_blob(self) -> None:
        if False:
            print('Hello World!')
        blob = cloud_storage_emulator.EmulatorBlob('name', 'string', None)
        self.assertEqual(blob.name, 'name')
        self.assertEqual(blob.download_as_bytes(), b'string')
        self.assertEqual(blob.content_type, 'application/octet-stream')

    def test_init_blob_with_content_type_audio_creates_blob(self) -> None:
        if False:
            return 10
        blob = cloud_storage_emulator.EmulatorBlob('name', 'string', 'audio/mp3')
        self.assertEqual(blob.name, 'name')
        self.assertEqual(blob.download_as_bytes(), b'string')
        self.assertEqual(blob.content_type, 'audio/mp3')

    def test_init_blob_with_content_type_images_webp_creates_blob(self) -> None:
        if False:
            while True:
                i = 10
        blob = cloud_storage_emulator.EmulatorBlob('name', 'string', 'image/webp')
        self.assertEqual(blob.name, 'name')
        self.assertEqual(blob.download_as_bytes(), b'string')
        self.assertEqual(blob.content_type, 'image/webp')

    def test_init_blob_with_wrong_mimetype_raise_exception(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(Exception, 'Content type contains unknown MIME type.'):
            cloud_storage_emulator.EmulatorBlob('name', b'string', 'png')

    def test_create_copy_creates_identical_copy(self) -> None:
        if False:
            return 10
        orig_blob = cloud_storage_emulator.EmulatorBlob('name', 'string', 'image/png')
        copy_blob = cloud_storage_emulator.EmulatorBlob.create_copy(orig_blob, 'new')
        self.assertNotEqual(orig_blob, copy_blob)
        self.assertNotEqual(orig_blob.name, copy_blob.name)
        self.assertEqual(orig_blob.download_as_bytes(), copy_blob.download_as_bytes())
        self.assertEqual(orig_blob.content_type, copy_blob.content_type)

    def test_to_dict_returns_correct_dictionary(self) -> None:
        if False:
            return 10
        blob = cloud_storage_emulator.EmulatorBlob('name', b'string', 'image/png')
        self.assertEqual(blob.to_dict(), {b'name': b'name', b'raw_bytes': b'string', b'content_type': b'image/png'})

    def test_from_dict_returns_blob(self) -> None:
        if False:
            return 10
        blob = cloud_storage_emulator.EmulatorBlob('name', b'string', 'image/png')
        self.assertEqual(blob, cloud_storage_emulator.EmulatorBlob.from_dict({b'name': b'name', b'raw_bytes': b'string', b'content_type': b'image/png'}))

    def test_compare_blob_and_int_is_false(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        orig_blob = cloud_storage_emulator.EmulatorBlob('name', 'string', 'image/png')
        self.assertFalse(orig_blob == 1)

    def test_repr_returns_correct_string_representation(self) -> None:
        if False:
            i = 10
            return i + 15
        orig_blob = cloud_storage_emulator.EmulatorBlob('name', 'string', 'image/png')
        self.assertEqual(repr(orig_blob), 'EmulatorBlob(name=name, content_type=image/png)')

class CloudStorageEmulatorUnitTests(test_utils.TestBase):
    """Tests for CloudStorageEmulator."""

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        super().setUp()
        self.emulator = cloud_storage_emulator.CloudStorageEmulator()
        self.emulator.namespace = 'namespace'
        self.emulator.reset()
        self.blob1 = cloud_storage_emulator.EmulatorBlob('/file/path.png', b'data', 'image/png')
        self.blob2 = cloud_storage_emulator.EmulatorBlob('/file/path2.png', b'data2', 'image/png')
        self.blob3 = cloud_storage_emulator.EmulatorBlob('/different/path.png', b'data2', 'image/png')

    def tearDown(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().tearDown()
        self.emulator.reset()

    def test_get_blob_retrieves_correct_blob_from_redis(self) -> None:
        if False:
            print('Hello World!')
        cloud_storage_emulator.REDIS_CLIENT.hset('namespace:/file/path.png', mapping=self.blob1.to_dict())
        self.assertEqual(self.emulator.get_blob('/file/path.png'), self.blob1)

    def test_upload_blob_saves_correct_blob_to_redis(self) -> None:
        if False:
            return 10
        self.emulator.upload_blob('/file/path.png', self.blob1)
        self.assertEqual(cloud_storage_emulator.EmulatorBlob.from_dict(cloud_storage_emulator.REDIS_CLIENT.hgetall('namespace:/file/path.png')), self.blob1)

    def test_delete_blob_removes_blob_from_redis(self) -> None:
        if False:
            while True:
                i = 10
        cloud_storage_emulator.REDIS_CLIENT.hset('namespace:/file/path.png', mapping=self.blob1.to_dict())
        self.emulator.delete_blob('/file/path.png')
        self.assertIsNone(cloud_storage_emulator.REDIS_CLIENT.get('/file/path.png'))

    def test_copy_blob_saves_copy_of_blob_to_redis(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        cloud_storage_emulator.REDIS_CLIENT.hset('namespace:/file/path.png', mapping=self.blob1.to_dict())
        self.emulator.copy_blob(cloud_storage_emulator.EmulatorBlob.from_dict(cloud_storage_emulator.REDIS_CLIENT.hgetall('namespace:/file/path.png')), '/different/path2.png')
        orig_blob = self.emulator.get_blob('/file/path.png')
        assert orig_blob is not None
        copy_blob = self.emulator.get_blob('/different/path2.png')
        assert copy_blob is not None
        self.assertNotEqual(orig_blob.name, copy_blob.name)
        self.assertEqual(orig_blob.download_as_bytes(), copy_blob.download_as_bytes())
        self.assertEqual(orig_blob.content_type, copy_blob.content_type)

    def test_list_blobs_returns_list_of_blobs_with_prefix(self) -> None:
        if False:
            print('Hello World!')
        cloud_storage_emulator.REDIS_CLIENT.hset('namespace:/file/path.png', mapping=self.blob1.to_dict())
        cloud_storage_emulator.REDIS_CLIENT.hset('namespace:/file/path2.png', mapping=self.blob2.to_dict())
        cloud_storage_emulator.REDIS_CLIENT.hset('namespace:/different/path.png', mapping=self.blob3.to_dict())
        self.assertItemsEqual(self.emulator.list_blobs('/'), [self.blob1, self.blob2, self.blob3])
        self.assertItemsEqual(self.emulator.list_blobs('/file'), [self.blob1, self.blob2])
        self.assertItemsEqual(self.emulator.list_blobs('/different'), [self.blob3])

    def test_reset_removes_all_values_from_redis(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        cloud_storage_emulator.REDIS_CLIENT.hset('namespace:/file/path.png', mapping=self.blob1.to_dict())
        self.emulator.reset()
        self.assertEqual(list(cloud_storage_emulator.REDIS_CLIENT.scan_iter(match='namespace:*')), [])