"""Tests for dev_mode_storage_services."""
from __future__ import annotations
from core.platform.storage import dev_mode_storage_services
from core.tests import test_utils

class DevModeStorageServicesTests(test_utils.TestBase):

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        dev_mode_storage_services.CLIENT.reset()
        super().setUp()

    def test_isfile_checks_if_file_exists(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        dev_mode_storage_services.commit('bucket', '/file/path.png', b'data', 'image/png')
        self.assertTrue(dev_mode_storage_services.isfile('bucket', '/file/path.png'))
        self.assertFalse(dev_mode_storage_services.isfile('bucket', '/file/path2.png'))

    def test_commit_and_get_with_bytes(self) -> None:
        if False:
            i = 10
            return i + 15
        dev_mode_storage_services.commit('bucket', '/file/path.png', b'data', 'image/png')
        self.assertEqual(dev_mode_storage_services.get('bucket', '/file/path.png'), b'data')

    def test_commit_and_get_with_str(self) -> None:
        if False:
            i = 10
            return i + 15
        dev_mode_storage_services.commit('bucket', '/file/path.png', 'data', 'image/png')
        self.assertEqual(dev_mode_storage_services.get('bucket', '/file/path.png'), b'data')

    def test_delete_correctly_deletes_file(self) -> None:
        if False:
            i = 10
            return i + 15
        dev_mode_storage_services.commit('bucket', '/file/path.png', b'data', 'image/png')
        self.assertTrue(dev_mode_storage_services.isfile('bucket', '/file/path.png'))
        dev_mode_storage_services.delete('bucket', '/file/path.png')
        self.assertFalse(dev_mode_storage_services.isfile('bucket', '/file/path.png'))

    def test_copy_with_existing_source_blob_is_successful(self) -> None:
        if False:
            i = 10
            return i + 15
        dev_mode_storage_services.commit('bucket', '/file/path.png', b'data', 'image/png')
        dev_mode_storage_services.copy('bucket', '/file/path.png', '/copy/path.png')
        self.assertTrue(dev_mode_storage_services.isfile('bucket', '/copy/path.png'))
        self.assertEqual(dev_mode_storage_services.get('bucket', '/file/path.png'), dev_mode_storage_services.get('bucket', '/copy/path.png'))

    def test_copy_with_non_existing_source_blob_fails(self) -> None:
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(Exception, 'Source asset does not exist'):
            dev_mode_storage_services.copy('bucket', '/file/path.png', '/copy/path.png')

    def test_listdir_with_slash_returns_all_blobs(self) -> None:
        if False:
            while True:
                i = 10
        dev_mode_storage_services.commit('bucket', '/file/path1.png', b'data1', 'image/png')
        dev_mode_storage_services.commit('bucket', '/file/path2.png', b'data2', 'image/png')
        dev_mode_storage_services.commit('bucket', '/different/path1.png', b'data3', 'image/png')
        blob_data = [blob.download_as_bytes() for blob in dev_mode_storage_services.listdir('bucket', '/')]
        self.assertItemsEqual(blob_data, [b'data1', b'data2', b'data3'])

    def test_listdir_with_specific_folder_returns_some_blobs(self) -> None:
        if False:
            while True:
                i = 10
        dev_mode_storage_services.commit('bucket', '/file/path1.png', b'data1', 'image/png')
        dev_mode_storage_services.commit('bucket', '/file/path2.png', b'data2', 'image/png')
        dev_mode_storage_services.commit('bucket', '/different/path1.png', b'data3', 'image/png')
        blob_data = [blob.download_as_bytes() for blob in dev_mode_storage_services.listdir('bucket', '/file')]
        self.assertItemsEqual(blob_data, [b'data1', b'data2'])