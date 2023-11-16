"""Tests for Azure Blob Storage client.
"""
import logging
import unittest
try:
    from apache_beam.io.azure import blobstorageio
except ImportError:
    blobstorageio = None

@unittest.skipIf(blobstorageio is None, 'Azure dependencies are not installed')
class TestAZFSPathParser(unittest.TestCase):
    BAD_AZFS_PATHS = ['azfs://azfs://storage-account/azfs://storage-account/**azfs://storage-account/**/*azfs://containerazfs:///nameazfs:///azfs:/blah/container/nameazfs://ab/container/nameazfs://accountwithmorethan24chars/container/nameazfs://***/container/nameazfs://storageaccount/my--container/nameazfs://storageaccount/CONTAINER/nameazfs://storageaccount/ct/name']

    def test_azfs_path(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(blobstorageio.parse_azfs_path('azfs://storageaccount/container/name', get_account=True), ('storageaccount', 'container', 'name'))
        self.assertEqual(blobstorageio.parse_azfs_path('azfs://storageaccount/container/name/sub', get_account=True), ('storageaccount', 'container', 'name/sub'))

    def test_bad_azfs_path(self):
        if False:
            print('Hello World!')
        for path in self.BAD_AZFS_PATHS:
            self.assertRaises(ValueError, blobstorageio.parse_azfs_path, path)
        self.assertRaises(ValueError, blobstorageio.parse_azfs_path, 'azfs://storageaccount/container/')

    def test_azfs_path_blob_optional(self):
        if False:
            while True:
                i = 10
        self.assertEqual(blobstorageio.parse_azfs_path('azfs://storageaccount/container/name', blob_optional=True, get_account=True), ('storageaccount', 'container', 'name'))
        self.assertEqual(blobstorageio.parse_azfs_path('azfs://storageaccount/container/', blob_optional=True, get_account=True), ('storageaccount', 'container', ''))

    def test_bad_azfs_path_blob_optional(self):
        if False:
            i = 10
            return i + 15
        for path in self.BAD_AZFS_PATHS:
            self.assertRaises(ValueError, blobstorageio.parse_azfs_path, path, True)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()