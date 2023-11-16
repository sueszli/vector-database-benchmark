"""Unit tests for MemoryMediaFileStorage"""
import unittest
from unittest import mock
from unittest.mock import MagicMock, mock_open
from parameterized import parameterized
from streamlit.runtime.media_file_storage import MediaFileKind, MediaFileStorageError
from streamlit.runtime.memory_media_file_storage import MemoryFile, MemoryMediaFileStorage, get_extension_for_mimetype

class MemoryMediaFileStorageTest(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.storage = MemoryMediaFileStorage(media_endpoint='/mock/media')

    @mock.patch('streamlit.runtime.memory_media_file_storage.open', mock_open(read_data=b'mock_bytes'))
    def test_load_with_path(self):
        if False:
            return 10
        'Adding a file by path creates a MemoryFile instance.'
        file_id = self.storage.load_and_get_id('mock/file/path', mimetype='video/mp4', kind=MediaFileKind.MEDIA, filename='file.mp4')
        self.assertEqual(MemoryFile(content=b'mock_bytes', mimetype='video/mp4', kind=MediaFileKind.MEDIA, filename='file.mp4'), self.storage.get_file(file_id))

    def test_load_with_bytes(self):
        if False:
            print('Hello World!')
        'Adding a file with bytes creates a MemoryFile instance.'
        file_id = self.storage.load_and_get_id(b'mock_bytes', mimetype='video/mp4', kind=MediaFileKind.MEDIA, filename='file.mp4')
        self.assertEqual(MemoryFile(content=b'mock_bytes', mimetype='video/mp4', kind=MediaFileKind.MEDIA, filename='file.mp4'), self.storage.get_file(file_id))

    def test_identical_files_have_same_id(self):
        if False:
            while True:
                i = 10
        'Two files with the same content, mimetype, and filename should share an ID.'
        file_id1 = self.storage.load_and_get_id(b'mock_bytes', mimetype='video/mp4', kind=MediaFileKind.MEDIA, filename='file.mp4')
        file_id2 = self.storage.load_and_get_id(b'mock_bytes', mimetype='video/mp4', kind=MediaFileKind.MEDIA, filename='file.mp4')
        self.assertEqual(file_id1, file_id2)
        changed_content = self.storage.load_and_get_id(b'mock_bytes_2', mimetype='video/mp4', kind=MediaFileKind.MEDIA, filename='file.mp4')
        self.assertNotEqual(file_id1, changed_content)
        changed_mimetype = self.storage.load_and_get_id(b'mock_bytes', mimetype='image/png', kind=MediaFileKind.MEDIA, filename='file.mp4')
        self.assertNotEqual(file_id1, changed_mimetype)
        changed_filename = self.storage.load_and_get_id(b'mock_bytes', mimetype='video/mp4', kind=MediaFileKind.MEDIA)
        self.assertNotEqual(file_id1, changed_filename)

    @mock.patch('streamlit.runtime.memory_media_file_storage.open', MagicMock(side_effect=Exception))
    def test_load_with_bad_path(self):
        if False:
            i = 10
            return i + 15
        "Adding a file by path raises a MediaFileStorageError if the file can't be read."
        with self.assertRaises(MediaFileStorageError):
            self.storage.load_and_get_id('mock/file/path', mimetype='video/mp4', kind=MediaFileKind.MEDIA, filename='file.mp4')

    @parameterized.expand([('video/mp4', '.mp4'), ('audio/wav', '.wav'), ('image/png', '.png'), ('image/jpeg', '.jpg')])
    def test_get_url(self, mimetype, extension):
        if False:
            print('Hello World!')
        'URLs should be formatted correctly, and have the expected extension.'
        file_id = self.storage.load_and_get_id(b'mock_bytes', mimetype=mimetype, kind=MediaFileKind.MEDIA)
        url = self.storage.get_url(file_id)
        self.assertEqual(f'/mock/media/{file_id}{extension}', url)

    def test_get_url_invalid_fileid(self):
        if False:
            for i in range(10):
                print('nop')
        'get_url raises if it gets a bad file_id.'
        with self.assertRaises(MediaFileStorageError):
            self.storage.get_url('not_a_file_id')

    def test_delete_file(self):
        if False:
            return 10
        'delete_file removes the file with the given ID.'
        file_id1 = self.storage.load_and_get_id(b'mock_bytes_1', mimetype='video/mp4', kind=MediaFileKind.MEDIA, filename='file.mp4')
        file_id2 = self.storage.load_and_get_id(b'mock_bytes_2', mimetype='video/mp4', kind=MediaFileKind.MEDIA, filename='file.mp4')
        self.storage.delete_file(file_id1)
        with self.assertRaises(Exception):
            self.storage.get_file(file_id1)
        self.assertIsNotNone(self.storage.get_file(file_id2))
        self.storage.delete_file(file_id2)
        with self.assertRaises(Exception):
            self.storage.get_file(file_id2)

    def test_delete_invalid_file_is_a_noop(self):
        if False:
            return 10
        "deleting a file that doesn't exist doesn't raise an error."
        self.storage.delete_file('mock_file_id')

    def test_cache_stats(self):
        if False:
            while True:
                i = 10
        'Test our CacheStatsProvider implementation.'
        self.assertEqual(0, len(self.storage.get_stats()))
        mock_data = b'some random mock binary data'
        num_files = 5
        for ii in range(num_files):
            self.storage.load_and_get_id(mock_data, mimetype='video/mp4', kind=MediaFileKind.MEDIA, filename=f'{ii}.mp4')
        stats = self.storage.get_stats()
        self.assertEqual(num_files, len(stats))
        self.assertEqual('st_memory_media_file_storage', stats[0].category_name)
        self.assertEqual(len(mock_data) * num_files, sum((stat.byte_length for stat in stats)))
        for file_id in list(self.storage._files_by_id.keys()):
            self.storage.delete_file(file_id)
        self.assertEqual(0, len(self.storage.get_stats()))

class MemoryMediaFileStorageUtilTest(unittest.TestCase):
    """Unit tests for utility functions in memory_media_file_storage.py"""

    @parameterized.expand([('video/mp4', '.mp4'), ('audio/wav', '.wav'), ('image/png', '.png'), ('image/jpeg', '.jpg')])
    def test_get_extension_for_mimetype(self, mimetype: str, expected_extension: str):
        if False:
            for i in range(10):
                print('nop')
        result = get_extension_for_mimetype(mimetype)
        self.assertEqual(expected_extension, result)