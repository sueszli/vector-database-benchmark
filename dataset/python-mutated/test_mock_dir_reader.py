import unittest
from tests.mock_dir_reader import MockDirReader

class TestMockDirReader(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.fs = MockDirReader()

    def test_empty(self):
        if False:
            i = 10
            return i + 15
        result = self.fs.entries_if_dir_exists('/')
        self.assertEqual([], result)

    def test_add_file_in_root(self):
        if False:
            i = 10
            return i + 15
        self.fs.add_file('/foo')
        result = self.fs.entries_if_dir_exists('/')
        self.assertEqual(['foo'], result)

    def test_mkdir(self):
        if False:
            for i in range(10):
                print('nop')
        self.fs.mkdir('/foo')
        result = self.fs.entries_if_dir_exists('/')
        self.assertEqual(['foo'], result)

    def test_add_file_in_dir(self):
        if False:
            while True:
                i = 10
        self.fs.mkdir('/foo')
        self.fs.add_file('/foo/bar')
        result = self.fs.entries_if_dir_exists('/')
        self.assertEqual(['foo'], result)
        result = self.fs.entries_if_dir_exists('/foo')
        self.assertEqual(['bar'], result)