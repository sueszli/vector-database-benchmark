import unittest
from tests.fake_file_system import FakeFileSystem

class TestFakeFileSystem(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.fs = FakeFileSystem()

    def test_you_can_read_from_files(self):
        if False:
            i = 10
            return i + 15
        self.fs.create_fake_file('/path/to/file', 'file contents')
        assert 'file contents' == self.fs.contents_of('/path/to/file')

    def test_when_creating_a_fake_file_it_creates_also_the_dir(self):
        if False:
            print('Hello World!')
        self.fs.create_fake_file('/dir/file')
        assert set(('file',)) == set(self.fs.entries_if_dir_exists('/dir'))

    def test_you_can_create_multiple_fake_file(self):
        if False:
            print('Hello World!')
        self.fs.create_fake_file('/path/to/file1', 'one')
        self.fs.create_fake_file('/path/to/file2', 'two')
        assert 'one' == self.fs.contents_of('/path/to/file1')
        assert 'two' == self.fs.contents_of('/path/to/file2')

    def test_no_file_exists_at_beginning(self):
        if False:
            for i in range(10):
                print('nop')
        assert not self.fs.exists('/filename')

    def test_after_a_creation_the_file_exists(self):
        if False:
            for i in range(10):
                print('nop')
        self.fs.create_fake_file('/filename')
        assert self.fs.exists('/filename')

    def test_create_fake_dir(self):
        if False:
            print('Hello World!')
        self.fs.create_fake_dir('/etc', 'passwd', 'shadow', 'hosts')
        assert set(['passwd', 'shadow', 'hosts']) == set(self.fs.entries_if_dir_exists('/etc'))