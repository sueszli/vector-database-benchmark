import unittest
from tests.test_put.support.fake_fs.fake_fs import FakeFs
from trashcli.put.describer import Describer

class TestDescriber(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.fs = FakeFs()
        self.describer = Describer(self.fs)

    def test_on_directories(self):
        if False:
            for i in range(10):
                print('nop')
        self.fs.mkdir('a-dir')
        assert 'directory' == self.describer.describe('.')
        assert 'directory' == self.describer.describe('..')
        assert 'directory' == self.describer.describe('a-dir')

    def test_on_dot_directories(self):
        if False:
            return 10
        self.fs.mkdir('a-dir')
        assert "'.' directory" == self.describer.describe('a-dir/.')
        assert "'.' directory" == self.describer.describe('./.')

    def test_on_dot_dot_directories(self):
        if False:
            i = 10
            return i + 15
        self.fs.mkdir('a-dir')
        assert "'..' directory" == self.describer.describe('./..')
        assert "'..' directory" == self.describer.describe('a-dir/..')

    def test_name_for_regular_files_non_empty_files(self):
        if False:
            for i in range(10):
                print('nop')
        self.fs.make_file('non-empty', 'contents')
        assert 'regular file' == self.describer.describe('non-empty')

    def test_name_for_empty_file(self):
        if False:
            while True:
                i = 10
        self.fs.make_file('empty')
        assert 'regular empty file' == self.describer.describe('empty')

    def test_name_for_symbolic_links(self):
        if False:
            while True:
                i = 10
        self.fs.symlink('nowhere', '/symlink')
        assert 'symbolic link' == self.describer.describe('symlink')

    def test_name_for_non_existent_entries(self):
        if False:
            for i in range(10):
                print('nop')
        assert 'non existent' == self.describer.describe('non-existent')