import os
import unittest
import pytest
from tests.support.files import make_empty_file, make_file, require_empty_dir
from tests.support.my_path import MyPath
from trashcli.put.describer import Describer
from trashcli.put.fs.real_fs import RealFs

@pytest.mark.slow
class TestDescriber(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.temp_dir = MyPath.make_temp_dir()
        self.describer = Describer(RealFs())

    def test_on_directories(self):
        if False:
            i = 10
            return i + 15
        require_empty_dir(self.temp_dir / 'a-dir')
        assert 'directory' == self.describer.describe('.')
        assert 'directory' == self.describer.describe('..')
        assert 'directory' == self.describer.describe(self.temp_dir / 'a-dir')

    def test_on_dot_directories(self):
        if False:
            i = 10
            return i + 15
        require_empty_dir(self.temp_dir / 'a-dir')
        assert "'.' directory" == self.describer.describe(self.temp_dir / 'a-dir/.')
        assert "'.' directory" == self.describer.describe('./.')

    def test_on_dot_dot_directories(self):
        if False:
            return 10
        require_empty_dir(self.temp_dir / 'a-dir')
        assert "'..' directory" == self.describer.describe('./..')
        assert "'..' directory" == self.describer.describe(self.temp_dir / 'a-dir/..')

    def test_name_for_regular_files_non_empty_files(self):
        if False:
            while True:
                i = 10
        make_file(self.temp_dir / 'non-empty', 'contents')
        assert 'regular file' == self.describer.describe(self.temp_dir / 'non-empty')

    def test_name_for_empty_file(self):
        if False:
            for i in range(10):
                print('nop')
        make_empty_file(self.temp_dir / 'empty')
        assert 'regular empty file' == self.describer.describe(self.temp_dir / 'empty')

    def test_name_for_symbolic_links(self):
        if False:
            return 10
        os.symlink('nowhere', self.temp_dir / 'symlink')
        assert 'symbolic link' == self.describer.describe(self.temp_dir / 'symlink')

    def test_name_for_non_existent_entries(self):
        if False:
            i = 10
            return i + 15
        assert not os.path.exists(self.temp_dir / 'non-existent')
        assert 'non existent' == self.describer.describe(self.temp_dir / 'non-existent')

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.temp_dir.clean_up()