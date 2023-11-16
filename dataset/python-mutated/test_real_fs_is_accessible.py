import os
import unittest
from tests.support.my_path import MyPath
from trashcli.put.fs.real_fs import RealFs

class TestRealFsIsAccessible(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.fs = RealFs()
        self.tmp_dir = MyPath.make_temp_dir()

    def test_dangling_link(self):
        if False:
            for i in range(10):
                print('nop')
        os.symlink('non-existent', self.tmp_dir / 'link')
        result = self.fs.is_accessible(self.tmp_dir / 'link')
        assert result is False

    def test_connected_link(self):
        if False:
            print('Hello World!')
        self.fs.make_file(self.tmp_dir / 'link-target', '')
        os.symlink('link-target', self.tmp_dir / 'link')
        result = self.fs.is_accessible(self.tmp_dir / 'link')
        assert result is True

    def test_dangling_link_with_lexists(self):
        if False:
            print('Hello World!')
        os.symlink('non-existent', self.tmp_dir / 'link')
        result = self.fs.lexists(self.tmp_dir / 'link')
        assert result is True

    def test_connected_link_with_lexists(self):
        if False:
            return 10
        self.fs.make_file(self.tmp_dir / 'link-target', '')
        os.symlink('link-target', self.tmp_dir / 'link')
        result = self.fs.lexists(self.tmp_dir / 'link')
        assert result is True