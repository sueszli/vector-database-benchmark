import unittest
import pytest
from tests.support.my_path import MyPath
from trashcli.put.fs.size_counter import SizeCounter
from trashcli.put.fs.real_fs import RealFs

@pytest.mark.slow
class TestSizeCounterOnRealFs(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.fs = RealFs()
        self.counter = SizeCounter(self.fs)
        self.tmp_dir = MyPath.make_temp_dir()

    def test_a_single_file(self):
        if False:
            print('Hello World!')
        self.fs.make_file(self.tmp_dir / 'file', 10 * 'a')
        assert self.counter.get_size_recursive(self.tmp_dir / 'file') == 10

    def test_two_files(self):
        if False:
            return 10
        self.fs.make_file(self.tmp_dir / 'a', 100 * 'a')
        self.fs.make_file(self.tmp_dir / 'b', 23 * 'b')
        assert self.counter.get_size_recursive(self.tmp_dir) == 123

    def test_recursive(self):
        if False:
            i = 10
            return i + 15
        self.fs.make_file(self.tmp_dir / 'a', 3 * '-')
        self.fs.makedirs(self.tmp_dir / 'dir', 511)
        self.fs.make_file(self.tmp_dir / 'dir' / 'a', 20 * '-')
        self.fs.makedirs(self.tmp_dir / 'dir' / 'dir', 511)
        self.fs.make_file(self.tmp_dir / 'dir' / 'dir' / 'b', 100 * '-')
        assert self.counter.get_size_recursive(self.tmp_dir) == 123

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.tmp_dir.clean_up()