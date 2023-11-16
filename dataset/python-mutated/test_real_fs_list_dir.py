import unittest
from tests.support.my_path import MyPath
from trashcli.put.fs.real_fs import RealFs

class TestRealFsListDir(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.fs = RealFs()
        self.tmp_dir = MyPath.make_temp_dir()

    def test(self):
        if False:
            i = 10
            return i + 15
        self.fs.make_file(self.tmp_dir / 'a', 'content')
        self.fs.make_file(self.tmp_dir / 'b', 'content')
        self.fs.make_file(self.tmp_dir / 'c', 'content')
        self.fs.makedirs(self.tmp_dir / 'd', 448)
        assert sorted(self.fs.listdir(self.tmp_dir)) == ['a', 'b', 'c', 'd']

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.tmp_dir.clean_up()