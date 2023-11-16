import unittest
from tests.support.my_path import MyPath
from tests.test_put.support.fake_fs.fake_fs import FakeFs

class TestFakeFsListDir(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.fs = FakeFs()
        self.tmp_dir = MyPath('/tmp')
        self.fs.makedirs(self.tmp_dir, 448)

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        self.fs.make_file(self.tmp_dir / 'a', 'content')
        self.fs.make_file(self.tmp_dir / 'b', 'content')
        self.fs.make_file(self.tmp_dir / 'c', 'content')
        self.fs.makedirs(self.tmp_dir / 'd', 448)
        assert sorted(self.fs.listdir(self.tmp_dir)) == ['a', 'b', 'c', 'd']