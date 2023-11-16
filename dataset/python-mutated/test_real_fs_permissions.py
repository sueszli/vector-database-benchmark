import unittest
from tests.support.capture_error import capture_error
from tests.support.my_path import MyPath
from trashcli.put.fs.real_fs import RealFs

class TestRealFsPermissions(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.fs = RealFs()
        self.tmp_dir = MyPath.make_temp_dir()

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        self.fs.makedirs(self.tmp_dir / 'dir', 0)
        error = capture_error(lambda : self.fs.make_file(self.tmp_dir / 'dir' / 'file', 'content'))
        assert str(error) == "[Errno 13] Permission denied: '%s'" % (self.tmp_dir / 'dir' / 'file')
        self.fs.chmod(self.tmp_dir / 'dir', 493)

    def test_chmod_and_get_mod(self):
        if False:
            return 10
        path = self.tmp_dir / 'file'
        self.fs.make_file(path, 'content')
        self.fs.chmod(path, 83)
        assert self.fs.get_mod(path) == 83

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.tmp_dir.clean_up()