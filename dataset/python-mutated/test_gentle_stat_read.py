import grp
import os
import pwd
import unittest
import six
from tests.support.files import make_file
from tests.support.my_path import MyPath
from trashcli.put.reporter import gentle_stat_read

class TestGentleStatRead(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.tmp_dir = MyPath.make_temp_dir()

    def test_file_non_found(self):
        if False:
            for i in range(10):
                print('nop')
        result = gentle_stat_read(self.tmp_dir / 'not-existent')
        six.assertRegex(self, result, "\\[Errno 2\\] No such file or directory: '/.*/not-existent'")

    def test_file(self):
        if False:
            while True:
                i = 10
        make_file(self.tmp_dir / 'pippo.txt')
        os.chmod(self.tmp_dir / 'pippo.txt', 345)
        result = gentle_stat_read(self.tmp_dir / 'pippo.txt')
        assert result == '531 %s %s' % (self.current_user(), self.current_group())

    def tearDown(self):
        if False:
            print('Hello World!')
        self.tmp_dir.clean_up()

    @staticmethod
    def current_user():
        if False:
            i = 10
            return i + 15
        return pwd.getpwuid(os.getuid()).pw_name

    @staticmethod
    def current_group():
        if False:
            return 10
        return grp.getgrgid(os.getgid()).gr_name