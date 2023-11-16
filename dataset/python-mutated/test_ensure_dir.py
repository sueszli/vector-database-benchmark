import unittest
from tests.test_put.support.fake_fs.fake_fs import FakeFs
from tests.test_put.support.format_mode import format_mode
from trashcli.put.dir_maker import DirMaker

class TestEnsureDir(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.fs = FakeFs('/')
        self.dir_maker = DirMaker(self.fs)

    def test_happy_path(self):
        if False:
            print('Hello World!')
        self.dir_maker.mkdir_p('/foo', 493)
        assert [self.fs.isdir('/foo'), format_mode(self.fs.get_mod('/foo'))] == [True, '0o755']

    def test_makedirs_honor_permissions(self):
        if False:
            return 10
        self.fs.makedirs('/foo', 0)
        assert [format_mode(self.fs.get_mod('/foo'))] == ['0o000']

    def test_bug_when_no_permissions_it_overrides_the_permissions(self):
        if False:
            print('Hello World!')
        self.fs.makedirs('/foo', 0)
        self.dir_maker.mkdir_p('/foo', 493)
        assert [self.fs.isdir('/foo'), format_mode(self.fs.get_mod('/foo'))] == [True, '0o000']