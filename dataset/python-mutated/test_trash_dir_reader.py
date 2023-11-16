import unittest
from tests.fake_file_system import FakeFileSystem
from trashcli.lib.trash_dir_reader import TrashDirReader

class TestTrashDirReader(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.fs = FakeFileSystem()
        self.trash_dir = TrashDirReader(self.fs)

    def test(self):
        if False:
            while True:
                i = 10
        self.fs.create_fake_file('/info/foo.trashinfo')
        result = list(self.trash_dir.list_orphans('/'))
        assert [] == result

    def test2(self):
        if False:
            while True:
                i = 10
        self.fs.create_fake_file('/files/foo')
        result = list(self.trash_dir.list_orphans('/'))
        assert ['/files/foo'] == result