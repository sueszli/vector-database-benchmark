import unittest
import pytest
from six import StringIO
from tests.fake_trash_dir import FakeTrashDir
from tests.support.my_path import MyPath
from trashcli.fstab.volume_listing import NoVolumesListing
from trashcli.rm.main import RealRmFileSystemReader
from trashcli.rm.rm_cmd import RmCmd

@pytest.mark.slow
class TestTrashRm(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.xdg_data_home = MyPath.make_temp_dir()
        self.stderr = StringIO()
        self.trash_rm = RmCmd(environ={'XDG_DATA_HOME': self.xdg_data_home}, getuid=lambda : 123, volumes_listing=NoVolumesListing(), stderr=self.stderr, file_reader=RealRmFileSystemReader())
        self.fake_trash_dir = FakeTrashDir(self.xdg_data_home / 'Trash')

    def test_issue69(self):
        if False:
            for i in range(10):
                print('nop')
        self.fake_trash_dir.add_trashinfo_without_path('foo')
        self.trash_rm.run(['trash-rm', 'ignored'], uid=None)
        assert self.stderr.getvalue() == "trash-rm: %s/Trash/info/foo.trashinfo: unable to parse 'Path'\n" % self.xdg_data_home

    def test_integration(self):
        if False:
            i = 10
            return i + 15
        self.fake_trash_dir.add_trashinfo_basename_path('del', 'to/be/deleted')
        self.fake_trash_dir.add_trashinfo_basename_path('keep', 'to/be/kept')
        self.trash_rm.run(['trash-rm', 'delete*'], uid=None)
        assert self.fake_trash_dir.ls_info() == ['keep.trashinfo']

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.xdg_data_home.clean_up()