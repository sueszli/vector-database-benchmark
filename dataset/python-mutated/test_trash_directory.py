import unittest
import pytest
import six
from tests.support.files import make_file, require_empty_dir
from tests.support.my_path import MyPath
from trashcli.restore.file_system import RealListingFileSystem
from trashcli.restore.info_files import InfoFiles

@pytest.mark.slow
class TestTrashDirectory(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.temp_dir = MyPath.make_temp_dir()
        require_empty_dir(self.temp_dir / 'trash-dir')
        self.info_files = InfoFiles(RealListingFileSystem())

    def test_should_list_a_trashinfo(self):
        if False:
            i = 10
            return i + 15
        make_file(self.temp_dir / 'trash-dir/info/foo.trashinfo')
        result = self.list_trashinfos()
        assert [('trashinfo', self.temp_dir / 'trash-dir/info/foo.trashinfo')] == result

    def test_should_list_multiple_trashinfo(self):
        if False:
            for i in range(10):
                print('nop')
        make_file(self.temp_dir / 'trash-dir/info/foo.trashinfo')
        make_file(self.temp_dir / 'trash-dir/info/bar.trashinfo')
        make_file(self.temp_dir / 'trash-dir/info/baz.trashinfo')
        result = self.list_trashinfos()
        six.assertCountEqual(self, [('trashinfo', self.temp_dir / 'trash-dir/info/foo.trashinfo'), ('trashinfo', self.temp_dir / 'trash-dir/info/baz.trashinfo'), ('trashinfo', self.temp_dir / 'trash-dir/info/bar.trashinfo')], result)

    def test_non_trashinfo_should_reported_as_a_warn(self):
        if False:
            return 10
        make_file(self.temp_dir / 'trash-dir/info/not-a-trashinfo')
        result = self.list_trashinfos()
        six.assertCountEqual(self, [('non_trashinfo', self.temp_dir / 'trash-dir/info/not-a-trashinfo')], result)

    def list_trashinfos(self):
        if False:
            i = 10
            return i + 15
        return list(self.info_files.all_info_files(self.temp_dir / 'trash-dir'))

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.temp_dir.clean_up()