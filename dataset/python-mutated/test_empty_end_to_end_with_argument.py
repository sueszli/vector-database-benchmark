import datetime
import unittest
import pytest
from .. import run_command
from ..fake_trash_dir import FakeTrashDir
from ..support.list_trash_dir import list_trash_dir
from ..support.my_path import MyPath

@pytest.mark.slow
class TestEmptyEndToEndWithArgument(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.tmp_dir = MyPath.make_temp_dir()
        self.xdg_data_home = self.tmp_dir / 'XDG_DATA_HOME'
        self.environ = {'XDG_DATA_HOME': self.xdg_data_home}
        self.trash_dir = self.xdg_data_home / 'Trash'
        self.fake_trash_dir = FakeTrashDir(self.trash_dir)

    def user_run_trash_empty(self, args):
        if False:
            while True:
                i = 10
        return run_command.run_command(self.tmp_dir, 'trash-empty', args, env=self.environ)

    def set_clock_at(self, yyyy_mm_dd):
        if False:
            for i in range(10):
                print('nop')
        self.environ['TRASH_DATE'] = '%sT00:00:00' % yyyy_mm_dd

    def test_set_clock(self):
        if False:
            for i in range(10):
                print('nop')
        self.set_clock_at('2000-01-01')
        result = self.user_run_trash_empty(['--print-time'])
        self.assertEqual(['2000-01-01T00:00:00\n', '', 0], result.all)

    def test_it_should_keep_files_newer_than_N_days(self):
        if False:
            for i in range(10):
                print('nop')
        self.fake_trash_dir.add_trashinfo_with_date('foo', datetime.date(2000, 1, 1))
        self.set_clock_at('2000-01-01')
        self.user_run_trash_empty(['2'])
        assert list_trash_dir(self.trash_dir) == ['info/foo.trashinfo']

    def test_it_should_remove_files_older_than_N_days(self):
        if False:
            return 10
        self.fake_trash_dir.add_trashinfo_with_date('foo', datetime.date(1999, 1, 1))
        self.set_clock_at('2000-01-01')
        self.user_run_trash_empty(['2'])
        assert list_trash_dir(self.trash_dir) == []

    def test_it_should_kept_files_with_invalid_deletion_date(self):
        if False:
            i = 10
            return i + 15
        self.fake_trash_dir.add_trashinfo_with_invalid_date('foo', 'Invalid Date')
        self.set_clock_at('2000-01-01')
        self.user_run_trash_empty(['2'])
        assert list_trash_dir(self.trash_dir) == ['info/foo.trashinfo']

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.tmp_dir.clean_up()