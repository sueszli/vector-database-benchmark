import datetime
import unittest
import pytest
from tests import run_command
from tests.fake_trash_dir import FakeTrashDir
from tests.support.help_reformatting import reformat_help_message
from tests.support.my_path import MyPath

@pytest.mark.slow
class TestEndToEndList(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.temp_dir = MyPath.make_temp_dir()
        self.trash_dir = self.temp_dir / 'trash-dir'
        self.fake_trash_dir = FakeTrashDir(self.trash_dir)

    def test_list(self):
        if False:
            i = 10
            return i + 15
        self.fake_trash_dir.add_trashinfo2('/file1', datetime.datetime(2000, 1, 1, 0, 0, 1))
        self.fake_trash_dir.add_trashinfo2('/file2', datetime.datetime(2000, 1, 1, 0, 0, 1))
        result = run_command.run_command(self.temp_dir, 'trash-list', ['--trash-dir', self.trash_dir])
        assert ['2000-01-01 00:00:01 /file1', '2000-01-01 00:00:01 /file2'] == sorted(result.stdout.splitlines())

    def test_list_trash_dirs(self):
        if False:
            return 10
        result = run_command.run_command(self.temp_dir, 'trash-list', ['--trash-dirs', '--trash-dir=/home/user/.local/share/Trash'])
        assert (result.stderr, sorted(result.stdout.splitlines()), result.exit_code) == ('', ['/home/user/.local/share/Trash'], 0)

    def test_list_with_paths(self):
        if False:
            i = 10
            return i + 15
        self.fake_trash_dir.add_trashinfo3('base1', '/file1', datetime.datetime(2000, 1, 1, 0, 0, 1))
        self.fake_trash_dir.add_trashinfo3('base2', '/file2', datetime.datetime(2000, 1, 1, 0, 0, 1))
        result = run_command.run_command(self.temp_dir, 'trash-list', ['--trash-dir', self.trash_dir, '--files'])
        assert ('', ['2000-01-01 00:00:01 /file1 -> %s/files/base1' % self.trash_dir, '2000-01-01 00:00:01 /file2 -> %s/files/base2' % self.trash_dir]) == (result.stderr, sorted(result.stdout.splitlines()))

    def test_help(self):
        if False:
            while True:
                i = 10
        result = run_command.run_command(self.temp_dir, 'trash-list', ['--help'])
        self.assertEqual(reformat_help_message("usage: trash-list [-h] [--print-completion {bash,zsh,tcsh}] [--version]\n                  [--volumes] [--trash-dirs] [--trash-dir TRASH_DIRS]\n                  [--all-users]\n\nList trashed files\n\noptions:\n  -h, --help            show this help message and exit\n  --print-completion {bash,zsh,tcsh}\n                        print shell completion script\n  --version             show program's version number and exit\n  --volumes             list volumes\n  --trash-dirs          list trash dirs\n  --trash-dir TRASH_DIRS\n                        specify the trash directory to use\n  --all-users           list trashcans of all the users\n\nReport bugs to https://github.com/andreafrancia/trash-cli/issues\n"), result.stderr + result.reformatted_help())

    def tearDown(self):
        if False:
            return 10
        self.temp_dir.clean_up()