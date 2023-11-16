import unittest
from trashcli import trash
from .. import run_command
from ..support.help_reformatting import reformat_help_message
from ..support.my_path import MyPath

class TestEmptyEndToEnd(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.tmp_dir = MyPath.make_temp_dir()

    def test_help(self):
        if False:
            return 10
        result = run_command.run_command(self.tmp_dir, 'trash-empty', ['--help'])
        self.assertEqual([reformat_help_message("usage: trash-empty [-h] [--print-completion {bash,zsh,tcsh}] [--version] [-v]\n                   [--trash-dir TRASH_DIR] [--all-users] [-i] [-f] [--dry-run]\n                   [days]\n\nPurge trashed files.\n\npositional arguments:\n  days\n\noptions:\n  -h, --help            show this help message and exit\n  --print-completion {bash,zsh,tcsh}\n                        print shell completion script\n  --version             show program's version number and exit\n  -v, --verbose         list files that will be deleted\n  --trash-dir TRASH_DIR\n                        specify the trash directory to use\n  --all-users           empty all trashcan of all the users\n  -i, --interactive     ask before emptying trash directories\n  -f                    don't ask before emptying trash directories\n  --dry-run             show which files would have been removed\n\nReport bugs to https://github.com/andreafrancia/trash-cli/issues\n"), '', 0], [result.reformatted_help(), result.stderr, result.exit_code])

    def test_h(self):
        if False:
            print('Hello World!')
        result = run_command.run_command(self.tmp_dir, 'trash-empty', ['-h'])
        self.assertEqual(['usage:', '', 0], [result.stdout[0:6], result.stderr, result.exit_code])

    def test_version(self):
        if False:
            while True:
                i = 10
        result = run_command.run_command(self.tmp_dir, 'trash-empty', ['--version'])
        self.assertEqual(['trash-empty %s\n' % trash.version, '', 0], [result.stdout, result.stderr, result.exit_code])

    def test_on_invalid_option(self):
        if False:
            i = 10
            return i + 15
        result = run_command.run_command(self.tmp_dir, 'trash-empty', ['--wrong-option'])
        self.assertEqual(['', 'trash-empty: error: unrecognized arguments: --wrong-option', 2], [result.stdout, result.stderr.splitlines()[-1], result.exit_code])

    def test_on_print_time(self):
        if False:
            while True:
                i = 10
        result = run_command.run_command(self.tmp_dir, 'trash-empty', ['--print-time'], env={'TRASH_DATE': '1970-12-31T23:59:59'})
        self.assertEqual(['1970-12-31T23:59:59\n', '', 0], result.all)

    def test_on_trash_date_not_parsable(self):
        if False:
            while True:
                i = 10
        result = run_command.run_command(self.tmp_dir, 'trash-empty', ['--print-time'], env={'TRASH_DATE': 'not a valid date'})
        self.assertEqual(['trash-empty: invalid TRASH_DATE: not a valid date\n', 0], [result.stderr, result.exit_code])

    def tearDown(self):
        if False:
            print('Hello World!')
        self.tmp_dir.clean_up()