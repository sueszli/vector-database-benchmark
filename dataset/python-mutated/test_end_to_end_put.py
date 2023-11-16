import unittest
from textwrap import dedent
import pytest
from tests import run_command
from tests.run_command import first_line_of, last_line_of
from trashcli.lib.exit_codes import EX_IOERR
from ..support.my_path import MyPath

@pytest.mark.slow
class TestEndToEndPut(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.tmp_dir = MyPath.make_temp_dir()

    def test_last_line_of_help(self):
        if False:
            for i in range(10):
                print('nop')
        result = run_command.run_command(self.tmp_dir, 'trash-put', ['--help'])
        assert last_line_of(result.stdout) == 'Report bugs to https://github.com/andreafrancia/trash-cli/issues'

    def test_without_args(self):
        if False:
            while True:
                i = 10
        result = run_command.run_command(self.tmp_dir, 'trash-put', [])
        assert [first_line_of(result.stderr), result.exit_code] == ['usage: trash-put [OPTION]... FILE...', 2]

    def test_wrong_option(self):
        if False:
            i = 10
            return i + 15
        result = run_command.run_command(self.tmp_dir, 'trash-put', ['--wrong-option'])
        assert [last_line_of(result.stderr), result.exit_code] == ['trash-put: error: unrecognized arguments: --wrong-option', 2]

    def test_on_help(self):
        if False:
            return 10
        result = run_command.run_command(self.tmp_dir, 'trash-put', ['--help'])
        assert [result.reformatted_help(), result.exit_code] == [dedent("                usage: trash-put [OPTION]... FILE...\n\n                Put files in trash\n\n                positional arguments:\n                  files\n\n                options:\n                  -h, --help            show this help message and exit\n                  --print-completion {bash,zsh,tcsh}\n                                        print shell completion script\n                  -d, --directory       ignored (for GNU rm compatibility)\n                  -f, --force           silently ignore nonexistent files\n                  -i, --interactive     prompt before every removal\n                  -r, -R, --recursive   ignored (for GNU rm compatibility)\n                  --trash-dir TRASHDIR  use TRASHDIR as trash folder\n                  -v, --verbose         explain what is being done\n                  --version             show program's version number and exit\n\n                To remove a file whose name starts with a '-', for example '-foo',\n                use one of these commands:\n\n                    trash -- -foo\n\n                    trash ./-foo\n\n                Report bugs to https://github.com/andreafrancia/trash-cli/issues\n            "), 0]

    def test_it_should_skip_dot_entry(self):
        if False:
            i = 10
            return i + 15
        result = run_command.run_command(self.tmp_dir, 'trash-put', ['.'])
        assert [result.stderr, result.exit_code] == ["trash-put: cannot trash directory '.'\n", EX_IOERR]

    def test_it_should_skip_dotdot_entry(self):
        if False:
            return 10
        result = run_command.run_command(self.tmp_dir, 'trash-put', ['..'])
        assert [result.stderr, result.exit_code] == ["trash-put: cannot trash directory '..'\n", EX_IOERR]

    def test_it_should_print_usage_on_no_argument(self):
        if False:
            i = 10
            return i + 15
        result = run_command.run_command(self.tmp_dir, 'trash-put', [])
        assert [result.stdout, result.stderr, result.exit_code] == ['', 'usage: trash-put [OPTION]... FILE...\ntrash-put: error: Please specify the files to trash.\n', 2]

    def test_it_should_skip_missing_files(self):
        if False:
            print('Hello World!')
        result = run_command.run_command(self.tmp_dir, 'trash-put', ['-f', 'this_file_does_not_exist', 'nor_does_this_file'])
        assert [result.stdout, result.stderr, result.exit_code] == ['', '', 0]

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.tmp_dir.clean_up()