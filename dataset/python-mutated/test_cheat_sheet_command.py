from __future__ import annotations
import contextlib
from io import StringIO
from unittest import mock
from airflow.cli import cli_parser
from airflow.cli.cli_config import ActionCommand, CLICommand, GroupCommand

def noop():
    if False:
        for i in range(10):
            print('nop')
    pass
MOCK_COMMANDS: list[CLICommand] = [GroupCommand(name='cmd_a', help='Help text A', subcommands=[ActionCommand(name='cmd_b', help='Help text B', func=noop, args=()), ActionCommand(name='cmd_c', help='Help text C', func=noop, args=())]), GroupCommand(name='cmd_e', help='Help text E', subcommands=[ActionCommand(name='cmd_f', help='Help text F', func=noop, args=()), ActionCommand(name='cmd_g', help='Help text G', func=noop, args=())]), ActionCommand(name='cmd_b', help='Help text D', func=noop, args=())]
ALL_COMMANDS = 'airflow cmd_b                             | Help text D\n'
SECTION_A = 'airflow cmd_a cmd_b                       | Help text B\nairflow cmd_a cmd_c                       | Help text C\n'
SECTION_E = 'airflow cmd_e cmd_f                       | Help text F\nairflow cmd_e cmd_g                       | Help text G\n'

class TestCheatSheetCommand:

    @classmethod
    def setup_class(cls):
        if False:
            return 10
        cls.parser = cli_parser.get_parser()

    @mock.patch('airflow.cli.cli_parser.airflow_commands', MOCK_COMMANDS)
    def test_should_display_index(self):
        if False:
            return 10
        with contextlib.redirect_stdout(StringIO()) as temp_stdout:
            args = self.parser.parse_args(['cheat-sheet'])
            args.func(args)
        output = temp_stdout.getvalue()
        assert ALL_COMMANDS in output
        assert SECTION_A in output
        assert SECTION_E in output