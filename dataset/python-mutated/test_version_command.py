from __future__ import annotations
from contextlib import redirect_stdout
from io import StringIO
import airflow.cli.commands.version_command
from airflow.cli import cli_parser
from airflow.version import version

class TestCliVersion:

    @classmethod
    def setup_class(cls):
        if False:
            for i in range(10):
                print('nop')
        cls.parser = cli_parser.get_parser()

    def test_cli_version(self):
        if False:
            i = 10
            return i + 15
        with redirect_stdout(StringIO()) as stdout:
            airflow.cli.commands.version_command.version(self.parser.parse_args(['version']))
        assert version in stdout.getvalue()