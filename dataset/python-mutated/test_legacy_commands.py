from __future__ import annotations
import contextlib
from argparse import ArgumentError
from io import StringIO
from unittest.mock import MagicMock
import pytest
from airflow.cli import cli_parser
from airflow.cli.commands import config_command
from airflow.cli.commands.legacy_commands import COMMAND_MAP, check_legacy_command
LEGACY_COMMANDS = ['worker', 'flower', 'trigger_dag', 'delete_dag', 'show_dag', 'list_dag', 'dag_status', 'backfill', 'list_dag_runs', 'pause', 'unpause', 'test', 'clear', 'list_tasks', 'task_failed_deps', 'task_state', 'run', 'render', 'initdb', 'resetdb', 'upgradedb', 'checkdb', 'shell', 'pool', 'list_users', 'create_user', 'delete_user']

class TestCliDeprecatedCommandsValue:

    @classmethod
    def setup_class(cls):
        if False:
            for i in range(10):
                print('nop')
        cls.parser = cli_parser.get_parser()

    def test_should_display_value(self):
        if False:
            while True:
                i = 10
        with pytest.raises(SystemExit) as ctx, contextlib.redirect_stderr(StringIO()) as temp_stderr:
            config_command.get_value(self.parser.parse_args(['worker']))
        assert 2 == ctx.value.code
        assert '`airflow worker` command, has been removed, please use `airflow celery worker`, see help above.' in temp_stderr.getvalue().strip()

    def test_command_map(self):
        if False:
            return 10
        for item in LEGACY_COMMANDS:
            assert COMMAND_MAP[item] is not None

    def test_check_legacy_command(self):
        if False:
            print('Hello World!')
        action = MagicMock()
        with pytest.raises(ArgumentError) as ctx:
            check_legacy_command(action, 'list_users')
        assert str(ctx.value) == 'argument : `airflow list_users` command, has been removed, please use `airflow users list`'