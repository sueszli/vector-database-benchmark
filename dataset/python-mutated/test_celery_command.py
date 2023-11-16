from __future__ import annotations
from importlib import reload
from unittest import mock
import pytest
from airflow.cli import cli_parser
from airflow.cli.commands import celery_command
from tests.test_utils.config import conf_vars

@pytest.mark.integration('celery')
@pytest.mark.backend('mysql', 'postgres')
class TestWorkerServeLogs:

    @classmethod
    def setup_class(cls):
        if False:
            for i in range(10):
                print('nop')
        with conf_vars({('core', 'executor'): 'CeleryExecutor'}):
            reload(cli_parser)
            cls.parser = cli_parser.get_parser()

    @conf_vars({('core', 'executor'): 'CeleryExecutor'})
    def test_serve_logs_on_worker_start(self):
        if False:
            return 10
        with mock.patch('airflow.cli.commands.celery_command.Process') as mock_process, mock.patch('airflow.providers.celery.executors.celery_executor.app'):
            args = self.parser.parse_args(['celery', 'worker', '--concurrency', '1'])
            with mock.patch('celery.platforms.check_privileges') as mock_privil:
                mock_privil.return_value = 0
                celery_command.worker(args)
                mock_process.assert_called()

    @conf_vars({('core', 'executor'): 'CeleryExecutor'})
    def test_skip_serve_logs_on_worker_start(self):
        if False:
            i = 10
            return i + 15
        with mock.patch('airflow.cli.commands.celery_command.Process') as mock_popen, mock.patch('airflow.providers.celery.executors.celery_executor.app'):
            args = self.parser.parse_args(['celery', 'worker', '--concurrency', '1', '--skip-serve-logs'])
            with mock.patch('celery.platforms.check_privileges') as mock_privil:
                mock_privil.return_value = 0
                celery_command.worker(args)
                mock_popen.assert_not_called()