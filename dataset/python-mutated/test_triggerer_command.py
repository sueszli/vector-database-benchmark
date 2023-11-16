from __future__ import annotations
from unittest import mock
import pytest
from airflow.cli import cli_parser
from airflow.cli.commands import triggerer_command
pytestmark = pytest.mark.db_test

class TestTriggererCommand:
    """
    Tests the CLI interface and that it correctly calls the TriggererJobRunner
    """

    @classmethod
    def setup_class(cls):
        if False:
            print('Hello World!')
        cls.parser = cli_parser.get_parser()

    @mock.patch('airflow.cli.commands.triggerer_command.TriggererJobRunner')
    @mock.patch('airflow.cli.commands.triggerer_command._serve_logs')
    def test_capacity_argument(self, mock_serve, mock_triggerer_job_runner):
        if False:
            i = 10
            return i + 15
        'Ensure that the capacity argument is passed correctly'
        mock_triggerer_job_runner.return_value.job_type = 'TriggererJob'
        args = self.parser.parse_args(['triggerer', '--capacity=42'])
        triggerer_command.triggerer(args)
        mock_serve.return_value.__enter__.assert_called_once()
        mock_serve.return_value.__exit__.assert_called_once()
        mock_triggerer_job_runner.assert_called_once_with(job=mock.ANY, capacity=42)