from __future__ import annotations
from unittest import mock
import pytest
from airflow.cli import cli_parser
from airflow.cli.commands import dag_processor_command
from airflow.configuration import conf
from tests.test_utils.config import conf_vars
pytestmark = pytest.mark.db_test

class TestDagProcessorCommand:
    """
    Tests the CLI interface and that it correctly calls the DagProcessor
    """

    @classmethod
    def setup_class(cls):
        if False:
            print('Hello World!')
        cls.parser = cli_parser.get_parser()

    @conf_vars({('scheduler', 'standalone_dag_processor'): 'True', ('core', 'load_examples'): 'False'})
    @mock.patch('airflow.cli.commands.dag_processor_command.DagProcessorJobRunner')
    @pytest.mark.skipif(conf.get_mandatory_value('database', 'sql_alchemy_conn').lower().startswith('sqlite'), reason="Standalone Dag Processor doesn't support sqlite.")
    def test_start_job(self, mock_dag_job):
        if False:
            return 10
        'Ensure that DagProcessorJobRunner is started'
        with conf_vars({('scheduler', 'standalone_dag_processor'): 'True'}):
            mock_dag_job.return_value.job_type = 'DagProcessorJob'
            args = self.parser.parse_args(['dag-processor'])
            dag_processor_command.dag_processor(args)
            mock_dag_job.return_value._execute.assert_called()