from __future__ import annotations
import itertools
from io import StringIO
from unittest.mock import call, patch
import pytest
from airflow.exceptions import AirflowException
from airflow.models import Connection
from airflow.providers.apache.spark.hooks.spark_sql import SparkSqlHook
from airflow.utils import db
from tests.test_utils.db import clear_db_connections
pytestmark = pytest.mark.db_test

def get_after(sentinel, iterable):
    if False:
        for i in range(10):
            print('nop')
    'Get the value after `sentinel` in an `iterable`'
    truncated = itertools.dropwhile(lambda el: el != sentinel, iterable)
    next(truncated)
    return next(truncated)

class TestSparkSqlHook:
    _config = {'conn_id': 'spark_default', 'executor_cores': 4, 'executor_memory': '22g', 'keytab': 'privileged_user.keytab', 'name': 'spark-job', 'num_executors': 10, 'verbose': True, 'sql': ' /path/to/sql/file.sql ', 'conf': 'key=value,PROP=VALUE'}

    @classmethod
    def setup_class(cls) -> None:
        if False:
            return 10
        clear_db_connections(add_default_connections_back=False)
        db.merge_conn(Connection(conn_id='spark_default', conn_type='spark', host='yarn://yarn-master'))

    @classmethod
    def teardown_class(cls) -> None:
        if False:
            while True:
                i = 10
        clear_db_connections(add_default_connections_back=True)

    def test_build_command(self):
        if False:
            i = 10
            return i + 15
        hook = SparkSqlHook(**self._config)
        cmd = ' '.join(hook._prepare_command(''))
        assert f"--executor-cores {self._config['executor_cores']}" in cmd
        assert f"--executor-memory {self._config['executor_memory']}" in cmd
        assert f"--keytab {self._config['keytab']}" in cmd
        assert f"--name {self._config['name']}" in cmd
        assert f"--num-executors {self._config['num_executors']}" in cmd
        sql_path = get_after('-f', hook._prepare_command(''))
        assert self._config['sql'].strip() == sql_path
        for key_value in self._config['conf'].split(','):
            (k, v) = key_value.split('=')
            assert f'--conf {k}={v}' in cmd
        if self._config['verbose']:
            assert '--verbose' in cmd

    @patch('airflow.providers.apache.spark.hooks.spark_sql.subprocess.Popen')
    def test_spark_process_runcmd(self, mock_popen):
        if False:
            print('Hello World!')
        mock_popen.return_value.stdout = StringIO('Spark-sql communicates using stdout')
        mock_popen.return_value.stderr = StringIO('stderr')
        mock_popen.return_value.wait.return_value = 0
        hook = SparkSqlHook(conn_id='spark_default', sql='SELECT 1')
        with patch.object(hook.log, 'debug') as mock_debug:
            with patch.object(hook.log, 'info') as mock_info:
                hook.run_query()
                mock_debug.assert_called_once_with('Spark-Sql cmd: %s', ['spark-sql', '-e', 'SELECT 1', '--master', 'yarn://yarn-master', '--name', 'default-name', '--verbose', '--queue', 'default'])
                mock_info.assert_called_once_with('Spark-sql communicates using stdout')
        assert mock_popen.mock_calls[0] == call(['spark-sql', '-e', 'SELECT 1', '--master', 'yarn://yarn-master', '--name', 'default-name', '--verbose', '--queue', 'default'], stderr=-2, stdout=-1, universal_newlines=True)

    @patch('airflow.providers.apache.spark.hooks.spark_sql.subprocess.Popen')
    def test_spark_process_runcmd_with_str(self, mock_popen):
        if False:
            return 10
        mock_popen.return_value.wait.return_value = 0
        hook = SparkSqlHook(conn_id='spark_default', sql='SELECT 1')
        hook.run_query('--deploy-mode cluster')
        assert mock_popen.mock_calls[0] == call(['spark-sql', '-e', 'SELECT 1', '--master', 'yarn://yarn-master', '--name', 'default-name', '--verbose', '--queue', 'default', '--deploy-mode', 'cluster'], stderr=-2, stdout=-1, universal_newlines=True)

    @patch('airflow.providers.apache.spark.hooks.spark_sql.subprocess.Popen')
    def test_spark_process_runcmd_with_list(self, mock_popen):
        if False:
            return 10
        mock_popen.return_value.wait.return_value = 0
        hook = SparkSqlHook(conn_id='spark_default', sql='SELECT 1')
        hook.run_query(['--deploy-mode', 'cluster'])
        assert mock_popen.mock_calls[0] == call(['spark-sql', '-e', 'SELECT 1', '--master', 'yarn://yarn-master', '--name', 'default-name', '--verbose', '--queue', 'default', '--deploy-mode', 'cluster'], stderr=-2, stdout=-1, universal_newlines=True)

    @patch('airflow.providers.apache.spark.hooks.spark_sql.subprocess.Popen')
    def test_spark_process_runcmd_and_fail(self, mock_popen):
        if False:
            print('Hello World!')
        sql = 'SELECT 1'
        master = 'local'
        params = '--deploy-mode cluster'
        status = 1
        mock_popen.return_value.wait.return_value = status
        with pytest.raises(AirflowException) as ctx:
            hook = SparkSqlHook(conn_id='spark_default', sql=sql, master=master)
            hook.run_query(params)
        assert str(ctx.value) == f"Cannot execute '{sql}' on {master} (additional parameters: '{params}'). Process exit code: {status}."