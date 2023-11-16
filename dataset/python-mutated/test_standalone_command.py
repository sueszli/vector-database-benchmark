from __future__ import annotations
from unittest import mock
import pytest
from airflow.cli.commands.standalone_command import StandaloneCommand
from airflow.executors.executor_constants import CELERY_EXECUTOR, CELERY_KUBERNETES_EXECUTOR, DASK_EXECUTOR, DEBUG_EXECUTOR, KUBERNETES_EXECUTOR, LOCAL_EXECUTOR, LOCAL_KUBERNETES_EXECUTOR, SEQUENTIAL_EXECUTOR

class TestStandaloneCommand:

    @pytest.mark.parametrize('conf_executor_name, conf_sql_alchemy_conn, expected_standalone_executor', [(LOCAL_EXECUTOR, 'sqlite_conn_string', LOCAL_EXECUTOR), (LOCAL_KUBERNETES_EXECUTOR, 'sqlite_conn_string', SEQUENTIAL_EXECUTOR), (SEQUENTIAL_EXECUTOR, 'sqlite_conn_string', SEQUENTIAL_EXECUTOR), (CELERY_EXECUTOR, 'sqlite_conn_string', SEQUENTIAL_EXECUTOR), (CELERY_KUBERNETES_EXECUTOR, 'sqlite_conn_string', SEQUENTIAL_EXECUTOR), (DASK_EXECUTOR, 'sqlite_conn_string', SEQUENTIAL_EXECUTOR), (KUBERNETES_EXECUTOR, 'sqlite_conn_string', SEQUENTIAL_EXECUTOR), (DEBUG_EXECUTOR, 'sqlite_conn_string', SEQUENTIAL_EXECUTOR), (LOCAL_EXECUTOR, 'other_db_conn_string', LOCAL_EXECUTOR), (LOCAL_KUBERNETES_EXECUTOR, 'other_db_conn_string', LOCAL_EXECUTOR), (SEQUENTIAL_EXECUTOR, 'other_db_conn_string', SEQUENTIAL_EXECUTOR), (CELERY_EXECUTOR, 'other_db_conn_string', LOCAL_EXECUTOR), (CELERY_KUBERNETES_EXECUTOR, 'other_db_conn_string', LOCAL_EXECUTOR), (DASK_EXECUTOR, 'other_db_conn_string', LOCAL_EXECUTOR), (KUBERNETES_EXECUTOR, 'other_db_conn_string', LOCAL_EXECUTOR), (DEBUG_EXECUTOR, 'other_db_conn_string', LOCAL_EXECUTOR)])
    def test_calculate_env(self, conf_executor_name, conf_sql_alchemy_conn, expected_standalone_executor):
        if False:
            while True:
                i = 10
        'Should always force a local executor compatible with the db.'
        with mock.patch.dict('os.environ', {'AIRFLOW__CORE__EXECUTOR': conf_executor_name, 'AIRFLOW__DATABASE__SQL_ALCHEMY_CONN': conf_sql_alchemy_conn}):
            env = StandaloneCommand().calculate_env()
            assert env['AIRFLOW__CORE__EXECUTOR'] == expected_standalone_executor