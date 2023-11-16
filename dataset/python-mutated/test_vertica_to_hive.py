from __future__ import annotations
import datetime
from unittest import mock
import pytest
from airflow.models.dag import DAG
from airflow.providers.apache.hive.transfers.vertica_to_hive import VerticaToHiveOperator
pytestmark = pytest.mark.db_test

def mock_get_conn():
    if False:
        for i in range(10):
            print('nop')
    commit_mock = mock.MagicMock()
    cursor_mock = mock.MagicMock(execute=[], fetchall=[['1', '2', '3']], description=['a', 'b', 'c'], iterate=[['1', '2', '3']])
    conn_mock = mock.MagicMock(commit=commit_mock, cursor=cursor_mock)
    return conn_mock

class TestVerticaToHiveTransfer:

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        args = {'owner': 'airflow', 'start_date': datetime.datetime(2017, 1, 1)}
        self.dag = DAG('test_dag_id', default_args=args)

    @mock.patch('airflow.providers.apache.hive.transfers.vertica_to_hive.VerticaHook.get_conn', side_effect=mock_get_conn)
    @mock.patch('airflow.providers.apache.hive.transfers.vertica_to_hive.HiveCliHook.load_file')
    def test_select_insert_transfer(self, *args):
        if False:
            print('Hello World!')
        '\n        Test check selection from vertica into memory and\n        after that inserting into mysql\n        '
        task = VerticaToHiveOperator(task_id='test_task_id', sql='select a, b, c', hive_table='test_table', vertica_conn_id='test_vertica_conn_id', hive_cli_conn_id='hive_cli_default', dag=self.dag)
        task.execute(None)