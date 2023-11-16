from __future__ import annotations
import datetime
from unittest import mock
import pytest
from airflow.models.dag import DAG
try:
    from airflow.providers.mysql.transfers.vertica_to_mysql import VerticaToMySqlOperator
except ImportError:
    pytest.skip('MySQL not available', allow_module_level=True)

def mock_get_conn():
    if False:
        while True:
            i = 10
    commit_mock = mock.MagicMock()
    cursor_mock = mock.MagicMock(execute=[], fetchall=[['1', '2', '3']], description=['a', 'b', 'c'], iterate=[['1', '2', '3']])
    conn_mock = mock.MagicMock(commit=commit_mock, cursor=cursor_mock)
    return conn_mock

class TestVerticaToMySqlTransfer:

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        args = {'owner': 'airflow', 'start_date': datetime.datetime(2017, 1, 1)}
        self.dag = DAG('test_dag_id', default_args=args)

    @mock.patch('airflow.providers.mysql.transfers.vertica_to_mysql.VerticaHook.get_conn', side_effect=mock_get_conn)
    @mock.patch('airflow.providers.mysql.transfers.vertica_to_mysql.MySqlHook.get_conn', side_effect=mock_get_conn)
    @mock.patch('airflow.providers.mysql.transfers.vertica_to_mysql.MySqlHook.insert_rows', return_value=True)
    def test_select_insert_transfer(self, *args):
        if False:
            i = 10
            return i + 15
        '\n        Test check selection from vertica into memory and\n        after that inserting into mysql\n        '
        task = VerticaToMySqlOperator(task_id='test_task_id', sql='select a, b, c', mysql_table='test_table', vertica_conn_id='test_vertica_conn_id', mysql_conn_id='test_mysql_conn_id', params={}, bulk_load=False, dag=self.dag)
        task.execute(None)

    @mock.patch('airflow.providers.mysql.transfers.vertica_to_mysql.VerticaHook.get_conn', side_effect=mock_get_conn)
    @mock.patch('airflow.providers.mysql.transfers.vertica_to_mysql.MySqlHook.get_conn', side_effect=mock_get_conn)
    def test_select_bulk_insert_transfer(self, *args):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test check selection from vertica into temporary file and\n        after that bulk inserting into mysql\n        '
        task = VerticaToMySqlOperator(task_id='test_task_id', sql='select a, b, c', mysql_table='test_table', vertica_conn_id='test_vertica_conn_id', mysql_conn_id='test_mysql_conn_id', params={}, bulk_load=True, dag=self.dag)
        task.execute(None)