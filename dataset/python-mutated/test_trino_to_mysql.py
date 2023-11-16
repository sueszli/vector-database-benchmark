from __future__ import annotations
import os
from datetime import datetime
from unittest.mock import patch
import pytest
from airflow.models.dag import DAG
from airflow.providers.mysql.transfers.trino_to_mysql import TrinoToMySqlOperator
DEFAULT_DATE = datetime(2022, 1, 1)

class TestTrinoToMySqlTransfer:

    def setup_method(self):
        if False:
            print('Hello World!')
        self.kwargs = dict(sql='sql', mysql_table='mysql_table', task_id='test_trino_to_mysql_transfer')
        args = {'owner': 'airflow', 'start_date': DEFAULT_DATE}
        self.dag = DAG('test_trino_to_mysql_transfer', default_args=args)

    @patch('airflow.providers.mysql.transfers.trino_to_mysql.MySqlHook')
    @patch('airflow.providers.mysql.transfers.trino_to_mysql.TrinoHook')
    def test_execute(self, mock_trino_hook, mock_mysql_hook):
        if False:
            print('Hello World!')
        TrinoToMySqlOperator(**self.kwargs).execute(context={})
        mock_trino_hook.return_value.get_records.assert_called_once_with(self.kwargs['sql'])
        mock_mysql_hook.return_value.insert_rows.assert_called_once_with(table=self.kwargs['mysql_table'], rows=mock_trino_hook.return_value.get_records.return_value)

    @patch('airflow.providers.mysql.transfers.trino_to_mysql.MySqlHook')
    @patch('airflow.providers.mysql.transfers.trino_to_mysql.TrinoHook')
    def test_execute_with_mysql_preoperator(self, mock_trino_hook, mock_mysql_hook):
        if False:
            for i in range(10):
                print('nop')
        self.kwargs.update(dict(mysql_preoperator='mysql_preoperator'))
        TrinoToMySqlOperator(**self.kwargs).execute(context={})
        mock_trino_hook.return_value.get_records.assert_called_once_with(self.kwargs['sql'])
        mock_mysql_hook.return_value.run.assert_called_once_with(self.kwargs['mysql_preoperator'])
        mock_mysql_hook.return_value.insert_rows.assert_called_once_with(table=self.kwargs['mysql_table'], rows=mock_trino_hook.return_value.get_records.return_value)

    @pytest.mark.skipif('AIRFLOW_RUNALL_TESTS' not in os.environ, reason='Skipped because AIRFLOW_RUNALL_TESTS is not set')
    def test_trino_to_mysql(self):
        if False:
            i = 10
            return i + 15
        op = TrinoToMySqlOperator(task_id='trino_to_mysql_check', sql='\n                SELECT name, count(*) as ccount\n                FROM airflow.static_babynames\n                GROUP BY name\n                ', mysql_table='test_static_babynames', mysql_preoperator='TRUNCATE TABLE test_static_babynames;', dag=self.dag)
        op.run(start_date=DEFAULT_DATE, end_date=DEFAULT_DATE, ignore_ti_state=True)