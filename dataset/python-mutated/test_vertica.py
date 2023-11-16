from __future__ import annotations
from unittest import mock
from airflow.providers.common.sql.hooks.sql import fetch_all_handler
from airflow.providers.vertica.operators.vertica import VerticaOperator

class TestVerticaOperator:

    @mock.patch('airflow.providers.common.sql.operators.sql.SQLExecuteQueryOperator.get_db_hook')
    def test_execute(self, mock_get_db_hook):
        if False:
            for i in range(10):
                print('nop')
        sql = 'select a, b, c'
        op = VerticaOperator(task_id='test_task_id', sql=sql)
        op.execute(None)
        mock_get_db_hook.return_value.run.assert_called_once_with(sql=sql, autocommit=False, handler=fetch_all_handler, parameters=None, return_last=True)