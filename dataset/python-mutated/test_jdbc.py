from __future__ import annotations
from unittest.mock import patch
from airflow.providers.common.sql.hooks.sql import fetch_all_handler
from airflow.providers.jdbc.operators.jdbc import JdbcOperator

class TestJdbcOperator:

    def setup_method(self):
        if False:
            print('Hello World!')
        self.kwargs = dict(sql='sql', task_id='test_jdbc_operator', dag=None)

    @patch('airflow.providers.common.sql.operators.sql.SQLExecuteQueryOperator.get_db_hook')
    def test_execute_do_push(self, mock_get_db_hook):
        if False:
            for i in range(10):
                print('nop')
        jdbc_operator = JdbcOperator(**self.kwargs, do_xcom_push=True)
        jdbc_operator.execute(context={})
        mock_get_db_hook.return_value.run.assert_called_once_with(sql=jdbc_operator.sql, autocommit=jdbc_operator.autocommit, handler=fetch_all_handler, parameters=jdbc_operator.parameters, return_last=True)

    @patch('airflow.providers.common.sql.operators.sql.SQLExecuteQueryOperator.get_db_hook')
    def test_execute_dont_push(self, mock_get_db_hook):
        if False:
            print('Hello World!')
        jdbc_operator = JdbcOperator(**self.kwargs, do_xcom_push=False)
        jdbc_operator.execute(context={})
        mock_get_db_hook.return_value.run.assert_called_once_with(sql=jdbc_operator.sql, autocommit=jdbc_operator.autocommit, parameters=jdbc_operator.parameters, handler=None, return_last=True)