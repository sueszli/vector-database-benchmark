from __future__ import annotations
from unittest import mock
from airflow.providers.exasol.hooks.exasol import exasol_fetch_all_handler
from airflow.providers.exasol.operators.exasol import ExasolOperator

class TestExasol:

    @mock.patch('airflow.providers.common.sql.operators.sql.SQLExecuteQueryOperator.get_db_hook')
    def test_overwrite_autocommit(self, mock_get_db_hook):
        if False:
            while True:
                i = 10
        operator = ExasolOperator(task_id='TEST', sql='SELECT 1', autocommit=True)
        operator.execute({})
        mock_get_db_hook.return_value.run.assert_called_once_with(sql='SELECT 1', autocommit=True, parameters=None, handler=exasol_fetch_all_handler, return_last=True)

    @mock.patch('airflow.providers.common.sql.operators.sql.SQLExecuteQueryOperator.get_db_hook')
    def test_pass_parameters(self, mock_get_db_hook):
        if False:
            return 10
        operator = ExasolOperator(task_id='TEST', sql='SELECT {value!s}', parameters={'value': 1})
        operator.execute({})
        mock_get_db_hook.return_value.run.assert_called_once_with(sql='SELECT {value!s}', autocommit=False, parameters={'value': 1}, handler=exasol_fetch_all_handler, return_last=True)

    @mock.patch('airflow.providers.common.sql.operators.sql.BaseSQLOperator.__init__')
    def test_overwrite_schema(self, mock_base_op):
        if False:
            return 10
        ExasolOperator(task_id='TEST', sql='SELECT 1', schema='dummy')
        mock_base_op.assert_called_once_with(conn_id='exasol_default', database=None, hook_params={'schema': 'dummy'}, default_args={}, task_id='TEST')