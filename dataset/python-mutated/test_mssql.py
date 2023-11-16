from __future__ import annotations
from unittest import mock
from unittest.mock import MagicMock, Mock
import pytest
from airflow.exceptions import AirflowException
try:
    from airflow.providers.microsoft.mssql.hooks.mssql import MsSqlHook
    from airflow.providers.microsoft.mssql.operators.mssql import MsSqlOperator
except ImportError:
    pytest.skip('MSSQL not available', allow_module_level=True)

class TestMsSqlOperator:

    @mock.patch('airflow.providers.common.sql.operators.sql.SQLExecuteQueryOperator.get_db_hook')
    def test_get_hook_from_conn(self, mock_get_db_hook):
        if False:
            i = 10
            return i + 15
        '\n        :class:`~.MsSqlOperator` should use the hook returned by :meth:`airflow.models.Connection.get_hook`\n        if one is returned.\n\n        This behavior is necessary in order to support usage of :class:`~.OdbcHook` with this operator.\n\n        Specifically we verify here that :meth:`~.MsSqlOperator.get_hook` returns the hook returned from a\n        call of ``get_hook`` on the object returned from :meth:`~.BaseHook.get_connection`.\n        '
        mock_hook = MagicMock()
        mock_get_db_hook.return_value = mock_hook
        op = MsSqlOperator(task_id='test', sql='')
        assert op.get_db_hook() == mock_hook

    @mock.patch('airflow.providers.common.sql.operators.sql.SQLExecuteQueryOperator.get_db_hook', autospec=MsSqlHook)
    def test_get_hook_default(self, mock_get_db_hook):
        if False:
            i = 10
            return i + 15
        '\n        If :meth:`airflow.models.Connection.get_hook` does not return a hook (e.g. because of an invalid\n        conn type), then :class:`~.MsSqlHook` should be used.\n        '
        mock_get_db_hook.return_value.side_effect = Mock(side_effect=AirflowException())
        op = MsSqlOperator(task_id='test', sql='')
        assert op.get_db_hook().__class__.__name__ == 'MsSqlHook'