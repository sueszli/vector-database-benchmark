from __future__ import annotations
import warnings
from typing import Sequence
from airflow.exceptions import AirflowProviderDeprecationWarning
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator

class MsSqlOperator(SQLExecuteQueryOperator):
    """
    Executes sql code in a specific Microsoft SQL database.

    .. seealso::
        For more information on how to use this operator, take a look at the guide:
        :ref:`howto/operator:MsSqlOperator`

    This operator may use one of two hooks, depending on the ``conn_type`` of the connection.

    If conn_type is ``'odbc'``, then :py:class:`~airflow.providers.odbc.hooks.odbc.OdbcHook`
    is used.  Otherwise, :py:class:`~airflow.providers.microsoft.mssql.hooks.mssql.MsSqlHook` is used.

    This class is deprecated.

    Please use :class:`airflow.providers.common.sql.operators.sql.SQLExecuteQueryOperator`.

    :param sql: the sql code to be executed (templated)
    :param mssql_conn_id: reference to a specific mssql database
    :param parameters: (optional) the parameters to render the SQL query with.
    :param autocommit: if True, each command is automatically committed.
        (default value: False)
    :param database: name of database which overwrite defined one in connection
    """
    template_fields: Sequence[str] = ('sql',)
    template_ext: Sequence[str] = ('.sql',)
    template_fields_renderers = {'sql': 'tsql'}
    ui_color = '#ededed'

    def __init__(self, *, mssql_conn_id: str='mssql_default', database: str | None=None, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        if database is not None:
            hook_params = kwargs.pop('hook_params', {})
            kwargs['hook_params'] = {'schema': database, **hook_params}
        super().__init__(conn_id=mssql_conn_id, **kwargs)
        warnings.warn("This class is deprecated.\n            Please use `airflow.providers.common.sql.operators.sql.SQLExecuteQueryOperator`.\n            Also, you can provide `hook_params={'schema': <database>}`.", AirflowProviderDeprecationWarning, stacklevel=2)