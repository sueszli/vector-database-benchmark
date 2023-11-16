from __future__ import annotations
import warnings
from typing import Sequence
from airflow.exceptions import AirflowProviderDeprecationWarning
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator

class JdbcOperator(SQLExecuteQueryOperator):
    """
    Executes sql code in a database using jdbc driver.

    Requires jaydebeapi.

    This class is deprecated.

    Please use :class:`airflow.providers.common.sql.operators.sql.SQLExecuteQueryOperator` instead.

    .. seealso::
        For more information on how to use this operator, take a look at the guide:
        :ref:`howto/operator:JdbcOperator`

    :param sql: the SQL code to be executed as a single string, or
        a list of str (sql statements), or a reference to a template file.
        Template references are recognized by str ending in '.sql'
    :param jdbc_conn_id: reference to a predefined database
    :param autocommit: if True, each command is automatically committed.
        (default value: False)
    :param parameters: (optional) the parameters to render the SQL query with.
    """
    template_fields: Sequence[str] = ('sql',)
    template_ext: Sequence[str] = ('.sql',)
    template_fields_renderers = {'sql': 'sql'}
    ui_color = '#ededed'

    def __init__(self, *, jdbc_conn_id: str='jdbc_default', **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(conn_id=jdbc_conn_id, **kwargs)
        warnings.warn('This class is deprecated.\n            Please use `airflow.providers.common.sql.operators.sql.SQLExecuteQueryOperator`.', AirflowProviderDeprecationWarning, stacklevel=2)