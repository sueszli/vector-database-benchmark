from __future__ import annotations
import warnings
from typing import Any, Sequence
from airflow.exceptions import AirflowProviderDeprecationWarning
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator

class VerticaOperator(SQLExecuteQueryOperator):
    """
    Executes sql code in a specific Vertica database.

    This class is deprecated.

    Please use :class:`airflow.providers.common.sql.operators.sql.SQLExecuteQueryOperator`.

    :param vertica_conn_id: reference to a specific Vertica database
    :param sql: the SQL code to be executed as a single string, or
        a list of str (sql statements), or a reference to a template file.
        Template references are recognized by str ending in '.sql'
    """
    template_fields: Sequence[str] = ('sql',)
    template_ext: Sequence[str] = ('.sql',)
    template_fields_renderers = {'sql': 'sql'}
    ui_color = '#b4e0ff'

    def __init__(self, *, vertica_conn_id: str='vertica_default', **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(conn_id=vertica_conn_id, **kwargs)
        warnings.warn('This class is deprecated.\n            Please use `airflow.providers.common.sql.operators.sql.SQLExecuteQueryOperator`.', AirflowProviderDeprecationWarning, stacklevel=2)