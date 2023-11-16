from __future__ import annotations
import warnings
from typing import Sequence
from airflow.exceptions import AirflowProviderDeprecationWarning
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator

class DrillOperator(SQLExecuteQueryOperator):
    """
    Executes the provided SQL in the identified Drill environment.

    This class is deprecated.

    Please use :class:`airflow.providers.common.sql.operators.sql.SQLExecuteQueryOperator`.

    .. seealso::
        For more information on how to use this operator, take a look at the guide:
        :ref:`howto/operator:DrillOperator`

    :param sql: the SQL code to be executed as a single string, or
        a list of str (sql statements), or a reference to a template file.
        Template references are recognized by str ending in '.sql'
    :param drill_conn_id: id of the connection config for the target Drill
        environment
    :param parameters: (optional) the parameters to render the SQL query with.
    """
    template_fields: Sequence[str] = ('sql',)
    template_fields_renderers = {'sql': 'sql'}
    template_ext: Sequence[str] = ('.sql',)
    ui_color = '#ededed'

    def __init__(self, *, drill_conn_id: str='drill_default', **kwargs) -> None:
        if False:
            return 10
        super().__init__(conn_id=drill_conn_id, **kwargs)
        warnings.warn('This class is deprecated.\n            Please use `airflow.providers.common.sql.operators.sql.SQLExecuteQueryOperator`.', AirflowProviderDeprecationWarning, stacklevel=2)