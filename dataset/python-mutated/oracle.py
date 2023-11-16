from __future__ import annotations
import re
import warnings
from typing import TYPE_CHECKING, Sequence
import oracledb
from airflow.exceptions import AirflowProviderDeprecationWarning
from airflow.models import BaseOperator
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from airflow.providers.oracle.hooks.oracle import OracleHook
if TYPE_CHECKING:
    from airflow.utils.context import Context

class OracleOperator(SQLExecuteQueryOperator):
    """
    Executes sql code in a specific Oracle database.

    This class is deprecated.

    Please use :class:`airflow.providers.common.sql.operators.sql.SQLExecuteQueryOperator`.

    :param sql: the sql code to be executed. Can receive a str representing a sql statement,
        a list of str (sql statements), or reference to a template file.
        Template reference are recognized by str ending in '.sql'
        (templated)
    :param oracle_conn_id: The :ref:`Oracle connection id <howto/connection:oracle>`
        reference to a specific Oracle database.
    :param parameters: (optional, templated) the parameters to render the SQL query with.
    :param autocommit: if True, each command is automatically committed.
        (default value: False)
    """
    template_fields: Sequence[str] = ('parameters', 'sql')
    template_ext: Sequence[str] = ('.sql',)
    template_fields_renderers = {'sql': 'sql'}
    ui_color = '#ededed'

    def __init__(self, *, oracle_conn_id: str='oracle_default', **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(conn_id=oracle_conn_id, **kwargs)
        warnings.warn('This class is deprecated.\n            Please use `airflow.providers.common.sql.operators.sql.SQLExecuteQueryOperator`.', AirflowProviderDeprecationWarning, stacklevel=2)

class OracleStoredProcedureOperator(BaseOperator):
    """
    Executes stored procedure in a specific Oracle database.

    :param procedure: name of stored procedure to call (templated)
    :param oracle_conn_id: The :ref:`Oracle connection id <howto/connection:oracle>`
        reference to a specific Oracle database.
    :param parameters: (optional, templated) the parameters provided in the call

    If *do_xcom_push* is *True*, the numeric exit code emitted by
    the database is pushed to XCom under key ``ORA`` in case of failure.
    """
    template_fields: Sequence[str] = ('parameters', 'procedure')
    ui_color = '#ededed'

    def __init__(self, *, procedure: str, oracle_conn_id: str='oracle_default', parameters: dict | list | None=None, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.oracle_conn_id = oracle_conn_id
        self.procedure = procedure
        self.parameters = parameters

    def execute(self, context: Context):
        if False:
            i = 10
            return i + 15
        self.log.info('Executing: %s', self.procedure)
        hook = OracleHook(oracle_conn_id=self.oracle_conn_id)
        try:
            return hook.callproc(self.procedure, autocommit=True, parameters=self.parameters)
        except oracledb.DatabaseError as e:
            if not self.do_xcom_push or not context:
                raise
            ti = context['ti']
            code_match = re.search('^ORA-(\\d+):.+', str(e))
            if code_match:
                ti.xcom_push(key='ORA', value=code_match.group(1))
            raise