from __future__ import annotations
import json
import os
from typing import TYPE_CHECKING, Any, Iterable, Mapping, TypeVar
import trino
from trino.exceptions import DatabaseError
from trino.transaction import IsolationLevel
from airflow.configuration import conf
from airflow.exceptions import AirflowException
from airflow.providers.common.sql.hooks.sql import DbApiHook
from airflow.utils.operator_helpers import AIRFLOW_VAR_NAME_FORMAT_MAPPING, DEFAULT_FORMAT_PREFIX
if TYPE_CHECKING:
    from airflow.models import Connection
T = TypeVar('T')

def generate_trino_client_info() -> str:
    if False:
        for i in range(10):
            print('nop')
    'Return json string with dag_id, task_id, execution_date and try_number.'
    context_var = {format_map['default'].replace(DEFAULT_FORMAT_PREFIX, ''): os.environ.get(format_map['env_var_format'], '') for format_map in AIRFLOW_VAR_NAME_FORMAT_MAPPING.values()}
    task_info = {'dag_id': context_var['dag_id'], 'task_id': context_var['task_id'], 'execution_date': context_var['execution_date'], 'try_number': context_var['try_number'], 'dag_run_id': context_var['dag_run_id'], 'dag_owner': context_var['dag_owner']}
    return json.dumps(task_info, sort_keys=True)

class TrinoException(Exception):
    """Trino exception."""

def _boolify(value):
    if False:
        i = 10
        return i + 15
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() == 'false':
            return False
        elif value.lower() == 'true':
            return True
    return value

class TrinoHook(DbApiHook):
    """
    Interact with Trino through trino package.

    >>> ph = TrinoHook()
    >>> sql = "SELECT count(1) AS num FROM airflow.static_babynames"
    >>> ph.get_records(sql)
    [[340698]]
    """
    conn_name_attr = 'trino_conn_id'
    default_conn_name = 'trino_default'
    conn_type = 'trino'
    hook_name = 'Trino'
    query_id = ''
    placeholder = '?'
    _test_connection_sql = 'select 1'

    def get_conn(self) -> Connection:
        if False:
            print('Hello World!')
        'Returns a connection object.'
        db = self.get_connection(self.trino_conn_id)
        extra = db.extra_dejson
        auth = None
        user = db.login
        if db.password and extra.get('auth') in ('kerberos', 'certs'):
            raise AirflowException(f"The {extra.get('auth')!r} authorization type doesn't support password.")
        elif db.password:
            auth = trino.auth.BasicAuthentication(db.login, db.password)
        elif extra.get('auth') == 'jwt':
            if 'jwt__file' in extra:
                with open(extra.get('jwt__file')) as jwt_file:
                    token = jwt_file.read()
            else:
                token = extra.get('jwt__token')
            auth = trino.auth.JWTAuthentication(token=token)
        elif extra.get('auth') == 'certs':
            auth = trino.auth.CertificateAuthentication(extra.get('certs__client_cert_path'), extra.get('certs__client_key_path'))
        elif extra.get('auth') == 'kerberos':
            auth = trino.auth.KerberosAuthentication(config=extra.get('kerberos__config', os.environ.get('KRB5_CONFIG')), service_name=extra.get('kerberos__service_name'), mutual_authentication=_boolify(extra.get('kerberos__mutual_authentication', False)), force_preemptive=_boolify(extra.get('kerberos__force_preemptive', False)), hostname_override=extra.get('kerberos__hostname_override'), sanitize_mutual_error_response=_boolify(extra.get('kerberos__sanitize_mutual_error_response', True)), principal=extra.get('kerberos__principal', conf.get('kerberos', 'principal')), delegate=_boolify(extra.get('kerberos__delegate', False)), ca_bundle=extra.get('kerberos__ca_bundle'))
        if _boolify(extra.get('impersonate_as_owner', False)):
            user = os.getenv('AIRFLOW_CTX_DAG_OWNER', None)
            if user is None:
                user = db.login
        http_headers = {'X-Trino-Client-Info': generate_trino_client_info()}
        trino_conn = trino.dbapi.connect(host=db.host, port=db.port, user=user, source=extra.get('source', 'airflow'), http_scheme=extra.get('protocol', 'http'), http_headers=http_headers, catalog=extra.get('catalog', 'hive'), schema=db.schema, auth=auth, isolation_level=self.get_isolation_level(), verify=_boolify(extra.get('verify', True)), session_properties=extra.get('session_properties') or None, client_tags=extra.get('client_tags') or None)
        return trino_conn

    def get_isolation_level(self) -> Any:
        if False:
            i = 10
            return i + 15
        'Returns an isolation level.'
        db = self.get_connection(self.trino_conn_id)
        isolation_level = db.extra_dejson.get('isolation_level', 'AUTOCOMMIT').upper()
        return getattr(IsolationLevel, isolation_level, IsolationLevel.AUTOCOMMIT)

    def get_records(self, sql: str | list[str]='', parameters: Iterable | Mapping[str, Any] | None=None) -> Any:
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(sql, str):
            raise ValueError(f'The sql in Trino Hook must be a string and is {sql}!')
        try:
            return super().get_records(self.strip_sql_string(sql), parameters)
        except DatabaseError as e:
            raise TrinoException(e)

    def get_first(self, sql: str | list[str]='', parameters: Iterable | Mapping[str, Any] | None=None) -> Any:
        if False:
            while True:
                i = 10
        if not isinstance(sql, str):
            raise ValueError(f'The sql in Trino Hook must be a string and is {sql}!')
        try:
            return super().get_first(self.strip_sql_string(sql), parameters)
        except DatabaseError as e:
            raise TrinoException(e)

    def get_pandas_df(self, sql: str='', parameters: Iterable | Mapping[str, Any] | None=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        import pandas as pd
        cursor = self.get_cursor()
        try:
            cursor.execute(self.strip_sql_string(sql), parameters)
            data = cursor.fetchall()
        except DatabaseError as e:
            raise TrinoException(e)
        column_descriptions = cursor.description
        if data:
            df = pd.DataFrame(data, **kwargs)
            df.rename(columns={n: c[0] for (n, c) in zip(df.columns, column_descriptions)}, inplace=True)
        else:
            df = pd.DataFrame(**kwargs)
        return df

    def insert_rows(self, table: str, rows: Iterable[tuple], target_fields: Iterable[str] | None=None, commit_every: int=0, replace: bool=False, **kwargs) -> None:
        if False:
            while True:
                i = 10
        '\n        A generic way to insert a set of tuples into a table.\n\n        :param table: Name of the target table\n        :param rows: The rows to insert into the table\n        :param target_fields: The names of the columns to fill in the table\n        :param commit_every: The maximum number of rows to insert in one\n            transaction. Set to 0 to insert all rows in one transaction.\n        :param replace: Whether to replace instead of insert\n        '
        if self.get_isolation_level() == IsolationLevel.AUTOCOMMIT:
            self.log.info('Transactions are not enable in trino connection. Please use the isolation_level property to enable it. Falling back to insert all rows in one transaction.')
            commit_every = 0
        super().insert_rows(table, rows, target_fields, commit_every, replace)

    @staticmethod
    def _serialize_cell(cell: Any, conn: Connection | None=None) -> Any:
        if False:
            i = 10
            return i + 15
        '\n        Trino will adapt all execute() args internally, hence we return cell without any conversion.\n\n        :param cell: The cell to insert into the table\n        :param conn: The database connection\n        :return: The cell\n        '
        return cell

    def get_openlineage_database_info(self, connection):
        if False:
            i = 10
            return i + 15
        'Returns Trino specific information for OpenLineage.'
        from airflow.providers.openlineage.sqlparser import DatabaseInfo
        return DatabaseInfo(scheme='trino', authority=DbApiHook.get_openlineage_authority_part(connection, default_port=trino.constants.DEFAULT_PORT), information_schema_columns=['table_schema', 'table_name', 'column_name', 'ordinal_position', 'data_type', 'table_catalog'], database=connection.extra_dejson.get('catalog', 'hive'), is_information_schema_cross_db=True)

    def get_openlineage_database_dialect(self, _):
        if False:
            return 10
        'Returns Trino dialect.'
        return 'trino'

    def get_openlineage_default_schema(self):
        if False:
            print('Hello World!')
        'Returns Trino default schema.'
        return trino.constants.DEFAULT_SCHEMA