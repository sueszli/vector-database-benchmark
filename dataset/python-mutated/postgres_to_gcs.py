"""PostgreSQL to GCS operator."""
from __future__ import annotations
import datetime
import json
import time
import uuid
from decimal import Decimal
import pendulum
from airflow.providers.google.cloud.transfers.sql_to_gcs import BaseSQLToGCSOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

class _PostgresServerSideCursorDecorator:
    """
    Inspired by `_PrestoToGCSPrestoCursorAdapter` to keep this consistent.

    Decorator for allowing description to be available for postgres cursor in case server side
    cursor is used. It doesn't provide other methods except those needed in BaseSQLToGCSOperator,
    which is more of a safety feature.
    """

    def __init__(self, cursor):
        if False:
            while True:
                i = 10
        self.cursor = cursor
        self.rows = []
        self.initialized = False

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __next__(self):
        if False:
            for i in range(10):
                print('nop')
        if self.rows:
            return self.rows.pop()
        else:
            self.initialized = True
            return next(self.cursor)

    @property
    def description(self):
        if False:
            return 10
        'Fetch first row to initialize cursor description when using server side cursor.'
        if not self.initialized:
            element = self.cursor.fetchone()
            if element is not None:
                self.rows.append(element)
            self.initialized = True
        return self.cursor.description

class PostgresToGCSOperator(BaseSQLToGCSOperator):
    """
    Copy data from Postgres to Google Cloud Storage in JSON, CSV or Parquet format.

    .. seealso::
        For more information on how to use this operator, take a look at the guide:
        :ref:`howto/operator:PostgresToGCSOperator`

    :param postgres_conn_id: Reference to a specific Postgres hook.
    :param use_server_side_cursor: If server-side cursor should be used for querying postgres.
        For detailed info, check https://www.psycopg.org/docs/usage.html#server-side-cursors
    :param cursor_itersize: How many records are fetched at a time in case of server-side cursor.
    """
    ui_color = '#a0e08c'
    type_map = {1114: 'DATETIME', 1184: 'TIMESTAMP', 1082: 'DATE', 1083: 'TIME', 1005: 'INTEGER', 1007: 'INTEGER', 1016: 'INTEGER', 20: 'INTEGER', 21: 'INTEGER', 23: 'INTEGER', 16: 'BOOL', 700: 'FLOAT', 701: 'FLOAT', 1700: 'FLOAT'}

    def __init__(self, *, postgres_conn_id='postgres_default', use_server_side_cursor=False, cursor_itersize=2000, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.postgres_conn_id = postgres_conn_id
        self.use_server_side_cursor = use_server_side_cursor
        self.cursor_itersize = cursor_itersize

    def _unique_name(self):
        if False:
            for i in range(10):
                print('nop')
        return f'{self.dag_id}__{self.task_id}__{uuid.uuid4()}' if self.use_server_side_cursor else None

    def query(self):
        if False:
            for i in range(10):
                print('nop')
        'Queries Postgres and returns a cursor to the results.'
        hook = PostgresHook(postgres_conn_id=self.postgres_conn_id)
        conn = hook.get_conn()
        cursor = conn.cursor(name=self._unique_name())
        cursor.execute(self.sql, self.parameters)
        if self.use_server_side_cursor:
            cursor.itersize = self.cursor_itersize
            return _PostgresServerSideCursorDecorator(cursor)
        return cursor

    def field_to_bigquery(self, field) -> dict[str, str]:
        if False:
            print('Hello World!')
        return {'name': field[0], 'type': self.type_map.get(field[1], 'STRING'), 'mode': 'REPEATED' if field[1] in (1009, 1005, 1007, 1016) else 'NULLABLE'}

    def convert_type(self, value, schema_type, stringify_dict=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Take a value from Postgres and convert it to a value safe for JSON/Google Cloud Storage/BigQuery.\n\n        Timezone aware Datetime are converted to UTC seconds.\n        Unaware Datetime, Date and Time are converted to ISO formatted strings.\n        Decimals are converted to floats.\n\n        :param value: Postgres column value.\n        :param schema_type: BigQuery data type.\n        :param stringify_dict: Specify whether to convert dict to string.\n        '
        if isinstance(value, datetime.datetime):
            iso_format_value = value.isoformat()
            if value.tzinfo is None:
                return iso_format_value
            return pendulum.parse(iso_format_value).float_timestamp
        if isinstance(value, datetime.date):
            return value.isoformat()
        if isinstance(value, datetime.time):
            formatted_time = time.strptime(str(value), '%H:%M:%S')
            time_delta = datetime.timedelta(hours=formatted_time.tm_hour, minutes=formatted_time.tm_min, seconds=formatted_time.tm_sec)
            return str(time_delta)
        if stringify_dict and isinstance(value, dict):
            return json.dumps(value)
        if isinstance(value, Decimal):
            return float(value)
        return value