from __future__ import annotations
import base64
import calendar
from datetime import date, datetime, timedelta
from decimal import Decimal
import oracledb
from airflow.providers.google.cloud.transfers.sql_to_gcs import BaseSQLToGCSOperator
from airflow.providers.oracle.hooks.oracle import OracleHook

class OracleToGCSOperator(BaseSQLToGCSOperator):
    """Copy data from Oracle to Google Cloud Storage in JSON, CSV or Parquet format.

    .. seealso::
        For more information on how to use this operator, take a look at the guide:
        :ref:`howto/operator:OracleToGCSOperator`

    :param oracle_conn_id: Reference to a specific
        :ref:`Oracle hook <howto/connection:oracle>`.
    :param ensure_utc: Ensure TIMESTAMP columns exported as UTC. If set to
        `False`, TIMESTAMP columns will be exported using the Oracle server's
        default timezone.
    """
    ui_color = '#a0e08c'
    type_map = {oracledb.DB_TYPE_BINARY_DOUBLE: 'DECIMAL', oracledb.DB_TYPE_BINARY_FLOAT: 'DECIMAL', oracledb.DB_TYPE_BINARY_INTEGER: 'INTEGER', oracledb.DB_TYPE_BOOLEAN: 'BOOLEAN', oracledb.DB_TYPE_DATE: 'TIMESTAMP', oracledb.DB_TYPE_NUMBER: 'NUMERIC', oracledb.DB_TYPE_TIMESTAMP: 'TIMESTAMP', oracledb.DB_TYPE_TIMESTAMP_LTZ: 'TIMESTAMP', oracledb.DB_TYPE_TIMESTAMP_TZ: 'TIMESTAMP'}

    def __init__(self, *, oracle_conn_id='oracle_default', ensure_utc=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.ensure_utc = ensure_utc
        self.oracle_conn_id = oracle_conn_id

    def query(self):
        if False:
            return 10
        'Queries Oracle and returns a cursor to the results.'
        oracle = OracleHook(oracle_conn_id=self.oracle_conn_id)
        conn = oracle.get_conn()
        cursor = conn.cursor()
        if self.ensure_utc:
            tz_query = "SET time_zone = '+00:00'"
            self.log.info('Executing: %s', tz_query)
            cursor.execute(tz_query)
        self.log.info('Executing: %s', self.sql)
        cursor.execute(self.sql)
        return cursor

    def field_to_bigquery(self, field) -> dict[str, str]:
        if False:
            for i in range(10):
                print('nop')
        field_type = self.type_map.get(field[1], 'STRING')
        field_mode = 'NULLABLE' if not field[6] or field_type == 'TIMESTAMP' else 'REQUIRED'
        return {'name': field[0], 'type': field_type, 'mode': field_mode}

    def convert_type(self, value, schema_type, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Take a value from Oracle db and convert it to a value safe for JSON/Google Cloud Storage/BigQuery.\n\n        * Datetimes are converted to UTC seconds.\n        * Decimals are converted to floats.\n        * Dates are converted to ISO formatted string if given schema_type is\n          DATE, or UTC seconds otherwise.\n        * Binary type fields are converted to integer if given schema_type is\n          INTEGER, or encoded with base64 otherwise. Imported BYTES data must\n          be base64-encoded according to BigQuery documentation:\n          https://cloud.google.com/bigquery/data-types\n\n        :param value: Oracle db column value\n        :param schema_type: BigQuery data type\n        '
        if value is None:
            return value
        if isinstance(value, datetime):
            value = calendar.timegm(value.timetuple())
        elif isinstance(value, timedelta):
            value = value.total_seconds()
        elif isinstance(value, Decimal):
            value = float(value)
        elif isinstance(value, date):
            if schema_type == 'DATE':
                value = value.isoformat()
            else:
                value = calendar.timegm(value.timetuple())
        elif isinstance(value, bytes):
            if schema_type == 'INTEGER':
                value = int.from_bytes(value, 'big')
            else:
                value = base64.standard_b64encode(value).decode('ascii')
        return value