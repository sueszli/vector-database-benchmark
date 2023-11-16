"""MsSQL to GCS operator."""
from __future__ import annotations
import datetime
import decimal
from typing import Sequence
from airflow.providers.google.cloud.transfers.sql_to_gcs import BaseSQLToGCSOperator
from airflow.providers.microsoft.mssql.hooks.mssql import MsSqlHook

class MSSQLToGCSOperator(BaseSQLToGCSOperator):
    """
    Copy data from Microsoft SQL Server to Google Cloud Storage in JSON, CSV or Parquet format.

    :param bit_fields: Sequence of fields names of MSSQL "BIT" data type,
        to be interpreted in the schema as "BOOLEAN". "BIT" fields that won't
        be included in this sequence, will be interpreted as "INTEGER" by
        default.
    :param mssql_conn_id: Reference to a specific MSSQL hook.

    **Example**:
        The following operator will export data from the Customers table
        within the given MSSQL Database and then upload it to the
        'mssql-export' GCS bucket (along with a schema file). ::

            export_customers = MsSqlToGoogleCloudStorageOperator(
                task_id='export_customers',
                sql='SELECT * FROM dbo.Customers;',
                bit_fields=['some_bit_field', 'another_bit_field'],
                bucket='mssql-export',
                filename='data/customers/export.json',
                schema_filename='schemas/export.json',
                mssql_conn_id='mssql_default',
                gcp_conn_id='google_cloud_default',
                dag=dag
            )

    .. seealso::
        For more information on how to use this operator, take a look at the guide:
        :ref:`howto/operator:MSSQLToGCSOperator`

    """
    ui_color = '#e0a98c'
    type_map = {2: 'BOOLEAN', 3: 'INTEGER', 4: 'TIMESTAMP', 5: 'NUMERIC'}

    def __init__(self, *, bit_fields: Sequence[str] | None=None, mssql_conn_id='mssql_default', **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.mssql_conn_id = mssql_conn_id
        self.bit_fields = bit_fields or []

    def query(self):
        if False:
            return 10
        '\n        Queries MSSQL and returns a cursor of results.\n\n        :return: mssql cursor\n        '
        mssql = MsSqlHook(mssql_conn_id=self.mssql_conn_id)
        conn = mssql.get_conn()
        cursor = conn.cursor()
        cursor.execute(self.sql)
        return cursor

    def field_to_bigquery(self, field) -> dict[str, str]:
        if False:
            for i in range(10):
                print('nop')
        if field[0] in self.bit_fields:
            field = (field[0], 2)
        return {'name': field[0].replace(' ', '_'), 'type': self.type_map.get(field[1], 'STRING'), 'mode': 'NULLABLE'}

    @classmethod
    def convert_type(cls, value, schema_type, **kwargs):
        if False:
            return 10
        '\n        Take a value from MSSQL and convert it to a value safe for JSON/Google Cloud Storage/BigQuery.\n\n        Datetime, Date and Time are converted to ISO formatted strings.\n        '
        if isinstance(value, decimal.Decimal):
            return float(value)
        if isinstance(value, (datetime.date, datetime.time)):
            return value.isoformat()
        return value