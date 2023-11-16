"""This module contains Google BigQuery to PostgreSQL operator."""
from __future__ import annotations
from typing import Sequence
from airflow.providers.google.cloud.transfers.bigquery_to_sql import BigQueryToSqlBaseOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

class BigQueryToPostgresOperator(BigQueryToSqlBaseOperator):
    """
    Fetch data from a BigQuery table (alternatively fetch selected columns) and insert into PostgreSQL table.

    .. seealso::
        For more information on how to use this operator, take a look at the guide:
        :ref:`howto/operator:BigQueryToPostgresOperator`

    :param target_table_name: target Postgres table (templated)
    :param postgres_conn_id: Reference to :ref:`postgres connection id <howto/connection:postgres>`.
    """
    template_fields: Sequence[str] = (*BigQueryToSqlBaseOperator.template_fields, 'dataset_id', 'table_id')

    def __init__(self, *, target_table_name: str, postgres_conn_id: str='postgres_default', **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(target_table_name=target_table_name, **kwargs)
        self.postgres_conn_id = postgres_conn_id

    def get_sql_hook(self) -> PostgresHook:
        if False:
            while True:
                i = 10
        return PostgresHook(schema=self.database, postgres_conn_id=self.postgres_conn_id)