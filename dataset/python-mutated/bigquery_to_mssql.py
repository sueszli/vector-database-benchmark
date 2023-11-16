"""This module contains Google BigQuery to MSSQL operator."""
from __future__ import annotations
import warnings
from typing import TYPE_CHECKING, Sequence
from airflow.exceptions import AirflowProviderDeprecationWarning
from airflow.providers.google.cloud.links.bigquery import BigQueryTableLink
from airflow.providers.google.cloud.transfers.bigquery_to_sql import BigQueryToSqlBaseOperator
from airflow.providers.microsoft.mssql.hooks.mssql import MsSqlHook
if TYPE_CHECKING:
    from airflow.utils.context import Context

class BigQueryToMsSqlOperator(BigQueryToSqlBaseOperator):
    """
    Fetch data from a BigQuery table (alternatively fetch selected columns) and insert it into a MSSQL table.

    .. seealso::
        For more information on how to use this operator, take a look at the guide:
        :ref:`howto/operator:BigQueryToMsSqlOperator`

    :param source_project_dataset_table: A dotted ``<project>.<dataset>.<table>``:
        the big query table of origin
    :param mssql_table: target MsSQL table. It is deprecated: use target_table_name instead. (templated)
    :param target_table_name: target MsSQL table. It takes precedence over mssql_table. (templated)
    :param mssql_conn_id: reference to a specific mssql hook
    """
    template_fields: Sequence[str] = (*BigQueryToSqlBaseOperator.template_fields, 'source_project_dataset_table')
    operator_extra_links = (BigQueryTableLink(),)

    def __init__(self, *, source_project_dataset_table: str, mssql_table: str | None=None, target_table_name: str | None=None, mssql_conn_id: str='mssql_default', **kwargs) -> None:
        if False:
            print('Hello World!')
        if mssql_table is not None:
            warnings.warn('The `mssql_table` parameter has been deprecated. Use `target_table_name` instead.', AirflowProviderDeprecationWarning)
            if target_table_name is not None:
                raise ValueError(f'Cannot set both arguments: mssql_table={mssql_table!r} and target_table_name={target_table_name!r}.')
            target_table_name = mssql_table
        try:
            (_, dataset_id, table_id) = source_project_dataset_table.split('.')
        except ValueError:
            raise ValueError(f'Could not parse {source_project_dataset_table} as <project>.<dataset>.<table>') from None
        super().__init__(target_table_name=target_table_name, dataset_table=f'{dataset_id}.{table_id}', **kwargs)
        self.mssql_conn_id = mssql_conn_id
        self.source_project_dataset_table = source_project_dataset_table

    def get_sql_hook(self) -> MsSqlHook:
        if False:
            i = 10
            return i + 15
        return MsSqlHook(schema=self.database, mysql_conn_id=self.mssql_conn_id)

    def persist_links(self, context: Context) -> None:
        if False:
            return 10
        (project_id, dataset_id, table_id) = self.source_project_dataset_table.split('.')
        BigQueryTableLink.persist(context=context, task_instance=self, dataset_id=dataset_id, project_id=project_id, table_id=table_id)