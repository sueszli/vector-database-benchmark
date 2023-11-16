"""This module contains Azure Data Explorer operators."""
from __future__ import annotations
from functools import cached_property
from typing import TYPE_CHECKING, Sequence
from deprecated.classic import deprecated
from airflow.configuration import conf
from airflow.exceptions import AirflowProviderDeprecationWarning
from airflow.models import BaseOperator
from airflow.providers.microsoft.azure.hooks.adx import AzureDataExplorerHook
if TYPE_CHECKING:
    from azure.kusto.data._models import KustoResultTable
    from airflow.utils.context import Context

class AzureDataExplorerQueryOperator(BaseOperator):
    """
    Operator for querying Azure Data Explorer (Kusto).

    :param query: KQL query to run (templated).
    :param database: Database to run the query on (templated).
    :param options: Optional query options. See:
      https://docs.microsoft.com/en-us/azure/kusto/api/netfx/request-properties#list-of-clientrequestproperties
    :param azure_data_explorer_conn_id: Reference to the
        :ref:`Azure Data Explorer connection<howto/connection:adx>`.
    """
    ui_color = '#00a1f2'
    template_fields: Sequence[str] = ('query', 'database')
    template_ext: Sequence[str] = ('.kql',)

    def __init__(self, *, query: str, database: str, options: dict | None=None, azure_data_explorer_conn_id: str='azure_data_explorer_default', **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.query = query
        self.database = database
        self.options = options
        self.azure_data_explorer_conn_id = azure_data_explorer_conn_id

    @cached_property
    def hook(self) -> AzureDataExplorerHook:
        if False:
            i = 10
            return i + 15
        'Returns new instance of AzureDataExplorerHook.'
        return AzureDataExplorerHook(self.azure_data_explorer_conn_id)

    @deprecated(reason='use `hook` property instead.', category=AirflowProviderDeprecationWarning)
    def get_hook(self) -> AzureDataExplorerHook:
        if False:
            return 10
        'Returns new instance of AzureDataExplorerHook.'
        return self.hook

    def execute(self, context: Context) -> KustoResultTable | str:
        if False:
            while True:
                i = 10
        '\n        Run KQL Query on Azure Data Explorer (Kusto).\n\n        Returns `PrimaryResult` of Query v2 HTTP response contents.\n\n        https://docs.microsoft.com/en-us/azure/kusto/api/rest/response2\n        '
        response = self.hook.run_query(self.query, self.database, self.options)
        if conf.getboolean('core', 'enable_xcom_pickling'):
            return response.primary_results[0]
        else:
            return str(response.primary_results[0])