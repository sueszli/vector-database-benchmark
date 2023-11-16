from __future__ import annotations
from typing import TYPE_CHECKING, Sequence
from airflow.providers.microsoft.azure.hooks.cosmos import AzureCosmosDBHook
from airflow.sensors.base import BaseSensorOperator
if TYPE_CHECKING:
    from airflow.utils.context import Context

class AzureCosmosDocumentSensor(BaseSensorOperator):
    """
    Checks for the existence of a document which matches the given query in CosmosDB.

    .. code-block:: python

        azure_cosmos_sensor = AzureCosmosDocumentSensor(
            database_name="somedatabase_name",
            collection_name="somecollection_name",
            document_id="unique-doc-id",
            azure_cosmos_conn_id="azure_cosmos_default",
            task_id="azure_cosmos_sensor",
        )

    :param database_name: Target CosmosDB database_name.
    :param collection_name: Target CosmosDB collection_name.
    :param document_id: The ID of the target document.
    :param azure_cosmos_conn_id: Reference to the
        :ref:`Azure CosmosDB connection<howto/connection:azure_cosmos>`.
    """
    template_fields: Sequence[str] = ('database_name', 'collection_name', 'document_id')

    def __init__(self, *, database_name: str, collection_name: str, document_id: str, azure_cosmos_conn_id: str='azure_cosmos_default', **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.azure_cosmos_conn_id = azure_cosmos_conn_id
        self.database_name = database_name
        self.collection_name = collection_name
        self.document_id = document_id

    def poke(self, context: Context) -> bool:
        if False:
            i = 10
            return i + 15
        self.log.info('*** Entering poke')
        hook = AzureCosmosDBHook(self.azure_cosmos_conn_id)
        return hook.get_document(self.document_id, self.database_name, self.collection_name) is not None