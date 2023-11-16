from typing import Optional
from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine

def import_documents_sample(project_id: str, location: str, data_store_id: str, gcs_uri: Optional[str]=None, bigquery_dataset: Optional[str]=None, bigquery_table: Optional[str]=None) -> str:
    if False:
        for i in range(10):
            print('nop')
    client_options = ClientOptions(api_endpoint=f'{location}-discoveryengine.googleapis.com') if location != 'global' else None
    client = discoveryengine.DocumentServiceClient(client_options=client_options)
    parent = client.branch_path(project=project_id, location=location, data_store=data_store_id, branch='default_branch')
    if gcs_uri:
        request = discoveryengine.ImportDocumentsRequest(parent=parent, gcs_source=discoveryengine.GcsSource(input_uris=[gcs_uri], data_schema='custom'), reconciliation_mode=discoveryengine.ImportDocumentsRequest.ReconciliationMode.INCREMENTAL)
    else:
        request = discoveryengine.ImportDocumentsRequest(parent=parent, bigquery_source=discoveryengine.BigQuerySource(project_id=project_id, dataset_id=bigquery_dataset, table_id=bigquery_table, data_schema='custom'), reconciliation_mode=discoveryengine.ImportDocumentsRequest.ReconciliationMode.INCREMENTAL)
    operation = client.import_documents(request=request)
    print(f'Waiting for operation to complete: {operation.operation.name}')
    response = operation.result()
    metadata = discoveryengine.ImportDocumentsMetadata(operation.metadata)
    print(response)
    print(metadata)
    return operation.operation.name