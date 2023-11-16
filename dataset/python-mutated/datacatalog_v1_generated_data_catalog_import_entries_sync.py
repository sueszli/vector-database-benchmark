from google.cloud import datacatalog_v1

def sample_import_entries():
    if False:
        print('Hello World!')
    client = datacatalog_v1.DataCatalogClient()
    request = datacatalog_v1.ImportEntriesRequest(gcs_bucket_path='gcs_bucket_path_value', parent='parent_value')
    operation = client.import_entries(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)