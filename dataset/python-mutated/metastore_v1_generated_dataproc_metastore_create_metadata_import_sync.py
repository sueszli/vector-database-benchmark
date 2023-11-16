from google.cloud import metastore_v1

def sample_create_metadata_import():
    if False:
        while True:
            i = 10
    client = metastore_v1.DataprocMetastoreClient()
    request = metastore_v1.CreateMetadataImportRequest(parent='parent_value', metadata_import_id='metadata_import_id_value')
    operation = client.create_metadata_import(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)