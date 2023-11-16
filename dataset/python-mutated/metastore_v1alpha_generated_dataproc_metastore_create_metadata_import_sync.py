from google.cloud import metastore_v1alpha

def sample_create_metadata_import():
    if False:
        for i in range(10):
            print('nop')
    client = metastore_v1alpha.DataprocMetastoreClient()
    request = metastore_v1alpha.CreateMetadataImportRequest(parent='parent_value', metadata_import_id='metadata_import_id_value')
    operation = client.create_metadata_import(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)