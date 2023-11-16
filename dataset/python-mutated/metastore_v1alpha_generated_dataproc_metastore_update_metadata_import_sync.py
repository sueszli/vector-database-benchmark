from google.cloud import metastore_v1alpha

def sample_update_metadata_import():
    if False:
        print('Hello World!')
    client = metastore_v1alpha.DataprocMetastoreClient()
    request = metastore_v1alpha.UpdateMetadataImportRequest()
    operation = client.update_metadata_import(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)