from google.cloud import metastore_v1

def sample_update_metadata_import():
    if False:
        while True:
            i = 10
    client = metastore_v1.DataprocMetastoreClient()
    request = metastore_v1.UpdateMetadataImportRequest()
    operation = client.update_metadata_import(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)