from google.cloud import metastore_v1

def sample_get_metadata_import():
    if False:
        for i in range(10):
            print('nop')
    client = metastore_v1.DataprocMetastoreClient()
    request = metastore_v1.GetMetadataImportRequest(name='name_value')
    response = client.get_metadata_import(request=request)
    print(response)