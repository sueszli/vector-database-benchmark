from google.cloud import metastore_v1beta

def sample_get_metadata_import():
    if False:
        i = 10
        return i + 15
    client = metastore_v1beta.DataprocMetastoreClient()
    request = metastore_v1beta.GetMetadataImportRequest(name='name_value')
    response = client.get_metadata_import(request=request)
    print(response)