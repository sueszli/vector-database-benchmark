from google.cloud import metastore_v1

def sample_list_metadata_imports():
    if False:
        while True:
            i = 10
    client = metastore_v1.DataprocMetastoreClient()
    request = metastore_v1.ListMetadataImportsRequest(parent='parent_value')
    page_result = client.list_metadata_imports(request=request)
    for response in page_result:
        print(response)