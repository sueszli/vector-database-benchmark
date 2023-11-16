from google.cloud import metastore_v1beta

def sample_list_metadata_imports():
    if False:
        print('Hello World!')
    client = metastore_v1beta.DataprocMetastoreClient()
    request = metastore_v1beta.ListMetadataImportsRequest(parent='parent_value')
    page_result = client.list_metadata_imports(request=request)
    for response in page_result:
        print(response)