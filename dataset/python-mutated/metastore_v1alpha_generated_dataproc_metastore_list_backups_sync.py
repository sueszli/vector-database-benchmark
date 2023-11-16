from google.cloud import metastore_v1alpha

def sample_list_backups():
    if False:
        return 10
    client = metastore_v1alpha.DataprocMetastoreClient()
    request = metastore_v1alpha.ListBackupsRequest(parent='parent_value')
    page_result = client.list_backups(request=request)
    for response in page_result:
        print(response)