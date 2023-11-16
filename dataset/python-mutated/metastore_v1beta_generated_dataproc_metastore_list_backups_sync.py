from google.cloud import metastore_v1beta

def sample_list_backups():
    if False:
        for i in range(10):
            print('nop')
    client = metastore_v1beta.DataprocMetastoreClient()
    request = metastore_v1beta.ListBackupsRequest(parent='parent_value')
    page_result = client.list_backups(request=request)
    for response in page_result:
        print(response)