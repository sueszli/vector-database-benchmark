from google.cloud import filestore_v1

def sample_list_backups():
    if False:
        return 10
    client = filestore_v1.CloudFilestoreManagerClient()
    request = filestore_v1.ListBackupsRequest(parent='parent_value')
    page_result = client.list_backups(request=request)
    for response in page_result:
        print(response)