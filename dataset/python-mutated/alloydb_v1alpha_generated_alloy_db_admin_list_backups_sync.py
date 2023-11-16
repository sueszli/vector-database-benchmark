from google.cloud import alloydb_v1alpha

def sample_list_backups():
    if False:
        while True:
            i = 10
    client = alloydb_v1alpha.AlloyDBAdminClient()
    request = alloydb_v1alpha.ListBackupsRequest(parent='parent_value')
    page_result = client.list_backups(request=request)
    for response in page_result:
        print(response)