from google.cloud import gke_backup_v1

def sample_list_restores():
    if False:
        print('Hello World!')
    client = gke_backup_v1.BackupForGKEClient()
    request = gke_backup_v1.ListRestoresRequest(parent='parent_value')
    page_result = client.list_restores(request=request)
    for response in page_result:
        print(response)