from google.cloud import gke_backup_v1

def sample_list_backups():
    if False:
        i = 10
        return i + 15
    client = gke_backup_v1.BackupForGKEClient()
    request = gke_backup_v1.ListBackupsRequest(parent='parent_value')
    page_result = client.list_backups(request=request)
    for response in page_result:
        print(response)