from google.cloud import gke_backup_v1

def sample_list_backup_plans():
    if False:
        for i in range(10):
            print('nop')
    client = gke_backup_v1.BackupForGKEClient()
    request = gke_backup_v1.ListBackupPlansRequest(parent='parent_value')
    page_result = client.list_backup_plans(request=request)
    for response in page_result:
        print(response)