from google.cloud import gke_backup_v1

def sample_list_restore_plans():
    if False:
        return 10
    client = gke_backup_v1.BackupForGKEClient()
    request = gke_backup_v1.ListRestorePlansRequest(parent='parent_value')
    page_result = client.list_restore_plans(request=request)
    for response in page_result:
        print(response)