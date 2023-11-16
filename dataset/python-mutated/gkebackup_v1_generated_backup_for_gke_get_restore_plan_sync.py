from google.cloud import gke_backup_v1

def sample_get_restore_plan():
    if False:
        return 10
    client = gke_backup_v1.BackupForGKEClient()
    request = gke_backup_v1.GetRestorePlanRequest(name='name_value')
    response = client.get_restore_plan(request=request)
    print(response)