from google.cloud import gke_backup_v1

def sample_delete_backup_plan():
    if False:
        return 10
    client = gke_backup_v1.BackupForGKEClient()
    request = gke_backup_v1.DeleteBackupPlanRequest(name='name_value')
    operation = client.delete_backup_plan(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)