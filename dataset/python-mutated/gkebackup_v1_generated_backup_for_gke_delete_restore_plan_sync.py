from google.cloud import gke_backup_v1

def sample_delete_restore_plan():
    if False:
        print('Hello World!')
    client = gke_backup_v1.BackupForGKEClient()
    request = gke_backup_v1.DeleteRestorePlanRequest(name='name_value')
    operation = client.delete_restore_plan(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)