from google.cloud import gke_backup_v1

def sample_delete_restore():
    if False:
        i = 10
        return i + 15
    client = gke_backup_v1.BackupForGKEClient()
    request = gke_backup_v1.DeleteRestoreRequest(name='name_value')
    operation = client.delete_restore(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)