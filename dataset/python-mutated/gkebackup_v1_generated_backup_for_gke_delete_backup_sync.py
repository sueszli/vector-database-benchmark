from google.cloud import gke_backup_v1

def sample_delete_backup():
    if False:
        print('Hello World!')
    client = gke_backup_v1.BackupForGKEClient()
    request = gke_backup_v1.DeleteBackupRequest(name='name_value')
    operation = client.delete_backup(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)