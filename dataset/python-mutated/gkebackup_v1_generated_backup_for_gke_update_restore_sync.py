from google.cloud import gke_backup_v1

def sample_update_restore():
    if False:
        while True:
            i = 10
    client = gke_backup_v1.BackupForGKEClient()
    restore = gke_backup_v1.Restore()
    restore.backup = 'backup_value'
    request = gke_backup_v1.UpdateRestoreRequest(restore=restore)
    operation = client.update_restore(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)