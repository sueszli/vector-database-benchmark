from google.cloud import gke_backup_v1

def sample_create_restore():
    if False:
        while True:
            i = 10
    client = gke_backup_v1.BackupForGKEClient()
    restore = gke_backup_v1.Restore()
    restore.backup = 'backup_value'
    request = gke_backup_v1.CreateRestoreRequest(parent='parent_value', restore=restore, restore_id='restore_id_value')
    operation = client.create_restore(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)