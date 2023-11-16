from google.cloud import gke_backup_v1

def sample_update_backup():
    if False:
        return 10
    client = gke_backup_v1.BackupForGKEClient()
    backup = gke_backup_v1.Backup()
    backup.all_namespaces = True
    request = gke_backup_v1.UpdateBackupRequest(backup=backup)
    operation = client.update_backup(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)