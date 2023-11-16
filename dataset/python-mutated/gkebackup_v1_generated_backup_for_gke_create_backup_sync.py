from google.cloud import gke_backup_v1

def sample_create_backup():
    if False:
        while True:
            i = 10
    client = gke_backup_v1.BackupForGKEClient()
    request = gke_backup_v1.CreateBackupRequest(parent='parent_value')
    operation = client.create_backup(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)