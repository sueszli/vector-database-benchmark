from google.cloud import gke_backup_v1

def sample_get_backup():
    if False:
        i = 10
        return i + 15
    client = gke_backup_v1.BackupForGKEClient()
    request = gke_backup_v1.GetBackupRequest(name='name_value')
    response = client.get_backup(request=request)
    print(response)