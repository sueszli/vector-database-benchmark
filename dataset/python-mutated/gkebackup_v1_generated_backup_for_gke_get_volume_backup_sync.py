from google.cloud import gke_backup_v1

def sample_get_volume_backup():
    if False:
        return 10
    client = gke_backup_v1.BackupForGKEClient()
    request = gke_backup_v1.GetVolumeBackupRequest(name='name_value')
    response = client.get_volume_backup(request=request)
    print(response)