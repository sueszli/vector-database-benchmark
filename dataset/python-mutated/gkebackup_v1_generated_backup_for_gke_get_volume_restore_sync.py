from google.cloud import gke_backup_v1

def sample_get_volume_restore():
    if False:
        while True:
            i = 10
    client = gke_backup_v1.BackupForGKEClient()
    request = gke_backup_v1.GetVolumeRestoreRequest(name='name_value')
    response = client.get_volume_restore(request=request)
    print(response)