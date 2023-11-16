from google.cloud import gke_backup_v1

def sample_get_restore():
    if False:
        for i in range(10):
            print('nop')
    client = gke_backup_v1.BackupForGKEClient()
    request = gke_backup_v1.GetRestoreRequest(name='name_value')
    response = client.get_restore(request=request)
    print(response)