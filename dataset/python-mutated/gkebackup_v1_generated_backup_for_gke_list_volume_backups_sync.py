from google.cloud import gke_backup_v1

def sample_list_volume_backups():
    if False:
        for i in range(10):
            print('nop')
    client = gke_backup_v1.BackupForGKEClient()
    request = gke_backup_v1.ListVolumeBackupsRequest(parent='parent_value')
    page_result = client.list_volume_backups(request=request)
    for response in page_result:
        print(response)