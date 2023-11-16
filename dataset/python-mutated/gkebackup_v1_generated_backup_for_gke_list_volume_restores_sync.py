from google.cloud import gke_backup_v1

def sample_list_volume_restores():
    if False:
        return 10
    client = gke_backup_v1.BackupForGKEClient()
    request = gke_backup_v1.ListVolumeRestoresRequest(parent='parent_value')
    page_result = client.list_volume_restores(request=request)
    for response in page_result:
        print(response)