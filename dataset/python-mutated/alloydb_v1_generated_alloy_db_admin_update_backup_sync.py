from google.cloud import alloydb_v1

def sample_update_backup():
    if False:
        print('Hello World!')
    client = alloydb_v1.AlloyDBAdminClient()
    backup = alloydb_v1.Backup()
    backup.cluster_name = 'cluster_name_value'
    request = alloydb_v1.UpdateBackupRequest(backup=backup)
    operation = client.update_backup(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)