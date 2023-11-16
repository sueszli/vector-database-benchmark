from google.cloud import alloydb_v1

def sample_create_backup():
    if False:
        while True:
            i = 10
    client = alloydb_v1.AlloyDBAdminClient()
    backup = alloydb_v1.Backup()
    backup.cluster_name = 'cluster_name_value'
    request = alloydb_v1.CreateBackupRequest(parent='parent_value', backup_id='backup_id_value', backup=backup)
    operation = client.create_backup(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)