from google.cloud import alloydb_v1alpha

def sample_create_backup():
    if False:
        print('Hello World!')
    client = alloydb_v1alpha.AlloyDBAdminClient()
    backup = alloydb_v1alpha.Backup()
    backup.cluster_name = 'cluster_name_value'
    request = alloydb_v1alpha.CreateBackupRequest(parent='parent_value', backup_id='backup_id_value', backup=backup)
    operation = client.create_backup(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)