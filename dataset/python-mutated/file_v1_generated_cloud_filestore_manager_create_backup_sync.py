from google.cloud import filestore_v1

def sample_create_backup():
    if False:
        i = 10
        return i + 15
    client = filestore_v1.CloudFilestoreManagerClient()
    request = filestore_v1.CreateBackupRequest(parent='parent_value', backup_id='backup_id_value')
    operation = client.create_backup(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)