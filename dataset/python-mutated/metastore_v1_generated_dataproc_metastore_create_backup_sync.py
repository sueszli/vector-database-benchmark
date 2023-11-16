from google.cloud import metastore_v1

def sample_create_backup():
    if False:
        print('Hello World!')
    client = metastore_v1.DataprocMetastoreClient()
    request = metastore_v1.CreateBackupRequest(parent='parent_value', backup_id='backup_id_value')
    operation = client.create_backup(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)