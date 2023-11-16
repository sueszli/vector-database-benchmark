from google.cloud import metastore_v1alpha

def sample_create_backup():
    if False:
        while True:
            i = 10
    client = metastore_v1alpha.DataprocMetastoreClient()
    request = metastore_v1alpha.CreateBackupRequest(parent='parent_value', backup_id='backup_id_value')
    operation = client.create_backup(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)