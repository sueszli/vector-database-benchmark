from google.cloud import metastore_v1

def sample_delete_backup():
    if False:
        return 10
    client = metastore_v1.DataprocMetastoreClient()
    request = metastore_v1.DeleteBackupRequest(name='name_value')
    operation = client.delete_backup(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)