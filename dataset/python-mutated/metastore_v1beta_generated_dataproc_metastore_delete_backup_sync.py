from google.cloud import metastore_v1beta

def sample_delete_backup():
    if False:
        for i in range(10):
            print('nop')
    client = metastore_v1beta.DataprocMetastoreClient()
    request = metastore_v1beta.DeleteBackupRequest(name='name_value')
    operation = client.delete_backup(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)