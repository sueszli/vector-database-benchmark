from google.cloud import metastore_v1alpha

def sample_delete_backup():
    if False:
        i = 10
        return i + 15
    client = metastore_v1alpha.DataprocMetastoreClient()
    request = metastore_v1alpha.DeleteBackupRequest(name='name_value')
    operation = client.delete_backup(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)