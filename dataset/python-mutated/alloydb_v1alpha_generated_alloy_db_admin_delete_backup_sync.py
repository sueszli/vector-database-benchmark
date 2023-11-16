from google.cloud import alloydb_v1alpha

def sample_delete_backup():
    if False:
        i = 10
        return i + 15
    client = alloydb_v1alpha.AlloyDBAdminClient()
    request = alloydb_v1alpha.DeleteBackupRequest(name='name_value')
    operation = client.delete_backup(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)