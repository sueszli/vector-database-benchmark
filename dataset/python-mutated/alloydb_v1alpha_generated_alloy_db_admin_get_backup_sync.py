from google.cloud import alloydb_v1alpha

def sample_get_backup():
    if False:
        print('Hello World!')
    client = alloydb_v1alpha.AlloyDBAdminClient()
    request = alloydb_v1alpha.GetBackupRequest(name='name_value')
    response = client.get_backup(request=request)
    print(response)