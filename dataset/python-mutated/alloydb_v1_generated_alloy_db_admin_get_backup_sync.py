from google.cloud import alloydb_v1

def sample_get_backup():
    if False:
        print('Hello World!')
    client = alloydb_v1.AlloyDBAdminClient()
    request = alloydb_v1.GetBackupRequest(name='name_value')
    response = client.get_backup(request=request)
    print(response)