from google.cloud import alloydb_v1beta

def sample_get_backup():
    if False:
        while True:
            i = 10
    client = alloydb_v1beta.AlloyDBAdminClient()
    request = alloydb_v1beta.GetBackupRequest(name='name_value')
    response = client.get_backup(request=request)
    print(response)