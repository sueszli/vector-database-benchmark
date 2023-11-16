from google.cloud import metastore_v1

def sample_get_backup():
    if False:
        return 10
    client = metastore_v1.DataprocMetastoreClient()
    request = metastore_v1.GetBackupRequest(name='name_value')
    response = client.get_backup(request=request)
    print(response)