from google.cloud import metastore_v1beta

def sample_get_backup():
    if False:
        i = 10
        return i + 15
    client = metastore_v1beta.DataprocMetastoreClient()
    request = metastore_v1beta.GetBackupRequest(name='name_value')
    response = client.get_backup(request=request)
    print(response)