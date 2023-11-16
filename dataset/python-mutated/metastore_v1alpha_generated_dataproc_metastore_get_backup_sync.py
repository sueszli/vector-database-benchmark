from google.cloud import metastore_v1alpha

def sample_get_backup():
    if False:
        return 10
    client = metastore_v1alpha.DataprocMetastoreClient()
    request = metastore_v1alpha.GetBackupRequest(name='name_value')
    response = client.get_backup(request=request)
    print(response)