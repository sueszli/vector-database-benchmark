from google.cloud import filestore_v1

def sample_get_backup():
    if False:
        print('Hello World!')
    client = filestore_v1.CloudFilestoreManagerClient()
    request = filestore_v1.GetBackupRequest(name='name_value')
    response = client.get_backup(request=request)
    print(response)