from google.cloud import filestore_v1

def sample_update_backup():
    if False:
        print('Hello World!')
    client = filestore_v1.CloudFilestoreManagerClient()
    request = filestore_v1.UpdateBackupRequest()
    operation = client.update_backup(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)