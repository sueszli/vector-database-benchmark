from google.cloud import filestore_v1

def sample_restore_instance():
    if False:
        print('Hello World!')
    client = filestore_v1.CloudFilestoreManagerClient()
    request = filestore_v1.RestoreInstanceRequest(source_backup='source_backup_value', name='name_value', file_share='file_share_value')
    operation = client.restore_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)