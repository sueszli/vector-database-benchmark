from google.cloud import securesourcemanager_v1

def sample_delete_repository():
    if False:
        while True:
            i = 10
    client = securesourcemanager_v1.SecureSourceManagerClient()
    request = securesourcemanager_v1.DeleteRepositoryRequest(name='name_value')
    operation = client.delete_repository(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)