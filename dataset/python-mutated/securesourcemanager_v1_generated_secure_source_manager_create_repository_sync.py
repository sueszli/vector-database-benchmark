from google.cloud import securesourcemanager_v1

def sample_create_repository():
    if False:
        i = 10
        return i + 15
    client = securesourcemanager_v1.SecureSourceManagerClient()
    request = securesourcemanager_v1.CreateRepositoryRequest(parent='parent_value', repository_id='repository_id_value')
    operation = client.create_repository(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)