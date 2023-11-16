from google.cloud import securesourcemanager_v1

def sample_delete_instance():
    if False:
        return 10
    client = securesourcemanager_v1.SecureSourceManagerClient()
    request = securesourcemanager_v1.DeleteInstanceRequest(name='name_value')
    operation = client.delete_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)