from google.cloud import securesourcemanager_v1

def sample_create_instance():
    if False:
        return 10
    client = securesourcemanager_v1.SecureSourceManagerClient()
    request = securesourcemanager_v1.CreateInstanceRequest(parent='parent_value', instance_id='instance_id_value')
    operation = client.create_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)