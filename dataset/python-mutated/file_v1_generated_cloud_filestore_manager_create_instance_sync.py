from google.cloud import filestore_v1

def sample_create_instance():
    if False:
        i = 10
        return i + 15
    client = filestore_v1.CloudFilestoreManagerClient()
    request = filestore_v1.CreateInstanceRequest(parent='parent_value', instance_id='instance_id_value')
    operation = client.create_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)