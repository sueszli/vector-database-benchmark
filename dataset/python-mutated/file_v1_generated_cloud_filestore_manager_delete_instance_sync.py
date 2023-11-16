from google.cloud import filestore_v1

def sample_delete_instance():
    if False:
        for i in range(10):
            print('nop')
    client = filestore_v1.CloudFilestoreManagerClient()
    request = filestore_v1.DeleteInstanceRequest(name='name_value')
    operation = client.delete_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)