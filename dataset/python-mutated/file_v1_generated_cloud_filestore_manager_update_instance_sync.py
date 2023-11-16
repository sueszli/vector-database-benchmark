from google.cloud import filestore_v1

def sample_update_instance():
    if False:
        return 10
    client = filestore_v1.CloudFilestoreManagerClient()
    request = filestore_v1.UpdateInstanceRequest()
    operation = client.update_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)