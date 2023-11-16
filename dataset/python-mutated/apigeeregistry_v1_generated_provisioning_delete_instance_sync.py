from google.cloud import apigee_registry_v1

def sample_delete_instance():
    if False:
        i = 10
        return i + 15
    client = apigee_registry_v1.ProvisioningClient()
    request = apigee_registry_v1.DeleteInstanceRequest(name='name_value')
    operation = client.delete_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)