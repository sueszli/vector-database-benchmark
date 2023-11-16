from google.cloud import apigee_registry_v1

def sample_get_instance():
    if False:
        while True:
            i = 10
    client = apigee_registry_v1.ProvisioningClient()
    request = apigee_registry_v1.GetInstanceRequest(name='name_value')
    response = client.get_instance(request=request)
    print(response)