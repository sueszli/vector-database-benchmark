from google.cloud import apigee_registry_v1

def sample_get_api():
    if False:
        return 10
    client = apigee_registry_v1.RegistryClient()
    request = apigee_registry_v1.GetApiRequest(name='name_value')
    response = client.get_api(request=request)
    print(response)