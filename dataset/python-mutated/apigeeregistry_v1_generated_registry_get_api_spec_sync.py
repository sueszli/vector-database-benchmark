from google.cloud import apigee_registry_v1

def sample_get_api_spec():
    if False:
        return 10
    client = apigee_registry_v1.RegistryClient()
    request = apigee_registry_v1.GetApiSpecRequest(name='name_value')
    response = client.get_api_spec(request=request)
    print(response)