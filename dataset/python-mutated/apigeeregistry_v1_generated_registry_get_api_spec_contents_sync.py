from google.cloud import apigee_registry_v1

def sample_get_api_spec_contents():
    if False:
        return 10
    client = apigee_registry_v1.RegistryClient()
    request = apigee_registry_v1.GetApiSpecContentsRequest(name='name_value')
    response = client.get_api_spec_contents(request=request)
    print(response)