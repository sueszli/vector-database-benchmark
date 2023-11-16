from google.cloud import apigee_registry_v1

def sample_create_api_spec():
    if False:
        i = 10
        return i + 15
    client = apigee_registry_v1.RegistryClient()
    request = apigee_registry_v1.CreateApiSpecRequest(parent='parent_value', api_spec_id='api_spec_id_value')
    response = client.create_api_spec(request=request)
    print(response)