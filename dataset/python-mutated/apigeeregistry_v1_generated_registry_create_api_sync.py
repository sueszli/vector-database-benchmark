from google.cloud import apigee_registry_v1

def sample_create_api():
    if False:
        return 10
    client = apigee_registry_v1.RegistryClient()
    request = apigee_registry_v1.CreateApiRequest(parent='parent_value', api_id='api_id_value')
    response = client.create_api(request=request)
    print(response)