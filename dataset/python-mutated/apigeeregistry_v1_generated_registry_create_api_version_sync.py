from google.cloud import apigee_registry_v1

def sample_create_api_version():
    if False:
        while True:
            i = 10
    client = apigee_registry_v1.RegistryClient()
    request = apigee_registry_v1.CreateApiVersionRequest(parent='parent_value', api_version_id='api_version_id_value')
    response = client.create_api_version(request=request)
    print(response)