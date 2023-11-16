from google.cloud import apigee_registry_v1

def sample_get_api_version():
    if False:
        while True:
            i = 10
    client = apigee_registry_v1.RegistryClient()
    request = apigee_registry_v1.GetApiVersionRequest(name='name_value')
    response = client.get_api_version(request=request)
    print(response)