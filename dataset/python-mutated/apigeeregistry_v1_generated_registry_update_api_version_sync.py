from google.cloud import apigee_registry_v1

def sample_update_api_version():
    if False:
        i = 10
        return i + 15
    client = apigee_registry_v1.RegistryClient()
    request = apigee_registry_v1.UpdateApiVersionRequest()
    response = client.update_api_version(request=request)
    print(response)