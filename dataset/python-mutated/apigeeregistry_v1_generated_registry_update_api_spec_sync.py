from google.cloud import apigee_registry_v1

def sample_update_api_spec():
    if False:
        return 10
    client = apigee_registry_v1.RegistryClient()
    request = apigee_registry_v1.UpdateApiSpecRequest()
    response = client.update_api_spec(request=request)
    print(response)