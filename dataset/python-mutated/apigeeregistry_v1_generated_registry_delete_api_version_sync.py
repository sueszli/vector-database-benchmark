from google.cloud import apigee_registry_v1

def sample_delete_api_version():
    if False:
        while True:
            i = 10
    client = apigee_registry_v1.RegistryClient()
    request = apigee_registry_v1.DeleteApiVersionRequest(name='name_value')
    client.delete_api_version(request=request)