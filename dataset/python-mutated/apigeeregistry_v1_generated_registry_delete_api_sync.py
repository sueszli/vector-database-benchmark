from google.cloud import apigee_registry_v1

def sample_delete_api():
    if False:
        return 10
    client = apigee_registry_v1.RegistryClient()
    request = apigee_registry_v1.DeleteApiRequest(name='name_value')
    client.delete_api(request=request)