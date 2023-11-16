from google.cloud import apigee_registry_v1

def sample_delete_api_spec():
    if False:
        for i in range(10):
            print('nop')
    client = apigee_registry_v1.RegistryClient()
    request = apigee_registry_v1.DeleteApiSpecRequest(name='name_value')
    client.delete_api_spec(request=request)