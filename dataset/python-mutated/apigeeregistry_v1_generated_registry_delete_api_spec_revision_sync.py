from google.cloud import apigee_registry_v1

def sample_delete_api_spec_revision():
    if False:
        i = 10
        return i + 15
    client = apigee_registry_v1.RegistryClient()
    request = apigee_registry_v1.DeleteApiSpecRevisionRequest(name='name_value')
    response = client.delete_api_spec_revision(request=request)
    print(response)