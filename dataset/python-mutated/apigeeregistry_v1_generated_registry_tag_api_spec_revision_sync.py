from google.cloud import apigee_registry_v1

def sample_tag_api_spec_revision():
    if False:
        i = 10
        return i + 15
    client = apigee_registry_v1.RegistryClient()
    request = apigee_registry_v1.TagApiSpecRevisionRequest(name='name_value', tag='tag_value')
    response = client.tag_api_spec_revision(request=request)
    print(response)