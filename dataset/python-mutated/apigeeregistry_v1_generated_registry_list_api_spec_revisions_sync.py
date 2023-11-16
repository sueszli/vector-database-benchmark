from google.cloud import apigee_registry_v1

def sample_list_api_spec_revisions():
    if False:
        for i in range(10):
            print('nop')
    client = apigee_registry_v1.RegistryClient()
    request = apigee_registry_v1.ListApiSpecRevisionsRequest(name='name_value')
    page_result = client.list_api_spec_revisions(request=request)
    for response in page_result:
        print(response)