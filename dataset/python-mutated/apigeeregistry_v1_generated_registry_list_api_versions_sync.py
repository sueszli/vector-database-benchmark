from google.cloud import apigee_registry_v1

def sample_list_api_versions():
    if False:
        return 10
    client = apigee_registry_v1.RegistryClient()
    request = apigee_registry_v1.ListApiVersionsRequest(parent='parent_value')
    page_result = client.list_api_versions(request=request)
    for response in page_result:
        print(response)