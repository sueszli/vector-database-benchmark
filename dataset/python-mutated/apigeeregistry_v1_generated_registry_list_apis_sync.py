from google.cloud import apigee_registry_v1

def sample_list_apis():
    if False:
        return 10
    client = apigee_registry_v1.RegistryClient()
    request = apigee_registry_v1.ListApisRequest(parent='parent_value')
    page_result = client.list_apis(request=request)
    for response in page_result:
        print(response)