from google.cloud import apigee_registry_v1

def sample_list_api_specs():
    if False:
        print('Hello World!')
    client = apigee_registry_v1.RegistryClient()
    request = apigee_registry_v1.ListApiSpecsRequest(parent='parent_value')
    page_result = client.list_api_specs(request=request)
    for response in page_result:
        print(response)