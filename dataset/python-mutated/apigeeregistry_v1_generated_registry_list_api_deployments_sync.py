from google.cloud import apigee_registry_v1

def sample_list_api_deployments():
    if False:
        i = 10
        return i + 15
    client = apigee_registry_v1.RegistryClient()
    request = apigee_registry_v1.ListApiDeploymentsRequest(parent='parent_value')
    page_result = client.list_api_deployments(request=request)
    for response in page_result:
        print(response)