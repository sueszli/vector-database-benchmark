from google.cloud import apigee_registry_v1

def sample_list_artifacts():
    if False:
        return 10
    client = apigee_registry_v1.RegistryClient()
    request = apigee_registry_v1.ListArtifactsRequest(parent='parent_value')
    page_result = client.list_artifacts(request=request)
    for response in page_result:
        print(response)