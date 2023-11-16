from google.cloud import apigee_registry_v1

def sample_list_api_deployment_revisions():
    if False:
        while True:
            i = 10
    client = apigee_registry_v1.RegistryClient()
    request = apigee_registry_v1.ListApiDeploymentRevisionsRequest(name='name_value')
    page_result = client.list_api_deployment_revisions(request=request)
    for response in page_result:
        print(response)