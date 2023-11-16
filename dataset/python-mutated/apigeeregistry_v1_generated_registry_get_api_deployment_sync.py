from google.cloud import apigee_registry_v1

def sample_get_api_deployment():
    if False:
        return 10
    client = apigee_registry_v1.RegistryClient()
    request = apigee_registry_v1.GetApiDeploymentRequest(name='name_value')
    response = client.get_api_deployment(request=request)
    print(response)