from google.cloud import apigee_registry_v1

def sample_update_api_deployment():
    if False:
        while True:
            i = 10
    client = apigee_registry_v1.RegistryClient()
    request = apigee_registry_v1.UpdateApiDeploymentRequest()
    response = client.update_api_deployment(request=request)
    print(response)