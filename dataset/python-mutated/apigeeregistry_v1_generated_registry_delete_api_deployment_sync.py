from google.cloud import apigee_registry_v1

def sample_delete_api_deployment():
    if False:
        print('Hello World!')
    client = apigee_registry_v1.RegistryClient()
    request = apigee_registry_v1.DeleteApiDeploymentRequest(name='name_value')
    client.delete_api_deployment(request=request)