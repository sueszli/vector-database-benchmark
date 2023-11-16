from google.cloud import apigee_registry_v1

def sample_create_api_deployment():
    if False:
        for i in range(10):
            print('nop')
    client = apigee_registry_v1.RegistryClient()
    request = apigee_registry_v1.CreateApiDeploymentRequest(parent='parent_value', api_deployment_id='api_deployment_id_value')
    response = client.create_api_deployment(request=request)
    print(response)