from google.cloud import apigee_registry_v1

def sample_delete_api_deployment_revision():
    if False:
        for i in range(10):
            print('nop')
    client = apigee_registry_v1.RegistryClient()
    request = apigee_registry_v1.DeleteApiDeploymentRevisionRequest(name='name_value')
    response = client.delete_api_deployment_revision(request=request)
    print(response)