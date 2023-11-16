from google.cloud import apigee_registry_v1

def sample_tag_api_deployment_revision():
    if False:
        for i in range(10):
            print('nop')
    client = apigee_registry_v1.RegistryClient()
    request = apigee_registry_v1.TagApiDeploymentRevisionRequest(name='name_value', tag='tag_value')
    response = client.tag_api_deployment_revision(request=request)
    print(response)