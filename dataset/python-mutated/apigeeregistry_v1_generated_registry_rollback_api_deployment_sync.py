from google.cloud import apigee_registry_v1

def sample_rollback_api_deployment():
    if False:
        i = 10
        return i + 15
    client = apigee_registry_v1.RegistryClient()
    request = apigee_registry_v1.RollbackApiDeploymentRequest(name='name_value', revision_id='revision_id_value')
    response = client.rollback_api_deployment(request=request)
    print(response)