from google.cloud import apigee_registry_v1

def sample_get_artifact_contents():
    if False:
        return 10
    client = apigee_registry_v1.RegistryClient()
    request = apigee_registry_v1.GetArtifactContentsRequest(name='name_value')
    response = client.get_artifact_contents(request=request)
    print(response)