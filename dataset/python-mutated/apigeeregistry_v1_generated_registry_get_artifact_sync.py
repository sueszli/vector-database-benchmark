from google.cloud import apigee_registry_v1

def sample_get_artifact():
    if False:
        i = 10
        return i + 15
    client = apigee_registry_v1.RegistryClient()
    request = apigee_registry_v1.GetArtifactRequest(name='name_value')
    response = client.get_artifact(request=request)
    print(response)