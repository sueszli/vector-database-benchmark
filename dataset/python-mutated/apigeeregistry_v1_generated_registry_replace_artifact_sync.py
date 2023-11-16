from google.cloud import apigee_registry_v1

def sample_replace_artifact():
    if False:
        return 10
    client = apigee_registry_v1.RegistryClient()
    request = apigee_registry_v1.ReplaceArtifactRequest()
    response = client.replace_artifact(request=request)
    print(response)