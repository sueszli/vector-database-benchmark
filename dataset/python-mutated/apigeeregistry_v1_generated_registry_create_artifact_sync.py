from google.cloud import apigee_registry_v1

def sample_create_artifact():
    if False:
        return 10
    client = apigee_registry_v1.RegistryClient()
    request = apigee_registry_v1.CreateArtifactRequest(parent='parent_value', artifact_id='artifact_id_value')
    response = client.create_artifact(request=request)
    print(response)