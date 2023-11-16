from google.cloud import apigee_registry_v1

def sample_delete_artifact():
    if False:
        return 10
    client = apigee_registry_v1.RegistryClient()
    request = apigee_registry_v1.DeleteArtifactRequest(name='name_value')
    client.delete_artifact(request=request)