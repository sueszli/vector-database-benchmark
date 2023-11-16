from google.cloud import artifactregistry_v1beta2

def sample_delete_version():
    if False:
        i = 10
        return i + 15
    client = artifactregistry_v1beta2.ArtifactRegistryClient()
    request = artifactregistry_v1beta2.DeleteVersionRequest()
    operation = client.delete_version(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)