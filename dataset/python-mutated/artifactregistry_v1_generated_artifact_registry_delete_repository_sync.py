from google.cloud import artifactregistry_v1

def sample_delete_repository():
    if False:
        i = 10
        return i + 15
    client = artifactregistry_v1.ArtifactRegistryClient()
    request = artifactregistry_v1.DeleteRepositoryRequest(name='name_value')
    operation = client.delete_repository(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)