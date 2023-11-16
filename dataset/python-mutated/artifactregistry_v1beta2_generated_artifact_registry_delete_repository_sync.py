from google.cloud import artifactregistry_v1beta2

def sample_delete_repository():
    if False:
        for i in range(10):
            print('nop')
    client = artifactregistry_v1beta2.ArtifactRegistryClient()
    request = artifactregistry_v1beta2.DeleteRepositoryRequest(name='name_value')
    operation = client.delete_repository(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)