from google.cloud import artifactregistry_v1

def sample_create_repository():
    if False:
        for i in range(10):
            print('nop')
    client = artifactregistry_v1.ArtifactRegistryClient()
    request = artifactregistry_v1.CreateRepositoryRequest(parent='parent_value', repository_id='repository_id_value')
    operation = client.create_repository(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)