from google.cloud import artifactregistry_v1beta2

def sample_create_repository():
    if False:
        while True:
            i = 10
    client = artifactregistry_v1beta2.ArtifactRegistryClient()
    request = artifactregistry_v1beta2.CreateRepositoryRequest(parent='parent_value')
    operation = client.create_repository(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)