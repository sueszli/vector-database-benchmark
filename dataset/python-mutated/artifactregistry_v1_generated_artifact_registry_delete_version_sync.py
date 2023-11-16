from google.cloud import artifactregistry_v1

def sample_delete_version():
    if False:
        for i in range(10):
            print('nop')
    client = artifactregistry_v1.ArtifactRegistryClient()
    request = artifactregistry_v1.DeleteVersionRequest()
    operation = client.delete_version(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)