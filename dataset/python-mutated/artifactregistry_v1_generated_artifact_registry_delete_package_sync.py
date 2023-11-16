from google.cloud import artifactregistry_v1

def sample_delete_package():
    if False:
        for i in range(10):
            print('nop')
    client = artifactregistry_v1.ArtifactRegistryClient()
    request = artifactregistry_v1.DeletePackageRequest(name='name_value')
    operation = client.delete_package(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)