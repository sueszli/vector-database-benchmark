from google.cloud import artifactregistry_v1beta2

def sample_delete_package():
    if False:
        while True:
            i = 10
    client = artifactregistry_v1beta2.ArtifactRegistryClient()
    request = artifactregistry_v1beta2.DeletePackageRequest()
    operation = client.delete_package(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)