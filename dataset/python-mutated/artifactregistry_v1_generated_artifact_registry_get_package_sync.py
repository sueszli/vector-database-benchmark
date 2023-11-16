from google.cloud import artifactregistry_v1

def sample_get_package():
    if False:
        for i in range(10):
            print('nop')
    client = artifactregistry_v1.ArtifactRegistryClient()
    request = artifactregistry_v1.GetPackageRequest(name='name_value')
    response = client.get_package(request=request)
    print(response)