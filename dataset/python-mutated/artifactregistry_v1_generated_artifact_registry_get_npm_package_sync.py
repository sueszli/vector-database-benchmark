from google.cloud import artifactregistry_v1

def sample_get_npm_package():
    if False:
        print('Hello World!')
    client = artifactregistry_v1.ArtifactRegistryClient()
    request = artifactregistry_v1.GetNpmPackageRequest(name='name_value')
    response = client.get_npm_package(request=request)
    print(response)