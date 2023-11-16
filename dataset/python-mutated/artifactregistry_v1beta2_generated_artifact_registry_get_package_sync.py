from google.cloud import artifactregistry_v1beta2

def sample_get_package():
    if False:
        return 10
    client = artifactregistry_v1beta2.ArtifactRegistryClient()
    request = artifactregistry_v1beta2.GetPackageRequest()
    response = client.get_package(request=request)
    print(response)