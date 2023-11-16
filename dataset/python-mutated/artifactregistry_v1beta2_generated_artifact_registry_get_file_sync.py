from google.cloud import artifactregistry_v1beta2

def sample_get_file():
    if False:
        return 10
    client = artifactregistry_v1beta2.ArtifactRegistryClient()
    request = artifactregistry_v1beta2.GetFileRequest()
    response = client.get_file(request=request)
    print(response)