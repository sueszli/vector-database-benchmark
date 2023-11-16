from google.cloud import artifactregistry_v1

def sample_get_file():
    if False:
        i = 10
        return i + 15
    client = artifactregistry_v1.ArtifactRegistryClient()
    request = artifactregistry_v1.GetFileRequest(name='name_value')
    response = client.get_file(request=request)
    print(response)