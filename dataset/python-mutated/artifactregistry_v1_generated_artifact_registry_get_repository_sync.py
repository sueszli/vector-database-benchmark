from google.cloud import artifactregistry_v1

def sample_get_repository():
    if False:
        i = 10
        return i + 15
    client = artifactregistry_v1.ArtifactRegistryClient()
    request = artifactregistry_v1.GetRepositoryRequest(name='name_value')
    response = client.get_repository(request=request)
    print(response)