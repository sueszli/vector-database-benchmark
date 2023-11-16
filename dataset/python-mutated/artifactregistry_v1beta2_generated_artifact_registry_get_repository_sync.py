from google.cloud import artifactregistry_v1beta2

def sample_get_repository():
    if False:
        while True:
            i = 10
    client = artifactregistry_v1beta2.ArtifactRegistryClient()
    request = artifactregistry_v1beta2.GetRepositoryRequest(name='name_value')
    response = client.get_repository(request=request)
    print(response)