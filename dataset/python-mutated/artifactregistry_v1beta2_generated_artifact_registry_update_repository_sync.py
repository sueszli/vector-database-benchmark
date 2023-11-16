from google.cloud import artifactregistry_v1beta2

def sample_update_repository():
    if False:
        while True:
            i = 10
    client = artifactregistry_v1beta2.ArtifactRegistryClient()
    request = artifactregistry_v1beta2.UpdateRepositoryRequest()
    response = client.update_repository(request=request)
    print(response)