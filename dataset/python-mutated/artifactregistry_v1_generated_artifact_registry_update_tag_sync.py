from google.cloud import artifactregistry_v1

def sample_update_tag():
    if False:
        while True:
            i = 10
    client = artifactregistry_v1.ArtifactRegistryClient()
    request = artifactregistry_v1.UpdateTagRequest()
    response = client.update_tag(request=request)
    print(response)