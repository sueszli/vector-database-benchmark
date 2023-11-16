from google.cloud import artifactregistry_v1beta2

def sample_update_tag():
    if False:
        i = 10
        return i + 15
    client = artifactregistry_v1beta2.ArtifactRegistryClient()
    request = artifactregistry_v1beta2.UpdateTagRequest()
    response = client.update_tag(request=request)
    print(response)