from google.cloud import artifactregistry_v1beta2

def sample_get_tag():
    if False:
        while True:
            i = 10
    client = artifactregistry_v1beta2.ArtifactRegistryClient()
    request = artifactregistry_v1beta2.GetTagRequest()
    response = client.get_tag(request=request)
    print(response)