from google.cloud import artifactregistry_v1

def sample_get_tag():
    if False:
        print('Hello World!')
    client = artifactregistry_v1.ArtifactRegistryClient()
    request = artifactregistry_v1.GetTagRequest()
    response = client.get_tag(request=request)
    print(response)