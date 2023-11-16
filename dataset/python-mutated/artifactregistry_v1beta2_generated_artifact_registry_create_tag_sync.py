from google.cloud import artifactregistry_v1beta2

def sample_create_tag():
    if False:
        print('Hello World!')
    client = artifactregistry_v1beta2.ArtifactRegistryClient()
    request = artifactregistry_v1beta2.CreateTagRequest()
    response = client.create_tag(request=request)
    print(response)