from google.cloud import artifactregistry_v1beta2

def sample_get_version():
    if False:
        while True:
            i = 10
    client = artifactregistry_v1beta2.ArtifactRegistryClient()
    request = artifactregistry_v1beta2.GetVersionRequest()
    response = client.get_version(request=request)
    print(response)