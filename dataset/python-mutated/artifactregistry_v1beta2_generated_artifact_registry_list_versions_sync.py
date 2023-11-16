from google.cloud import artifactregistry_v1beta2

def sample_list_versions():
    if False:
        i = 10
        return i + 15
    client = artifactregistry_v1beta2.ArtifactRegistryClient()
    request = artifactregistry_v1beta2.ListVersionsRequest()
    page_result = client.list_versions(request=request)
    for response in page_result:
        print(response)