from google.cloud import artifactregistry_v1

def sample_list_versions():
    if False:
        print('Hello World!')
    client = artifactregistry_v1.ArtifactRegistryClient()
    request = artifactregistry_v1.ListVersionsRequest()
    page_result = client.list_versions(request=request)
    for response in page_result:
        print(response)