from google.cloud import artifactregistry_v1beta2

def sample_list_tags():
    if False:
        i = 10
        return i + 15
    client = artifactregistry_v1beta2.ArtifactRegistryClient()
    request = artifactregistry_v1beta2.ListTagsRequest()
    page_result = client.list_tags(request=request)
    for response in page_result:
        print(response)