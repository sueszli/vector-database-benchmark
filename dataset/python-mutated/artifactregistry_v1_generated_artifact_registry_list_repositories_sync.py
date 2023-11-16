from google.cloud import artifactregistry_v1

def sample_list_repositories():
    if False:
        for i in range(10):
            print('nop')
    client = artifactregistry_v1.ArtifactRegistryClient()
    request = artifactregistry_v1.ListRepositoriesRequest(parent='parent_value')
    page_result = client.list_repositories(request=request)
    for response in page_result:
        print(response)