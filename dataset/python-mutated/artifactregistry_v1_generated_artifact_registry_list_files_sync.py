from google.cloud import artifactregistry_v1

def sample_list_files():
    if False:
        for i in range(10):
            print('nop')
    client = artifactregistry_v1.ArtifactRegistryClient()
    request = artifactregistry_v1.ListFilesRequest(parent='parent_value')
    page_result = client.list_files(request=request)
    for response in page_result:
        print(response)