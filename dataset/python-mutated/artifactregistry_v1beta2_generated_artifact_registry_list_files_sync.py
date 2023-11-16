from google.cloud import artifactregistry_v1beta2

def sample_list_files():
    if False:
        print('Hello World!')
    client = artifactregistry_v1beta2.ArtifactRegistryClient()
    request = artifactregistry_v1beta2.ListFilesRequest()
    page_result = client.list_files(request=request)
    for response in page_result:
        print(response)