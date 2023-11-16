from google.cloud import artifactregistry_v1

def sample_list_packages():
    if False:
        while True:
            i = 10
    client = artifactregistry_v1.ArtifactRegistryClient()
    request = artifactregistry_v1.ListPackagesRequest(parent='parent_value')
    page_result = client.list_packages(request=request)
    for response in page_result:
        print(response)