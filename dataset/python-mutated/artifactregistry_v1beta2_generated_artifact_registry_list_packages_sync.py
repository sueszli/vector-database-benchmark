from google.cloud import artifactregistry_v1beta2

def sample_list_packages():
    if False:
        for i in range(10):
            print('nop')
    client = artifactregistry_v1beta2.ArtifactRegistryClient()
    request = artifactregistry_v1beta2.ListPackagesRequest()
    page_result = client.list_packages(request=request)
    for response in page_result:
        print(response)